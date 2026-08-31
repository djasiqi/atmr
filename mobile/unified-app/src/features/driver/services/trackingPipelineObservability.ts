/**
 * JZ-R1 — instrumentation remote-first (lecture seule).
 * Aucune mutation du pipeline GPS : enregistrement d'horodatages + snapshot heartbeat.
 */

import { AppState, Platform } from "react-native";

import { resolveDeviceRuntimeMetadata } from "../../../core/device/deviceRuntimeMetadata";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import {
  computeLocationFixAgeSeconds,
  computeTaskInvokeAgeSeconds,
  computeWatchCallbackAgeSeconds,
} from "./trackingObservabilityHealth";
import { getTrackingRuntimeSnapshot } from "./trackingRuntime";
import { captureActiveRuntime } from "./trackingRuntimeRegistry";

export const PIPELINE_SNAPSHOT_VERSION = 1;

export const PIPELINE_STALE_SECONDS = 120;

export type PipelineJ3Result = "accepted" | "rejected" | "unknown";

export type PipelineFirstSuspect =
  | "BG_TASK"
  | "J1_HANDLER"
  | "J3_GATE"
  | "ENQUEUE"
  | "FLUSH"
  | "ACK"
  | "UNKNOWN";

export type TrackingPipelineSnapshot = {
  pipeline_snapshot_version: number;
  desired_mode: "mission_live" | "availability_presence" | "off" | null;
  mission_id: number | null;
  tracking_required: boolean;
  is_available: boolean | null;

  bridge_last_fix_age_s: number | null;
  bg_task_last_invoke_age_s: number | null;
  watch_callback_age_s: number | null;

  j1_handler_age_s: number | null;
  j3_accepted_age_s: number | null;
  j3_last_result: PipelineJ3Result;
  j3_last_reject_reason: string | null;

  queue_last_enqueue_age_s: number | null;
  queue_depth: number | null;
  queue_head_age_s: number | null;

  flush_last_attempt_age_s: number | null;
  flush_last_sent_age_s: number | null;

  durable_ack_age_s: number | null;

  owner_present: boolean;
  owner_generation: string | null;
  background_task_registered: boolean | null;

  app_state: string | null;
  platform: string | null;
  app_version: string | null;
  native_build_version: string | null;
  runtime_version: string | null;
  ota_update_id: string | null;

  last_recovery_reason: string | null;
  last_recovery_age_s: number | null;
  recovery_count_15m: number;

  tracking_runtime_age_s: number | null;

  /** Diagnostic only — not FIRST_STOP certified. */
  first_suspect: PipelineFirstSuspect;
};

type BridgeSnapshotLike = {
  missionId: number | null;
  missionStatus?: string | null;
  appState?: string;
  isRunning?: boolean;
  lastSentAt?: string | null;
  lastEnqueuedAt?: string | null;
  lastTransportAttemptAt?: string | null;
  lastAckAt?: string | null;
  lastPersistedAt?: string | null;
  queueDepth?: number | null;
  lastWatchAtMs?: number | null;
  lastFixProducedAtMs?: number | null;
};

let lastJ1HandlerAtMs: number | null = null;
let lastJ3AcceptedAtMs: number | null = null;
let lastJ3Result: PipelineJ3Result = "unknown";
let lastJ3RejectReason: string | null = null;
let lastRecoveryReason: string | null = null;
let lastRecoveryAtMs: number | null = null;
const recoveryTimestampsMs: number[] = [];

function parseIsoAgeSeconds(iso: string | null | undefined, nowMs: number): number | null {
  if (!iso) return null;
  const ts = Date.parse(iso);
  if (!Number.isFinite(ts)) return null;
  const ageMs = nowMs - ts;
  if (ageMs < -120_000) return null;
  if (ageMs < 0) return 0;
  return Math.round(ageMs / 1000);
}

function readBridgeSnapshot(): BridgeSnapshotLike | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./driverTrackingBridge") as typeof import("./driverTrackingBridge");
    if (typeof mod.getDriverTrackingBridgeSnapshot !== "function") return null;
    return mod.getDriverTrackingBridgeSnapshot() as BridgeSnapshotLike;
  } catch {
    return null;
  }
}

function readTrackingStartedAtMs(): number | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./driverTrackingBridge") as {
      getDriverTrackingStartedAtMs?: () => number | null;
    };
    return mod.getDriverTrackingStartedAtMs?.() ?? null;
  } catch {
    return null;
  }
}

function readDriverAvailable(): boolean | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./driverAvailabilityBridge") as {
      getDriverAvailabilityActive?: () => boolean | null;
    };
    return mod.getDriverAvailabilityActive?.() ?? null;
  } catch {
    return null;
  }
}

function readTrackingRequired(_appState: string): boolean {
  if (readDriverAvailable() !== true) return false;
  const bridge = readBridgeSnapshot();
  if (!bridge) return false;
  // Proxy du tracking désiré : bridge actif (EN SERVICE + mode capture attendu).
  return Boolean(bridge.isRunning);
}

function resolveDesiredMode(
  missionId: number | null,
  missionStatus: string | null | undefined
): TrackingPipelineSnapshot["desired_mode"] {
  if (missionId == null) {
    return "availability_presence";
  }
  const status = String(missionStatus ?? "").toUpperCase();
  if (["IN_PROGRESS", "EN_ROUTE", "ARRIVED", "ACCEPTED", "ASSIGNED"].includes(status)) {
    return "mission_live";
  }
  if (status === "COMPLETED" || status === "CANCELLED") {
    return "availability_presence";
  }
  return "mission_live";
}

async function readBackgroundTaskRegistered(): Promise<boolean | null> {
  if (Platform.OS === "web") return null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./backgroundLocationTask") as typeof import("./backgroundLocationTask");
    if (typeof mod.getNativeTaskLifecycleStatus !== "function") return null;
    const lifecycle = await mod.getNativeTaskLifecycleStatus();
    return Boolean(lifecycle?.taskDefined);
  } catch {
    return null;
  }
}

async function readQueueHeadAgeSeconds(nowMs: number): Promise<number | null> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./driverTrackingQueue") as typeof import("./driverTrackingQueue");
    if (typeof mod.driverTrackingQueue?.getSnapshot !== "function") return null;
    const snap = await mod.driverTrackingQueue.getSnapshot();
    if (snap.oldestItemAgeMs == null) return null;
    return Math.round(snap.oldestItemAgeMs / 1000);
  } catch {
    return null;
  }
}

function pruneRecoveryTimestamps(nowMs: number): void {
  const cutoff = nowMs - 15 * 60 * 1000;
  while (recoveryTimestampsMs.length > 0 && recoveryTimestampsMs[0]! < cutoff) {
    recoveryTimestampsMs.shift();
  }
}

export function recordPipelineJ1Handler(nowMs: number = Date.now()): void {
  lastJ1HandlerAtMs = nowMs;
}

export function recordPipelineJ3Decision(input: {
  result: PipelineJ3Result;
  reason?: string | null;
  nowMs?: number;
}): void {
  const nowMs = input.nowMs ?? Date.now();
  lastJ3Result = input.result;
  if (input.result === "accepted") {
    lastJ3AcceptedAtMs = nowMs;
    lastJ3RejectReason = null;
    return;
  }
  if (input.result === "rejected") {
    lastJ3RejectReason = input.reason?.slice(0, 128) ?? "rejected";
  }
}

export function recordPipelineRecoveryReason(reason: string, nowMs: number = Date.now()): void {
  lastRecoveryReason = reason.slice(0, 128);
  lastRecoveryAtMs = nowMs;
  pruneRecoveryTimestamps(nowMs);
  recoveryTimestampsMs.push(nowMs);
}

export function computePipelineFirstSuspect(
  pipeline: Pick<
    TrackingPipelineSnapshot,
    | "bridge_last_fix_age_s"
    | "bg_task_last_invoke_age_s"
    | "watch_callback_age_s"
    | "j1_handler_age_s"
    | "j3_accepted_age_s"
    | "j3_last_result"
    | "queue_last_enqueue_age_s"
    | "flush_last_attempt_age_s"
    | "durable_ack_age_s"
  >
): PipelineFirstSuspect {
  const stale = (age: number | null | undefined) =>
    age === null || age === undefined || age > PIPELINE_STALE_SECONDS;
  const fresh = (age: number | null | undefined) =>
    age !== null && age !== undefined && age <= PIPELINE_STALE_SECONDS;

  if (pipeline.j3_last_result === "rejected") {
    return "J3_GATE";
  }

  if (!stale(pipeline.durable_ack_age_s)) {
    return "UNKNOWN";
  }

  if (fresh(pipeline.flush_last_attempt_age_s) && stale(pipeline.durable_ack_age_s)) {
    return "ACK";
  }
  if (fresh(pipeline.queue_last_enqueue_age_s) && stale(pipeline.flush_last_attempt_age_s)) {
    return "FLUSH";
  }
  if (
    pipeline.j3_last_result === "accepted"
    && fresh(pipeline.j3_accepted_age_s)
    && stale(pipeline.queue_last_enqueue_age_s)
  ) {
    return "ENQUEUE";
  }
  if (fresh(pipeline.j1_handler_age_s) && stale(pipeline.j3_accepted_age_s)) {
    return "J3_GATE";
  }
  if (fresh(pipeline.watch_callback_age_s) && stale(pipeline.j1_handler_age_s)) {
    return "J1_HANDLER";
  }
  if (
    fresh(pipeline.bridge_last_fix_age_s)
    && stale(pipeline.bg_task_last_invoke_age_s)
    && stale(pipeline.watch_callback_age_s)
  ) {
    return "BG_TASK";
  }
  if (fresh(pipeline.bridge_last_fix_age_s) && stale(pipeline.watch_callback_age_s)) {
    return "J1_HANDLER";
  }
  return "UNKNOWN";
}

export async function collectTrackingPipelineSnapshot(
  nowMs: number = Date.now()
): Promise<TrackingPipelineSnapshot | null> {
  if (!isFeatureEnabled("tracking_pipeline_remote_observability_enabled")) {
    return null;
  }

  const bridge = readBridgeSnapshot();
  const runtime = getTrackingRuntimeSnapshot();
  const runtimeMeta = resolveDeviceRuntimeMetadata();
  const ownerRuntime = captureActiveRuntime();
  const appState = bridge?.appState ?? AppState.currentState;
  const isAvailable = readDriverAvailable();
  const trackingRequired = readTrackingRequired(appState);
  const trackingStartedAtMs = readTrackingStartedAtMs();

  const bridgeLastFixAge = computeLocationFixAgeSeconds(
    bridge?.lastFixProducedAtMs ?? null,
    nowMs
  );
  const bgTaskInvokeAge = computeTaskInvokeAgeSeconds(runtime.lastTaskInvokedAt, nowMs);
  const watchCallbackAge = computeWatchCallbackAgeSeconds(bridge?.lastWatchAtMs ?? null, nowMs);

  const queueHeadAge = await readQueueHeadAgeSeconds(nowMs);
  const backgroundTaskRegistered = await readBackgroundTaskRegistered();

  const durableAckAge =
    parseIsoAgeSeconds(bridge?.lastAckAt ?? bridge?.lastPersistedAt, nowMs);

  const pipelineBase = {
    pipeline_snapshot_version: PIPELINE_SNAPSHOT_VERSION,
    desired_mode: resolveDesiredMode(bridge?.missionId ?? null, bridge?.missionStatus),
    mission_id: bridge?.missionId ?? null,
    tracking_required: trackingRequired,
    is_available: isAvailable,

    bridge_last_fix_age_s: bridgeLastFixAge,
    bg_task_last_invoke_age_s: bgTaskInvokeAge,
    watch_callback_age_s: watchCallbackAge,

    j1_handler_age_s: computeTaskInvokeAgeSeconds(lastJ1HandlerAtMs, nowMs),
    j3_accepted_age_s: computeTaskInvokeAgeSeconds(lastJ3AcceptedAtMs, nowMs),
    j3_last_result: lastJ3Result,
    j3_last_reject_reason: lastJ3RejectReason,

    queue_last_enqueue_age_s: parseIsoAgeSeconds(bridge?.lastEnqueuedAt, nowMs),
    queue_depth: bridge?.queueDepth ?? null,
    queue_head_age_s: queueHeadAge,

    flush_last_attempt_age_s: parseIsoAgeSeconds(bridge?.lastTransportAttemptAt, nowMs),
    flush_last_sent_age_s: parseIsoAgeSeconds(bridge?.lastSentAt, nowMs),

    durable_ack_age_s: durableAckAge,

    owner_present: Boolean(ownerRuntime),
    owner_generation: ownerRuntime?.identity.trackingGenerationId ?? null,
    background_task_registered: backgroundTaskRegistered,

    app_state: appState,
    platform: Platform.OS,
    app_version: runtimeMeta.appVersion ?? null,
    native_build_version: runtimeMeta.appBuild ?? null,
    runtime_version: runtimeMeta.expoRuntimeVersion ?? null,
    ota_update_id: runtimeMeta.otaUpdateId ?? null,

    last_recovery_reason: lastRecoveryReason,
    last_recovery_age_s: computeTaskInvokeAgeSeconds(lastRecoveryAtMs, nowMs),
    recovery_count_15m: (() => {
      pruneRecoveryTimestamps(nowMs);
      return recoveryTimestampsMs.length;
    })(),

    tracking_runtime_age_s: computeTaskInvokeAgeSeconds(trackingStartedAtMs, nowMs),
  };

  return {
    ...pipelineBase,
    first_suspect: computePipelineFirstSuspect(pipelineBase),
  };
}

/** Tests uniquement. */
export function __resetTrackingPipelineObservabilityForTests(): void {
  lastJ1HandlerAtMs = null;
  lastJ3AcceptedAtMs = null;
  lastJ3Result = "unknown";
  lastJ3RejectReason = null;
  lastRecoveryReason = null;
  lastRecoveryAtMs = null;
  recoveryTimestampsMs.length = 0;
}
