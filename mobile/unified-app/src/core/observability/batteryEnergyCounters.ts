/**
 * Instrumentation batterie T0 — coût minimal.
 * Compteurs mémoire + 1 emit / minute. Flag OFF = no-op (pas de timer).
 * Ne mute jamais le pipeline GPS.
 */
import { AppState, Platform } from "react-native";

import { resolveDeviceRuntimeMetadata } from "../device/deviceRuntimeMetadata";
import { isFeatureEnabled } from "../featureFlags/registry";
import { emitDriverTelemetry } from "./driverTelemetry";

export type BatteryEnqueueSource =
  | "native_task"
  | "bridge_tick"
  | "bridge_fallback_fix";

export type BatteryCallbackSource = "native_task" | "js_watch";

type LatencyAcc = { sum: number; count: number; samples: number[] };

type DepthTrack = {
  min: number | null;
  max: number | null;
  last: number | null;
  samples: number[];
};

type MinuteBucket = {
  nativeCallbacks: number;
  jsCallbacks: number;
  uniqueFixes: number;
  duplicateFixes: number;
  enqueues: number;
  putSuccess: number;
  enqueueNative: number;
  enqueueBridgeTick: number;
  enqueueBridgeFallback: number;
  queueAtEnqueue: DepthTrack;
  queueAtDrain: DepthTrack;
  callbackToEnqueue: LatencyAcc;
  enqueueToUpload: LatencyAcc;
  recordedToUpload: LatencyAcc;
  seenRecordedAt: Set<string>;
  lastMode: string | null;
  lastAppState: string | null;
};

const MAX_SAMPLES = 48;
const MAX_PENDING_ENQUEUE = 120;

let testEnabledOverride: boolean | null = null;
let watchActive = false;
let nativeTaskActive = false;
let minuteStartMs = 0;
let flushTimer: ReturnType<typeof setTimeout> | null = null;
let bucket: MinuteBucket = emptyBucket();
const pendingEnqueueAt = new Map<string, number>();
const pendingCallbackAt = new Map<string, number>();

function emptyLatency(): LatencyAcc {
  return { sum: 0, count: 0, samples: [] };
}

function emptyDepth(): DepthTrack {
  return { min: null, max: null, last: null, samples: [] };
}

function emptyBucket(): MinuteBucket {
  return {
    nativeCallbacks: 0,
    jsCallbacks: 0,
    uniqueFixes: 0,
    duplicateFixes: 0,
    enqueues: 0,
    putSuccess: 0,
    enqueueNative: 0,
    enqueueBridgeTick: 0,
    enqueueBridgeFallback: 0,
    queueAtEnqueue: emptyDepth(),
    queueAtDrain: emptyDepth(),
    callbackToEnqueue: emptyLatency(),
    enqueueToUpload: emptyLatency(),
    recordedToUpload: emptyLatency(),
    seenRecordedAt: new Set(),
    lastMode: null,
    lastAppState: null,
  };
}

function isEnabled(): boolean {
  if (testEnabledOverride != null) return testEnabledOverride;
  return isFeatureEnabled("tracking_battery_energy_instrumentation_enabled");
}

function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx] ?? null;
}

function pushLatency(acc: LatencyAcc, ms: number): void {
  if (!Number.isFinite(ms) || ms < 0) return;
  acc.sum += ms;
  acc.count += 1;
  if (acc.samples.length < MAX_SAMPLES) {
    acc.samples.push(ms);
  }
}

function pushDepth(track: DepthTrack, depth: number | null | undefined): void {
  if (typeof depth !== "number" || !Number.isFinite(depth) || depth < 0) return;
  const value = Math.floor(depth);
  track.last = value;
  track.min = track.min == null ? value : Math.min(track.min, value);
  track.max = track.max == null ? value : Math.max(track.max, value);
  if (track.samples.length < MAX_SAMPLES) {
    track.samples.push(value);
  }
}

function depthSnapshot(track: DepthTrack, prefix: string): Record<string, number | null> {
  return {
    [`${prefix}_min`]: track.min,
    [`${prefix}_max`]: track.max,
    [`${prefix}_last`]: track.last,
    [`${prefix}_p50`]: percentile(track.samples, 50),
    [`${prefix}_p95`]: percentile(track.samples, 95),
  };
}

function remember(map: Map<string, number>, key: string, value: number): void {
  if (map.size >= MAX_PENDING_ENQUEUE) {
    const first = map.keys().next().value;
    if (typeof first === "string") map.delete(first);
  }
  map.set(key, value);
}

function parseRecordedMs(recordedAt: string | null | undefined): number | null {
  if (!recordedAt) return null;
  const ms = Date.parse(recordedAt);
  return Number.isFinite(ms) ? ms : null;
}

function ensureMinute(nowMs: number): void {
  const start = nowMs - (nowMs % 60_000);
  if (minuteStartMs === 0) {
    minuteStartMs = start;
    scheduleFlush(start);
    return;
  }
  if (start !== minuteStartMs) {
    emitMinuteSnapshot(minuteStartMs);
    bucket = emptyBucket();
    minuteStartMs = start;
    scheduleFlush(start);
  }
}

function scheduleFlush(startMs: number): void {
  if (flushTimer) {
    clearTimeout(flushTimer);
    flushTimer = null;
  }
  const delay = Math.max(250, startMs + 60_000 - Date.now() + 40);
  flushTimer = setTimeout(() => {
    flushTimer = null;
    if (!isEnabled()) return;
    if (minuteStartMs === startMs) {
      emitMinuteSnapshot(startMs);
      bucket = emptyBucket();
      minuteStartMs = 0;
    }
  }, delay);
}

function emitMinuteSnapshot(startMs: number): void {
  const meta = resolveDeviceRuntimeMetadata();
  const elapsedMin = Math.max(1 / 60, (Date.now() - startMs) / 60_000);
  const native = bucket.nativeCallbacks;
  const js = bucket.jsCallbacks;
  const uniques = bucket.uniqueFixes;
  const dups = bucket.duplicateFixes;
  const enq = bucket.enqueues;
  const puts = bucket.putSuccess;
  emitDriverTelemetry("tracking.battery.minute", {
    source: "driver.observability.battery_energy",
    platform: Platform.OS,
    device_model: meta.model,
    app_version: meta.appVersion,
    tracking_mode: bucket.lastMode,
    app_state: bucket.lastAppState ?? AppState.currentState,
    provider: Platform.OS === "ios" ? "core_location" : "fused_location",
    native_callbacks: native,
    js_callbacks: js,
    unique_fixes: uniques,
    duplicate_fixes: dups,
    enqueues: enq,
    put_success: puts,
    enqueue_native: bucket.enqueueNative,
    enqueue_bridge_tick: bucket.enqueueBridgeTick,
    enqueue_bridge_fallback: bucket.enqueueBridgeFallback,
    native_callbacks_per_min: native / elapsedMin,
    js_callbacks_per_min: js / elapsedMin,
    unique_fixes_per_min: uniques / elapsedMin,
    duplicate_fixes_per_min: dups / elapsedMin,
    enqueues_per_min: enq / elapsedMin,
    put_success_per_min: puts / elapsedMin,
    ...depthSnapshot(bucket.queueAtEnqueue, "queue_depth_enqueue"),
    ...depthSnapshot(bucket.queueAtDrain, "queue_depth_drain"),
    same_recorded_at_reused: dups > 0,
    layers_not_collapsed: native !== uniques || uniques !== enq,
    callback_to_enqueue_p50_ms: percentile(bucket.callbackToEnqueue.samples, 50),
    enqueue_to_upload_p50_ms: percentile(bucket.enqueueToUpload.samples, 50),
    recorded_to_upload_p50_ms: percentile(bucket.recordedToUpload.samples, 50),
    native_task_active: nativeTaskActive,
    js_watch_active: watchActive,
    window_start_ms: startMs,
  });
}

function touchContext(mode?: string | null, appState?: string | null): void {
  if (mode) bucket.lastMode = mode;
  if (appState) bucket.lastAppState = appState;
}

export function setBatteryEnergyInstrEnabledForTests(value: boolean | null): void {
  testEnabledOverride = value;
}

export function resetBatteryEnergyCountersForTests(): void {
  if (flushTimer) {
    clearTimeout(flushTimer);
    flushTimer = null;
  }
  bucket = emptyBucket();
  minuteStartMs = 0;
  pendingEnqueueAt.clear();
  pendingCallbackAt.clear();
  watchActive = false;
  nativeTaskActive = false;
}

export function setBatteryWatchActive(active: boolean): void {
  watchActive = active;
}

export function setBatteryNativeTaskActive(active: boolean): void {
  nativeTaskActive = active;
}

export function recordBatteryCallback(input: {
  source: BatteryCallbackSource;
  recordedAt?: string | null;
  callbackAtMs?: number;
  trackingMode?: string | null;
  appState?: string | null;
}): void {
  if (!isEnabled()) return;
  const now = input.callbackAtMs ?? Date.now();
  ensureMinute(now);
  touchContext(input.trackingMode, input.appState);
  if (input.source === "native_task") {
    bucket.nativeCallbacks += 1;
  } else {
    bucket.jsCallbacks += 1;
  }
  if (input.recordedAt) {
    remember(pendingCallbackAt, input.recordedAt, now);
  }
}

export function recordBatteryEnqueue(input: {
  source: BatteryEnqueueSource;
  recordedAt?: string | null;
  eventId?: string | null;
  enqueueAtMs?: number;
  queueDepth?: number | null;
  trackingMode?: string | null;
  appState?: string | null;
}): void {
  if (!isEnabled()) return;
  const now = input.enqueueAtMs ?? Date.now();
  ensureMinute(now);
  touchContext(input.trackingMode, input.appState);
  bucket.enqueues += 1;
  if (input.source === "native_task") bucket.enqueueNative += 1;
  else if (input.source === "bridge_tick") bucket.enqueueBridgeTick += 1;
  else bucket.enqueueBridgeFallback += 1;
  pushDepth(bucket.queueAtEnqueue, input.queueDepth);
  const recordedAt = input.recordedAt ?? null;
  if (recordedAt) {
    if (bucket.seenRecordedAt.has(recordedAt)) {
      bucket.duplicateFixes += 1;
    } else {
      bucket.seenRecordedAt.add(recordedAt);
      bucket.uniqueFixes += 1;
    }
    const cbAt = pendingCallbackAt.get(recordedAt);
    if (cbAt != null) {
      pushLatency(bucket.callbackToEnqueue, now - cbAt);
    }
  }
  if (input.eventId) {
    remember(pendingEnqueueAt, input.eventId, now);
  }
}

export function recordBatteryPutSuccess(input: {
  eventId?: string | null;
  recordedAt?: string | null;
  queuedAtMs?: number | null;
  uploadAtMs?: number;
  queueDepth?: number | null;
  trackingMode?: string | null;
  appState?: string | null;
}): void {
  if (!isEnabled()) return;
  const now = input.uploadAtMs ?? Date.now();
  ensureMinute(now);
  touchContext(input.trackingMode, input.appState);
  bucket.putSuccess += 1;
  pushDepth(bucket.queueAtDrain, input.queueDepth);
  const queuedAt =
    input.queuedAtMs ??
    (input.eventId ? pendingEnqueueAt.get(input.eventId) : undefined);
  if (queuedAt != null) {
    pushLatency(bucket.enqueueToUpload, now - queuedAt);
    if (input.eventId) pendingEnqueueAt.delete(input.eventId);
  }
  const recordedMs = parseRecordedMs(input.recordedAt);
  if (recordedMs != null) {
    pushLatency(bucket.recordedToUpload, now - recordedMs);
  }
}

export function flushBatteryEnergyMinuteNowForTests(): void {
  if (minuteStartMs === 0) return;
  emitMinuteSnapshot(minuteStartMs);
  bucket = emptyBucket();
  minuteStartMs = 0;
  if (flushTimer) {
    clearTimeout(flushTimer);
    flushTimer = null;
  }
}
