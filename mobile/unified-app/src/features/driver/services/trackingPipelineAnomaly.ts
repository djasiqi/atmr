/**
 * JZ-R1 — snapshots anomalie pipeline (anti-spam, instrumentation-only).
 */

import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import {
  PIPELINE_STALE_SECONDS,
  type PipelineFirstSuspect,
  type TrackingPipelineSnapshot,
  collectTrackingPipelineSnapshot,
  computePipelineFirstSuspect,
} from "./trackingPipelineObservability";
import { sendDeviceHealth, type DeviceHealthRequestPayload } from "./deviceHealthHeartbeat";

export type PipelineAnomalyKind = "ANOMALY" | "RECOVERED";

const ANOMALY_COOLDOWN_MS = 5 * 60 * 1000;

let anomalyActive = false;
let lastAnomalyEmitAtMs = 0;
let lastFirstSuspect: PipelineFirstSuspect | null = null;

function isAckStale(pipeline: TrackingPipelineSnapshot): boolean {
  if (pipeline.durable_ack_age_s != null) {
    return pipeline.durable_ack_age_s > PIPELINE_STALE_SECONDS;
  }
  if (pipeline.tracking_runtime_age_s != null) {
    return pipeline.tracking_runtime_age_s > PIPELINE_STALE_SECONDS;
  }
  return false;
}

export function shouldPipelineBeTracked(pipeline: TrackingPipelineSnapshot): boolean {
  if (!pipeline.tracking_required) return false;
  if (pipeline.is_available !== true) return false;
  const runtimeAge = pipeline.tracking_runtime_age_s;
  if (runtimeAge == null || runtimeAge <= PIPELINE_STALE_SECONDS) return false;
  return isAckStale(pipeline);
}

export function evaluatePipelineAnomaly(
  pipeline: TrackingPipelineSnapshot,
  nowMs: number = Date.now()
): PipelineAnomalyKind | null {
  if (!isFeatureEnabled("tracking_pipeline_remote_observability_enabled")) {
    return null;
  }

  const broken = shouldPipelineBeTracked(pipeline);
  if (!broken) {
    if (anomalyActive) {
      anomalyActive = false;
      lastFirstSuspect = null;
      return "RECOVERED";
    }
    return null;
  }

  const suspect = computePipelineFirstSuspect(pipeline);
  const suspectChanged =
    lastFirstSuspect != null && suspect !== lastFirstSuspect && suspect !== "UNKNOWN";

  if (!anomalyActive) {
    anomalyActive = true;
    lastAnomalyEmitAtMs = nowMs;
    lastFirstSuspect = suspect;
    return "ANOMALY";
  }

  if (suspectChanged) {
    lastAnomalyEmitAtMs = nowMs;
    lastFirstSuspect = suspect;
    return "ANOMALY";
  }

  if (nowMs - lastAnomalyEmitAtMs >= ANOMALY_COOLDOWN_MS) {
    lastAnomalyEmitAtMs = nowMs;
    lastFirstSuspect = suspect;
    return "ANOMALY";
  }

  return null;
}

export async function maybeEmitPipelineAnomalySnapshot(
  baseHealth: DeviceHealthRequestPayload,
  nowMs: number = Date.now()
): Promise<void> {
  const pipeline = await collectTrackingPipelineSnapshot(nowMs);
  if (!pipeline) return;

  const kind = evaluatePipelineAnomaly(pipeline, nowMs);
  if (!kind) return;

  const enrichedPipeline = {
    ...pipeline,
    anomaly_kind: kind,
    first_suspect: computePipelineFirstSuspect(pipeline),
  };

  const triggerReason =
    kind === "RECOVERED"
      ? "tracking.pipeline.recovered"
      : "tracking.pipeline.anomaly_snapshot";

  emitDriverTelemetry(
    kind === "RECOVERED"
      ? "tracking.recovery.recovered"
      : "tracking.bridge.health",
    {
      source: "driver.tracking.pipeline_anomaly",
      reason: triggerReason,
      first_suspect: enrichedPipeline.first_suspect,
      mission_id: pipeline.mission_id,
      durable_ack_age_s: pipeline.durable_ack_age_s,
    }
  );

  await sendDeviceHealth({
    ...baseHealth,
    tracking_pipeline: enrichedPipeline,
    trigger_reason: triggerReason,
  });
}

/** Tests uniquement. */
export function __resetPipelineAnomalyForTests(): void {
  anomalyActive = false;
  lastAnomalyEmitAtMs = 0;
  lastFirstSuspect = null;
}
