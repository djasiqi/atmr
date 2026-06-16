import { updateDriverMissionStatus } from "./api";
import { applyArrivedMilestoneFromStatusResponse } from "./domain/missionMilestoneOverlay";
import { DriverTransitionStatus } from "./types";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../core/featureFlags/registry";
import {
  OfflineMutationQueue,
  OfflineMutationAction,
  OfflineMutationFlushResult,
} from "../../core/offline/offlineMutationQueue";
import { driverTrackingQueue } from "./services/driverTrackingQueue";

type QueuedDriverAction = OfflineMutationAction & {
  missionId: number;
  targetStatus: DriverTransitionStatus;
  reason?: string | null;
  eventSequence?: number;
};

const STORAGE_KEY = "driver_pending_actions_v1";
const MAX_RETRIES = 8;
const BACKOFF_BASE_MS = 1500;
const BACKOFF_MAX_MS = 30_000;

const DEFAULT_QUEUE_REPLAY_WINDOW_SECONDS = 600;
const TUNED_QUEUE_REPLAY_WINDOW_SECONDS = Number(
  process.env.EXPO_PUBLIC_DRIVER_TRANSITION_REPLAY_WINDOW_SECONDS ?? "1800"
);
export const QUEUE_REPLAY_WINDOW_SECONDS = isFeatureEnabled("driver_transition_window_tuning_enabled")
  ? TUNED_QUEUE_REPLAY_WINDOW_SECONDS
  : DEFAULT_QUEUE_REPLAY_WINDOW_SECONDS;
const QUEUE_REPLAY_WINDOW_MS = Math.max(60_000, QUEUE_REPLAY_WINDOW_SECONDS * 1000);
const MAX_REPLAY_ATTEMPTS_PER_TRANSITION = Number(
  process.env.EXPO_PUBLIC_DRIVER_TRANSITION_MAX_REPLAY_ATTEMPTS ?? "5"
);
const LONG_PENDING_THRESHOLD_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_TRANSITION_LONG_PENDING_THRESHOLD_MS ?? "300000"
);

function createActionId() {
  return `drv_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

class DriverOfflineQueue {
  private readonly queue = new OfflineMutationQueue<QueuedDriverAction>({
    storageKey: STORAGE_KEY,
    maxRetries: MAX_RETRIES,
    replayWindowMs: QUEUE_REPLAY_WINDOW_MS,
    maxReplayAttemptsPerAction: MAX_REPLAY_ATTEMPTS_PER_TRANSITION,
    backoffBaseMs: BACKOFF_BASE_MS,
    backoffMaxMs: BACKOFF_MAX_MS,
    execute: async (action) => {
      const res = await updateDriverMissionStatus({
        missionId: action.missionId,
        targetStatus: action.targetStatus,
        idempotencyKey: action.id,
        reason: action.reason ?? null,
      });
      applyArrivedMilestoneFromStatusResponse(action.missionId, res);
    },
    onExpired: (action) => {
      console.warn("[offline_action_expired]", {
        mission_id: action.missionId,
        action_id: action.id,
        queue_replay_window_seconds: QUEUE_REPLAY_WINDOW_SECONDS,
      });
      emitDriverTelemetry("transition.queue.failure", {
        source: "driver.offline.queue",
        mission_id: action.missionId,
        retry_count: action.retryCount,
        reason: "expired_replay_window",
        mission_transition_queue_expired_total: 1,
      });
    },
    onSuccess: (action) => {
      emitDriverTelemetry("transition.queue.flush", {
        source: "driver.offline.queue",
        mission_id: action.missionId,
        retry_count: action.retryCount,
      });
    },
    onRetry: (action, retryCount) => {
      emitDriverTelemetry("transition.queue.retry", {
        source: "driver.offline.queue",
        mission_id: action.missionId,
        retry_count: retryCount,
      });
    },
    onFailure: (action, retryCount) => {
      emitDriverTelemetry("transition.queue.failure", {
        source: "driver.offline.queue",
        mission_id: action.missionId,
        retry_count: retryCount,
        reason: "max_retries_reached",
      });
    },
  });

  async enqueue(
    missionId: number,
    targetStatus: DriverTransitionStatus,
    reason?: string | null
  ) {
    const action: QueuedDriverAction = {
      id: createActionId(),
      missionId,
      targetStatus,
      reason: reason ?? null,
      queuedAt: Date.now(),
      retryCount: 0,
    };
    const queued = await this.queue.enqueue(action);
    emitDriverTelemetry("transition.queue.retry", {
      source: "driver.offline.queue",
      mission_id: missionId,
      retry_count: 0,
      queue_replay_window_seconds: QUEUE_REPLAY_WINDOW_SECONDS,
      max_replay_attempts_per_transition: MAX_REPLAY_ATTEMPTS_PER_TRANSITION,
    });
    return queued;
  }

  async purgeMission(missionId: number) {
    await this.queue.removeWhere((action) => action.missionId === missionId);
  }

  async count() {
    return this.queue.count();
  }

  async getSnapshot() {
    const [queuedCount, oldestAgeMs, trackingSnapshot] = await Promise.all([
      this.queue.count(),
      this.queue.oldestAgeMs(),
      driverTrackingQueue.getSnapshot(),
    ]);
    const divergenceMs =
      oldestAgeMs > 0 && trackingSnapshot.oldestItemAgeMs
        ? Math.abs(oldestAgeMs - trackingSnapshot.oldestItemAgeMs)
        : null;
    return {
      queuedCount,
      oldestAgeMs,
      replayWindowMs: QUEUE_REPLAY_WINDOW_MS,
      longPending: oldestAgeMs >= LONG_PENDING_THRESHOLD_MS,
      divergenceMs,
    };
  }

  async flush(): Promise<OfflineMutationFlushResult> {
    const snapshotBefore = await this.getSnapshot();
    if (snapshotBefore.queuedCount > 0) {
      emitDriverTelemetry("transition.queue.flush", {
        source: "driver.offline.queue",
        queue_depth: snapshotBefore.queuedCount,
        mission_transition_queue_age_ms: snapshotBefore.oldestAgeMs,
        mission_transition_vs_tracking_divergence_ms: snapshotBefore.divergenceMs,
      });
      if (snapshotBefore.longPending) {
        emitDriverTelemetry("transition.queue.failure", {
          source: "driver.offline.queue",
          reason: "long_pending_transition",
          mission_transition_queue_age_ms: snapshotBefore.oldestAgeMs,
          queue_replay_window_seconds: QUEUE_REPLAY_WINDOW_SECONDS,
        });
      }
    }
    return this.queue.flush();
  }
}

export const driverOfflineQueue = new DriverOfflineQueue();

