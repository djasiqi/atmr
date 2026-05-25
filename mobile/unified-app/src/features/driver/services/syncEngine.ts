import { QueryClient } from "@tanstack/react-query";
import { evaluateConnectivityPolicy } from "../../../core/network/connectivityPolicy";
import { getNetworkSnapshot, subscribeNetworkState } from "../../../core/network/networkState";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { flushTrackingQueue, getTrackingSnapshot } from "../tracking";
import { scheduleDriverMissionSync } from "./missionSyncOrchestrator";
import { AppState, AppStateStatus } from "react-native";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { shouldSkipMissionPolling } from "../../../core/realtime/transportAuthority";

type RuntimeEngine = {
  timer: ReturnType<typeof setInterval> | null;
  stopNetworkSubscription: (() => void) | null;
  stopAppStateSubscription: (() => void) | null;
  currentIntervalMs: number;
  rawAppState: AppStateStatus;
  stableAppState: AppStateStatus;
  appStateChangedAtMs: number;
  lastResumeResyncAtMs: number;
  noMissionSinceMs: number | null;
};

type MissionPresenceSnapshot = {
  hasRelevantMission: boolean;
  missionCount: number;
};

type SyncEngineOptions = {
  getMissionPresence?: () => MissionPresenceSnapshot;
};

const engines = new Map<string, RuntimeEngine>();
const DEFAULT_INTERVAL_MS = 15_000;
const IDLE_HEARTBEAT_MS = Number(process.env.EXPO_PUBLIC_DRIVER_IDLE_HEARTBEAT_MS ?? "120000");
const BACKGROUND_HEARTBEAT_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_BACKGROUND_HEARTBEAT_MS ?? "180000"
);
const DEEP_IDLE_HEARTBEAT_MS = Number(process.env.EXPO_PUBLIC_DRIVER_DEEP_IDLE_HEARTBEAT_MS ?? "300000");
const APPSTATE_TRANSITION_STABILITY_WINDOW_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_APPSTATE_STABILITY_WINDOW_MS ?? "8000"
);
const DEEP_IDLE_ENTRY_STABILITY_WINDOW_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_DEEP_IDLE_ENTRY_STABILITY_WINDOW_MS ?? "300000"
);
const FOREGROUND_RESUME_RESYNC_GUARD_WINDOW_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_FOREGROUND_RESYNC_GUARD_WINDOW_MS ?? "5000"
);
const IDLE_ASSIGNMENT_DETECTION_TARGET_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_IDLE_ASSIGNMENT_DETECTION_TARGET_MS ?? "60000"
);

function getOrCreateEngine(contextId: string): RuntimeEngine {
  const existing = engines.get(contextId);
  if (existing) return existing;
  const created: RuntimeEngine = {
    timer: null,
    stopNetworkSubscription: null,
    stopAppStateSubscription: null,
    currentIntervalMs: DEFAULT_INTERVAL_MS,
    rawAppState: AppState.currentState,
    stableAppState: AppState.currentState,
    appStateChangedAtMs: Date.now(),
    lastResumeResyncAtMs: 0,
    noMissionSinceMs: null,
  };
  engines.set(contextId, created);
  return created;
}

function stopHeartbeat(engine: RuntimeEngine) {
  if (!engine.timer) return;
  clearInterval(engine.timer);
  engine.timer = null;
}

function getStableAppState(engine: RuntimeEngine): AppStateStatus {
  if (Date.now() - engine.appStateChangedAtMs >= APPSTATE_TRANSITION_STABILITY_WINDOW_MS) {
    engine.stableAppState = engine.rawAppState;
  }
  return engine.stableAppState;
}

function targetIntervalForAuthority(
  authority: "mission_active_loop" | "idle_loop" | "background_loop" | "deep_idle_loop",
  recommendedMs: number
) {
  if (authority === "mission_active_loop") return recommendedMs;
  if (authority === "idle_loop") return Math.max(IDLE_HEARTBEAT_MS, recommendedMs);
  if (authority === "background_loop") return Math.max(BACKGROUND_HEARTBEAT_MS, recommendedMs);
  return Math.max(DEEP_IDLE_HEARTBEAT_MS, recommendedMs);
}

function startHeartbeat(
  queryClient: QueryClient,
  contextId: string,
  engine: RuntimeEngine,
  options: SyncEngineOptions
) {
  stopHeartbeat(engine);
  engine.timer = setInterval(() => {
    const snapshot = getNetworkSnapshot();
    const policy = evaluateConnectivityPolicy(snapshot);
    const stableAppState = getStableAppState(engine);
    const missionPresence = options.getMissionPresence?.() ?? {
      hasRelevantMission: true,
      missionCount: 0,
    };
    if (!missionPresence.hasRelevantMission) {
      engine.noMissionSinceMs = engine.noMissionSinceMs ?? Date.now();
    } else {
      engine.noMissionSinceMs = null;
    }
    const tracking = getTrackingSnapshot();
    const realtime = realtimeManager.getSnapshot();
    const bgGating = isFeatureEnabled("driver_background_network_gating_enabled");
    const idleGating = isFeatureEnabled("driver_network_idle_gating_enabled");
    const deepIdleCandidate =
      bgGating &&
      stableAppState !== "active" &&
      !missionPresence.hasRelevantMission &&
      tracking.queueDepth === 0 &&
      realtime.actualTransport === "socket" &&
      !realtime.degradedMode &&
      (engine.noMissionSinceMs ? Date.now() - engine.noMissionSinceMs > DEEP_IDLE_ENTRY_STABILITY_WINDOW_MS : false);
    const authority: "mission_active_loop" | "idle_loop" | "background_loop" | "deep_idle_loop" =
      missionPresence.hasRelevantMission
        ? "mission_active_loop"
        : deepIdleCandidate
          ? "deep_idle_loop"
          : stableAppState === "active"
            ? "idle_loop"
            : "background_loop";
    const targetIntervalMs = targetIntervalForAuthority(authority, policy.recommendedSyncIntervalMs);
    if (idleGating && targetIntervalMs !== engine.currentIntervalMs) {
      engine.currentIntervalMs = targetIntervalMs;
      startHeartbeat(queryClient, contextId, engine, options);
      return;
    }

    emitDriverTelemetry("driver.network.tick", {
      source: "driver.sync_engine",
      context_id: contextId,
      app_state: stableAppState,
      network_activity_authority: authority,
      mission_presence: missionPresence.hasRelevantMission ? "relevant" : "none",
      driver_network_tick_total: 1,
    });
    emitDriverTelemetry("driver.network.wake", {
      source: "driver.sync_engine",
      context_id: contextId,
      driver_network_wake_cause: "sync_engine",
      network_activity_authority: authority,
    });
    if (!missionPresence.hasRelevantMission) {
      emitDriverTelemetry("driver.network.tick", {
        source: "driver.sync_engine",
        context_id: contextId,
        driver_no_mission_network_tick_total: 1,
      });
    }
    if (stableAppState !== "active") {
      emitDriverTelemetry("driver.network.tick", {
        source: "driver.sync_engine",
        context_id: contextId,
        driver_background_network_tick_total: 1,
      });
    }
    const shouldFlushQueue = policy.allowGpsFlush && (tracking.queueDepth > 0 || tracking.isRunning);
    if (shouldFlushQueue) {
      void flushTrackingQueue();
    } else {
      emitDriverTelemetry("driver.sync_engine.flush.skipped", {
        source: "driver.sync_engine",
        context_id: contextId,
        reason: "queue_empty_or_tracking_idle",
      });
    }
    if (!shouldSkipMissionPolling(realtime)) {
      scheduleDriverMissionSync(queryClient, contextId, "manual");
      realtimeManager.setTransportAuthority("reconcile", "sync_engine_tick");
    }
    if (engine.noMissionSinceMs && missionPresence.hasRelevantMission) {
      emitDriverTelemetry("driver.network.tick", {
        source: "driver.sync_engine",
        context_id: contextId,
        mission_assignment_detection_latency_ms: Date.now() - engine.noMissionSinceMs,
        idle_assignment_detection_target_ms: IDLE_ASSIGNMENT_DETECTION_TARGET_MS,
      });
      engine.noMissionSinceMs = null;
    }
    emitDriverTelemetry("driver.sync_engine.heartbeat", {
      source: "driver.sync_engine",
      context_id: contextId,
      network_mode: policy.mode,
      allow_socket: policy.allowSocket,
      allow_gps_flush: policy.allowGpsFlush,
      app_state: stableAppState,
      network_activity_authority: authority,
      driver_sync_engine_active_ms: engine.currentIntervalMs,
      interval_ms: engine.currentIntervalMs,
    });
  }, engine.currentIntervalMs);
}

export function startDriverSyncEngine(
  queryClient: QueryClient,
  contextId: string,
  options: SyncEngineOptions = {}
): () => void {
  const engine = getOrCreateEngine(contextId);
  if (!engine.stopAppStateSubscription) {
    const subscription = AppState.addEventListener("change", (next) => {
      const previous = engine.rawAppState;
      engine.rawAppState = next;
      engine.appStateChangedAtMs = Date.now();
      if (previous !== "active" && next === "active") {
        const now = Date.now();
        if (now - engine.lastResumeResyncAtMs >= FOREGROUND_RESUME_RESYNC_GUARD_WINDOW_MS) {
          engine.lastResumeResyncAtMs = now;
          emitDriverTelemetry("driver.foreground.resume.resync", {
            source: "driver.sync_engine",
            context_id: contextId,
            foreground_resume_resync_total: 1,
          });
          scheduleDriverMissionSync(queryClient, contextId, "foreground");
        } else {
          emitDriverTelemetry("driver.foreground.resume.resync.coalesced", {
            source: "driver.sync_engine",
            context_id: contextId,
            foreground_resume_resync_coalesced_total: 1,
          });
        }
      }
    });
    engine.stopAppStateSubscription = () => subscription.remove();
  }
  if (!engine.stopNetworkSubscription) {
    engine.stopNetworkSubscription = subscribeNetworkState((snapshot) => {
      const policy = evaluateConnectivityPolicy(snapshot);
      if (policy.recommendedSyncIntervalMs !== engine.currentIntervalMs) {
        engine.currentIntervalMs = policy.recommendedSyncIntervalMs;
        startHeartbeat(queryClient, contextId, engine, options);
      }
    });
  }
  startHeartbeat(queryClient, contextId, engine, options);
  return () => {
    stopHeartbeat(engine);
    if (engine.stopNetworkSubscription) {
      engine.stopNetworkSubscription();
      engine.stopNetworkSubscription = null;
    }
    if (engine.stopAppStateSubscription) {
      engine.stopAppStateSubscription();
      engine.stopAppStateSubscription = null;
    }
    engines.delete(contextId);
  };
}
