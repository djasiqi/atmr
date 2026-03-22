/**
 * syncEngine — Orchestre socket / HTTP / queue / flush selon la connectivityPolicy.
 * Plan migration 2G/3G — Phase 1–3.
 * Singleton ; start/stop dans _layout.tsx selon isDriverAuthenticated.
 */

import { AppState, InteractionManager, Platform } from "react-native";
import { getLogger } from "@/utils/logger";
import {
  initConnectivityPolicy,
  stopConnectivityPolicy,
  subscribeToMode,
  type NetworkMode,
} from "./connectivityPolicy";
import { subscribeToNetworkState, getNetworkStateSnapshot } from "./networkState";
import { addSocketConnectListener, getSocket, getSocketRole, sendDriverHeartbeat, triggerMissionResync } from "./socket";
import { MissionStateManager } from "./missionState";
import { syncLocationQueue, flushLatestPositionViaHttp } from "./locationQueue";
import {
  getAdaptiveLocationTracker,
  getCurrentLocationMode,
  getLastPermissionSnapshot,
} from "./locationTracker";
import { resolveLocationModeFromState, resolvePresenceState } from "./locationPresenceFsm";
import {
  isMissionTrackingActiveStatus,
  isMissionTrackingEligibleNow,
} from "./missionTrackingPolicy";
import { getLastResolvedTrackingPolicy } from "./trackingRuntime";

const log = getLogger("SyncEngine");
const trackLog = getLogger("TRACK");

/** Raisons stables pour LOCATION_FLUSH_SKIPPED (analyse prod / stats). */
export type LocationFlushSkipReason =
  | "socket_role_not_driver"
  | "background_no_presence_no_mission"
  | "permission_snapshot_missing"
  | "app_not_active_and_not_eligible"
  | "backoff_active"
  | "mission_status_not_eligible"
  | "policy_background_flush_blocked";

let lastLoggedCriticalFailureReason: string | null = null;
let lastFlushSkipReason: string | null = null;

function noteMissionCriticalFailure(
  detail: string,
  extra: Record<string, unknown>
): void {
  if (lastLoggedCriticalFailureReason === detail) return;
  lastLoggedCriticalFailureReason = detail;
  trackLog.info("MISSION_CRITICAL_EVAL_SKIPPED", { detail, ...extra });
  if (detail === "snapshot_uninitialized") {
    trackLog.info("PERMISSION_SNAPSHOT_MISSING", extra);
  }
}

function logLocationFlushSkipped(
  reason: LocationFlushSkipReason,
  extra: Record<string, unknown>
): void {
  if (lastFlushSkipReason === reason) return;
  lastFlushSkipReason = reason;
  trackLog.info("LOCATION_FLUSH_SKIPPED", { reason, ...extra });
}

let instance: SyncEngineImpl | null = null;
let modeUnsub: (() => void) | null = null;
let networkUnsub: (() => void) | null = null;
let appStateUnsub: (() => void) | null = null;
let socketConnectUnsub: (() => void) | null = null;
let locationFlushInterval: ReturnType<typeof setInterval> | null = null;
let reconciliationInterval: ReturnType<typeof setInterval> | null = null;
let missionFallbackInterval: ReturnType<typeof setInterval> | null = null;
let missionHeartbeatInterval: ReturnType<typeof setInterval> | null = null;
let presenceHeartbeatInterval: ReturnType<typeof setInterval> | null = null;

function isMissionCriticalTrackingMode(): boolean {
  if (getSocketRole() !== "driver") {
    return false;
  }

  const policy = getLastResolvedTrackingPolicy();
  if (policy !== null) {
    return (
      policy.shouldEscalateMissionPriority &&
      (policy.mode === "FOREGROUND_MISSION" || policy.mode === "BACKGROUND_MISSION")
    );
  }

  const snapshot = getLastPermissionSnapshot();
  const missionActive = MissionStateManager.isActive();
  const ms = MissionStateManager.getState();
  const missionStatus = missionActive ? ms.currentStatus : null;
  const eligible =
    missionActive &&
    missionStatus !== null &&
    isMissionTrackingActiveStatus(missionStatus);

  const commonCtx = {
    appState: AppState.currentState,
    missionActive,
    missionStatus,
    currentLocationMode: getCurrentLocationMode(),
  };

  if (!snapshot) {
    noteMissionCriticalFailure("snapshot_uninitialized", commonCtx);
    return false;
  }

  const fgOk = snapshot.fg === "granted";
  const bgOk = Platform.OS === "ios" ? snapshot.bg === "granted" : true;

  if (!fgOk) {
    noteMissionCriticalFailure("permission_fg_denied", {
      ...commonCtx,
      fg: snapshot.fg,
    });
    return false;
  }
  if (Platform.OS === "ios" && !bgOk) {
    noteMissionCriticalFailure("permission_bg_denied_ios", {
      ...commonCtx,
      bg: snapshot.bg,
    });
    return false;
  }

  if (!eligible) {
    noteMissionCriticalFailure("mission_not_eligible", commonCtx);
    return false;
  }

  const fsmState = resolvePresenceState({
    isAuthenticated: true,
    isDriver: true,
    hasFgPermission: fgOk,
    hasBgPermission: bgOk,
    appInBackground: AppState.currentState !== "active",
    hasActiveMission: missionActive,
    availabilityPresenceEnabled: true,
  });
  const mode = resolveLocationModeFromState(fsmState);
  if (mode !== "mission_live") {
    noteMissionCriticalFailure("fsm_not_mission_live", {
      ...commonCtx,
      presenceDerivedMode: mode,
    });
    return false;
  }

  lastLoggedCriticalFailureReason = null;
  return true;
}

function isAvailabilityPresenceMode(): boolean {
  return getSocketRole() === "driver" && getCurrentLocationMode() === "availability_presence";
}

/** Présence : policy si disponible, sinon mode legacy. */
function isPresencePresenceModeFromPolicyOrLegacy(): boolean {
  const p = getLastResolvedTrackingPolicy();
  if (p) {
    return p.mode === "FOREGROUND_PRESENCE" || p.mode === "BACKGROUND_PRESENCE";
  }
  return isAvailabilityPresenceMode();
}

const DEFAULT_FLUSH_INTERVAL_MS = 15000;
const DEFAULT_MISSION_HEARTBEAT_MS = 60000;
const DEFAULT_PRESENCE_HEARTBEAT_MS = 180000;

function runMissionHeartbeatTick(): void {
  if (AppState.currentState !== "active" && !isMissionCriticalTrackingMode()) return;
  if (getSocketRole() !== "driver") return;
  const socket = getSocket();
  if (!MissionStateManager.isActive()) return;
  if (!socket?.connected) {
    triggerLocationFlush();
    return;
  }
  const state = MissionStateManager.getState();
  const missionId = state?.activeMission?.id;
  if (!missionId) return;
  const lastPos = getAdaptiveLocationTracker().getLastPosition();
  const location = lastPos?.coords
    ? { lat: lastPos.coords.latitude, lon: lastPos.coords.longitude }
    : undefined;
  sendDriverHeartbeat({ last_mission_id: missionId, location }).catch((e) => {
    log.warn("sendDriverHeartbeat error", { error: e });
  });
}

function runPresenceHeartbeatTick(): void {
  if (AppState.currentState === "active") return;
  if (!isPresencePresenceModeFromPolicyOrLegacy()) return;
  if (MissionStateManager.isActive()) return;
  flushLatestPositionViaHttp().catch((e: unknown) => {
    const err = e as { message?: string; response?: { status?: number }; code?: string };
    log.warn("presence heartbeat fallback error", {
      message: err?.message ?? String(e),
      status: err?.response?.status,
      code: err?.code,
    });
  });
}

function schedulePolicyDrivenIntervals(): void {
  const policy = getLastResolvedTrackingPolicy();
  const flushMs = policy?.flushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS;
  const missionHbMs = policy?.missionHeartbeatIntervalMs ?? DEFAULT_MISSION_HEARTBEAT_MS;
  const presenceHbMs = policy?.presenceHeartbeatIntervalMs ?? DEFAULT_PRESENCE_HEARTBEAT_MS;

  if (locationFlushInterval) {
    clearInterval(locationFlushInterval);
    locationFlushInterval = null;
  }
  if (missionHeartbeatInterval) {
    clearInterval(missionHeartbeatInterval);
    missionHeartbeatInterval = null;
  }
  if (presenceHeartbeatInterval) {
    clearInterval(presenceHeartbeatInterval);
    presenceHeartbeatInterval = null;
  }

  locationFlushInterval = setInterval(triggerLocationFlush, flushMs);
  missionHeartbeatInterval = setInterval(runMissionHeartbeatTick, missionHbMs);
  presenceHeartbeatInterval = setInterval(runPresenceHeartbeatTick, presenceHbMs);
}

function triggerPendingActionsFlush(): void {
  MissionStateManager.syncPendingActions().catch((e: unknown) => {
    const err = e as { message?: string };
    log.warn("syncPendingActions error", { message: err?.message ?? String(e) });
  });
}

let locationFlushConsecutiveFailures = 0;
const LOCATION_FLUSH_BACKOFF_THRESHOLD = 2;

function triggerLocationFlush(): void {
  if (getSocketRole() !== "driver") {
    logLocationFlushSkipped("socket_role_not_driver", {
      appState: AppState.currentState,
    });
    return;
  }
  const resolvedPolicy = getLastResolvedTrackingPolicy();
  if (resolvedPolicy?.transportPreference === "deferred") {
    log.debug("location flush skipped (policy deferred)", { mode: resolvedPolicy.mode });
    return;
  }

  if (AppState.currentState !== "active" && resolvedPolicy) {
    if (!resolvedPolicy.shouldAllowBackgroundFlush) {
      logLocationFlushSkipped("policy_background_flush_blocked", {
        appState: AppState.currentState,
        mode: resolvedPolicy.mode,
        shouldEscalate: resolvedPolicy.shouldEscalateMissionPriority,
      });
      return;
    }
  } else if (AppState.currentState !== "active" && !resolvedPolicy) {
    const missionCritical = isMissionCriticalTrackingMode();
    const presenceMode = isAvailabilityPresenceMode();
    if (!missionCritical && !presenceMode) {
      const snapshot = getLastPermissionSnapshot();
      if (!snapshot) {
        logLocationFlushSkipped("permission_snapshot_missing", {
          appState: AppState.currentState,
          missionCritical,
          presenceMode,
          currentLocationMode: getCurrentLocationMode(),
          missionActive: MissionStateManager.isActive(),
          missionStatus: MissionStateManager.isActive()
            ? MissionStateManager.getState().currentStatus
            : null,
        });
      } else if (MissionStateManager.isActive() && !isMissionTrackingEligibleNow()) {
        logLocationFlushSkipped("mission_status_not_eligible", {
          appState: AppState.currentState,
          currentLocationMode: getCurrentLocationMode(),
          missionStatus: MissionStateManager.getState().currentStatus,
        });
      } else if (
        MissionStateManager.isActive() &&
        isMissionTrackingEligibleNow() &&
        !missionCritical
      ) {
        logLocationFlushSkipped("app_not_active_and_not_eligible", {
          appState: AppState.currentState,
          currentLocationMode: getCurrentLocationMode(),
          missionStatus: MissionStateManager.getState().currentStatus,
        });
      } else {
        logLocationFlushSkipped("background_no_presence_no_mission", {
          appState: AppState.currentState,
          missionCritical,
          presenceMode,
          currentLocationMode: getCurrentLocationMode(),
        });
      }
      return;
    }
  }

  const missionCritical = isMissionCriticalTrackingMode();
  // Backoff : éviter de hammer si erreurs répétées (401, réseau, etc.)
  if (locationFlushConsecutiveFailures >= LOCATION_FLUSH_BACKOFF_THRESHOLD) {
    log.debug("location flush skipped (backoff)", { failures: locationFlushConsecutiveFailures });
    logLocationFlushSkipped("backoff_active", {
      failures: locationFlushConsecutiveFailures,
      appState: AppState.currentState,
    });
    return;
  }
  const socket = getSocket();
  const doFlush = () => {
    if (socket?.connected) {
      syncLocationQueue(socket)
        .then(() => {
          locationFlushConsecutiveFailures = 0;
          lastFlushSkipReason = null;
        })
        .catch((e: unknown) => {
          locationFlushConsecutiveFailures++;
          const err = e as { message?: string; response?: { status?: number }; code?: string };
          log.warn("syncLocationQueue error", {
            message: err?.message ?? String(e),
            status: err?.response?.status,
            code: err?.code,
            failures: locationFlushConsecutiveFailures,
          });
        });
    } else {
      // Socket déconnecté (ex. app en arrière-plan) : fallback HTTP pour maintenir "en ligne"
      flushLatestPositionViaHttp()
        .then(() => {
          locationFlushConsecutiveFailures = 0;
          lastFlushSkipReason = null;
        })
        .catch((e: unknown) => {
          locationFlushConsecutiveFailures++;
          const err = e as { message?: string; response?: { status?: number }; code?: string };
          log.warn("HTTP location fallback error", {
            message: err?.message ?? String(e),
            status: err?.response?.status,
            code: err?.code,
            failures: locationFlushConsecutiveFailures,
          });
        });
    }
  };
  // En arrière-plan mission critical: flush immédiat.
  if (AppState.currentState !== "active" && missionCritical) {
    doFlush();
    return;
  }
  // En arrière-plan non critical (si appelé): différer.
  if (AppState.currentState !== "active") {
    InteractionManager.runAfterInteractions(doFlush);
  } else {
    doFlush();
  }
}

class SyncEngineImpl {
  private started = false;

  start(): void {
    if (this.started) {
      log.debug("syncEngine already started (double mount, no-op)");
      return;
    }
    initConnectivityPolicy();
    modeUnsub = subscribeToMode((mode: NetworkMode) => {
      log.debug("mode changed", { mode });
    });

    // Phase 3+5 : flush pendingActionsQueue + locationQueue sur online, foreground, socket connect
    networkUnsub = subscribeToNetworkState(() => {
      if (AppState.currentState !== "active") return;
      const net = getNetworkStateSnapshot();
      if (net?.isConnected === true && net?.isInternetReachable !== false) {
        locationFlushConsecutiveFailures = 0; // Reset backoff when network restored
        triggerPendingActionsFlush();
        triggerLocationFlush();
      }
    });

    const appSub = AppState.addEventListener("change", (next) => {
      if (next === "active") {
        locationFlushConsecutiveFailures = 0; // Reset backoff on foreground
        InteractionManager.runAfterInteractions(() => {
          setTimeout(() => {
            triggerPendingActionsFlush();
            triggerLocationFlush();
          }, 150);
        });
      }
    });
    appStateUnsub = () => appSub.remove();

    socketConnectUnsub = addSocketConnectListener(() => {
      locationFlushConsecutiveFailures = 0; // Reset backoff on reconnect
      triggerPendingActionsFlush();
      triggerLocationFlush();
    });

    schedulePolicyDrivenIntervals();

    reconciliationInterval = setInterval(() => {
      if (AppState.currentState !== "active") return;
      MissionStateManager.reconcileNow().catch((e: unknown) => {
        const err = e as { message?: string };
        log.warn("reconcileNow error", { message: err?.message ?? String(e) });
      });
    }, 3 * 60 * 1000);

    missionFallbackInterval = setInterval(() => {
      if (AppState.currentState !== "active" && !isMissionCriticalTrackingMode()) return;
      if (getSocketRole() === "driver") {
        triggerMissionResync(true).catch((e: unknown) => {
          const err = e as { message?: string };
          log.warn("triggerMissionResync error", { message: err?.message ?? String(e) });
        });
      }
    }, 60000);

    this.started = true;
    log.success("syncEngine started");
  }

  stop(): void {
    if (!this.started) {
      log.debug("syncEngine stop skipped (not started)");
      return;
    }
    modeUnsub?.();
    modeUnsub = null;
    networkUnsub?.();
    networkUnsub = null;
    appStateUnsub?.();
    appStateUnsub = null;
    socketConnectUnsub?.();
    socketConnectUnsub = null;
    if (locationFlushInterval) {
      clearInterval(locationFlushInterval);
      locationFlushInterval = null;
    }
    if (reconciliationInterval) {
      clearInterval(reconciliationInterval);
      reconciliationInterval = null;
    }
    if (missionFallbackInterval) {
      clearInterval(missionFallbackInterval);
      missionFallbackInterval = null;
    }
    if (missionHeartbeatInterval) {
      clearInterval(missionHeartbeatInterval);
      missionHeartbeatInterval = null;
    }
    if (presenceHeartbeatInterval) {
      clearInterval(presenceHeartbeatInterval);
      presenceHeartbeatInterval = null;
    }
    stopConnectivityPolicy();
    this.started = false;
    log.info("syncEngine stopped");
  }

  isStarted(): boolean {
    return this.started;
  }

  /** Recharge flush + heartbeats depuis `getLastResolvedTrackingPolicy()` (appelé après reconcile). */
  rescheduleIntervalsFromPolicy(): void {
    if (!this.started) return;
    schedulePolicyDrivenIntervals();
  }
}

/**
 * Retourne l'instance singleton du syncEngine.
 * Ne jamais créer de new SyncEngine() ailleurs.
 */
export function getSyncEngine(): SyncEngineImpl {
  if (!instance) {
    instance = new SyncEngineImpl();
  }
  return instance;
}

/** Après mise à jour de la policy ; no-op si syncEngine arrêté. */
export function rescheduleSyncEngineIntervalsFromPolicy(): void {
  const inst = getSyncEngine();
  if (inst.isStarted()) {
    inst.rescheduleIntervalsFromPolicy();
  }
}
