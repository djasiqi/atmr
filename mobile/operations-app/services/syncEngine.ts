/**
 * syncEngine — Orchestre socket / HTTP / queue / flush selon la connectivityPolicy.
 * Plan migration 2G/3G — Phase 1–3.
 * Singleton ; start/stop dans _layout.tsx selon isDriverAuthenticated.
 */

import { AppState, InteractionManager } from "react-native";
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
import { getAdaptiveLocationTracker, getCurrentLocationMode } from "./locationTracker";
import { resolveLocationModeFromState, resolvePresenceState } from "./locationPresenceFsm";

const log = getLogger("SyncEngine");

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
  if (getSocketRole() !== "driver") return false;
  const state = resolvePresenceState({
    isAuthenticated: true,
    isDriver: true,
    hasFgPermission: true,
    hasBgPermission: true,
    appInBackground: AppState.currentState !== "active",
    hasActiveMission: MissionStateManager.isActive(),
    availabilityPresenceEnabled: true,
  });
  return resolveLocationModeFromState(state) === "mission_live";
}

function isAvailabilityPresenceMode(): boolean {
  return getSocketRole() === "driver" && getCurrentLocationMode() === "availability_presence";
}

function triggerPendingActionsFlush(): void {
  MissionStateManager.syncPendingActions().catch((e) => {
    log.warn("syncPendingActions error", { error: e });
  });
}

function triggerLocationFlush(): void {
  if (getSocketRole() !== "driver") return;
  const missionCritical = isMissionCriticalTrackingMode();
  const presenceMode = isAvailabilityPresenceMode();
  if (AppState.currentState !== "active" && !missionCritical && !presenceMode) return;
  const socket = getSocket();
  const doFlush = () => {
    if (socket?.connected) {
      syncLocationQueue(socket).catch((e) => {
        log.warn("syncLocationQueue error", { error: e });
      });
    } else {
      // Socket déconnecté (ex. app en arrière-plan) : fallback HTTP pour maintenir "en ligne"
      flushLatestPositionViaHttp().catch((e) => {
        log.warn("HTTP location fallback error", { error: e });
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
        triggerPendingActionsFlush();
        triggerLocationFlush();
      }
    });

    const appSub = AppState.addEventListener("change", (next) => {
      if (next === "active") {
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
      triggerPendingActionsFlush();
      triggerLocationFlush();
    });

    locationFlushInterval = setInterval(triggerLocationFlush, 15000);

    reconciliationInterval = setInterval(() => {
      if (AppState.currentState !== "active") return;
      MissionStateManager.reconcileNow().catch((e) => {
        log.warn("reconcileNow error", { error: e });
      });
    }, 3 * 60 * 1000);

    missionFallbackInterval = setInterval(() => {
      if (AppState.currentState !== "active" && !isMissionCriticalTrackingMode()) return;
      if (getSocketRole() === "driver") {
        triggerMissionResync(true).catch((e) => {
          log.warn("triggerMissionResync error", { error: e });
        });
      }
    }, 60000);

    missionHeartbeatInterval = setInterval(() => {
      if (AppState.currentState !== "active" && !isMissionCriticalTrackingMode()) return;
      if (getSocketRole() !== "driver") return;
      const socket = getSocket();
      if (!MissionStateManager.isActive()) return;
      if (!socket?.connected) {
        // En mission critical background, maintenir la présence via fallback HTTP.
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
    }, 60000);

    // Heartbeat présence hors mission: 180s en background (HTTP only).
    presenceHeartbeatInterval = setInterval(() => {
      if (AppState.currentState === "active") return;
      if (!isAvailabilityPresenceMode()) return;
      if (MissionStateManager.isActive()) return;
      flushLatestPositionViaHttp().catch((e) => {
        log.warn("presence heartbeat fallback error", { error: e });
      });
    }, 180000);

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
