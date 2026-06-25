import { QueryClient } from "@tanstack/react-query";
import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { contextRealtimeRouter } from "../../../core/realtime/contextRealtimeRouter";
import { driverQueryKeys } from "../queryKeys";
import {
  applyDriverSocketEvent,
  startDriverRealtimePollingWithOptions,
  stopDriverRealtimePolling,
} from "../realtime";
import { DriverMission, DriverSocketEvent } from "../types";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { flushTrackingQueue } from "../tracking";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import {
  disposeDriverMissionSyncOrchestrator,
  scheduleDriverMissionSync,
} from "./missionSyncOrchestrator";
import { driverTrackingQueue } from "./driverTrackingQueue";
import {
  syncBridgeQueueDepthFromPersistence,
  forceRestartTrackingWatchFromBridge,
  hardRestartDriverTrackingBridge,
} from "./driverTrackingBridge";
import { pickTrackingMission } from "../domain/pickTrackingMission";
import { isTrackingActiveStatus } from "../domain/status";
import { normalizeDriverMissionStatus } from "../statusDictionary";

type BridgeOptions = {
  enableSocket: boolean;
  getMissionPresence?: () => { hasRelevantMission: boolean; missionCount: number };
};

/**
 * Cooldown local des remote kicks. Le backend throttle déjà à 10 min/driver,
 * mais ce garde-fou évite le churn (stop/restart watch+FGS) en cas de rafale
 * d'events `force_tracking_restart` (replay socket à la reconnexion, plusieurs
 * workers backend avant la pose du throttle Redis).
 */
const REMOTE_KICK_COOLDOWN_MS = Number(
  process.env.EXPO_PUBLIC_REMOTE_KICK_COOLDOWN_MS ?? "60000"
);
let lastRemoteKickAtMs = 0;

/**
 * Kick backend `force_tracking_restart`. Le serveur n'émet ce kick que lorsqu'il
 * détecte un vrai problème (watchdog : pas de position fraîche). On fait donc
 * confiance au signal et on applique la **récupération forte** :
 *  - Si une mission live est résolvable (état du bridge OU cache React Query) →
 *    `hardRestartDriverTrackingBridge` : teardown complet du FGS natif + watch +
 *    engine, puis reconstruction. Couvre à la fois le cas « arrêté à froid »
 *    (après login/logout) et le FGS zombie (service vivant mais souscription
 *    GPS morte) — qu'un simple redémarrage de watch ne ressusciterait pas.
 *  - Sinon (aucune mission active, ex. présence) → redémarrage watch/FGS léger.
 */
function handleForceTrackingRestart(
  queryClient: QueryClient,
  contextId: string
): void {
  const nowMs = Date.now();
  if (nowMs - lastRemoteKickAtMs < REMOTE_KICK_COOLDOWN_MS) {
    emitDriverTelemetry("tracking.remote_kick.throttled", {
      source: "driver.realtime.bridge",
      context_id: contextId,
      since_last_ms: nowMs - lastRemoteKickAtMs,
      cooldown_ms: REMOTE_KICK_COOLDOWN_MS,
    });
    return;
  }
  lastRemoteKickAtMs = nowMs;

  // Mission de repli (utile uniquement si le bridge est à froid : missionId null).
  let fallback: Parameters<typeof hardRestartDriverTrackingBridge>[0] = null;
  try {
    const missions = queryClient.getQueryData(
      driverQueryKeys.missions(contextId)
    ) as DriverMission[] | undefined;
    const active = pickTrackingMission(missions);
    if (active?.id != null) {
      const normalized = normalizeDriverMissionStatus(active.status);
      if (isTrackingActiveStatus(normalized)) {
        fallback = {
          missionId: active.id,
          status: normalized,
          scheduling: {
            scheduled_time: active.scheduled_time ?? null,
            time_confirmed: active.time_confirmed ?? null,
            scheduling: active.scheduling ?? null,
          },
        };
      }
    }
  } catch {
    fallback = null;
  }

  void hardRestartDriverTrackingBridge(fallback, "backend_remote_kick").then(
    (restarted) => {
    if (restarted) {
      emitDriverTelemetry("tracking.remote_kick.hard_restart", {
        source: "driver.realtime.bridge",
        context_id: contextId,
        mission_id: fallback?.missionId ?? null,
      });
      return;
    }
    // Aucune mission live → repli léger (présence/legacy).
    void forceRestartTrackingWatchFromBridge("backend_remote_kick");
  });
}

const DEFAULT_OPTIONS: BridgeOptions = {
  enableSocket: isFeatureEnabled("realtime_socket_enabled"),
};

export function startDriverRealtimeBridge(
  queryClient: QueryClient,
  contextId: string,
  options: Partial<BridgeOptions> = {}
) {
  const effective = { ...DEFAULT_OPTIONS, ...options };
  let wasConnected = false;
  let lastReconnectResyncAtMs = 0;
  const reconnectResyncThrottleMs = Number(
    process.env.EXPO_PUBLIC_REALTIME_RECONNECT_RESYNC_THROTTLE_MS ?? "3000"
  );
  realtimeManager.connect(contextId, { enableSocket: effective.enableSocket });
  startDriverRealtimePollingWithOptions(queryClient, contextId, {
    getMissionPresence: effective.getMissionPresence,
  });

  const unsubscribeLifecycle = realtimeManager.subscribe((snapshot) => {
    const justReconnected = !wasConnected && snapshot.connected;
    const transitionGateEnabled = isFeatureEnabled("realtime_resync_transition_gate_enabled");
    const shouldResync = snapshot.activeContextId === contextId && justReconnected;
    if (shouldResync) {
      const now = Date.now();
      if (transitionGateEnabled && now - lastReconnectResyncAtMs < reconnectResyncThrottleMs) {
        return;
      }
      lastReconnectResyncAtMs = now;
      scheduleDriverMissionSync(queryClient, contextId, "reconnect");
      if (isFeatureEnabled("tracking_resume_resync_enabled")) {
        void flushTrackingQueue();
      }
      queryClient.setQueryData(driverQueryKeys.syncState(contextId), {
        last_sync_at: new Date().toISOString(),
        mode: snapshot.mode,
      });
    }
    const justDisconnected = wasConnected && !snapshot.connected;
    if (
      justDisconnected &&
      isFeatureEnabled("tracking_real_ack_semantics_enabled")
    ) {
      void driverTrackingQueue.releaseSocketEmittedForHttpRetry().then(() =>
        flushTrackingQueue().then(() => syncBridgeQueueDepthFromPersistence())
      );
    }
    wasConnected = snapshot.connected;
  });

  const unsubscribeContextEvents = contextRealtimeRouter.subscribe(contextId, (rawEvent) => {
    const event = rawEvent as DriverSocketEvent;
    if (!event || typeof event !== "object" || typeof event.mission_id !== "number") {
      return;
    }
    applyDriverSocketEvent(queryClient, contextId, event);
  });

  const unsubscribeDriverEvents = realtimeManager.subscribeDriverEvents((rawEvent) => {
    const event = rawEvent as DriverSocketEvent | { event_type?: string; payload?: unknown };
    if (!event || typeof event !== "object") {
      return;
    }
    if (event.event_type === "driver_location_batch_ack") {
      const payload = (event as { payload?: unknown }).payload as
        | {
            tracking_event_ids?: unknown;
            tracking_event_id?: unknown;
            ack_last_sequence_id?: unknown;
          }
        | undefined;
      const trackingEventIds = Array.isArray(payload?.tracking_event_ids)
        ? payload?.tracking_event_ids.filter((value): value is string => typeof value === "string")
        : typeof payload?.tracking_event_id === "string"
          ? [payload.tracking_event_id]
          : [];
      if (trackingEventIds.length > 0) {
        void driverTrackingQueue.markBackendAckedByIds(trackingEventIds).then(() =>
          syncBridgeQueueDepthFromPersistence()
        );
      }
      const ackLastSequenceId =
        typeof payload?.ack_last_sequence_id === "number"
          ? payload.ack_last_sequence_id
          : typeof payload?.ack_last_sequence_id === "string"
            ? Number(payload.ack_last_sequence_id)
            : null;
      if (ackLastSequenceId && Number.isFinite(ackLastSequenceId)) {
        void driverTrackingQueue
          .markBackendAckedByWatermark(ackLastSequenceId)
          .then(() => syncBridgeQueueDepthFromPersistence());
      }
      emitDriverTelemetry("tracking.batch.ack", {
        source: "driver.realtime.bridge",
        context_id: contextId,
        backend_acked_count: trackingEventIds.length,
        ack_last_sequence_id: ackLastSequenceId,
      });
      return;
    }
    if (event.event_type === "rate_limit_exceeded") {
      void driverTrackingQueue.releaseSocketEmittedForHttpRetry().then(() =>
        flushTrackingQueue().then(() => syncBridgeQueueDepthFromPersistence())
      );
      emitDriverTelemetry("tracking.batch.rate_limited", {
        source: "driver.realtime.bridge",
        context_id: contextId,
      });
      return;
    }
    if (event.event_type === "force_tracking_restart") {
      handleForceTrackingRestart(queryClient, contextId);
      emitDriverTelemetry("tracking.remote_kick.received", {
        source: "driver.realtime.bridge",
        context_id: contextId,
      });
      return;
    }
    if (event.event_type === "eta_changed" && typeof (event as DriverSocketEvent).mission_id === "number") {
      applyDriverSocketEvent(queryClient, contextId, event as DriverSocketEvent);
      return;
    }
    if (typeof (event as DriverSocketEvent).mission_id !== "number") return;
    const payload = event.payload as { context_id?: unknown } | undefined;
    const eventContextId = typeof payload?.context_id === "string" ? payload.context_id : contextId;
    contextRealtimeRouter.dispatch(eventContextId, event as DriverSocketEvent);
  });

  return () => {
    unsubscribeLifecycle();
    unsubscribeContextEvents();
    unsubscribeDriverEvents();
    stopDriverRealtimePolling();
    disposeDriverMissionSyncOrchestrator(contextId);
    // Ne pas appeler disconnect() ici : le socket est partagé (layout chauffeur, chat).
    // La déconnexion est gérée par sessionProvider (logout / changement de contexte).
  };
}
