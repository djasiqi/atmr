import { QueryClient } from "@tanstack/react-query";
import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { contextRealtimeRouter } from "../../../core/realtime/contextRealtimeRouter";
import { driverQueryKeys } from "../queryKeys";
import {
  applyDriverSocketEvent,
  startDriverRealtimePollingWithOptions,
  stopDriverRealtimePolling,
} from "../realtime";
import { DriverSocketEvent } from "../types";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { flushTrackingQueue } from "../tracking";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import {
  disposeDriverMissionSyncOrchestrator,
  scheduleDriverMissionSync,
} from "./missionSyncOrchestrator";
import { driverTrackingQueue } from "./driverTrackingQueue";
import { syncBridgeQueueDepthFromPersistence } from "./driverTrackingBridge";

type BridgeOptions = {
  enableSocket: boolean;
  getMissionPresence?: () => { hasRelevantMission: boolean; missionCount: number };
};

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
