import { QueryClient } from "@tanstack/react-query";
import { getDriverMissions } from "./api/driverHttp";
import { driverQueryKeys } from "./queryKeys";
import { DriverMission, DriverSocketEvent } from "./types";
import { missionRuntimeManager } from "./services/missionRuntimeManager";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { emitPerfKpi } from "../../core/observability/perfKpi";
import { scheduleDriverMissionSync } from "./services/missionSyncOrchestrator";
import { mapDriverMission } from "./domain/missionMappers";
import { resolveMissionReassignConvergence } from "./services/missionReassignConvergence";
import { normalizeDriverEventType } from "../../core/realtime/eventContracts";
import { realtimeManager } from "../../core/realtime/realtimeManager";
import { isFeatureEnabled } from "../../core/featureFlags/registry";
import { AppState } from "react-native";
import { evaluateConnectivityPolicy } from "../../core/network/connectivityPolicy";
import { getNetworkSnapshot } from "../../core/network/networkState";
import { isTrackingActiveStatus } from "./domain/status";
import { normalizeDriverMissionStatus } from "./statusDictionary";
import { getTrackingSnapshot, updateDriverTrackingStatus } from "./tracking";

type RuntimeState = {
  contextId: string | null;
  timer: ReturnType<typeof setInterval> | null;
  lastFullPollingAtMs: number;
  lastNetworkTickAtMs: number;
  noMissionSinceMs: number | null;
};

type MissionPresenceSnapshot = {
  hasRelevantMission: boolean;
  missionCount: number;
};

type DriverRealtimePollingOptions = {
  getMissionPresence?: () => MissionPresenceSnapshot;
};

const runtime: RuntimeState = {
  contextId: null,
  timer: null,
  lastFullPollingAtMs: 0,
  lastNetworkTickAtMs: 0,
  noMissionSinceMs: null,
};
const RESYNC_COOLDOWN_MS = 10_000;
const POLLING_INTERVAL_MS = 15_000;
const HEALTHY_SOCKET_MIN_FULL_POLL_MS = Number(
  process.env.EXPO_PUBLIC_REALTIME_HEALTHY_SOCKET_MIN_FULL_POLL_MS ?? "60000"
);
const IDLE_HEARTBEAT_MS = Number(process.env.EXPO_PUBLIC_DRIVER_IDLE_HEARTBEAT_MS ?? "120000");
const BACKGROUND_HEARTBEAT_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_BACKGROUND_HEARTBEAT_MS ?? "180000"
);
const DEEP_IDLE_HEARTBEAT_MS = Number(process.env.EXPO_PUBLIC_DRIVER_DEEP_IDLE_HEARTBEAT_MS ?? "300000");
const IDLE_ASSIGNMENT_DETECTION_TARGET_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_IDLE_ASSIGNMENT_DETECTION_TARGET_MS ?? "60000"
);
const lastResyncAtByMission = new Map<number, number>();

function scheduleMissionResync(queryClient: QueryClient, contextId: string, missionId: number) {
  const now = Date.now();
  const last = lastResyncAtByMission.get(missionId) ?? 0;
  if (now - last < RESYNC_COOLDOWN_MS) return;
  lastResyncAtByMission.set(missionId, now);
  scheduleDriverMissionSync(queryClient, contextId, "manual");
}

function parseIso(input: string | null | undefined): number {
  if (!input) return Number.NaN;
  const value = Date.parse(input);
  return Number.isFinite(value) ? value : Number.NaN;
}

function emitMissionFreshnessMetric(
  source: string,
  contextId: string,
  mode: "socket" | "polling" | "reconcile",
  missionUpdatedAt: string | null | undefined
) {
  const updatedAtMs = parseIso(missionUpdatedAt ?? null);
  if (!Number.isFinite(updatedAtMs)) return;
  emitDriverTelemetry("realtime.mission.freshness", {
    source,
    context_id: contextId,
    realtime_transport_mode: mode,
    mission_state_freshness_ms: Math.max(0, Date.now() - updatedAtMs),
  });
}

function shouldApplyEvent(
  localMission: DriverMission | undefined,
  event: DriverSocketEvent
): boolean {
  const decision = missionRuntimeManager.shouldApplyRealtimeEvent(event);
  if (!decision.apply) {
    emitDriverTelemetry("realtime.event.ignored", {
      source: "driver.realtime",
      mission_id: event.mission_id,
      reason: decision.reason,
      event_sequence: event.event_sequence ?? null,
      event_id: event.event_id ?? null,
    });
    return false;
  }
  if (decision.gapDetected) {
    emitDriverTelemetry("realtime.event.sequence_gap", {
      source: "driver.realtime",
      mission_id: event.mission_id,
      event_sequence: event.event_sequence ?? null,
    });
  }

  const localUpdatedAt = parseIso((localMission?.updated_at as string | undefined) ?? null);
  const eventUpdatedAt = parseIso(event.updated_at);

  if (Number.isFinite(localUpdatedAt) && Number.isFinite(eventUpdatedAt)) {
    if (eventUpdatedAt < localUpdatedAt) {
      emitDriverTelemetry("realtime.event.ignored", {
        source: "driver.realtime",
        mission_id: event.mission_id,
        reason: "updated_at_old",
        event_sequence: event.event_sequence ?? null,
      });
      return false;
    }
  } else if (localMission) {
    // Fallback ordering: if we cannot guarantee ordering, force refetch before transition.
    return false;
  }
  return true;
}

function maybeStopTrackingOnTerminalMissionStatus(
  missionId: number,
  payload: Record<string, unknown>,
  localMission: DriverMission | undefined
): void {
  const rawStatus =
    typeof payload.status === "string"
      ? payload.status
      : typeof localMission?.status === "string"
        ? localMission.status
        : null;
  const normalized = normalizeDriverMissionStatus(rawStatus);
  if (isTrackingActiveStatus(normalized)) return;
  const tracking = getTrackingSnapshot();
  if (tracking.missionId !== missionId) return;
  updateDriverTrackingStatus(normalized);
}

export function applyDriverSocketEvent(
  queryClient: QueryClient,
  contextId: string,
  event: DriverSocketEvent
) {
  const canonicalType = normalizeDriverEventType(event.event_type);
  if (!canonicalType) {
    return;
  }
  const canonicalEvent: DriverSocketEvent = {
    ...event,
    event_type: canonicalType,
  };
  queryClient.setQueryData(driverQueryKeys.missions(contextId), (previous: unknown) => {
    const missions = Array.isArray(previous) ? ([...previous] as DriverMission[]) : [];
    const index = missions.findIndex((mission) => mission.id === canonicalEvent.mission_id);
    const localMission = index >= 0 ? missions[index] : undefined;
    const convergence = resolveMissionReassignConvergence(localMission, canonicalEvent);
    if (convergence.shouldForceResync) {
      emitDriverTelemetry("driver.runtime.resync", {
        source: "driver.realtime.reassign_convergence",
        context_id: contextId,
        mission_id: canonicalEvent.mission_id,
        trigger: convergence.reason,
      });
      void queryClient.invalidateQueries({
        queryKey: driverQueryKeys.missionDetail(contextId, canonicalEvent.mission_id),
      });
      scheduleMissionResync(queryClient, contextId, canonicalEvent.mission_id);
      return missions;
    }

    if (!shouldApplyEvent(localMission, canonicalEvent)) {
      // If ordering cannot be guaranteed, refetch snapshot before applying transition.
      void queryClient.invalidateQueries({
        queryKey: driverQueryKeys.missionDetail(contextId, canonicalEvent.mission_id),
      });
      scheduleMissionResync(queryClient, contextId, canonicalEvent.mission_id);
      return missions;
    }

    missionRuntimeManager.registerRealtimeEvent(canonicalEvent);

    const payload = canonicalEvent.payload ?? {};
    if (localMission) {
      missions[index] = mapDriverMission({
        ...localMission,
        ...payload,
        updated_at: canonicalEvent.updated_at ?? (localMission.updated_at as string | undefined) ?? null,
      });
      missionRuntimeManager.registerSnapshot(
        canonicalEvent.mission_id,
        String(missions[index]?.updated_at ?? canonicalEvent.updated_at ?? null)
      );
      emitMissionFreshnessMetric(
        "driver.realtime.socket",
        contextId,
        "socket",
        String(missions[index]?.updated_at ?? canonicalEvent.updated_at ?? null)
      );
      return missions;
    }
    missions.unshift(
      mapDriverMission({
        id: canonicalEvent.mission_id,
        status: String(payload.status ?? "ASSIGNED"),
        updated_at: canonicalEvent.updated_at ?? new Date().toISOString(),
        ...payload,
      })
    );
    missionRuntimeManager.registerSnapshot(canonicalEvent.mission_id, canonicalEvent.updated_at ?? null);
    emitMissionFreshnessMetric("driver.realtime.socket", contextId, "socket", canonicalEvent.updated_at);
    if (process.env.EXPO_PUBLIC_BOOKING_SOCKET_FIELD_LOG === "1") {
      const payload = canonicalEvent.payload ?? {};
      emitPerfKpi("perf.mission_received_to_ui", {
        source: "driver.realtime.socket",
        context_id: contextId,
        mission_id: canonicalEvent.mission_id,
        payload_keys: Object.keys(payload).slice(0, 40).join(","),
      });
    }
    return missions;
  });

  const payload = canonicalEvent.payload ?? {};
  if (typeof payload.status === "string" || canonicalType === "mission_status_changed") {
    const missions =
      (queryClient.getQueryData(driverQueryKeys.missions(contextId)) as DriverMission[] | undefined) ??
      [];
    const localMission = missions.find((mission) => mission.id === canonicalEvent.mission_id);
    maybeStopTrackingOnTerminalMissionStatus(
      canonicalEvent.mission_id,
      payload as Record<string, unknown>,
      localMission
    );
  }
}

export function startDriverRealtimePolling(queryClient: QueryClient, contextId: string) {
  return startDriverRealtimePollingWithOptions(queryClient, contextId, {});
}

export function startDriverRealtimePollingWithOptions(
  queryClient: QueryClient,
  contextId: string,
  options: DriverRealtimePollingOptions
) {
  runtime.contextId = contextId;
  if (runtime.timer) return;
  runtime.timer = setInterval(async () => {
    try {
      const snapshot = realtimeManager.getSnapshot();
      const missionPresence = options.getMissionPresence?.() ?? {
        hasRelevantMission: true,
        missionCount: 0,
      };
      const appState = AppState.currentState;
      const idleGating = isFeatureEnabled("driver_network_idle_gating_enabled");
      const bgGating = isFeatureEnabled("driver_background_network_gating_enabled");
      const harmonized = isFeatureEnabled("driver_sync_poll_harmonization_enabled");
      const adaptivePollingEnabled = isFeatureEnabled("realtime_adaptive_polling_enabled");
      const network2gCadenceEnabled = isFeatureEnabled("driver_network_2g_cadence_enabled");
      const connectivityPolicy = evaluateConnectivityPolicy(getNetworkSnapshot());
      const socketHealthy =
        snapshot.connected &&
        snapshot.actualTransport === "socket" &&
        !snapshot.degradedMode &&
        snapshot.authExhausted === false;
      if (!missionPresence.hasRelevantMission) {
        runtime.noMissionSinceMs = runtime.noMissionSinceMs ?? Date.now();
      } else {
        runtime.noMissionSinceMs = null;
      }
      const canEnterDeepIdle =
        bgGating &&
        appState !== "active" &&
        !missionPresence.hasRelevantMission &&
        socketHealthy &&
        snapshot.transportAuthority !== "degraded";
      const authority: "mission_active_loop" | "idle_loop" | "background_loop" | "deep_idle_loop" =
        missionPresence.hasRelevantMission
          ? "mission_active_loop"
          : canEnterDeepIdle
            ? "deep_idle_loop"
            : appState === "active"
              ? "idle_loop"
              : "background_loop";
      let minIntervalMs =
        authority === "mission_active_loop"
          ? POLLING_INTERVAL_MS
          : authority === "idle_loop"
            ? IDLE_HEARTBEAT_MS
            : authority === "background_loop"
              ? BACKGROUND_HEARTBEAT_MS
              : DEEP_IDLE_HEARTBEAT_MS;
      if (network2gCadenceEnabled) {
        minIntervalMs = Math.max(minIntervalMs, connectivityPolicy.recommendedSyncIntervalMs);
      }
      const now = Date.now();
      const shouldRateLimit = idleGating && now - runtime.lastNetworkTickAtMs < minIntervalMs;
      emitDriverTelemetry("driver.network.tick", {
        source: "driver.realtime",
        context_id: contextId,
        app_state: appState,
        network_activity_authority: authority,
        network_profile_active: connectivityPolicy.mode === "degraded" ? "poor" : connectivityPolicy.mode,
        mission_presence: missionPresence.hasRelevantMission ? "relevant" : "none",
        driver_network_tick_total: 1,
      });
      if (network2gCadenceEnabled) {
        emitDriverTelemetry("driver.network.profile", {
          source: "driver.realtime",
          context_id: contextId,
          network_profile_active: connectivityPolicy.mode === "degraded" ? "poor" : connectivityPolicy.mode,
          recommended_sync_interval_ms: connectivityPolicy.recommendedSyncIntervalMs,
        });
      }
      emitDriverTelemetry("driver.network.wake", {
        source: "driver.realtime",
        context_id: contextId,
        driver_network_wake_cause: "polling_tick",
        network_activity_authority: authority,
      });
      if (shouldRateLimit) {
        emitDriverTelemetry("driver.network.tick.skipped", {
          source: "driver.realtime",
          context_id: contextId,
          network_activity_authority: authority,
          driver_network_tick_skipped_total: 1,
          reason: "idle_rate_limit",
        });
        return;
      }
      if (
        adaptivePollingEnabled &&
        socketHealthy &&
        Date.now() - runtime.lastFullPollingAtMs < HEALTHY_SOCKET_MIN_FULL_POLL_MS
      ) {
        emitDriverTelemetry("realtime.polling.skipped", {
          source: "driver.realtime",
          context_id: contextId,
          realtime_transport_mode: snapshot.actualTransport,
          realtime_transport_authority: snapshot.transportAuthority,
          network_activity_authority: authority,
          mission_presence: missionPresence.hasRelevantMission ? "relevant" : "none",
        });
        return;
      }
      if (harmonized && snapshot.transportAuthority === "reconcile") {
        emitDriverTelemetry("driver.network.tick.skipped", {
          source: "driver.realtime",
          context_id: contextId,
          network_activity_authority: authority,
          driver_network_tick_skipped_total: 1,
          reason: "reconcile_authority",
        });
        return;
      }
      const missions = await getDriverMissions();
      runtime.lastNetworkTickAtMs = now;
      runtime.lastFullPollingAtMs = Date.now();
      realtimeManager.setTransportAuthority("polling", "full_polling_tick");
      emitDriverTelemetry("realtime.polling.full_refetch", {
        source: "driver.realtime",
        context_id: contextId,
        mission_poll_full_refetch_total: 1,
        driver_http_calls_per_hour: 1,
        network_activity_authority: authority,
      });
      const hasRelevantMissionAfterPoll = missions.some((mission) =>
        ["ASSIGNED", "EN_ROUTE", "ARRIVED", "IN_PROGRESS"].includes(String(mission.status ?? ""))
      );
      if (runtime.noMissionSinceMs && hasRelevantMissionAfterPoll) {
        emitDriverTelemetry("driver.network.tick", {
          source: "driver.realtime",
          context_id: contextId,
          mission_assignment_detection_latency_ms: Math.max(0, now - runtime.noMissionSinceMs),
          idle_assignment_detection_target_ms: IDLE_ASSIGNMENT_DETECTION_TARGET_MS,
        });
        runtime.noMissionSinceMs = null;
      }
      if (missions.length > 0) {
        emitMissionFreshnessMetric(
          "driver.realtime.polling",
          contextId,
          "polling",
          String(missions[0]?.updated_at ?? null)
        );
      }
      missions.forEach((mission) =>
        missionRuntimeManager.registerSnapshot(mission.id, String(mission.updated_at ?? null))
      );
      queryClient.setQueryData(driverQueryKeys.missions(contextId), missions);
    } catch (error) {
      emitDriverTelemetry("realtime.polling.failure", {
        source: "driver.realtime",
        context_id: contextId,
        reason: error instanceof Error ? error.message : "polling_failed",
      });
    }
  }, POLLING_INTERVAL_MS);
}

export function stopDriverRealtimePolling() {
  if (runtime.timer) {
    clearInterval(runtime.timer);
    runtime.timer = null;
  }
  runtime.contextId = null;
  runtime.lastFullPollingAtMs = 0;
  runtime.lastNetworkTickAtMs = 0;
  runtime.noMissionSinceMs = null;
}

