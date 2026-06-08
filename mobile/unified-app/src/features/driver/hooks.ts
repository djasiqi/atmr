import { useCallback, useEffect, useRef } from "react";
import { useFocusEffect } from "@react-navigation/native";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSession } from "../../core/sessionProvider";
import { realtimeManager } from "../../core/realtime/realtimeManager";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { getRuntimeFlagsVersion, isFeatureEnabled } from "../../core/featureFlags/registry";
import { QUERY_STALE_TIME_MS } from "../../core/queryStaleTimes";
import {
  getDriverCompanyBookingsToday,
  getDriverMissionEta,
  getDriverMissionDetail,
  getDriverMissions,
  getDriverMissionsSince,
  registerDriverPushToken,
  updateDriverAvailability,
  updateDriverMissionStatus,
} from "./api/driverHttp";
import { driverOfflineQueue } from "./offlineQueue";
import { driverQueryKeys, invalidateDriverMissionScope } from "./queryKeys";
import { startDriverRealtimeBridge } from "./services/driverRealtimeBridge";
import { missionRuntimeManager } from "./services/missionRuntimeManager";
import { startDriverSyncEngine } from "./services/syncEngine";
import {
  applyArrivedMilestoneFromStatusResponse,
  markDriverArrivedAtPickupMilestone,
  unmarkDriverArrivedAtPickupMilestone,
} from "./domain/missionMilestoneOverlay";
import {
  clearAllStepperApproachBaselines,
  clearPickupApproachForMission,
  resetPickupApproachBaseline,
  resetStepperApproachBaseline,
} from "./domain/missionStepperApproach";
import { scheduleDriverMissionSync } from "./services/missionSyncOrchestrator";
import { reconcileTrackingRuntime } from "./services/trackingReconcile";
import {
  flushTrackingQueue,
  getTrackingSnapshot,
  startDriverTracking,
  stopDriverTracking,
  updateDriverTrackingStatus,
} from "./tracking";
import { DriverMission, DriverMissionStatus, DriverTransitionStatus } from "./types";
import { normalizeDriverMissionStatus } from "./statusDictionary";
import { useDriverRuntimeResume } from "./runtimeResume";

export function useActiveDriverContextId(): string | null {
  const { activeContext } = useSession();
  if (!activeContext || activeContext.context_type !== "driver") return null;
  return activeContext.context_id;
}

export function useDriverMissionsQuery() {
  const contextId = useActiveDriverContextId();
  return useQuery({
    queryKey: contextId ? driverQueryKeys.missions(contextId) : ["driver-missions", "disabled"],
    queryFn: getDriverMissions,
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

function getLocalDayStartIso(): string {
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  return now.toISOString();
}

/** Missions modifiées depuis le début de journée locale (inclut statuts terminaux). */
export function useDriverTodayMissionsQuery() {
  const contextId = useActiveDriverContextId();
  return useQuery({
    queryKey: contextId
      ? [...driverQueryKeys.missions(contextId), "today-since-local-day-start"]
      : ["driver-missions-today", "disabled"],
    queryFn: () => getDriverMissionsSince(getLocalDayStartIso()),
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

/** Transports de l’entreprise prévus aujourd’hui (collègues inclus) — `/driver/me/company-bookings/today`. */
export function useDriverCompanyBookingsTodayQuery() {
  const contextId = useActiveDriverContextId();
  return useQuery({
    queryKey: contextId
      ? driverQueryKeys.companyBookingsToday(contextId)
      : ["driver-company-bookings-today", "disabled"],
    queryFn: getDriverCompanyBookingsToday,
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

export function useDriverMissionDetailQuery(missionId: number | null) {
  const contextId = useActiveDriverContextId();
  return useQuery({
    queryKey:
      contextId && missionId
        ? driverQueryKeys.missionDetail(contextId, missionId)
        : ["driver-mission-detail", "disabled", missionId ?? "none"],
    queryFn: () => getDriverMissionDetail(missionId as number),
    enabled: Boolean(contextId && missionId),
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

export function useDynamicEtaQuery(
  missionId: number | null,
  options?: { missionStatus?: string | null }
) {
  const contextId = useActiveDriverContextId();
  return useQuery({
    queryKey:
      contextId && missionId
        ? ["driver-mission-eta", contextId, missionId, options?.missionStatus ?? "unknown"]
        : ["driver-mission-eta", "disabled"],
    queryFn: () =>
      getDriverMissionEta(missionId as number, {
        missionStatus: options?.missionStatus ?? null,
      }),
    enabled: Boolean(contextId && missionId),
    refetchInterval: () => {
      const status = String(options?.missionStatus ?? "").toUpperCase();
      if (
        status === "COMPLETED" ||
        status === "CANCELLED" ||
        status === "RELEASED" ||
        status === "NO_SHOW"
      ) {
        return 60_000;
      }
      if (status === "EN_ROUTE" || status === "IN_PROGRESS" || status === "ARRIVED") {
        return 30_000;
      }
      return 60_000;
    },
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

export function useDriverStatusTransition() {
  const queryClient = useQueryClient();
  const contextId = useActiveDriverContextId();

  return useMutation({
    mutationFn: async (params: { missionId: number; targetStatus: DriverTransitionStatus; reason?: string | null }) => {
      if (!missionRuntimeManager.beginTransition(params.missionId, params.targetStatus)) {
        emitDriverTelemetry("transition.queue.failure", {
          source: "driver.hooks.transition",
          mission_id: params.missionId,
          reason: "transition_in_flight",
        });
        throw new Error("Transition already in-flight for this mission");
      }
      const queued = await driverOfflineQueue.enqueue(params.missionId, params.targetStatus);
      try {
        const res = await updateDriverMissionStatus({
          missionId: params.missionId,
          targetStatus: params.targetStatus,
          idempotencyKey: queued.id,
          reason: params.reason ?? null,
        });
        applyArrivedMilestoneFromStatusResponse(params.missionId, res);
      } catch (error) {
        const apiError = error as { retryable?: boolean; code?: string; message?: string };
        if (apiError.retryable === false) {
          await driverOfflineQueue.purgeMission(params.missionId);
          emitDriverTelemetry("transition.queue.failure", {
            source: "driver.hooks.transition",
            mission_id: params.missionId,
            reason: String(apiError.code ?? "transition_non_retryable"),
          });
          throw error;
        }
        // offline or transient backend issue: keep action queued for replay
      } finally {
        missionRuntimeManager.completeTransition(params.missionId);
      }
      return queued;
    },
    onMutate: async (params) => {
      if (!contextId) {
        return undefined;
      }
      const { missionId, targetStatus } = params;
      const detailKey = driverQueryKeys.missionDetail(contextId, missionId);
      const listKey = driverQueryKeys.missions(contextId);
      await queryClient.cancelQueries({ queryKey: detailKey });
      await queryClient.cancelQueries({ queryKey: listKey });
      const previousDetail = queryClient.getQueryData<DriverMission | undefined>(detailKey);
      const previousMissions = queryClient.getQueryData<DriverMission[] | undefined>(listKey);
      if (targetStatus === "ARRIVED") {
        markDriverArrivedAtPickupMilestone(missionId);
        clearPickupApproachForMission(missionId);
      }
      if (targetStatus === "EN_ROUTE") {
        resetPickupApproachBaseline(missionId);
      }
      if (targetStatus === "IN_PROGRESS") {
        resetStepperApproachBaseline(missionId, "dropoff");
      }
      if (targetStatus === "COMPLETED") {
        clearAllStepperApproachBaselines(missionId);
      }
      const nextStatus: DriverMissionStatus = targetStatus === "ARRIVED" ? "ARRIVED" : (targetStatus as DriverMissionStatus);
      queryClient.setQueryData<DriverMission | undefined>(detailKey, (old) => {
        if (!old || old.id !== missionId) {
          return old;
        }
        return { ...old, status: nextStatus };
      });
      queryClient.setQueryData<DriverMission[] | undefined>(listKey, (old) => {
        if (!Array.isArray(old)) {
          return old;
        }
        return old.map((m) => (m.id === missionId ? { ...m, status: nextStatus } : m));
      });
      updateDriverTrackingStatus(nextStatus);
      return { contextId, previousDetail, previousMissions, missionId, targetStatus };
    },
    onError: (_err, params, snap) => {
      if (!snap) {
        return;
      }
      queryClient.setQueryData(
        driverQueryKeys.missionDetail(snap.contextId, snap.missionId),
        snap.previousDetail
      );
      queryClient.setQueryData(driverQueryKeys.missions(snap.contextId), snap.previousMissions);
      if (params.targetStatus === "ARRIVED") {
        unmarkDriverArrivedAtPickupMilestone(params.missionId);
      }
    },
    onSuccess: (_, variables) => {
      if (!contextId) {
        updateDriverTrackingStatus(variables.targetStatus as DriverMissionStatus);
        return;
      }
      invalidateDriverMissionScope(queryClient, contextId, variables.missionId);
    },
  });
}

export function useDriverRealtimeSync() {
  const queryClient = useQueryClient();
  const contextId = useActiveDriverContextId();
  const { status } = useSession();
  const enableSocket = isFeatureEnabled("realtime_socket_enabled");

  const getMissionPresence = useCallback(() => {
    if (!contextId) {
      return {
        hasRelevantMission: false,
        missionCount: 0,
      };
    }
    const missions =
      (queryClient.getQueryData(driverQueryKeys.missions(contextId)) as DriverMission[] | undefined) ?? [];
    const hasRelevantMission = missions.some((mission) =>
      ["ASSIGNED", "EN_ROUTE", "ARRIVED", "IN_PROGRESS"].includes(String(mission.status ?? ""))
    );
    return {
      hasRelevantMission,
      missionCount: missions.length,
    };
  }, [contextId, queryClient]);

  useEffect(() => {
    if (!contextId) return;
    const dispose = startDriverRealtimeBridge(queryClient, contextId, {
      enableSocket,
      getMissionPresence,
    });
    const disposeSyncEngine = startDriverSyncEngine(queryClient, contextId, {
      getMissionPresence,
    });
    return () => {
      if (typeof dispose === "function") {
        dispose();
      }
      if (typeof disposeSyncEngine === "function") {
        disposeSyncEngine();
      }
    };
  }, [contextId, queryClient, enableSocket, getMissionPresence]);

  useDriverRuntimeResume({
    contextId,
    enabled: isFeatureEnabled("tracking_resume_resync_enabled"),
    onForegroundResume: async () => {
      await flushTrackingQueue();
    },
  });

  useEffect(() => {
    if (!contextId) return;
    const heartbeat = setInterval(() => {
      const realtime = realtimeManager.getSnapshot();
      const tracking = getTrackingSnapshot();
      const runtime = missionRuntimeManager.snapshot();
      emitDriverTelemetry("driver.runtime.heartbeat", {
        source: "driver.hooks.realtime",
        context_id: contextId,
        session_valid: status === "ready",
        socket_connected: realtime.connected,
        tracking_active: tracking.isRunning,
        app_state: tracking.appState,
        tracking_queue_depth: tracking.queueDepth,
        mission_runtime_known: runtime.knownMissions,
        mission_runtime_in_flight: runtime.inFlightTransitions,
        feature_flags_version: getRuntimeFlagsVersion(),
      });
    }, 60_000);
    return () => {
      clearInterval(heartbeat);
    };
  }, [contextId, status]);
}

export function useDriverTracking(missionId: number | null, missionStatus: string | null | undefined) {
  const contextId = useActiveDriverContextId();
  const normalized = normalizeDriverMissionStatus(missionStatus);
  useEffect(() => {
    if (contextId) {
      void reconcileTrackingRuntime({
        contextId,
        missionId,
        missionStatus: missionId ? normalized : null,
      });
    }
    if (!missionId) {
      stopDriverTracking();
      return;
    }
    startDriverTracking(missionId, normalized);
    return () => {
      stopDriverTracking();
    };
  }, [contextId, missionId, normalized]);
}

/**
 * À chaque retour de focus sur un écran qui dépend de la liste des missions
 * (dashboard, liste missions actives), on déclenche un resync best-effort :
 *   - schedule un sync orchestrator (delta /since si dispo, full sinon)
 *   - invalide explicitement le cache RQ pour forcer un re-fetch immédiat
 *
 * Garantit que les modifications faites par le dispatch (étage, code porte,
 * mobilité, horaire, adresse, montant…) sont visibles dès que le chauffeur
 * revient sur l'écran, même si le socket realtime est gated par feature flag
 * ou si l'event a été perdu (reconnexion réseau, app en background, etc.).
 */
const CONTEXT_SWITCH_RESYNC_GRACE_MS = 3000;

export function useDriverMissionsListFocusResync() {
  const queryClient = useQueryClient();
  const contextId = useActiveDriverContextId();
  const lastResyncAtRef = useRef(0);
  const contextSwitchGraceUntilRef = useRef(0);
  const previousContextIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!contextId) return;
    if (previousContextIdRef.current !== contextId) {
      previousContextIdRef.current = contextId;
      contextSwitchGraceUntilRef.current = Date.now() + CONTEXT_SWITCH_RESYNC_GRACE_MS;
    }
  }, [contextId]);

  useFocusEffect(
    useCallback(() => {
      if (!contextId) return;
      const now = Date.now();
      if (now < contextSwitchGraceUntilRef.current) return;
      // Anti-doublon : si on a déjà resync il y a < 1.5 s, on saute
      // (évite les boucles avec d'autres useEffect au mount).
      if (now - lastResyncAtRef.current < 1500) return;
      lastResyncAtRef.current = now;
      scheduleDriverMissionSync(queryClient, contextId, "manual");
      void queryClient.invalidateQueries({
        queryKey: driverQueryKeys.missions(contextId),
      });
      emitDriverTelemetry("driver.runtime.resync", {
        source: "driver.hooks.missions_list_focus",
        context_id: contextId,
        trigger: "missions_list_focus",
      });
    }, [contextId, queryClient])
  );
}

export function useDriverMissionDetailOpenResync(missionId: number | null) {
  const queryClient = useQueryClient();
  const contextId = useActiveDriverContextId();

  useEffect(() => {
    if (!contextId || !missionId) return;
    scheduleDriverMissionSync(queryClient, contextId, "manual");
    invalidateDriverMissionScope(queryClient, contextId, missionId);
    emitDriverTelemetry("driver.runtime.resync", {
      source: "driver.hooks.mission_detail_open",
      context_id: contextId,
      trigger: "mission_detail_open",
      mission_id: missionId,
    });
  }, [contextId, missionId, queryClient]);
}

export function useDriverPushRegistration() {
  return useMutation({
    mutationFn: registerDriverPushToken,
    onSuccess: (_, variables) => {
      emitDriverTelemetry("push.token.registered", {
        source: "driver.hooks.push",
        driver_id: String(variables.driverId),
      });
    },
  });
}

export function useDriverAvailabilityMutation() {
  return useMutation({
    mutationFn: (isAvailable: boolean) => updateDriverAvailability(isAvailable),
    onSuccess: (_, isAvailable) => {
      emitDriverTelemetry("driver.availability.updated", {
        source: "driver.hooks.availability",
        available: isAvailable,
      });
    },
  });
}

export { useDriverAvailability } from "./hooks/useDriverAvailability";
export { useTrackingState } from "./hooks/useTrackingState";
export { useSocketStatus } from "./hooks/useSocketStatus";
export { useNotifications } from "./hooks/useNotifications";
export { useNotificationActions } from "./hooks/useNotificationActions";
export { LIVE_TRACKING_TRANSITIONS } from "./services/missionLiveTrackingEligibility";
export { useMissionLiveTrackingGuard } from "./hooks/useMissionLiveTrackingGuard";
export { useTrackingAttentionState } from "./hooks/useTrackingAttentionState";

