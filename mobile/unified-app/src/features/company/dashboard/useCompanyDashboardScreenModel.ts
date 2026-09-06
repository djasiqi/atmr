import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AppState } from "react-native";
import { isAxiosError } from "axios";
import { useFocusEffect } from "@react-navigation/native";
import { useQueryClient } from "@tanstack/react-query";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { useSession } from "../../../core/sessionProvider";
import { contextRealtimeRouter } from "../../../core/realtime/contextRealtimeRouter";
import { normalizeCompanyEventType } from "../../../core/realtime/eventContracts";
import {
  useCompanyFallbackPolling,
  useCompanyDashboardQuery,
  useCompanyDispatchMissionsQuery,
  useCompanyDispatchStatusQuery,
  useCompanyRealtimeInvalidation,
  useCompanyRealtimeStatus,
  useCompanyOptimizerStatusQuery,
  useCompanyInboxQuery,
} from "../hooks";
import { useCompanyDriverLiveTracking } from "../realtime/useCompanyDriverLiveTracking";
import { getDispatchApiErrorMessage } from "../api/companyApi";
import { resolveDriverStatus } from "../utils/companyDriverMapStatus";
import { emitCompanyDispatchTelemetry } from "../telemetry/companyTelemetry";
import { setStickyRidesFetchReason } from "../utils/ridesFetchReason";
import { useRetainDispatchQueryCache } from "../utils/dispatchQueryRetention";
import { endPageLoad, startPageLoad } from "../../../core/observability/perfKpi";
import { markBootMilestone } from "../../../core/observability/bootMilestones";
import { isCompanySessionNetworkReady } from "../../../core/network/companySessionNetworkGate";
import {
  markCompanyScreenUsable,
  measureCompanyDashboardPhase,
  recordCompanyDashboardPhase,
} from "../observability/companyDashboardPhases";
import { useCompanyBackgroundBootReady } from "../boot/companyColdStartPhase";
import {
  applyCompanyColdStartSnapshot,
  peekCompanyColdStartSnapshot,
} from "../boot/companyColdStartSnapshot";
import { usePersistCompanyColdStartSnapshot } from "../boot/usePersistCompanyColdStartSnapshot";
import {
  buildDashboardPresentation,
  getDashboardModeConfig,
  type CompanyDispatchMode,
  type CompanyOptimizerRuntime,
  type DashboardRuntimeMetrics,
} from "./dispatchDashboardPresentation";
import { buildCompanyDashboardUiModel } from "./companyDashboardViewModel";
import { IN_FLIGHT_MISSION_STATUSES, missionHasConfirmedPickupTime, toEpoch } from "./companyDashboardMissionUi";
import { OPTIMIZER_ENABLED } from "../dispatch/dispatchModeLock";

const HEALTHY_FRESHNESS_WINDOW_MS = 30_000;
const FOCUS_REFRESH_THROTTLE_MS = 3_000;
const STALE_DATA_MS = 30_000;

function getTodayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function resolveMissionIdFromEvent(payload: {
  mission_id?: unknown;
  booking_id?: unknown;
  id?: unknown;
}): number | undefined {
  const candidate = payload.mission_id ?? payload.booking_id ?? payload.id;
  if (typeof candidate === "number" && Number.isFinite(candidate)) return candidate;
  if (typeof candidate === "string") {
    const parsed = Number.parseInt(candidate, 10);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

export function useCompanyDashboardScreenModel() {
  const { activeContext } = useSession();
  const activeContextId = activeContext?.context_id ?? null;
  const queryClient = useQueryClient();
  const backgroundReady = useCompanyBackgroundBootReady();

  const [selectedDate, setSelectedDate] = useState(() => getTodayIsoDate());
  const [dateSheetOpen, setDateSheetOpen] = useState(false);

  const hydratedContextRef = useRef<string | null>(null);
  if (activeContextId && hydratedContextRef.current !== activeContextId) {
    const snapshot = peekCompanyColdStartSnapshot(activeContextId);
    if (snapshot) {
      applyCompanyColdStartSnapshot(queryClient, snapshot);
      markBootMilestone("CACHE_READY", { source: "cold_start_snapshot" });
    }
    hydratedContextRef.current = activeContextId;
  }
  const renderStartedAtRef = useRef(Date.now());
  renderStartedAtRef.current = Date.now();

  useRetainDispatchQueryCache("cockpit", activeContextId, selectedDate);
  const missionsQuery = useCompanyDispatchMissionsQuery({ date: selectedDate });
  const dashboardQuery = useCompanyDashboardQuery(selectedDate);
  const dispatchStatusQuery = useCompanyDispatchStatusQuery(selectedDate);
  const optimizerQuery = useCompanyOptimizerStatusQuery();
  const inboxQuery = useCompanyInboxQuery({ enabled: backgroundReady });
  const liveDrivers = useCompanyDriverLiveTracking();
  const realtime = useCompanyRealtimeStatus();
  const { invalidate } = useCompanyRealtimeInvalidation();
  const previousRealtimeStatus = useRef<string | null>(null);
  const lastOptimizerStatusTelemetryAtRef = useRef(0);
  const lastFocusRefreshAtRef = useRef(0);

  const missionsRefetch = missionsQuery.refetch;
  const dashboardRefetch = dashboardQuery.refetch;
  const dispatchStatusRefetch = dispatchStatusQuery.refetch;
  const optimizerRefetch = optimizerQuery.refetch;
  const liveDriversRefetch = liveDrivers.refetch;

  const refreshStaleOnly = useCallback(async (options?: { includeMissions?: boolean }) => {
    const now = Date.now();
    const tasks: Promise<unknown>[] = [];
    // OPT-04E : focus / AppState ne refetch pas les rides (J±1 + double GET).
    if (options?.includeMissions && now - missionsQuery.dataUpdatedAt > STALE_DATA_MS) {
      tasks.push(missionsRefetch());
    }
    if (now - dashboardQuery.dataUpdatedAt > STALE_DATA_MS) tasks.push(dashboardRefetch());
    if (now - dispatchStatusQuery.dataUpdatedAt > STALE_DATA_MS) {
      tasks.push(dispatchStatusRefetch());
    }
    // GPS flotte : refetch si snapshot absent ou trop vieux (secours sans event socket).
    const snapshotAgeMs = liveDrivers.snapshotRefreshedAt
      ? now - Date.parse(liveDrivers.snapshotRefreshedAt)
      : Number.POSITIVE_INFINITY;
    if (!Number.isFinite(snapshotAgeMs) || snapshotAgeMs > STALE_DATA_MS) {
      tasks.push(liveDriversRefetch());
    }
    if (tasks.length === 0) return;
    await Promise.all(tasks);
  }, [
    dashboardQuery.dataUpdatedAt,
    dashboardRefetch,
    dispatchStatusQuery.dataUpdatedAt,
    dispatchStatusRefetch,
    liveDrivers.snapshotRefreshedAt,
    liveDriversRefetch,
    missionsQuery.dataUpdatedAt,
    missionsRefetch,
  ]);

  const refreshAll = useCallback(async () => {
    setStickyRidesFetchReason("manual");
    const tasks: Promise<unknown>[] = [
      missionsRefetch(),
      dashboardRefetch(),
      dispatchStatusRefetch(),
      liveDriversRefetch(),
    ];
    if (OPTIMIZER_ENABLED) {
      const now = Date.now();
      if (now - lastOptimizerStatusTelemetryAtRef.current >= 1500) {
        lastOptimizerStatusTelemetryAtRef.current = now;
        emitCompanyDispatchTelemetry(
          "company.dispatch.optimizer_status_requested",
          {
            source: "company.dashboard.refresh",
            context_type: "company",
            context_id: activeContextId,
          },
          { allowWhenDisabled: true }
        );
      }
      tasks.push(optimizerRefetch());
    }
    await Promise.all(tasks);
  }, [
    activeContextId,
    dashboardRefetch,
    dispatchStatusRefetch,
    liveDriversRefetch,
    missionsRefetch,
    optimizerRefetch,
  ]);

  useEffect(() => {
    startPageLoad("company.dashboard");
    markBootMilestone("DASHBOARD_MOUNTED");
    emitCompanyDispatchTelemetry(
      "company.dispatch.opened",
      {
        source: "company.dashboard.screen",
        context_type: "company",
        context_id: activeContextId,
      },
      { allowWhenDisabled: true }
    );
  }, [activeContextId]);

  useEffect(() => {
    const firstScreenReady =
      missionsQuery.data != null ||
      dashboardQuery.data != null ||
      liveDrivers.drivers.length > 0 ||
      dashboardQuery.isSuccess ||
      missionsQuery.isSuccess;
    if (!firstScreenReady) return;
    endPageLoad("company.dashboard", "company.dashboard.data_ready");
    markBootMilestone("DASHBOARD_DATA_READY", {
      meaning: "cache_or_first_data",
      interactive: false,
    });
  }, [
    dashboardQuery.data,
    dashboardQuery.isSuccess,
    liveDrivers.drivers.length,
    missionsQuery.data,
    missionsQuery.isSuccess,
  ]);

  useEffect(() => {
    if (realtime.transportStatus !== "healthy") return;
    markBootMilestone("SOCKET_HEALTHY", { transport_status: realtime.transportStatus });
  }, [realtime.transportStatus]);

  useLayoutEffect(() => {
    recordCompanyDashboardPhase(
      "react_commit",
      Date.now() - renderStartedAtRef.current,
      {
        surface: "dashboard_model",
        driver_count: liveDrivers.drivers.length,
        mission_count: missionsQuery.data?.missions?.length ?? 0,
      }
    );
  });

  useLayoutEffect(() => {
    const firstScreenReady =
      missionsQuery.data != null ||
      dashboardQuery.data != null ||
      liveDrivers.drivers.length > 0 ||
      dashboardQuery.isSuccess ||
      missionsQuery.isSuccess;
    if (!firstScreenReady || !isCompanySessionNetworkReady()) return;
    markCompanyScreenUsable("company.dashboard", { trigger: "model_commit" });
  }, [
    dashboardQuery.data,
    dashboardQuery.isSuccess,
    liveDrivers.drivers.length,
    missionsQuery.data,
    missionsQuery.isSuccess,
  ]);

  useFocusEffect(
    useCallback(() => {
      const now = Date.now();
      if (now - lastFocusRefreshAtRef.current < FOCUS_REFRESH_THROTTLE_MS) return;
      lastFocusRefreshAtRef.current = now;
      void refreshStaleOnly();
    }, [refreshStaleOnly])
  );

  const pollWhenRealtimeDown = useCallback(
    () => refreshStaleOnly({ includeMissions: true }),
    [refreshStaleOnly]
  );
  useCompanyFallbackPolling(pollWhenRealtimeDown);

  useFocusEffect(
    useCallback(() => {
      if (!activeContext || activeContext.context_type !== "company") return undefined;
      return contextRealtimeRouter.subscribe(activeContext.context_id, (event) => {
        if (!event || typeof event !== "object") return;
        const eventPayload = event as {
          event_type?: string;
          booking_id?: unknown;
          mission_id?: unknown;
          id?: unknown;
        };
        const missionId = resolveMissionIdFromEvent(eventPayload);
        const normalizedEventType = normalizeCompanyEventType(eventPayload.event_type);
        if (normalizedEventType === "booking_created") {
          invalidate("booking_created", missionId);
        } else if (normalizedEventType === "booking_updated") {
          invalidate("booking_updated", missionId);
        } else if (normalizedEventType === "booking_cancelled") {
          invalidate("booking_cancelled", missionId);
        } else if (normalizedEventType === "urgent_alert") {
          invalidate("urgent_alert", missionId);
        } else if (normalizedEventType === "driver_location_update") {
          invalidate("driver_location_update");
        } else if (normalizedEventType === "optimizer_status_changed") {
          if (OPTIMIZER_ENABLED) {
            invalidate("optimizer_status_changed");
          }
        } else if (normalizedEventType === "delay_invalidated") {
          invalidate("delay_invalidated", missionId);
        } else if (normalizedEventType === "company_dispatch_update") {
          void refreshAll();
        } else if (normalizedEventType === "dispatch_assignment") {
          // Phase 2 PR B/C — gate D3.1 : event critical, invalidation ciblée
          // (dashboard + missions + ride detail si missionId).
          invalidate("dispatch_assignment", missionId);
        } else if (normalizedEventType === "dispatch_run_lifecycle") {
          // Phase 2 PR B/C — gate D3.1 : run started/completed/failed.
          invalidate("dispatch_run_lifecycle", missionId);
        }
      });
    }, [activeContext, invalidate, refreshAll])
  );

  useEffect(() => {
    const previousState = previousRealtimeStatus.current;
    emitCompanyDispatchTelemetry(
      "company.dispatch.socket_state_changed",
      {
        source: "company.dashboard.realtime",
        context_type: "company",
        context_id: activeContextId,
        previous_state: previousState,
        state: realtime.status,
        connected: realtime.connected,
        reason: realtime.lastError,
      },
      { allowWhenDisabled: true }
    );
    previousRealtimeStatus.current = realtime.status;
  }, [activeContextId, realtime.connected, realtime.lastError, realtime.status]);

  const hasUsableCockpitData =
    missionsQuery.data != null ||
    dashboardQuery.data != null ||
    liveDrivers.drivers.length > 0;

  const loading =
    !hasUsableCockpitData &&
    (missionsQuery.isLoading || dashboardQuery.isLoading || liveDrivers.isLoading);

  const error =
    missionsQuery.error ??
    dashboardQuery.error ??
    liveDrivers.error;

  const errMsg = !error
    ? ""
    : error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : "Erreur inconnue";
  const axiosStatus = isAxiosError(error) ? error.response?.status : undefined;
  const isAuthFailure = axiosStatus === 401 || axiosStatus === 403;
  const displayErrMsg = !error ? "" : getDispatchApiErrorMessage(error, errMsg || "Erreur inconnue");
  const isLikelyNetworkError = Boolean(
    errMsg &&
      !isAuthFailure &&
      /network|Network|fetch|Failed to fetch|connexion|Connexion|internet|Internet/i.test(errMsg)
  );

  const lastKnownSyncAt = useMemo(() => {
    const candidates = [
      missionsQuery.data?.refreshed_at ?? null,
      dashboardQuery.data?.refreshed_at ?? null,
      dispatchStatusQuery.data?.refreshed_at ?? null,
      optimizerQuery.data?.refreshed_at ?? null,
      liveDrivers.snapshotRefreshedAt ?? null,
      realtime.lastEventAt ?? null,
    ];
    return candidates
      .filter((candidate): candidate is string => typeof candidate === "string" && candidate.length > 0)
      .sort((a, b) => toEpoch(b) - toEpoch(a))[0] ?? null;
  }, [
    dashboardQuery.data?.refreshed_at,
    dispatchStatusQuery.data?.refreshed_at,
    liveDrivers.snapshotRefreshedAt,
    missionsQuery.data?.refreshed_at,
    optimizerQuery.data?.refreshed_at,
    realtime.lastEventAt,
  ]);

  const isPotentiallyStale =
    !isAuthFailure &&
    realtime.status !== "healthy" &&
    (!!error || (lastKnownSyncAt ? Date.now() - toEpoch(lastKnownSyncAt) > HEALTHY_FRESHNESS_WINDOW_MS : true));

  const missions = useMemo(() => missionsQuery.data?.missions ?? [], [missionsQuery.data?.missions]);

  const clientDelayedCount = useMemo(() => {
    const now = Date.now();
    let c = 0;
    for (const m of missions) {
      if (m.status === "completed" || m.status === "cancelled") continue;
      // Exclure les courses sans heure confirmée (legacy « À définir ») : sans horaire
      // de prise en charge fixé, elles ne peuvent pas être en retard.
      if (!missionHasConfirmedPickupTime(m)) continue;
      if (toEpoch(m.scheduled_at) < now) c += 1;
    }
    return c;
  }, [missions]);

  const { missionsPending, missionsInProgress, hasPendingOverdue, fleet } = useMemo(() => {
    let p = 0;
    let e = 0;
    const now = Date.now();
    for (const m of missions) {
      if (m.status === "pending" || m.status === "proposed" || m.status === "accepted") p += 1;
      if (IN_FLIGHT_MISSION_STATUSES.includes(m.status)) e += 1;
    }
    const hasPending = missions.some(
      (m) =>
        (m.status === "pending" || m.status === "proposed" || m.status === "accepted") &&
        missionHasConfirmedPickupTime(m) &&
        toEpoch(m.scheduled_at) < now
    );
    let enMission = 0;
    let dispo = 0;
    let off = 0;
    for (const d of liveDrivers.drivers) {
      const s = resolveDriverStatus(d);
      if (s === "en_mission") enMission += 1;
      else if (s === "available") dispo += 1;
      else off += 1;
    }
    return {
      missionsPending: p,
      missionsInProgress: e,
      hasPendingOverdue: hasPending,
      fleet: { enMission, dispo, off },
    };
  }, [missions, liveDrivers.drivers]);

  const onlineCount = useMemo(() => {
    let c = 0;
    for (const d of liveDrivers.drivers) {
      if (resolveDriverStatus(d) !== "offline") c += 1;
    }
    return c;
  }, [liveDrivers.drivers]);

  const driversAvailableCount = useMemo(() => {
    let a = 0;
    for (const d of liveDrivers.drivers) {
      if (resolveDriverStatus(d) === "available") a += 1;
    }
    return a;
  }, [liveDrivers.drivers]);

  const dataHealthLabel: "Temps réel" | "Repli" = useMemo(
    () => (!isPotentiallyStale && realtime.status === "healthy" && !error ? "Temps réel" : "Repli"),
    [error, isPotentiallyStale, realtime.status]
  );

  const dispatchMode: CompanyDispatchMode = useMemo(() => {
    const m = dispatchStatusQuery.data?.dispatch_mode ?? "unknown";
    if (m === "manual" || m === "semi_auto" || m === "fully_auto") return m;
    return "unknown";
  }, [dispatchStatusQuery.data?.dispatch_mode]);

  const headerMode =
    dispatchMode === "manual" || dispatchMode === "semi_auto" || dispatchMode === "fully_auto"
      ? dispatchMode
      : null;

  const dispatchState = dispatchStatusQuery.data?.dispatch_state ?? "unknown";
  const optimizer: CompanyOptimizerRuntime = useMemo(() => {
    if (!OPTIMIZER_ENABLED) {
      return { optimizerEnabled: false, optimizerState: "idle" };
    }
    const s = optimizerQuery.data?.status;
    const st = s?.optimizer_state;
    return {
      optimizerEnabled: s?.optimizer_enabled === true,
      optimizerState: st === "failed" || st === "degraded" || st === "running" || st === "idle" ? st : "idle",
    };
  }, [optimizerQuery.data?.status]);

  const config = useMemo(() => getDashboardModeConfig(dispatchMode), [dispatchMode]);
  const realtimeHealthyData = useMemo(
    () => realtime.status === "healthy" && !isPotentiallyStale,
    [isPotentiallyStale, realtime.status]
  );

  const presentationMetrics: DashboardRuntimeMetrics = useMemo(() => {
    const dash = dashboardQuery.data;
    const useClientDelayed = dispatchMode === "manual" || dispatchMode === "unknown";
    const delayedBookings = useClientDelayed ? clientDelayedCount : (dash?.delayed_bookings ?? 0);
    const delayedBookingsMetricsAvailable = useClientDelayed
      ? true
      : Boolean(
          dashboardQuery.isSuccess && dash != null && dash.delayed_bookings_metrics_available === true
        );
    const opportunitiesMetricsAvailable =
      dispatchMode !== "semi_auto"
        ? true
        : Boolean(
            dashboardQuery.isSuccess && dash != null && dash.opportunities_metrics_available === true
          );
    return {
      missions,
      missionsPending,
      missionsInProgress,
      delayedBookings,
      delayedBookingsMetricsAvailable,
      opportunities: dash?.opportunities ?? 0,
      opportunitiesMetricsAvailable,
      advancedCounts: undefined,
      driversAvailable: driversAvailableCount,
      driversEnMission: fleet.enMission,
      driversOffline: fleet.off,
      onlineDrivers: onlineCount,
      totalDrivers: liveDrivers.drivers.length,
      isPotentiallyStale,
      hasPendingOverdue,
      isLikelyNetworkError,
      errMsg: displayErrMsg,
      isAuthFailure,
      dataHealthLabel,
      realtimeHealthyData,
    };
  }, [
    clientDelayedCount,
    dashboardQuery.isSuccess,
    dashboardQuery.data,
    dataHealthLabel,
    dispatchMode,
    displayErrMsg,
    driversAvailableCount,
    fleet.enMission,
    fleet.off,
    hasPendingOverdue,
    isAuthFailure,
    isLikelyNetworkError,
    isPotentiallyStale,
    liveDrivers.drivers.length,
    missions,
    missionsInProgress,
    missionsPending,
    onlineCount,
    realtimeHealthyData,
  ]);

  const hasDispatchScreen = isFeatureEnabled("company_dispatch_screen_enabled");
  const presentationView = useMemo(
    () =>
      measureCompanyDashboardPhase("presentation", () =>
        buildDashboardPresentation({
          config,
          dispatchState: dispatchState as "idle" | "running" | "degraded" | "failed" | "unknown",
          optimizer,
          socketStatus: realtime.status,
          connected: realtime.connected,
          metrics: presentationMetrics,
          hasDispatchScreen,
        })
      ),
    [
      config,
      dispatchState,
      hasDispatchScreen,
      optimizer,
      presentationMetrics,
      realtime.connected,
      realtime.status,
    ]
  );

  const alertTexts = useMemo(
    () =>
      presentationView.alertLines.map((a) => ({
        id: a.id,
        text: a.text,
        isError: a.severity === "error",
      })),
    [presentationView.alertLines]
  );

  usePersistCompanyColdStartSnapshot({
    contextId: activeContextId,
    date: selectedDate,
    missions: missionsQuery.data,
    dashboard: dashboardQuery.data,
    drivers: liveDrivers.drivers,
    driversRefreshedAt: liveDrivers.snapshotRefreshedAt,
    dispatchStatus: dispatchStatusQuery.data,
  });

  const uiModel = useMemo(
    () =>
      measureCompanyDashboardPhase(
        "view_model",
        () =>
          buildCompanyDashboardUiModel({
            isLive: realtimeHealthyData,
            missions,
            drivers: liveDrivers.drivers,
            missionsPending,
            missionsInProgress,
            delayedBookings: presentationMetrics.delayedBookings,
            driversAvailable: driversAvailableCount,
            presentationView,
            alertTexts,
            inboxNotifications: inboxQuery.data?.notifications ?? [],
            selectedDateIso: selectedDate,
            loading,
          }),
        { driver_count: liveDrivers.drivers.length, mission_count: missions.length }
      ),
    [
      alertTexts,
      driversAvailableCount,
      inboxQuery.data?.notifications,
      liveDrivers.drivers,
      loading,
      missions,
      missionsInProgress,
      missionsPending,
      presentationMetrics.delayedBookings,
      presentationView,
      realtimeHealthyData,
      selectedDate,
    ]
  );

  useEffect(() => {
    const subscription = AppState.addEventListener("change", (nextState) => {
      if (nextState !== "active") return;
      if (
        realtime.status === "healthy" &&
        lastKnownSyncAt &&
        Date.now() - toEpoch(lastKnownSyncAt) < HEALTHY_FRESHNESS_WINDOW_MS
      ) {
        return;
      }
      void refreshStaleOnly();
    });
    return () => subscription.remove();
  }, [lastKnownSyncAt, realtime.status, refreshStaleOnly]);

  return {
    selectedDate,
    setSelectedDate,
    dateSheetOpen,
    setDateSheetOpen,
    headerMode,
    refreshAll,
    loading,
    error,
    displayErrMsg,
    isLikelyNetworkError,
    isAuthFailure,
    isPotentiallyStale,
    realtime,
    liveDrivers,
    missions,
    uiModel,
    presentationView,
    primaryCta: presentationView.primaryCta,
  };
}
