import { useCallback, useEffect, useMemo, useRef, useState, type ComponentProps } from "react";
import { isAxiosError } from "axios";
import { AppState, Platform, Pressable, RefreshControl, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { PermissionGuard } from "../../../src/core/guards";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { useSession } from "../../../src/core/sessionProvider";
import {
  useActiveCompanyContextId,
  useCompanyFallbackPolling,
  useCompanyDashboardQuery,
  useCompanyDispatchMissionsQuery,
  useCompanyDispatchStatusQuery,
  useCompanyRealtimeInvalidation,
  useCompanyRealtimeStatus,
  useCompanyOptimizerStatusQuery,
} from "../../../src/features/company/hooks";
import { emitCompanyDispatchTelemetry } from "../../../src/features/company/telemetry/companyTelemetry";
import { contextRealtimeRouter } from "../../../src/core/realtime/contextRealtimeRouter";
import { useCompanyDriverLiveTracking } from "../../../src/features/company/realtime/useCompanyDriverLiveTracking";
import { normalizeCompanyEventType } from "../../../src/core/realtime/eventContracts";
import type { CompanyDispatchMissionStatus } from "../../../src/features/company/api/contracts";
import { getDispatchApiErrorMessage, switchCompanyDispatchMode } from "../../../src/features/company/api/companyApi";
import { resolveDriverStatus } from "../../../src/features/company/utils/companyDriverMapStatus";
import {
  buildDashboardPresentation,
  getDashboardModeConfig,
  type CompanyDispatchMode,
  type CompanyOptimizerRuntime,
  type DashboardRuntimeMetrics,
} from "../../../src/features/company/dashboard/dispatchDashboardPresentation";
import { AppText, Screen } from "../../../src/design/responsive";
import { semanticDanger, semanticWarning } from "../../../src/design/responsive/colors";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";
import { createShadow } from "../../../src/styles/shadowStyles";
import { EnterpriseDriversMap } from "../../../src/features/company/components/EnterpriseDriversMap";
import { EnterpriseHeader } from "../../../src/features/company/components/EnterpriseHeader";
import { DayPickerSheet } from "../../../src/features/company/components/DayPickerSheet";
import { DispatchModeSheet, type DispatchModeValue } from "../../../src/features/company/components/DispatchModeSheet";

function getTodayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function toEpoch(value: string | null | undefined): number {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

const HEALTHY_FRESHNESS_WINDOW_MS = 30_000;

/** Même découpe que la pastille En course (listes) : seulement trajet actif. */
const IN_FLIGHT: CompanyDispatchMissionStatus[] = ["en_route", "in_progress"];

/**
 * Référence UX/UI : `operations-app/app/(enterprise)/dashboard.tsx` + `_layout.tsx`.
 * - Fond scroll : `enterprisePalette.background` → même valeur que `E.BG` (#f4f7fc).
 * - Bordures cartes / sections : `rgba(0,121,107,0.08)` → `E.BORDER`.
 * - Ombres sections : même `createShadow` que les `<Section>` operations-app (opacité 0.04).
 */
const dashboardSurfaceShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

/** Couleurs locales (alertes / hero carte) : palette alignée operations-app / enterpriseOpsTheme. */
const C = {
  pageBg: E.BG,
  cardBg: E.CARD,
  text: E.TEXT,
  textMuted: E.TEXT_MUTED,
  textSub: E.TEXT_SEC,
  border: E.BORDER,
  brand: E.BRAND,
  brandSoft: "rgba(0, 121, 107, 0.1)",
  err: E.DANGER,
  mapHeroBg: "#0F172A",
  mapHeroMuted: "rgba(226, 232, 240, 0.85)",
  mapHeroAccent: "#2DD4BF",
  networkHint: "#C2410C",
} as const;

type KpiIconName = ComponentProps<typeof Ionicons>["name"];

const SWISS_TZ = "Europe/Zurich";

/** Aperçu prochaine course : date + heure (Suisse) sur une ligne courte. */
function formatNextCourseWhen(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (!Number.isFinite(d.getTime())) return "—";
  return d.toLocaleString("fr-CH", {
    timeZone: SWISS_TZ,
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Premiers mots d’adresse (avant la première virgule) pour rester lisible. */
function conciseAddressSegment(s: string | null | undefined, maxLen = 48): string {
  const t = s?.trim() ?? "";
  if (!t) return "—";
  const head = t.split(",")[0]?.trim() || t;
  if (head.length <= maxLen) return head;
  return `${head.slice(0, Math.max(0, maxLen - 1))}…`;
}

function resolveMissionIdFromEvent(payload: {
  mission_id?: unknown;
  booking_id?: unknown;
  id?: unknown;
}): number | undefined {
  const candidate = payload.mission_id ?? payload.booking_id ?? payload.id;
  if (typeof candidate === "number" && Number.isFinite(candidate)) {
    return candidate;
  }
  if (typeof candidate === "string") {
    const parsed = Number.parseInt(candidate, 10);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function KpiTile({
  def,
  display,
}: {
  def: { label: string; icon: KpiIconName };
  display: { kind: "value" | "unavailable" | "hidden"; line1: string; line2?: string };
}) {
  if (display.kind === "hidden") return null;
  const isUnavailable = display.kind === "unavailable";
  return (
    <View style={styles.kpiStat} accessibilityLabel={def.label}>
      <View style={styles.kpiTopRow}>
        <View style={styles.kpiIconWrap} accessibilityElementsHidden>
          <Ionicons name={def.icon} size={16} color={C.brand} />
        </View>
        <View style={styles.kpiTextCol}>
          <AppText variant="caption" style={styles.kpiLabel} numberOfLines={1}>
            {def.label}
          </AppText>
          {isUnavailable ? (
            <View>
              <AppText variant="sectionTitle" style={styles.kpiValue} accessibilityLabel="Non disponible —">
                —
              </AppText>
              <AppText variant="caption" style={styles.kpiSubUnavailable}>
                Non disponible
              </AppText>
            </View>
          ) : (
            <AppText variant="sectionTitle" style={styles.kpiValue} numberOfLines={1} adjustsFontSizeToFit>
              {display.line1}
            </AppText>
          )}
        </View>
      </View>
    </View>
  );
}

function AlertNotificationRow({
  severity,
  text,
}: {
  severity: "error" | "warning";
  text: string;
}) {
  const isErr = severity === "error";
  return (
    <View
      style={[styles.alertNotifRow, isErr ? styles.alertNotifRowErr : styles.alertNotifRowWarn]}
      accessibilityRole="text"
    >
      <View style={styles.alertNotifIconSlot} accessibilityElementsHidden>
        <Ionicons
          name={isErr ? "alert-circle" : "warning-outline"}
          size={17}
          color={isErr ? semanticDanger.fgStrong : semanticWarning.fg}
        />
      </View>
      <AppText
        variant="caption"
        style={[styles.alertNotifText, isErr ? styles.alertNotifTextErr : styles.alertNotifTextWarn]}
        numberOfLines={5}
      >
        {text}
      </AppText>
    </View>
  );
}

export default function CompanyDashboardScreen() {
  const { activeContext, can } = useSession();
  const activeContextId = activeContext?.context_id ?? null;
  const contextId = useActiveCompanyContextId();
  const roleGuardsEnabled = isFeatureEnabled("company_mobile_role_guards_enabled");
  const contextPermissions = activeContext?.permissions ?? [];
  const canRunSensitiveAction = (permission: string, fallbackPermission: string) => {
    if (!roleGuardsEnabled) return true;
    if (contextPermissions.includes(permission)) return can(permission);
    return can(fallbackPermission);
  };
  const canDispatchManage = canRunSensitiveAction("company:dispatch:manage", "company:dashboard:read");
  const [selectedDate, setSelectedDate] = useState(() => getTodayIsoDate());
  const [dateSheetOpen, setDateSheetOpen] = useState(false);
  const [modeSheetOpen, setModeSheetOpen] = useState(false);
  const missionsQuery = useCompanyDispatchMissionsQuery({ date: selectedDate });
  const dashboardQuery = useCompanyDashboardQuery(selectedDate);
  const dispatchStatusQuery = useCompanyDispatchStatusQuery(selectedDate);
  const optimizerQuery = useCompanyOptimizerStatusQuery();
  const liveDrivers = useCompanyDriverLiveTracking();
  const liveDriversRefetch = liveDrivers.refetch;
  const realtime = useCompanyRealtimeStatus();
  const previousRealtimeStatus = useRef<string | null>(null);
  const lastOptimizerStatusTelemetryAtRef = useRef(0);
  const { invalidate } = useCompanyRealtimeInvalidation();
  const missionsRefetch = missionsQuery.refetch;
  const dashboardRefetch = dashboardQuery.refetch;
  const dispatchStatusRefetch = dispatchStatusQuery.refetch;
  const optimizerRefetch = optimizerQuery.refetch;
  const router = useRouter();

  const applyDispatchModeFromSheet = useCallback(
    async (mode: DispatchModeValue) => {
      if (!contextId || !canDispatchManage) return;
      try {
        await switchCompanyDispatchMode({ contextId, mode });
        setModeSheetOpen(false);
        await Promise.all([dispatchStatusRefetch(), dashboardRefetch(), missionsRefetch()]);
      } catch {
        // Garder la feuille ouverte pour réessayer.
      }
    },
    [canDispatchManage, contextId, dashboardRefetch, dispatchStatusRefetch, missionsRefetch]
  );

  const refreshAll = useCallback(async () => {
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
    await Promise.all([
      missionsRefetch(),
      dashboardRefetch(),
      dispatchStatusRefetch(),
      optimizerRefetch(),
      liveDriversRefetch(),
    ]);
  }, [
    activeContextId,
    dashboardRefetch,
    dispatchStatusRefetch,
    liveDriversRefetch,
    missionsRefetch,
    optimizerRefetch,
  ]);

  useEffect(() => {
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

  useFocusEffect(
    useCallback(() => {
      void refreshAll();
    }, [refreshAll])
  );

  useCompanyFallbackPolling(refreshAll);

  useEffect(() => {
    if (!activeContext || activeContext.context_type !== "company") return;
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
      if (normalizedEventType === "booking_updated") {
        invalidate("booking_updated", missionId);
      } else if (normalizedEventType === "booking_cancelled") {
        invalidate("booking_cancelled", missionId);
      } else if (normalizedEventType === "driver_location_update") {
        invalidate("driver_location_update");
      } else if (normalizedEventType === "optimizer_status_changed") {
        invalidate("optimizer_status_changed");
      } else if (normalizedEventType === "delay_invalidated") {
        invalidate("delay_invalidated", missionId);
      } else if (normalizedEventType === "company_dispatch_update") {
        void refreshAll();
      }
    });
  }, [activeContext, invalidate, refreshAll]);

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

  const loading =
    missionsQuery.isLoading ||
    dashboardQuery.isLoading ||
    dispatchStatusQuery.isLoading ||
    optimizerQuery.isLoading ||
    liveDrivers.isLoading;
  const error =
    missionsQuery.error ??
    dashboardQuery.error ??
    dispatchStatusQuery.error ??
    optimizerQuery.error ??
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

  const isPotentiallyStale =
    !isAuthFailure &&
    realtime.status !== "healthy" &&
    (!!error || (lastKnownSyncAt ? Date.now() - toEpoch(lastKnownSyncAt) > HEALTHY_FRESHNESS_WINDOW_MS : true));

  const missions = useMemo(
    () => missionsQuery.data?.missions ?? [],
    [missionsQuery.data?.missions]
  );

  const clientDelayedCount = useMemo(() => {
    const now = Date.now();
    let c = 0;
    for (const m of missions) {
      if (m.status === "completed" || m.status === "cancelled") continue;
      if (!m.scheduled_at) continue;
      if (toEpoch(m.scheduled_at) < now) c += 1;
    }
    return c;
  }, [missions]);

  const { missionsPending, missionsInProgress, hasPendingOverdue, fleet, nextMissions } = useMemo(() => {
    let p = 0;
    let e = 0;
    for (const m of missions) {
      if (m.status === "pending" || m.status === "proposed" || m.status === "accepted") p += 1;
      if (IN_FLIGHT.includes(m.status)) e += 1;
    }
    const now = Date.now();
    const hasPending =
      missions.some(
        (m) =>
          (m.status === "pending" || m.status === "proposed" || m.status === "accepted") &&
          Boolean(m.scheduled_at) &&
          toEpoch(m.scheduled_at) < now
      ) ?? false;
    let enMission = 0;
    let dispo = 0;
    let off = 0;
    for (const d of liveDrivers.drivers) {
      const s = resolveDriverStatus(d);
      if (s === "en_mission") enMission += 1;
      else if (s === "available") dispo += 1;
      else off += 1;
    }
    const list = missions
      .filter((m) => m.status !== "completed" && m.status !== "cancelled")
      .sort((a, b) => toEpoch(a.scheduled_at) - toEpoch(b.scheduled_at))
      .slice(0, 3);
    return {
      missionsPending: p,
      missionsInProgress: e,
      hasPendingOverdue: hasPending,
      fleet: { enMission, dispo, off },
      nextMissions: list,
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
    () =>
      !isPotentiallyStale && realtime.status === "healthy" && !error
        ? "Temps réel"
        : "Repli",
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
          dashboardQuery.isSuccess &&
            dash != null &&
            dash.delayed_bookings_metrics_available === true
        );
    const opportunitiesMetricsAvailable =
      dispatchMode !== "semi_auto"
        ? true
        : Boolean(
            dashboardQuery.isSuccess &&
              dash != null &&
              dash.opportunities_metrics_available === true
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
  const view = useMemo(
    () =>
      buildDashboardPresentation({
        config,
        dispatchState: dispatchState as "idle" | "running" | "degraded" | "failed" | "unknown",
        optimizer,
        socketStatus: realtime.status,
        connected: realtime.connected,
        metrics: presentationMetrics,
        hasDispatchScreen,
      }),
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
      void refreshAll();
    });
    return () => subscription.remove();
  }, [lastKnownSyncAt, realtime.status, refreshAll]);

  const onAllRides = useCallback(() => {
    router.push("/(app)/(company)/rides");
  }, [router]);

  return (
    <PermissionGuard permission="company:dashboard:read">
      <Screen
        scroll
        backgroundColor={C.pageBg}
        withHorizontalPadding
        stickyHeader={
          <EnterpriseHeader
            metaDetail="networkOnly"
            date={selectedDate}
            mode={headerMode}
            realtimeStatus={realtime.status}
            onOpenDatePicker={() => setDateSheetOpen(true)}
            onOpenModePicker={canDispatchManage ? () => setModeSheetOpen(true) : undefined}
          />
        }
        extraScrollBottomPadding={80}
        contentContainerStyle={[styles.page, { backgroundColor: C.pageBg }]}
        refreshControl={
          <RefreshControl refreshing={loading} onRefresh={() => void refreshAll()} tintColor={C.brand} />
        }
      >
        <View style={styles.mapHeroShell}>
          <EnterpriseDriversMap
            drivers={liveDrivers.drivers}
            mapHeight={236}
            containerStyle={styles.mapHeroInner}
          />
        </View>

        <View style={styles.kpiRow} accessibilityLabel="Indicateurs clés">
          {view.kpi
            .map((row) => {
              const d = row.display;
              if (d.kind === "value") {
                return (
                  <KpiTile
                    key={row.def.key}
                    def={row.def}
                    display={{ kind: "value", line1: d.line1, line2: d.line2 }}
                  />
                );
              }
              if (d.kind === "unavailable") {
                return <KpiTile key={row.def.key} def={row.def} display={{ kind: "unavailable", line1: "—" }} />;
              }
              return null;
            })
            .filter(Boolean)}
        </View>

        {missions.length === 0 && !loading ? (
          <View style={styles.emptyState} accessibilityRole="text">
            <View style={styles.emptyStateIcon} accessibilityElementsHidden>
              <Ionicons name="car-outline" size={22} color={C.brand} />
            </View>
            <AppText variant="body" style={styles.emptyStateTitle}>
              Aucune course planifiée
            </AppText>
            <AppText variant="caption" style={styles.emptyStateSubtitle}>
              Les courses apparaîtront ici dès qu’elles seront créées.
            </AppText>
          </View>
        ) : null}

        {nextMissions.length > 0 ? (
          <View style={styles.dashboardSection}>
            <View style={styles.summaryHeaderRow}>
              <View style={styles.sectionIconWrap} accessibilityElementsHidden>
                <Ionicons name="time-outline" size={16} color={C.brand} />
              </View>
              <AppText variant="sectionTitle" style={styles.summaryTitle}>
                Prochaines courses
              </AppText>
            </View>
            <View style={styles.sectionBody}>
              {nextMissions.map((m, index) => (
                <View
                  key={m.mission_id}
                  style={[
                    styles.missionBlock,
                    index < nextMissions.length - 1 && styles.missionBlockSep,
                  ]}
                  accessibilityLabel={`Prochaine course ${m.mission_id}`}
                >
                  <AppText variant="label" style={styles.missionWhen} numberOfLines={1}>
                    {formatNextCourseWhen(m.scheduled_at)}
                  </AppText>
                  <AppText variant="label" style={styles.missionClientName} numberOfLines={1}>
                    {m.client_name?.trim() ? m.client_name.trim() : "Invité"}
                  </AppText>
                  <AppText variant="caption" style={styles.missionAddressLine} numberOfLines={2}>
                    <AppText variant="caption" style={styles.missionAddressKey}>
                      Départ :{" "}
                    </AppText>
                    {conciseAddressSegment(m.pickup_label)}
                  </AppText>
                  <AppText variant="caption" style={styles.missionAddressLine} numberOfLines={2}>
                    <AppText variant="caption" style={styles.missionAddressKey}>
                      Arrivée :{" "}
                    </AppText>
                    {conciseAddressSegment(m.dropoff_label)}
                  </AppText>
                </View>
              ))}
              <Pressable onPress={onAllRides} style={({ pressed }) => [styles.linkCtaRow, styles.linkCtaRowFirst, pressed && styles.linkCtaRowPressed]}>
                <AppText variant="label" style={styles.fleetCtaText}>
                  Voir toutes les courses
                </AppText>
                <Ionicons name="chevron-forward" size={18} color={C.brand} />
              </Pressable>
            </View>
          </View>
        ) : null}

        {view.alertLines.length > 0 || (error && !isLikelyNetworkError && !isAuthFailure) ? (
          <View style={styles.alertsBlock} accessibilityLabel="Alertes">
            <View style={styles.alertsHeaderCompact}>
              <View style={styles.alertsHeaderIconBox} accessibilityElementsHidden>
                <Ionicons name="notifications-outline" size={15} color={semanticDanger.fgStrong} />
              </View>
              <AppText variant="caption" style={styles.alertsHeaderTitle}>
                Alertes
              </AppText>
            </View>
            <View style={styles.alertsNotifStack}>
              {view.alertLines.map((a) => (
                <AlertNotificationRow key={a.id} severity={a.severity} text={a.text} />
              ))}
              {error && !isLikelyNetworkError && !isAuthFailure && displayErrMsg ? (
                <AlertNotificationRow severity="error" text={displayErrMsg} />
              ) : null}
            </View>
          </View>
        ) : null}
      </Screen>
      <DayPickerSheet
        visible={dateSheetOpen}
        selectedDate={selectedDate}
        onClose={() => setDateSheetOpen(false)}
        onSelectDate={(iso) => {
          setSelectedDate(iso);
          setDateSheetOpen(false);
        }}
      />
      <DispatchModeSheet
        visible={modeSheetOpen}
        mode={headerMode}
        onClose={() => setModeSheetOpen(false)}
        onSelectMode={(mode) => void applyDispatchModeFromSheet(mode)}
        switchingEnabled={canDispatchManage}
      />
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  /** Réf. operations-app `styles.content` : padding 16, espacement vertical entre blocs ~14. */
  page: {
    flexGrow: 1,
    paddingTop: 16,
    paddingBottom: 8,
    gap: 14,
  },
  mapHeroShell: {
    borderRadius: 16,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: C.border,
    backgroundColor: C.cardBg,
    ...dashboardSurfaceShadow,
  },
  mapHeroInner: {
    borderWidth: 0,
    borderRadius: 0,
    ...(Platform.OS === "web"
      ? ({ boxShadow: "none" } as const)
      : { elevation: 0, shadowOpacity: 0, shadowRadius: 0, shadowOffset: { width: 0, height: 0 } }),
  },
  kpiRow: { flexDirection: "row" as const, flexWrap: "wrap" as const, gap: 6 },
  kpiStat: {
    flexGrow: 1,
    minWidth: "40%",
    maxWidth: "100%",
    backgroundColor: C.cardBg,
    borderWidth: 1,
    borderColor: C.border,
    /** Réf. operations-app `statusCard` : rayon 16. */
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 14,
    ...dashboardSurfaceShadow,
  },
  kpiTopRow: { flexDirection: "row" as const, alignItems: "center" as const, gap: 8 },
  kpiIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    /** Réf. operations-app `sectionIconWrap`. */
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  kpiTextCol: { flex: 1, minWidth: 0, justifyContent: "center" as const },
  kpiLabel: {
    color: C.textSub,
    fontWeight: "700" as const,
    letterSpacing: 0.5,
    textTransform: "uppercase" as const,
    fontSize: 12,
  },
  kpiValue: {
    marginTop: 1,
    color: C.text,
    fontWeight: "800" as const,
    lineHeight: 22,
  },
  kpiSubUnavailable: { color: C.textMuted, fontWeight: "600", marginTop: 1 },
  /** Réf. operations-app `emptyState` / `emptyStateIcon` / titres. */
  emptyState: {
    alignItems: "center" as const,
    paddingVertical: 28,
    gap: 6,
  },
  emptyStateIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
    marginBottom: 4,
  },
  emptyStateTitle: {
    color: C.text,
    fontSize: 14,
    fontWeight: "600" as const,
    textAlign: "center" as const,
  },
  emptyStateSubtitle: {
    color: C.textMuted,
    fontSize: 12,
    lineHeight: 17,
    textAlign: "center" as const,
    maxWidth: 300,
  },
  /** Réf. operations-app `styles.section` (surface blanche, padding 16, pas de fond soft AppCard). */
  dashboardSection: {
    backgroundColor: C.cardBg,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: C.border,
    padding: 16,
    ...dashboardSurfaceShadow,
  },
  /** Sans filet : même flux que operations-app après `sectionTitleRow` (marge 12 uniquement). */
  sectionBody: {
    gap: 2,
    paddingTop: 0,
    paddingBottom: 2,
  },
  /** Réf. `sectionTitleRow` : gap 8, marge sous le titre 12. */
  summaryHeaderRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 8,
    marginBottom: 12,
  },
  sectionIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  summaryTitle: {
    color: C.text,
    fontSize: 16,
    fontWeight: "700" as const,
    flex: 1,
    minWidth: 0,
  },
  linkCtaRow: {
    marginTop: 8,
    minHeight: 44,
    paddingVertical: 10,
    paddingHorizontal: 2,
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
  },
  linkCtaRowFirst: { marginTop: 10 },
  linkCtaRowPressed: { opacity: 0.88 },
  fleetCtaText: { color: C.brand, fontSize: 14, fontWeight: "800" as const },
  missionBlock: { gap: 3, paddingBottom: 10 },
  missionBlockSep: {
    marginBottom: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: C.border,
  },
  missionWhen: {
    color: C.brand,
    fontWeight: "800" as const,
  },
  missionClientName: { color: C.text, fontWeight: "700" as const, marginTop: 2 },
  missionAddressLine: { color: C.textSub, lineHeight: 16, fontWeight: "500" as const, marginTop: 2 },
  missionAddressKey: { color: C.textMuted, fontWeight: "700" as const },
  alertsBlock: {
    backgroundColor: C.cardBg,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: C.border,
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 8,
    ...dashboardSurfaceShadow,
  },
  alertsHeaderCompact: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 8,
  },
  alertsHeaderIconBox: {
    width: 26,
    height: 26,
    borderRadius: 8,
    backgroundColor: "rgba(180, 35, 24, 0.08)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  alertsHeaderTitle: {
    fontSize: 11,
    fontWeight: "700" as const,
    letterSpacing: 0.9,
    textTransform: "uppercase" as const,
    color: C.textMuted,
  },
  alertsNotifStack: { gap: 6 },
  alertNotifRow: {
    flexDirection: "row" as const,
    alignItems: "flex-start" as const,
    gap: 10,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 8,
    borderWidth: StyleSheet.hairlineWidth,
  },
  alertNotifRowErr: {
    borderLeftWidth: 3,
    borderLeftColor: semanticDanger.border,
    backgroundColor: "rgba(217, 45, 32, 0.05)",
    borderColor: "rgba(217, 45, 32, 0.14)",
  },
  alertNotifRowWarn: {
    borderLeftWidth: 3,
    borderLeftColor: semanticWarning.border,
    backgroundColor: semanticWarning.bg,
    borderColor: "rgba(224, 184, 108, 0.45)",
  },
  alertNotifIconSlot: { paddingTop: 1 },
  alertNotifText: {
    flex: 1,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: "600" as const,
  },
  alertNotifTextErr: { color: semanticDanger.fg },
  alertNotifTextWarn: { color: semanticWarning.fg },
});
