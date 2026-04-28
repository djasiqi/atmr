import { useCallback, useEffect, useMemo, useRef, useState, type ComponentProps } from "react";
import { AppState, Platform, Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useRouter, type Href } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/fr";
import { PermissionGuard } from "../../../src/core/guards";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { useSession } from "../../../src/core/sessionProvider";
import {
  useCompanyFallbackPolling,
  useCompanyDashboardQuery,
  useCompanyDispatchMissionsQuery,
  useCompanyDispatchStatusQuery,
  useCompanyRealtimeInvalidation,
  useCompanyRealtimeStatus,
  useCompanyOptimizerStatusQuery,
} from "../../../src/features/company/hooks";
import { emitCompanyDispatchTelemetry } from "../../../src/features/company/telemetry/companyTelemetry";
import { companyRealtimeBridge } from "../../../src/features/company/realtime/companyRealtimeBridge";
import { contextRealtimeRouter } from "../../../src/core/realtime/contextRealtimeRouter";
import { useCompanyDriverLiveTracking } from "../../../src/features/company/realtime/useCompanyDriverLiveTracking";
import { normalizeCompanyEventType } from "../../../src/core/realtime/eventContracts";
import type { CompanyDispatchMissionStatus } from "../../../src/features/company/api/contracts";
import { resolveDriverStatus } from "../../../src/features/company/utils/companyDriverMapStatus";
import {
  buildDashboardPresentation,
  getDashboardModeConfig,
  type CompanyDispatchMode,
  type CompanyOptimizerRuntime,
  type DashboardRuntimeMetrics,
} from "../../../src/features/company/dashboard/dispatchDashboardPresentation";

dayjs.extend(relativeTime);
dayjs.locale("fr");

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

const C = {
  pageBg: "#EAF3F1",
  cardBg: "#FFFFFF",
  text: "#163A34",
  textMuted: "#5F7369",
  textSub: "#6B7A72",
  border: "rgba(145, 165, 157, 0.45)",
  brand: "#0A8F7A",
  brandSoft: "rgba(10, 143, 122, 0.12)",
  warnBg: "rgba(234, 179, 8, 0.12)",
  warnText: "#92400e",
  err: "#B42318",
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

function mapSocketFr(
  status: string,
  connected: boolean,
  companyRealtimeEnabled: boolean
): string {
  if (!companyRealtimeEnabled) {
    return "Désactivé (repli HTTP)";
  }
  if (status === "connecting" || status === "reconnecting") {
    return "Connexion…";
  }
  if (status === "degraded") {
    return "Dégradé";
  }
  if (status === "failed") {
    return "Indisponible (vérif. session / réseau)";
  }
  if (status === "healthy" && connected) {
    return "Connecté";
  }
  if (!connected) {
    return "Hors ligne";
  }
  switch (status) {
    case "healthy":
      return "Connecté";
    case "idle":
      return "Déconnecté";
    default:
      return status;
  }
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
          <Text style={styles.kpiLabel} numberOfLines={1}>
            {def.label}
          </Text>
          {isUnavailable ? (
            <View>
              <Text style={styles.kpiValue} accessibilityLabel="Non disponible —">
                —
              </Text>
              <Text style={styles.kpiSubUnavailable}>Non disponible</Text>
            </View>
          ) : (
            <Text style={styles.kpiValue} numberOfLines={1} adjustsFontSizeToFit>
              {display.line1}
            </Text>
          )}
        </View>
      </View>
    </View>
  );
}

export default function CompanyDashboardScreen() {
  const { activeContext } = useSession();
  const activeContextId = activeContext?.context_id ?? null;
  const [tick, setTick] = useState(0);
  const date = useMemo(() => getTodayIsoDate(), []);
  const missionsQuery = useCompanyDispatchMissionsQuery({ date });
  const dashboardQuery = useCompanyDashboardQuery(date);
  const dispatchStatusQuery = useCompanyDispatchStatusQuery(date);
  const optimizerQuery = useCompanyOptimizerStatusQuery();
  const liveDrivers = useCompanyDriverLiveTracking();
  const liveDriversRefetch = liveDrivers.refetch;
  const realtime = useCompanyRealtimeStatus();
  const companyRealtimeEnabled = isFeatureEnabled("company_realtime_enabled");
  const previousRealtimeStatus = useRef<string | null>(null);
  const lastOptimizerStatusTelemetryAtRef = useRef(0);
  const { invalidate } = useCompanyRealtimeInvalidation();
  const missionsRefetch = missionsQuery.refetch;
  const dashboardRefetch = dashboardQuery.refetch;
  const dispatchStatusRefetch = dispatchStatusQuery.refetch;
  const optimizerRefetch = optimizerQuery.refetch;
  const router = useRouter();

  useEffect(() => {
    const t = setInterval(() => {
      setTick((x) => x + 1);
    }, 12_000);
    return () => clearInterval(t);
  }, []);

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
  const isLikelyNetworkError = Boolean(
    errMsg && /network|Network|fetch|Failed to fetch|connexion|Connexion|internet|Internet/i.test(errMsg)
  );

  const isPotentiallyStale =
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

  // tick : rafraîchit le relatif toutes ~12s sans relancer fetch
  const lastSyncLabel = useMemo(() => {
    if (!lastKnownSyncAt) return "—";
    return dayjs(lastKnownSyncAt).fromNow();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `tick` force le recalcul
  }, [lastKnownSyncAt, tick]);

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
      errMsg,
      dataHealthLabel,
      realtimeHealthyData,
    };
  }, [
    clientDelayedCount,
    dashboardQuery.isSuccess,
    dashboardQuery.data,
    dataHealthLabel,
    dispatchMode,
    driversAvailableCount,
    errMsg,
    fleet.enMission,
    fleet.off,
    hasPendingOverdue,
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

  /** Bandeau coloré seulement si le repli HTTP ne suffit pas (données vieilles) ou flux vraiment dégradé. */
  const bandStrongAlert =
    companyRealtimeEnabled &&
    (isPotentiallyStale ||
      realtime.status === "degraded" ||
      (!realtime.connected && realtime.status !== "failed") ||
      (realtime.status === "failed" && isPotentiallyStale));

  const onPrimaryCta = useCallback(() => {
    const p = view.primaryCta;
    if (p.params && p.params.filter) {
      router.push({ pathname: p.path as Href, params: p.params } as Href);
    } else {
      router.push(p.path as Href);
    }
  }, [router, view.primaryCta]);

  const onFleetMap = useCallback(() => {
    if (Platform.OS === "web") return;
    router.push("/(app)/(company)/fleet-map");
  }, [router]);

  const onAllRides = useCallback(() => {
    router.push("/(app)/(company)/rides");
  }, [router]);

  return (
    <PermissionGuard permission="company:dashboard:read">
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.page}
        refreshControl={
          <RefreshControl refreshing={loading} onRefresh={() => void refreshAll()} tintColor={C.brand} />
        }
      >
        <View
          style={[styles.statusStrip, bandStrongAlert && styles.statusStripAlert]}
          accessibilityLabel="Statut flux temps réel"
        >
          <View style={styles.statusRow}>
            <Text style={styles.statusStripLabel}>Flux</Text>
            <Text style={styles.statusStripValue}>
              {mapSocketFr(realtime.status, realtime.connected, companyRealtimeEnabled)}
            </Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusStripLabel}>Chauffeurs en ligne</Text>
            <Text style={styles.statusStripValue} accessibilityLabel={`Chauffeurs en ligne`}>
              {onlineCount} / {liveDrivers.drivers.length}
            </Text>
          </View>
          <View style={styles.statusRow}>
            <Text style={styles.statusStripLabel}>Dernière synchro</Text>
            <Text style={styles.statusStripValue} accessibilityLiveRegion="polite">
              {lastSyncLabel}
            </Text>
          </View>
          <Pressable
            onPress={() => {
              if (companyRealtimeEnabled) companyRealtimeBridge.reconnect();
            }}
            disabled={!companyRealtimeEnabled}
            style={({ pressed }) => [
              styles.reconnectIconBtn,
              !companyRealtimeEnabled && { opacity: 0.35 },
              pressed && companyRealtimeEnabled && { opacity: 0.8 },
            ]}
            hitSlop={8}
            accessibilityLabel="Reconnecter le flux"
            accessibilityState={{ disabled: !companyRealtimeEnabled }}
          >
            <Ionicons
              name="refresh-outline"
              size={18}
              color={!companyRealtimeEnabled ? C.textMuted : bandStrongAlert ? C.err : C.brand}
            />
          </Pressable>
        </View>

        <View style={styles.contextCard}>
          {view.showAutomationCaution ? (
            <View style={styles.cautionChip}>
              <Ionicons name="warning-outline" size={16} color={C.warnText} />
              <Text style={styles.cautionChipText}>Vérification recommandée</Text>
            </View>
          ) : null}
          <Text style={styles.contextQ}>{view.operationalQuestion}</Text>
          <Text style={styles.contextTitle}>{view.contextTitle}</Text>
          <Text style={styles.contextMessage}>{view.contextMessage}</Text>
          <Pressable
            onPress={onPrimaryCta}
            style={({ pressed }) => [styles.primaryCta, pressed && { opacity: 0.92 }]}
            accessibilityRole="button"
            accessibilityLabel={view.primaryCta.label}
          >
            <Text style={styles.primaryCtaText}>{view.primaryCta.label}</Text>
            <Ionicons name="chevron-forward" size={18} color="#FFFFFF" />
          </Pressable>
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
            <View style={styles.emptyIconRing}>
              <Ionicons name="car-outline" size={30} color="#94A3B8" />
            </View>
            <Text style={styles.emptyTitle}>Aucune course planifiée</Text>
            <Text style={styles.emptySub}>
              Les courses apparaîtront ici dès qu’elles seront créées.
            </Text>
          </View>
        ) : null}

        <View style={styles.summaryCard}>
          <View style={styles.summaryHeaderRow}>
            <Ionicons name="people-outline" size={15} color={C.brand} />
            <Text style={styles.summaryTitle}>Aperçu flotte</Text>
          </View>
          <View style={styles.fleetLine}>
            <Text style={styles.fleetText}>En mission {fleet.enMission}</Text>
            <Text style={styles.fleetText}>·</Text>
            <Text style={styles.fleetText}>Disponibles {fleet.dispo}</Text>
            <Text style={styles.fleetText}>·</Text>
            <Text style={styles.fleetText}>Hors ligne {fleet.off}</Text>
          </View>
          {Platform.OS === "web" ? (
            <Text style={styles.fleetWebHint}>
              La carte n’est pas disponible sur le web. Utilisez l’app mobile.
            </Text>
          ) : (
            <Pressable onPress={onFleetMap} style={({ pressed }) => [styles.fleetCta, pressed && { opacity: 0.9 }]}>
              <Text style={styles.fleetCtaText}>Voir la carte</Text>
              <Ionicons name="chevron-forward" size={16} color={C.brand} />
            </Pressable>
          )}
        </View>

        <View style={styles.summaryCard}>
          <View style={styles.summaryHeaderRow}>
            <Ionicons name="options-outline" size={15} color={C.brand} />
            <Text style={styles.summaryTitle}>Lecture moteur dispatch</Text>
          </View>
          {view.technicalLines.map((r, i) => (
            <View
              key={r.label}
              style={[
                styles.kvDenseRow,
                i < view.technicalLines.length - 1 && styles.kvDenseRowBorder,
              ]}
            >
              <Text style={styles.kvDenseKey}>{r.label}</Text>
              <Text style={styles.kvDenseVal} numberOfLines={2}>
                {r.value}
              </Text>
            </View>
          ))}
        </View>

        {nextMissions.length > 0 ? (
          <View style={styles.summaryCard}>
            <View style={styles.summaryHeaderRow}>
              <Ionicons name="time-outline" size={15} color={C.brand} />
              <Text style={styles.summaryTitle}>Prochaines courses</Text>
            </View>
            {nextMissions.map((m) => (
              <View key={m.mission_id} style={styles.missionBlock} accessibilityLabel={`Prochaine course ${m.mission_id}`}>
                <Text style={styles.missionWhen} numberOfLines={1}>
                  {formatNextCourseWhen(m.scheduled_at)}
                </Text>
                <Text style={styles.missionClientName} numberOfLines={1}>
                  {m.client_name?.trim() ? m.client_name.trim() : "Invité"}
                </Text>
                <Text style={styles.missionAddressLine} numberOfLines={2}>
                  <Text style={styles.missionAddressKey}>Départ : </Text>
                  {conciseAddressSegment(m.pickup_label)}
                </Text>
                <Text style={styles.missionAddressLine} numberOfLines={2}>
                  <Text style={styles.missionAddressKey}>Arrivée : </Text>
                  {conciseAddressSegment(m.dropoff_label)}
                </Text>
              </View>
            ))}
            <Pressable onPress={onAllRides} style={({ pressed }) => [styles.fleetCta, pressed && { opacity: 0.9 }]}>
              <Text style={styles.fleetCtaText}>Voir toutes les courses</Text>
              <Ionicons name="chevron-forward" size={16} color={C.brand} />
            </Pressable>
          </View>
        ) : null}

        {view.alertLines.length > 0 || (error && !isLikelyNetworkError) ? (
          <View style={styles.alertsBlock}>
            <View style={styles.alertsHeader}>
              <Ionicons name="alert-circle" size={18} color={C.err} />
              <Text style={styles.alertsTitle}>Alertes</Text>
            </View>
            {view.alertLines.map((a) => (
              <View
                key={a.id}
                style={[styles.alertItem, a.severity === "error" ? styles.alertItemErr : styles.alertItemWarn]}
              >
                <Text style={styles.alertText}>{a.text}</Text>
              </View>
            ))}
            {error && !isLikelyNetworkError && errMsg ? (
              <View style={[styles.alertItem, styles.alertItemErr]}>
                <Text style={styles.alertText}>{errMsg}</Text>
              </View>
            ) : null}
          </View>
        ) : null}
      </ScrollView>
    </PermissionGuard>
  );
}

const kpiCardShadow = Platform.select({
  web: { boxShadow: "0 2px 10px rgba(22, 58, 52, 0.06)" },
  default: {
    shadowColor: "#163A34",
    shadowOpacity: 0.06,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
});

const styles = StyleSheet.create({
  scroll: {
    flex: 1,
    backgroundColor: C.pageBg,
  },
  page: {
    padding: 20,
    paddingBottom: 100,
    gap: 12,
  },
  statusStrip: {
    backgroundColor: C.cardBg,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: C.border,
    paddingVertical: 10,
    paddingHorizontal: 12,
    gap: 4,
    ...kpiCardShadow,
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    alignItems: "center" as const,
  },
  statusStripAlert: {
    borderColor: "rgba(180, 35, 24, 0.35)",
    backgroundColor: "rgba(194, 65, 12, 0.08)",
  },
  statusRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 4,
    marginRight: 12,
    minWidth: "30%",
  },
  statusStripLabel: { color: C.textMuted, fontSize: 10, fontWeight: "600", textTransform: "uppercase" as const },
  statusStripValue: { color: C.text, fontSize: 12, fontWeight: "800" },
  reconnectIconBtn: { marginLeft: "auto", padding: 4 },
  contextCard: {
    backgroundColor: C.cardBg,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: C.border,
    padding: 14,
    gap: 6,
    ...kpiCardShadow,
  },
  contextQ: { color: C.textSub, fontSize: 12, fontWeight: "600" },
  contextTitle: { color: C.text, fontSize: 18, fontWeight: "800" },
  contextMessage: { color: C.textMuted, fontSize: 13, lineHeight: 18 },
  primaryCta: {
    marginTop: 4,
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    gap: 6,
    backgroundColor: C.brand,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 14,
  },
  primaryCtaText: { color: "#FFFFFF", fontSize: 15, fontWeight: "800" },
  cautionChip: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    alignSelf: "flex-start" as const,
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
    backgroundColor: C.warnBg,
    borderRadius: 8,
  },
  cautionChipText: { color: C.warnText, fontSize: 12, fontWeight: "800" },
  kpiRow: { flexDirection: "row" as const, flexWrap: "wrap" as const, gap: 6 },
  kpiStat: {
    flexGrow: 1,
    minWidth: "40%",
    maxWidth: "100%",
    backgroundColor: C.cardBg,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 12,
    paddingVertical: 8,
    paddingHorizontal: 10,
    ...kpiCardShadow,
  },
  kpiTopRow: { flexDirection: "row" as const, alignItems: "center" as const, gap: 8 },
  kpiIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: C.brandSoft,
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  kpiTextCol: { flex: 1, minWidth: 0, justifyContent: "center" as const },
  kpiLabel: {
    color: C.textSub,
    fontSize: 10,
    fontWeight: "700" as const,
    letterSpacing: 0.25,
    textTransform: "uppercase" as const,
  },
  kpiValue: {
    marginTop: 1,
    color: C.text,
    fontSize: 18,
    fontWeight: "800" as const,
    lineHeight: 22,
  },
  kpiSubUnavailable: { color: C.textMuted, fontSize: 10, fontWeight: "600", marginTop: 1 },
  emptyState: { alignItems: "center" as const, marginVertical: 4, gap: 4, paddingVertical: 8 },
  emptyIconRing: {
    width: 56,
    height: 56,
    borderRadius: 16,
    backgroundColor: "rgba(148, 163, 184, 0.2)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  emptyTitle: { color: C.text, fontSize: 16, fontWeight: "800", textAlign: "center" as const },
  emptySub: { color: C.textMuted, fontSize: 12, lineHeight: 17, textAlign: "center" as const, maxWidth: 300 },
  summaryCard: {
    backgroundColor: C.cardBg,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 10,
    ...kpiCardShadow,
  },
  summaryHeaderRow: { flexDirection: "row" as const, alignItems: "center" as const, gap: 6, marginBottom: 8 },
  summaryTitle: { color: C.text, fontSize: 14, fontWeight: "700" as const },
  fleetLine: { flexDirection: "row" as const, flexWrap: "wrap" as const, alignItems: "center" as const, gap: 4 },
  fleetText: { color: C.textMuted, fontSize: 12, fontWeight: "600" as const },
  fleetWebHint: { color: C.mapHeroMuted, fontSize: 12, lineHeight: 16, marginTop: 4 },
  fleetCta: {
    marginTop: 8,
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
  },
  fleetCtaText: { color: C.brand, fontSize: 13, fontWeight: "800" as const },
  kvDenseRow: {
    flexDirection: "row" as const,
    justifyContent: "space-between" as const,
    alignItems: "flex-start" as const,
    gap: 8,
    paddingVertical: 5,
  },
  kvDenseRowBorder: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: C.border,
  },
  kvDenseKey: { flex: 1, minWidth: 0, color: C.textMuted, fontSize: 11, fontWeight: "600" as const },
  kvDenseVal: { maxWidth: "58%", color: C.text, fontSize: 12, fontWeight: "700" as const, textAlign: "right" as const },
  missionBlock: { gap: 3, marginBottom: 8 },
  missionWhen: {
    color: C.brand,
    fontSize: 13,
    fontWeight: "800" as const,
  },
  missionClientName: { color: C.text, fontSize: 14, fontWeight: "700" as const, marginTop: 2 },
  missionAddressLine: { color: C.textSub, fontSize: 12, lineHeight: 16, fontWeight: "500" as const, marginTop: 2 },
  missionAddressKey: { color: C.textMuted, fontWeight: "700" as const },
  alertsBlock: {
    backgroundColor: "rgba(180, 35, 24, 0.06)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(180, 35, 24, 0.2)",
    padding: 10,
  },
  alertsHeader: { flexDirection: "row" as const, alignItems: "center" as const, gap: 6, marginBottom: 4 },
  alertsTitle: { color: C.err, fontSize: 15, fontWeight: "800" as const },
  alertItem: { borderRadius: 8, padding: 8, marginTop: 4 },
  alertItemErr: { backgroundColor: "rgba(180, 35, 24, 0.1)" },
  alertItemWarn: { backgroundColor: C.warnBg },
  alertText: { color: C.text, fontSize: 12, lineHeight: 16, fontWeight: "600" as const },
});
