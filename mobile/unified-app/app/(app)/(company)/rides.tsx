import { lazy, Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentProps } from "react";
import {
  ActivityIndicator,
  Pressable,
  Platform,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useFocusEffect, useIsFocused } from "@react-navigation/native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import {
  AppText,
  Screen,
  computeCompanyFloatingBottomPad,
  useAppViewport,
  useFloatingBarClearance,
} from "../../../src/design/responsive";
import { useQueryClient } from "@tanstack/react-query";
import { useSession } from "../../../src/core/sessionProvider";
import {
  useActiveCompanyContextId,
  useCompanyDispatchDelaysQuery,
  useCompanyDispatchMissionsQuery,
  useCompanyRideActions,
  useCompanyRealtimeInvalidation,
  useCompanyRealtimeStatus,
} from "../../../src/features/company/hooks";
import { emitCompanyDispatchTelemetry } from "../../../src/features/company/telemetry/companyTelemetry";
import { contextRealtimeRouter } from "../../../src/core/realtime/contextRealtimeRouter";
import { normalizeCompanyEventType } from "../../../src/core/realtime/eventContracts";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import {
  beginTapFeedback,
  endTapLocal,
  endTapNavigation,
  recordMissionDetailsPhase,
  recordScreenRender,
} from "../../../src/core/observability/perfResponsiveness";
import { useCompanyBackgroundBootReady } from "../../../src/features/company/boot/companyColdStartPhase";
import { markCompanyScreenUsable } from "../../../src/features/company/observability/companyDashboardPhases";
import { EnterpriseHeader } from "../../../src/features/company/components/EnterpriseHeader";
import { DayPickerSheet } from "../../../src/features/company/components/DayPickerSheet";
import { CompanyRidesMissionFlatList } from "../../../src/features/company/components/rides/CompanyRidesMissionFlatList";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";
import { filterMissionsByDispatchListChip } from "../../../src/features/company/utils/rideListStatusFilter";
import { filterDayMissionsForLocalSearch } from "../../../src/features/company/utils/filterMissionsByLocalSearch";
import { prefetchAdjacentDispatchMissions } from "../../../src/features/company/utils/prefetchAdjacentDispatchMissions";
import { remainingUnloadedCount } from "../../../src/features/company/utils/dispatchDayPagination";
import { useRetainDispatchQueryCache } from "../../../src/features/company/utils/dispatchQueryRetention";
import { setStickyRidesFetchReason } from "../../../src/features/company/utils/ridesFetchReason";
import {
  COMPANY_OFFLINE_ACTION_MESSAGE,
  isCompanyNetworkRequestError,
  resolveDispatchDayEmptyKind,
} from "../../../src/features/company/utils/companyOfflinePolicy";
import { useCompanyNetworkOffline } from "../../../src/features/company/utils/useCompanyNetworkOffline";
import { rememberRideDetailSnapshot } from "../../../src/features/company/utils/rideDetailSnapshotStore";
import {
  flattenCompanyDispatchDelays,
  pickupDelaysByBookingLastWins,
  pickupEtaIsoByBookingId,
} from "../../../src/features/company/utils/dispatchWebAlignment";
import {
  getTodayIsoDateInZurich,
  missionBelongsToSelectedDay,
} from "../../../src/features/company/utils/companyDateUtils";
import { hasConfirmedPickupTime, missionHasRenderableSchedule } from "../../../src/features/company/utils/pickupSentinel";
import {
  cancelCompanyRide,
  getCompanyAvailableDrivers,
  getCompanyPartnershipsForTransfer,
  getDispatchApiErrorMessage,
  markCompanyRideUrgent,
  transferCompanyRide,
} from "../../../src/features/company/api/companyApi";
import { MaybeCompanyRidesEngineActions } from "../../../src/features/company/components/rides/MaybeCompanyRidesEngineActions";
import type { CompanyDispatchMission } from "../../../src/features/company/api/contracts";
import { createShadow } from "../../../src/styles/shadowStyles";
import { FONT_SIZE } from "../../../src/design/responsive/typographyTokens";

const RideCreateModal = lazy(() =>
  import("../../../src/features/company/components/rides/RideCreateModal").then((m) => ({
    default: m.RideCreateModal,
  }))
);
const RideEditModal = lazy(() =>
  import("../../../src/features/company/components/rides/RideEditModal").then((m) => ({
    default: m.RideEditModal,
  }))
);
const AssignDriverModal = lazy(() =>
  import("../../../src/features/company/components/rides/AssignDriverModal").then((m) => ({
    default: m.AssignDriverModal,
  }))
);
const RideScheduleModal = lazy(() =>
  import("../../../src/features/company/components/rides/RideScheduleModal").then((m) => ({
    default: m.RideScheduleModal,
  }))
);
const TransferRideModal = lazy(() =>
  import("../../../src/features/company/components/transfers/TransferRideModal").then((m) => ({
    default: m.TransferRideModal,
  }))
);

const searchBarShadowOps = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.03,
  shadowRadius: 4,
  elevation: 1,
});

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

type LabeledOption = { id: number; label: string };

function parseId(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function extractLabeledRows(
  payload: unknown,
  preferredIdKeys: string[],
  preferredLabelKeys: string[]
): LabeledOption[] {
  if (!payload || typeof payload !== "object") return [];
  const source = payload as Record<string, unknown>;
  const candidates = [
    source.items,
    source.results,
    source.data,
    source.drivers,
    source.partnerships,
    source.companies,
  ];
  const rows = candidates.find((value) => Array.isArray(value));
  if (!Array.isArray(rows)) return [];
  return rows
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null;
      const raw = entry as Record<string, unknown>;
      const idCandidate = preferredIdKeys.map((key) => raw[key]).find((value) => value != null);
      const id = parseId(idCandidate);
      if (id == null) return null;
      const fromKeys = preferredLabelKeys
        .map((key) => raw[key])
        .find((value) => typeof value === "string" && value.trim().length > 0);
      const first = typeof raw.first_name === "string" ? raw.first_name.trim() : "";
      const last = typeof raw.last_name === "string" ? raw.last_name.trim() : "";
      const fromParts = [first, last].filter(Boolean).join(" ").trim();
      const labelCandidate = (fromKeys ?? (fromParts.length > 0 ? fromParts : null)) ?? `#${id}`;
      return { id, label: String(labelCandidate) };
    })
    .filter((value): value is LabeledOption => value !== null);
}

type RideFilterIconName = ComponentProps<typeof Ionicons>["name"];

/** `key` = paramètre `status` pour GET /company_mobile/dispatch/v1/rides (et fallback client). */
const RIDE_STATUS_FILTERS: {
  key: string;
  label: string;
  hint?: string;
  icon: RideFilterIconName;
}[] = [
  { key: "all", label: "Tous", hint: "Journée complète", icon: "apps-outline" },
  {
    key: "pending",
    label: "En attente",
    hint: "En attente de prise en charge, offre (proposée) ou acceptée",
    icon: "time-outline",
  },
  { key: "assigned", label: "Affectés", hint: "Chauffeur connu, pas terminé", icon: "person-outline" },
  {
    key: "in_flight",
    label: "En course",
    hint: "Uniquement : statut en route ou en mission",
    icon: "navigate-outline",
  },
  {
    key: "completed",
    label: "Terminés",
    hint: "Terminé ou aller-retour clôturé",
    icon: "checkmark-done-outline",
  },
  { key: "cancelled", label: "Annulés", icon: "close-circle-outline" },
];

const RIDE_STATUS_FILTER_KEYS = new Set(RIDE_STATUS_FILTERS.map((item) => item.key));

function firstSearchParam(value: string | string[] | undefined): string | undefined {
  if (value == null) return undefined;
  return Array.isArray(value) ? value[0] : value;
}

/** Même écart vertical qu’entre cartes Clients / Facturation (`READONLY_LIST_CARD_STACK_GAP`). */
const DISPATCH_RIDE_LIST_CARD_GAP = 4;

const rideStyles = StyleSheet.create({
  /** Contenu scroll : rythme uniforme avec header / cartes. */
  page: { paddingTop: 10, paddingBottom: 20, gap: 16 },
  searchBar: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: E.CARD,
    borderRadius: 14,
    minHeight: 44,
    paddingHorizontal: 14,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.22)",
    ...searchBarShadowOps,
  },
  searchClearHit: { padding: 6 },
  searchInput: {
    flex: 1,
    minWidth: 0,
    borderWidth: 0,
    backgroundColor: "transparent",
    color: E.TEXT,
    paddingVertical: Platform.OS === "ios" ? 10 : 8,
    paddingHorizontal: 8,
    fontSize: FONT_SIZE.px15,
    outlineStyle: Platform.OS === "web" ? ("none" as const) : undefined,
  },
  /** Bandeau filtres — bordure slate légère comme la barre de recherche. */
  tabsPanel: {
    alignSelf: "stretch",
    marginTop: 8,
    ...(Platform.OS === "web" ? ({ minWidth: 0 } as const) : {}),
    backgroundColor: E.CARD,
    borderRadius: 14,
    paddingVertical: 6,
    paddingHorizontal: 8,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.22)",
    ...searchBarShadowOps,
  },
  /** Pas de flexGrow vertical ici (colonne parente) : une seule ligne, largeur = parent. */
  tabsHeaderScrollView: {
    alignSelf: "stretch",
    width: "100%" as const,
    flexShrink: 0,
    ...(Platform.OS === "web" ? ({ minWidth: 0 } as const) : {}),
  },
  tabsHeaderScrollContent: {
    flexDirection: "row" as const,
    flexWrap: "nowrap" as const,
    alignItems: "center" as const,
    flexGrow: 1,
    gap: 6,
    paddingVertical: 2,
    paddingHorizontal: 2,
  },
  /** Une ligne : icône + compteur, état actif = léger lavis vert (pas de plein vert). */
  tabButton: {
    flexShrink: 0,
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    gap: 4,
    minHeight: 36,
    paddingVertical: 6,
    paddingHorizontal: 11,
    borderRadius: 11,
    backgroundColor: "transparent",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "transparent",
    ...(Platform.OS === "web"
      ? ({ userSelect: "none", cursor: "pointer" } as const)
      : {}),
  },
  tabButtonActive: {
    backgroundColor: "rgba(0, 121, 107, 0.1)",
    borderColor: "rgba(0, 121, 107, 0.28)",
  },
  tabCount: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "600" as const,
    color: E.TEXT_SEC,
    fontVariant: ["tabular-nums"] as const,
    minWidth: 10,
    textAlign: "left" as const,
    ...(Platform.OS === "android" ? { includeFontPadding: false } : {}),
  },
  tabCountActive: {
    color: E.BRAND,
  },
  tabCountZero: {
    opacity: 0.38,
  },
  tabCountActiveZero: {
    opacity: 0.55,
  },
  exceptionsRouteHint: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px12,
    lineHeight: 17,
    marginBottom: 0,
    padding: 14,
    backgroundColor: E.CARD,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.22)",
    ...searchBarShadowOps,
  },
  pressed: { opacity: 0.88 },
  /** Réf. `ridesListContainer` operations : marge haute portée par `page.gap`. */
  listTop: { marginTop: 0, gap: DISPATCH_RIDE_LIST_CARD_GAP },
  dayPaginationHint: {
    paddingVertical: 12,
    alignItems: "center",
    gap: 8,
  },
  dayPaginationHintText: { color: E.TEXT_MUTED, textAlign: "center" },
  dayPaginationRetry: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: E.BRAND,
  },
  dayPaginationRetryLabel: { color: E.BRAND, fontWeight: "600" },
  loadingBox: {
    alignItems: "center",
    paddingVertical: 48,
    gap: 10,
  },
  loadingText: { color: E.TEXT_MUTED, fontSize: FONT_SIZE.px13 },
  emptyState: {
    alignItems: "center",
    paddingVertical: 56,
    paddingHorizontal: 24,
  },
  emptyIcon: {
    width: 52,
    height: 52,
    borderRadius: 14,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
  },
  emptyTitle: {
    color: E.TEXT,
    fontWeight: "600",
    fontSize: FONT_SIZE.px15,
  },
  emptySubtitle: {
    color: E.TEXT_MUTED,
    fontSize: FONT_SIZE.px13,
    textAlign: "center",
    marginTop: 6,
    lineHeight: 19,
  },
  /** Bandeau ops (flux WS, erreurs chargement) — liseré critique aligné cartes retard. */
  opsAlert: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
    padding: 14,
    backgroundColor: "#FEF2F2",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(220, 38, 38, 0.22)",
    borderLeftWidth: 3,
    borderLeftColor: "#DC2626",
    ...searchBarShadowOps,
  },
  opsAlertIconWell: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: "rgba(220, 38, 38, 0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  opsAlertBodyCol: {
    flex: 1,
    minWidth: 0,
    gap: 4,
  },
  opsAlertTitle: {
    color: E.TEXT,
    fontSize: FONT_SIZE.px14,
    fontWeight: "600",
    letterSpacing: 0.1,
    lineHeight: 19,
  },
  opsAlertMessage: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px13,
    fontWeight: "500",
    lineHeight: 18,
  },
  opsAlertRetry: {
    alignSelf: "flex-start",
    marginTop: 8,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 10,
    backgroundColor: "rgba(0, 121, 107, 0.1)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.25)",
  },
  opsAlertRetryDisabled: { opacity: 0.55 },
  opsAlertRetryLabel: {
    color: E.BRAND_DARK,
    fontSize: FONT_SIZE.px13,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  errorBlock: { color: E.DANGER, lineHeight: 19 },
  mutedText: { color: E.TEXT_SEC },
});

export default function CompanyRidesScreen() {
  recordScreenRender("company.rides");
  const router = useRouter();
  const params = useLocalSearchParams<{ filter?: string; create?: string; status?: string }>();
  const { activeContext, can, bootstrapSession } = useSession();
  const { bottomInset } = useAppViewport();
  const ridesBottomClearance = useFloatingBarClearance(
    "company",
    computeCompanyFloatingBottomPad(bottomInset)
  );
  const [sessionRetryBusy, setSessionRetryBusy] = useState(false);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const [assignModalMissionId, setAssignModalMissionId] = useState<number | null>(null);
  const [transferModalMissionId, setTransferModalMissionId] = useState<number | null>(null);
  const [modalError, setModalError] = useState<string | null>(null);
  const [modalPending, setModalPending] = useState(false);
  const [drivers, setDrivers] = useState<LabeledOption[]>([]);
  const [selectedDriverId, setSelectedDriverId] = useState<number | null>(null);
  const [partnerships, setPartnerships] = useState<LabeledOption[]>([]);
  const [selectedPartnerId, setSelectedPartnerId] = useState<number | null>(null);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [editMissionId, setEditMissionId] = useState<number | null>(null);
  const [scheduleMissionId, setScheduleMissionId] = useState<number | null>(null);
  const [missionActionPendingId, setMissionActionPendingId] = useState<number | null>(null);
  const [expandedMissionId, setExpandedMissionId] = useState<number | null>(null);
  const [selectedDate, setSelectedDate] = useState(() => getTodayIsoDateInZurich());
  const [dateSheetOpen, setDateSheetOpen] = useState(false);
  const filterNorm = useMemo(() => firstSearchParam(params.filter)?.toLowerCase() ?? "", [params.filter]);
  const statusParam = firstSearchParam(params.status);

  useEffect(() => {
    if (filterNorm === "delayed" || filterNorm === "exceptions" || filterNorm === "urgent") {
      setStatus("all");
    }
  }, [filterNorm]);

  useEffect(() => {
    if (filterNorm === "delayed" || filterNorm === "exceptions" || filterNorm === "urgent") return;
    if (statusParam && RIDE_STATUS_FILTER_KEYS.has(statusParam)) {
      setStatus(statusParam);
    }
  }, [filterNorm, statusParam]);

  const contextId = useActiveCompanyContextId();
  const queryClient = useQueryClient();
  const ridesFocused = useIsFocused();
  const backgroundReady = useCompanyBackgroundBootReady();
  const networkOffline = useCompanyNetworkOffline();
  useRetainDispatchQueryCache("rides", contextId, selectedDate);
  const rideActions = useCompanyRideActions();
  const missionsQuery = useCompanyDispatchMissionsQuery({
    date: selectedDate,
    completeDay: ridesFocused,
  });
  const missionsRefetch = missionsQuery.refetch;
  const dispatchDelaysQuery = useCompanyDispatchDelaysQuery({
    date: selectedDate,
    enabled: ridesFocused || backgroundReady,
  });
  const delaysRefetch = dispatchDelaysQuery.refetch;
  const ridesUsableMarkedRef = useRef(false);
  useLayoutEffect(() => {
    if (!ridesFocused) {
      ridesUsableMarkedRef.current = false;
      return;
    }
    if (ridesUsableMarkedRef.current) return;
    ridesUsableMarkedRef.current = true;
    markCompanyScreenUsable("company.rides", { trigger: "focus_commit" });
  }, [ridesFocused]);
  const { invalidate } = useCompanyRealtimeInvalidation();
  const realtimeStatus = useCompanyRealtimeStatus();
  const lastOpenedTelemetryKeyRef = useRef<string | null>(null);
  const lastFocusRefreshAtRef = useRef(0);
  const FOCUS_REFRESH_THROTTLE_MS = 3000;
  const STALE_DATA_MS = 60_000;

  const createParam = firstSearchParam(params.create);
  useEffect(() => {
    if (String(createParam) !== "1") return;
    setCreateModalVisible(true);
    const f = firstSearchParam(params.filter);
    void router.setParams({ create: undefined, filter: f ?? undefined });
  }, [createParam, params.filter, router]);

  const refresh = useCallback(async () => {
    setStickyRidesFetchReason("manual");
    await Promise.all([missionsRefetch(), delaysRefetch()]);
  }, [missionsRefetch, delaysRefetch]);

  const refreshStaleOnly = useCallback(async () => {
    // OPT-04E : J se réconcilie via refetchOnWindowFocus (si stale), pas un 2e GET ici.
    if (Date.now() - dispatchDelaysQuery.dataUpdatedAt > STALE_DATA_MS) {
      await delaysRefetch();
    }
  }, [delaysRefetch, dispatchDelaysQuery.dataUpdatedAt]);

  const clearRouteFilterOverride = useCallback(() => {
    if (filterNorm === "exceptions" || filterNorm === "delayed" || filterNorm === "urgent") {
      void router.setParams({ filter: undefined });
    }
  }, [filterNorm, router]);

  const retryRealtimeSession = useCallback(async () => {
    setSessionRetryBusy(true);
    try {
      await bootstrapSession();
      await refresh();
    } finally {
      setSessionRetryBusy(false);
    }
  }, [bootstrapSession, refresh]);

  const showRealtimeFluxAlert =
    Boolean(realtimeStatus.lastError?.trim()) || realtimeStatus.status === "failed";

  useFocusEffect(
    useCallback(() => {
      const focusKey = String(activeContext?.context_id ?? "none");
      if (lastOpenedTelemetryKeyRef.current !== focusKey) {
        lastOpenedTelemetryKeyRef.current = focusKey;
        emitCompanyDispatchTelemetry(
          "company.dispatch.opened",
          {
            source: "company.rides.screen",
            context_type: "company",
            context_id: activeContext?.context_id ?? null,
          },
          { allowWhenDisabled: true }
        );
      }
      const now = Date.now();
      if (now - lastFocusRefreshAtRef.current < FOCUS_REFRESH_THROTTLE_MS) return;
      lastFocusRefreshAtRef.current = now;
      void refreshStaleOnly();
      return () => {
        lastOpenedTelemetryKeyRef.current = null;
      };
    }, [activeContext?.context_id, refreshStaleOnly])
  );

  useFocusEffect(
    useCallback(() => {
      if (!activeContext || activeContext.context_type !== "company") return undefined;
      return contextRealtimeRouter.subscribe(activeContext.context_id, (event) => {
        if (!event || typeof event !== "object") return;
        const payload = event as {
          event_type?: string;
          booking_id?: unknown;
          mission_id?: unknown;
          id?: unknown;
        };
        const missionId = resolveMissionIdFromEvent(payload);
        const eventType = normalizeCompanyEventType(payload.event_type);
        if (eventType === "booking_updated") {
          invalidate("booking_updated", missionId);
        } else if (eventType === "booking_cancelled") {
          invalidate("booking_cancelled", missionId);
        } else if (eventType === "company_dispatch_update") {
          void refresh();
        } else if (eventType === "delay_invalidated") {
          emitCompanyDispatchTelemetry(
            "company.dispatch.delay_invalidated",
            {
              source: "company.rides.realtime",
              context_id: activeContext.context_id,
              mission_id: missionId ?? null,
              context_type: "company",
            },
            { allowWhenDisabled: true }
          );
          invalidate("delay_invalidated", missionId);
        }
      });
    }, [activeContext, invalidate, refresh])
  );

  useEffect(() => {
    if (!contextId || !missionsQuery.isSuccess) return;
    if (!ridesFocused && !backgroundReady) return;
    const handle = setTimeout(() => {
      prefetchAdjacentDispatchMissions(queryClient, contextId, selectedDate);
    }, 220);
    return () => clearTimeout(handle);
  }, [backgroundReady, contextId, missionsQuery.isSuccess, queryClient, ridesFocused, selectedDate]);

  const dayMissions = useMemo(() => {
    const raw = missionsQuery.data?.missions ?? [];
    return raw.filter((mission) => missionBelongsToSelectedDay(mission, selectedDate));
  }, [missionsQuery.data?.missions, selectedDate]);
  const searchedDayMissions = useMemo(
    () => filterDayMissionsForLocalSearch(dayMissions, selectedDate, search),
    [dayMissions, search, selectedDate]
  );
  const missions = useMemo(
    () => filterMissionsByDispatchListChip(searchedDayMissions, status),
    [searchedDayMissions, status]
  );
  const delayPickupByBookingId = useMemo(() => {
    const rows = flattenCompanyDispatchDelays(dispatchDelaysQuery.data ?? []);
    return pickupDelaysByBookingLastWins(rows);
  }, [dispatchDelaysQuery.data]);
  const pickupEtaByBookingId = useMemo(
    () => pickupEtaIsoByBookingId(dispatchDelaysQuery.data ?? []),
    [dispatchDelaysQuery.data],
  );
  const missionBeingEdited = useMemo(() => {
    if (editMissionId == null) return null;
    return missions.find((item) => item.mission_id === editMissionId) ?? null;
  }, [editMissionId, missions]);

  const missionBeingScheduled = useMemo(() => {
    if (scheduleMissionId == null) return null;
    return missions.find((item) => item.mission_id === scheduleMissionId) ?? null;
  }, [scheduleMissionId, missions]);

  const rideEditInitial = useMemo(() => {
    if (missionBeingEdited == null) return null;
    return {
      clientId: null,
      clientLabel: missionBeingEdited.client_name ?? null,
      pickup: missionBeingEdited.pickup_label ?? "",
      dropoff: missionBeingEdited.dropoff_label ?? "",
      scheduledAt: missionBeingEdited.scheduled_at ?? null,
      notes: null,
    };
  }, [missionBeingEdited]);
  const allMissions = searchedDayMissions;
  const allMissionsWithSchedule = useMemo(
    () => allMissions.filter(missionHasRenderableSchedule),
    [allMissions]
  );
  const filterCountByKey = useMemo(() => {
    const o: Record<string, number> = {};
    for (const item of RIDE_STATUS_FILTERS) {
      o[item.key] = filterMissionsByDispatchListChip(allMissionsWithSchedule, item.key).length;
    }
    return o;
  }, [allMissionsWithSchedule]);
  const roleGuardsEnabled = isFeatureEnabled("company_mobile_role_guards_enabled");
  const contextPermissions = activeContext?.permissions ?? [];
  const canRunSensitiveAction = (permission: string, fallbackPermission: string) => {
    if (!roleGuardsEnabled) return true;
    if (contextPermissions.includes(permission)) return can(permission);
    return can(fallbackPermission);
  };
  const canAssignRide = canRunSensitiveAction("company:rides:assign", "company:rides:read");
  const canEditRide = canRunSensitiveAction("company:rides:edit", "company:rides:read");
  const canTransferRide = canRunSensitiveAction("company:rides:transfer", "company:rides:read");
  const canUrgentRide = canRunSensitiveAction("company:rides:urgent", "company:rides:read");
  const canCancelRide = canRunSensitiveAction("company:rides:cancel", "company:rides:read");
  const canScheduleRide = canRunSensitiveAction("company:rides:schedule", "company:rides:read");
  const filteredMissions = useMemo(() => {
    const missionsWithSchedule = missions.filter(missionHasRenderableSchedule);
    const f = filterNorm;
    if (f === "urgent") {
      return [...missionsWithSchedule].sort((left, right) => {
        const urgentLike = (m: (typeof left)) =>
          m.status === "pending" ||
          m.status === "proposed" ||
          m.status === "accepted" ||
          m.status === "assigned";
        const leftUrgent = urgentLike(left);
        const rightUrgent = urgentLike(right);
        if (leftUrgent === rightUrgent) return 0;
        return leftUrgent ? -1 : 1;
      });
    }
    if (f === "exceptions") {
      const now = Date.now();
      return missionsWithSchedule.filter((m) => {
        if (m.status === "completed" || m.status === "cancelled") return false;
        if ((m.status === "pending" || m.status === "proposed" || m.status === "accepted") && m.scheduled_at) {
          const t = Date.parse(m.scheduled_at);
          if (Number.isFinite(t) && t < now) return true;
        }
        return false;
      });
    }
    if (f === "delayed") {
      const now = Date.now();
      return missionsWithSchedule.filter((m) => {
        if (m.status === "completed" || m.status === "cancelled") return false;
        // Exclure les courses sans heure confirmée du calcul des retards.
        if (!hasConfirmedPickupTime(m)) return false;
        const scheduledAt = m.scheduled_at;
        if (!scheduledAt) return false;
        const t = Date.parse(scheduledAt);
        if (!Number.isFinite(t)) return false;
        return t < now;
      });
    }
    return missionsWithSchedule;
  }, [missions, filterNorm]);
  const ridesListContentStyle = useMemo(
    () => [rideStyles.page, rideStyles.listTop, { paddingBottom: Math.max(20, ridesBottomClearance) }],
    [ridesBottomClearance]
  );
  const dayLoaded = missionsQuery.data?.loaded ?? 0;
  const dayTotal = missionsQuery.data?.total ?? dayLoaded;
  const isDayComplete = missionsQuery.data?.is_complete ?? Boolean(missionsQuery.data);
  const dayRemaining = remainingUnloadedCount({ loaded: dayLoaded, total: dayTotal });
  const paginationError = missionsQuery.data?.pagination_error === true;
  const dayEmptyKind = resolveDispatchDayEmptyKind({
    missionCount: filteredMissions.length,
    hasCachedDay: missionsQuery.data != null,
    isLoading: missionsQuery.isLoading,
    isError: missionsQuery.isError,
    isOffline: networkOffline,
    isNetworkError: isCompanyNetworkRequestError(missionsQuery.error),
    searchActive: search.trim().length > 0,
    isDayComplete,
  });

  const openAssignModal = useCallback(
    async (missionId: number) => {
      if (!contextId) return;
      if (networkOffline) {
        setAssignModalMissionId(missionId);
        setModalError(COMPANY_OFFLINE_ACTION_MESSAGE);
        setModalPending(false);
        return;
      }
      setAssignModalMissionId(missionId);
      setModalError(null);
      setModalPending(true);
      try {
        const payload = await getCompanyAvailableDrivers({ contextId });
        const options = extractLabeledRows(payload, ["driver_id", "id"], [
          "driver_name",
          "display_name",
          "name",
          "full_name",
        ]);
        setDrivers(options);
        setSelectedDriverId(options[0]?.id ?? null);
      } catch (error) {
        setDrivers([]);
        setSelectedDriverId(null);
        setModalError(getDispatchApiErrorMessage(error, "Chargement des chauffeurs impossible."));
      } finally {
        setModalPending(false);
      }
    },
    [contextId, networkOffline]
  );

  const openTransferModal = useCallback(
    async (missionId: number) => {
      if (!contextId) return;
      if (networkOffline) {
        setTransferModalMissionId(missionId);
        setModalError(COMPANY_OFFLINE_ACTION_MESSAGE);
        setModalPending(false);
        return;
      }
      setTransferModalMissionId(missionId);
      setModalError(null);
      setModalPending(true);
      try {
        const { items } = await getCompanyPartnershipsForTransfer({ contextId });
        setPartnerships(items);
        setSelectedPartnerId(items[0]?.id ?? null);
      } catch (error) {
        setModalError(getDispatchApiErrorMessage(error, "Chargement des partenaires impossible."));
      } finally {
        setModalPending(false);
      }
    },
    [contextId, networkOffline]
  );

  const applyAssign = useCallback(async () => {
    if (!contextId || !assignModalMissionId || !selectedDriverId) return;
    const mission = missions.find((item) => item.mission_id === assignModalMissionId);
    setModalPending(true);
    try {
      await rideActions.assign.mutateAsync({
        missionId: assignModalMissionId,
        driverId: selectedDriverId,
      });
      emitCompanyDispatchTelemetry(
        "company.dispatch.driver_selected",
        {
          source: "company.rides.assign",
          context_type: "company",
          context_id: activeContext?.context_id ?? null,
          mission_id: assignModalMissionId,
          driver_id: selectedDriverId,
          previous_driver_id: mission?.driver_id ?? null,
        },
        { allowWhenDisabled: true }
      );
      setAssignModalMissionId(null);
      await refresh();
    } catch (error) {
      setModalError(error instanceof Error ? error.message : "Assignation impossible.");
      void refresh();
    } finally {
      setModalPending(false);
    }
  }, [activeContext?.context_id, assignModalMissionId, contextId, missions, refresh, rideActions.assign, selectedDriverId]);

  const applyTransfer = useCallback(async () => {
    if (!contextId || !transferModalMissionId || !selectedPartnerId) return;
    setModalPending(true);
    try {
      await transferCompanyRide({
        contextId,
        missionId: transferModalMissionId,
        partnershipId: selectedPartnerId,
      });
      emitCompanyDispatchTelemetry(
        "company.dispatch.driver_selected",
        {
          source: "company.rides.transfer",
          context_type: "company",
          context_id: activeContext?.context_id ?? null,
          mission_id: transferModalMissionId,
          target_company_id: selectedPartnerId,
        },
        { allowWhenDisabled: true }
      );
      setTransferModalMissionId(null);
      await refresh();
    } catch (error) {
      setModalError(getDispatchApiErrorMessage(error, "Transfert impossible."));
    } finally {
      setModalPending(false);
    }
  }, [activeContext?.context_id, contextId, refresh, selectedPartnerId, transferModalMissionId]);

  const markUrgentNow = useCallback(
    async (missionId: number) => {
      if (!contextId) return;
      setMissionActionPendingId(missionId);
      try {
        await markCompanyRideUrgent({
          contextId,
          missionId,
          payload: {
            urgent: true,
            reason_code: "manual_company_priority",
            source: "rides_list",
          },
        });
        await refresh();
      } catch (error) {
        // 409: course deja planifiee (heure reelle) — backend refuse l’urgence sentinelle.
        if (__DEV__) {
          console.warn("company.rides.urgent_failed", error);
        }
      } finally {
        setMissionActionPendingId(null);
      }
    },
    [contextId, refresh]
  );

  const cancelRideNow = useCallback(
    async (missionId: number) => {
      if (!contextId) return;
      setMissionActionPendingId(missionId);
      try {
        await cancelCompanyRide({
          contextId,
          missionId,
          reasonCode: "company_manual_cancel",
          note: "Cancellation requested from company rides list",
        });
        await refresh();
      } finally {
        setMissionActionPendingId(null);
      }
    },
    [contextId, refresh]
  );

  const openScheduleModal = useCallback((missionId: number) => {
    setScheduleMissionId(missionId);
  }, []);

  const handleToggleMissionExpand = useCallback((missionId: number) => {
    const tapId = beginTapFeedback("rides.expand", "company.rides");
    setExpandedMissionId((prev) => (prev === missionId ? null : missionId));
    endTapLocal(tapId);
  }, []);

  const goRideDetails = useCallback(
    (mission: CompanyDispatchMission) => {
      const tapId = beginTapFeedback("rides.details", "company.rides");
      rememberRideDetailSnapshot(mission);
      recordMissionDetailsPhase("tap");
      emitCompanyDispatchTelemetry(
        "company.dispatch.driver_selected",
        {
          source: "company.rides.open-detail",
          context_type: "company",
          context_id: activeContext?.context_id ?? null,
          mission_id: mission.mission_id,
          driver_id: mission.driver_id ?? null,
        },
        { allowWhenDisabled: true }
      );
      router.push({
        pathname: "/(app)/(company)/ride-details",
        params: { rideId: String(mission.mission_id) },
      });
      endTapNavigation(tapId);
      recordMissionDetailsPhase("navigation");
    },
    [activeContext?.context_id, router]
  );

  return (
    <PermissionGuard permission="company:rides:read">
      <Screen
        scroll={false}
        backgroundColor={E.BG}
        pageTransition={false}
        withHorizontalPadding
        stickyHeader={
          <EnterpriseHeader
            metaDetail="networkOnly"
            date={selectedDate}
            mode="manual"
            realtimeStatus={realtimeStatus.status}
            showModeChip={false}
            onOpenDatePicker={() => setDateSheetOpen(true)}
          />
        }
        extraScrollBottomPadding={100}
        contentContainerStyle={rideStyles.page}
      >
        <CompanyRidesMissionFlatList
          missions={filteredMissions}
          isLoading={missionsQuery.isLoading}
          expandedMissionId={expandedMissionId}
          missionActionPendingId={missionActionPendingId}
          contextId={contextId}
          dispatchDelaysFetched={dispatchDelaysQuery.isFetched}
          delayPickupByBookingId={delayPickupByBookingId}
          pickupEtaByBookingId={pickupEtaByBookingId}
          canAssignRide={canAssignRide}
          canEditRide={canEditRide}
          canTransferRide={canTransferRide}
          canUrgentRide={canUrgentRide}
          canCancelRide={canCancelRide}
          canScheduleRide={canScheduleRide}
          contentContainerStyle={ridesListContentStyle}
          searchQuery={search}
          isDayComplete={isDayComplete}
          remainingUnloadedCount={dayRemaining}
          emptyKind={dayEmptyKind}
          refreshControl={
            <RefreshControl
              refreshing={missionsQuery.isFetching && !missionsQuery.isLoading}
              onRefresh={() => void refresh()}
              tintColor={E.BRAND}
            />
          }
          listHeaderComponent={
            <>
        {showRealtimeFluxAlert ? (
          <View style={rideStyles.opsAlert} accessibilityRole="alert">
            <View style={rideStyles.opsAlertIconWell}>
              <Ionicons name="pulse-outline" size={20} color="#B91C1C" />
            </View>
            <View style={rideStyles.opsAlertBodyCol}>
              <Text style={rideStyles.opsAlertTitle}>Temps réel indisponible</Text>
              <Text style={rideStyles.opsAlertMessage}>
                {realtimeStatus.lastError?.trim() ||
                  "Connexion temps réel interrompue. Les mises à jour peuvent être retardées."}
              </Text>
              <Pressable
                onPress={() => void retryRealtimeSession()}
                disabled={sessionRetryBusy}
                style={({ pressed }) => [
                  rideStyles.opsAlertRetry,
                  sessionRetryBusy ? rideStyles.opsAlertRetryDisabled : null,
                  pressed && !sessionRetryBusy ? rideStyles.pressed : null,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Rétablir la session et recharger les courses"
              >
                {sessionRetryBusy ? (
                  <ActivityIndicator size="small" color={E.BRAND_DARK} />
                ) : (
                  <Text style={rideStyles.opsAlertRetryLabel}>Rétablir la session</Text>
                )}
              </Pressable>
            </View>
          </View>
        ) : null}
        <View style={rideStyles.searchBar}>
          <Ionicons name="search-outline" size={16} color={E.TEXT_MUTED} />
          <TextInput
            value={search}
            onChangeText={setSearch}
            placeholder="Client, adresse ou chauffeur…"
            placeholderTextColor={E.TEXT_MUTED}
            style={rideStyles.searchInput}
            returnKeyType="search"
            enterKeyHint="search"
            autoCorrect={false}
            autoCapitalize="none"
          />
          {search.length > 0 ? (
            <Pressable
              onPress={() => setSearch("")}
              style={rideStyles.searchClearHit}
              accessibilityLabel="Effacer la recherche"
              hitSlop={8}
            >
              <Ionicons name="close-circle" size={18} color={E.TEXT_MUTED} />
            </Pressable>
          ) : null}
        </View>
        {/* LOCK OFF : gate avant instanciation — aucun hook / GET / CTA moteur. */}
        <MaybeCompanyRidesEngineActions
          contextId={contextId}
          selectedDate={selectedDate}
          onRan={refresh}
        />
        {filterNorm === "exceptions" ? (
          <AppText variant="caption" style={rideStyles.exceptionsRouteHint} accessibilityRole="text">
            Vue « exceptions » : filtre local sur l’échéance (pas le comptage moteur). Les puces
            ci-dessous restent les statuts de mission.
          </AppText>
        ) : filterNorm === "delayed" ? (
          <AppText variant="caption" style={rideStyles.exceptionsRouteHint} accessibilityRole="text">
            Vue « retards » : courses actives dont l’horaire de prise en charge est dépassé (même
            logique que le tableau de bord en mode manuel).
          </AppText>
        ) : null}
        <View style={rideStyles.tabsPanel}>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
            style={rideStyles.tabsHeaderScrollView}
            contentContainerStyle={rideStyles.tabsHeaderScrollContent}
            accessible={false}
            {...(Platform.OS === "web" ? ({ "data-active-tab": status } as object) : {})}
          >
            {RIDE_STATUS_FILTERS.map((item) => {
              const on = status === item.key;
              const c = filterCountByKey[item.key] ?? 0;
              return (
                <Pressable
                  key={item.key}
                  onPress={() => {
                    setStatus(item.key);
                    clearRouteFilterOverride();
                  }}
                  style={({ pressed }) => [
                    rideStyles.tabButton,
                    on && rideStyles.tabButtonActive,
                    pressed && rideStyles.pressed,
                  ]}
                  accessibilityState={{ selected: on }}
                  accessibilityLabel={`Filtrer : ${item.label} (${c})`}
                  accessibilityHint={item.hint}
                  testID={`ride-filter-tab-${item.key}`}
                  {...(Platform.OS === "web" ? ({ "data-tour-id": `tab-${item.key}` } as object) : {})}
                >
                  <Ionicons
                    name={item.icon}
                    size={17}
                    color={on ? E.BRAND : E.TEXT_SEC}
                    accessibilityElementsHidden
                    importantForAccessibility="no"
                  />
                  <Text
                    style={[
                      rideStyles.tabCount,
                      on && rideStyles.tabCountActive,
                      !on && c === 0 && rideStyles.tabCountZero,
                      on && c === 0 && rideStyles.tabCountActiveZero,
                    ]}
                    accessibilityElementsHidden
                  >
                    {c}
                  </Text>
                </Pressable>
              );
            })}
          </ScrollView>
        </View>
            </>
          }
          onToggleExpand={handleToggleMissionExpand}
          onOpenAssign={openAssignModal}
          onGoDetails={goRideDetails}
          onEdit={(missionId) => {
            const tapId = beginTapFeedback("rides.edit", "company.rides");
            setEditMissionId(missionId);
            endTapLocal(tapId);
          }}
          onSchedule={openScheduleModal}
          onTransfer={openTransferModal}
          onCancel={cancelRideNow}
          onMarkUrgent={markUrgentNow}
          listFooterComponent={
            missionsQuery.error && !missionsQuery.data && dayEmptyKind !== "offline_unavailable" ? (
          <View style={rideStyles.opsAlert} accessibilityRole="alert">
            <View style={rideStyles.opsAlertIconWell}>
              <Ionicons name="alert-circle" size={20} color="#B91C1C" />
            </View>
            <View style={rideStyles.opsAlertBodyCol}>
              <Text style={rideStyles.opsAlertTitle}>Courses inaccessibles</Text>
              <Text style={rideStyles.opsAlertMessage}>
                {getDispatchApiErrorMessage(missionsQuery.error, "Erreur de chargement des courses.")}
              </Text>
            </View>
          </View>
            ) : paginationError ? (
              <View style={rideStyles.dayPaginationHint} accessibilityRole="alert">
                <Text style={rideStyles.dayPaginationHintText}>
                  Impossible de charger les courses restantes. Les {dayLoaded} déjà reçues restent visibles.
                </Text>
                <Pressable
                  onPress={() => missionsQuery.retryDayPagination()}
                  style={({ pressed }) => [
                    rideStyles.dayPaginationRetry,
                    pressed ? rideStyles.pressed : null,
                  ]}
                  accessibilityRole="button"
                  accessibilityLabel="Reprendre le chargement des courses restantes"
                >
                  <Text style={rideStyles.dayPaginationRetryLabel}>Réessayer</Text>
                </Pressable>
              </View>
            ) : dayRemaining > 0 && !isDayComplete ? (
              <View style={rideStyles.dayPaginationHint} accessibilityRole="text">
                <Text style={rideStyles.dayPaginationHintText}>
                  Chargement de {dayRemaining} courses supplémentaires…
                </Text>
              </View>
            ) : null
          }
        />

      </Screen>
      <Suspense fallback={null}>
        {assignModalMissionId != null ? (
          <AssignDriverModal
            visible
            pending={modalPending}
            drivers={drivers}
            selectedDriverId={selectedDriverId}
            error={modalError}
            onSelect={setSelectedDriverId}
            onConfirm={() => void applyAssign()}
            onClose={() => setAssignModalMissionId(null)}
            mode={
              missions.some((m) => m.mission_id === assignModalMissionId && m.driver_id != null)
                ? "reassign"
                : "assign"
            }
          />
        ) : null}
        {transferModalMissionId != null ? (
          <TransferRideModal
            visible
            pending={modalPending}
            options={partnerships}
            selectedPartnerId={selectedPartnerId}
            error={modalError}
            onSelect={setSelectedPartnerId}
            onConfirm={() => void applyTransfer()}
            onClose={() => {
              if (!modalPending) setTransferModalMissionId(null);
            }}
          />
        ) : null}
        {createModalVisible ? (
          <RideCreateModal
            visible
            onClose={() => setCreateModalVisible(false)}
            onCreated={() => void refresh()}
          />
        ) : null}
        {scheduleMissionId != null ? (
          <RideScheduleModal
            visible
            missionId={scheduleMissionId}
            initialScheduledAt={missionBeingScheduled?.scheduled_at ?? null}
            onClose={() => setScheduleMissionId(null)}
            onSaved={() => void refresh()}
          />
        ) : null}
        {editMissionId != null ? (
          <RideEditModal
            visible
            missionId={editMissionId}
            detailDate={selectedDate}
            isGuestMission
            initial={rideEditInitial}
            onClose={() => setEditMissionId(null)}
            onSaved={() => void refresh()}
          />
        ) : null}
      </Suspense>
      <DayPickerSheet
        visible={dateSheetOpen}
        selectedDate={selectedDate}
        onClose={() => setDateSheetOpen(false)}
        onSelectDate={(iso) => {
          const tapId = beginTapFeedback("rides.date", "company.rides");
          setSelectedDate(iso);
          setDateSheetOpen(false);
          endTapLocal(tapId);
        }}
      />
    </PermissionGuard>
  );
}
