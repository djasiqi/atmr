import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pressable, Platform, RefreshControl, ScrollView, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useFocusEffect } from "@react-navigation/native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { useSession } from "../../../src/core/sessionProvider";
import {
  useActiveCompanyContextId,
  useCompanyDispatchMissionsQuery,
  useCompanyRideActions,
  useCompanyRealtimeInvalidation,
  useCompanyRealtimeStatus,
} from "../../../src/features/company/hooks";
import { emitCompanyDispatchTelemetry } from "../../../src/features/company/telemetry/companyTelemetry";
import { contextRealtimeRouter } from "../../../src/core/realtime/contextRealtimeRouter";
import { normalizeCompanyEventType } from "../../../src/core/realtime/eventContracts";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { Button, InputField, Loader, Modal } from "../../../src/components/ui";
import { RideCreateModal } from "../../../src/features/company/components/rides/RideCreateModal";
import { RideEditModal } from "../../../src/features/company/components/rides/RideEditModal";
import {
  EnterpriseActionChip,
  EnterpriseFooterActionRow,
  EnterpriseRoundIconAction,
} from "../../../src/features/company/components/EnterpriseActionChip";
import { CompanyInboxButton } from "../../../src/features/company/components/CompanyInboxButton";
import { EnterpriseHeader } from "../../../src/features/company/components/EnterpriseHeader";
import { DispatchRideListCard } from "../../../src/features/company/components/DispatchRideListCard";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";
import { isDispatchCompleted, isDispatchCancelled } from "../../../src/features/company/utils/companyDispatchStatus";
import { filterMissionsByDispatchListChip } from "../../../src/features/company/utils/rideListStatusFilter";
import { isPickupSentinel } from "../../../src/features/company/utils/pickupSentinel";
import { createShadow } from "../../../src/styles/shadowStyles";
import { TransferRideModal } from "../../../src/features/company/components/transfers/TransferRideModal";
import {
  cancelCompanyRide,
  getCompanyAvailableDrivers,
  getCompanyDispatchModes,
  getCompanyPartnershipsForTransfer,
  markCompanyRideUrgent,
  runCompanyDispatch,
  runCompanyOptimizer,
  scheduleCompanyRide,
  transferCompanyRide,
} from "../../../src/features/company/api/companyApi";
import type { CompanyDispatchMission } from "../../../src/features/company/api/contracts";

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

/** `key` = paramètre `status` pour GET /company_mobile/dispatch/v1/rides (et fallback client). */
const RIDE_STATUS_FILTERS: { key: string; label: string; hint?: string }[] = [
  { key: "all", label: "Tous", hint: "Journée complète" },
  {
    key: "pending",
    label: "En attente",
    hint: "En attente de prise en charge, offre (proposée) ou acceptée",
  },
  { key: "assigned", label: "Affectés", hint: "Chauffeur connu, pas terminé" },
  { key: "in_flight", label: "En course", hint: "Uniquement : statut en route ou en mission" },
  { key: "completed", label: "Terminés", hint: "Terminé ou aller-retour clôturé" },
  { key: "cancelled", label: "Annulés" },
];

const searchBarShadow = createShadow({
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.03,
  shadowRadius: 4,
  elevation: 1,
});

const tabActiveSurfaceShadow = createShadow({
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.08,
  shadowRadius: 2,
  elevation: 1,
});

const rideStyles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: E.BG },
  page: { padding: 16, paddingBottom: 32, gap: 10 },
  searchBar: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: E.CARD,
    borderRadius: 12,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: E.BORDER,
    marginTop: 2,
    ...searchBarShadow,
  },
  searchInput: {
    flex: 1,
    borderWidth: 0,
    backgroundColor: "transparent",
    color: E.TEXT,
    paddingVertical: 10,
    paddingHorizontal: 8,
    fontSize: 14,
    minHeight: 40,
  },
  actionsRow: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  actionBtn: { flexGrow: 1, minWidth: 108 },
  /** Barre d’onglets (style segment type web CompanyDashboard) */
  tabsHeader: {
    width: "100%" as const,
    flexDirection: "row" as const,
    alignItems: "stretch" as const,
    gap: 2,
    backgroundColor: "rgba(15, 23, 42, 0.04)",
    borderWidth: 2,
    borderColor: "rgba(15, 23, 42, 0.1)",
    borderRadius: 10,
    padding: 2,
    marginTop: 2,
  },
  tabButton: {
    flex: 1,
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    gap: 4,
    minWidth: 0,
    minHeight: 32,
    paddingVertical: 5,
    paddingHorizontal: 4,
    borderRadius: 8,
    backgroundColor: "transparent",
  },
  tabButtonActive: {
    backgroundColor: E.CARD,
  },
  tabLabel: { color: E.TEXT_SEC, fontSize: 10, fontWeight: "600" as const },
  tabLabelActive: { color: E.BRAND, fontWeight: "800" as const, fontSize: 10 },
  tabBadge: {
    backgroundColor: E.TEXT_MUTED,
    borderRadius: 100,
    paddingVertical: 1,
    paddingHorizontal: 5,
    minWidth: 20,
    alignItems: "center" as const,
  },
  tabBadgeActive: { backgroundColor: E.BRAND },
  tabBadgeText: { color: "#FFFFFF", fontSize: 9, fontWeight: "700" as const, lineHeight: 12 },
  exceptionsRouteHint: {
    color: E.TEXT_SEC,
    fontSize: 12,
    lineHeight: 17,
    marginBottom: 4,
  },
  pressed: { opacity: 0.88 },
  rideCardWrapper: { marginBottom: 10 },
  listTop: { marginTop: 4 },
  emptyHint: { color: E.TEXT_MUTED, fontSize: 13, textAlign: "center" as const, paddingVertical: 8 },
  errorBlock: { color: E.DANGER, fontSize: 13, lineHeight: 19 },
  mutedText: { color: E.TEXT_SEC, fontSize: 13 },
  modalRow: { borderWidth: 1, borderRadius: 10, padding: 8, marginBottom: 5 },
  modalRowSelected: { borderColor: E.BRAND, backgroundColor: "rgba(0,121,107,0.08)" },
  modalRowNormal: { borderColor: E.BORDER, backgroundColor: E.CARD },
  modalRowText: { color: E.TEXT, fontSize: 14, fontWeight: "600" },
  modalRowTextSelected: { color: E.BRAND, fontSize: 14, fontWeight: "800" },
});

export default function CompanyRidesScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ filter?: string; create?: string }>();
  const { activeContext, can } = useSession();
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const [actionPending, setActionPending] = useState<null | "dispatch" | "optimizer">(null);
  const [activeMode, setActiveMode] = useState<"manual" | "semi_auto" | "fully_auto" | null>(null);
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
  const [missionActionPendingId, setMissionActionPendingId] = useState<number | null>(null);
  const [expandedMissionId, setExpandedMissionId] = useState<number | null>(null);
  const date = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const contextId = useActiveCompanyContextId();
  const rideActions = useCompanyRideActions();
  const missionsQuery = useCompanyDispatchMissionsQuery({ date, search, status });
  const missionsRefetch = missionsQuery.refetch;
  const allMissionsForCountsQuery = useCompanyDispatchMissionsQuery({
    date,
    search,
    status: "all",
  });
  const allMissionsRefetch = allMissionsForCountsQuery.refetch;
  const { invalidate } = useCompanyRealtimeInvalidation();
  const realtimeStatus = useCompanyRealtimeStatus();
  const lastOpenedTelemetryAtRef = useRef(0);

  const createParam = Array.isArray(params.create) ? params.create[0] : params.create;
  useEffect(() => {
    if (String(createParam) !== "1") return;
    setCreateModalVisible(true);
    const f = Array.isArray(params.filter) ? params.filter[0] : params.filter;
    void router.setParams({ create: undefined, filter: f ?? undefined });
  }, [createParam, params.filter, router]);

  const refresh = useCallback(async () => {
    await Promise.all([missionsRefetch(), allMissionsRefetch()]);
  }, [missionsRefetch, allMissionsRefetch]);

  useFocusEffect(
    useCallback(() => {
      const now = Date.now();
      if (now - lastOpenedTelemetryAtRef.current >= 1500) {
        lastOpenedTelemetryAtRef.current = now;
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
      void refresh();
    }, [activeContext?.context_id, refresh])
  );

  useEffect(() => {
    if (!activeContext || activeContext.context_type !== "company") return;
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
  }, [activeContext, invalidate, refresh]);

  const missions = useMemo(() => missionsQuery.data?.missions ?? [], [missionsQuery.data?.missions]);
  const allMissions = useMemo(
    () => allMissionsForCountsQuery.data?.missions ?? [],
    [allMissionsForCountsQuery.data?.missions]
  );
  const filterCountByKey = useMemo(() => {
    const o: Record<string, number> = {};
    for (const item of RIDE_STATUS_FILTERS) {
      o[item.key] = filterMissionsByDispatchListChip(allMissions, item.key).length;
    }
    return o;
  }, [allMissions]);
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
  const canDispatchManage = canRunSensitiveAction("company:dispatch:manage", "company:rides:read");
  const filteredMissions = useMemo(() => {
    const f = params.filter?.toLowerCase() ?? "";
    if (f === "urgent") {
      return [...missions].sort((left, right) => {
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
      return missions.filter((m) => {
        if (m.status === "completed" || m.status === "cancelled") return false;
        if ((m.status === "pending" || m.status === "proposed" || m.status === "accepted") && m.scheduled_at) {
          const t = Date.parse(m.scheduled_at);
          if (Number.isFinite(t) && t < now) return true;
        }
        return false;
      });
    }
    return missions;
  }, [missions, params.filter]);

  const loadDispatchMode = useCallback(async () => {
    if (!contextId) return;
    try {
      const payload = await getCompanyDispatchModes({ contextId });
      if (!payload || typeof payload !== "object") return;
      const obj = payload as Record<string, unknown>;
      const nextMode = obj.mode ?? obj.current_mode ?? obj.dispatch_mode ?? null;
      if (nextMode === "manual" || nextMode === "semi_auto" || nextMode === "fully_auto") {
        setActiveMode(nextMode);
      }
    } catch {
      // Mode illisible : l’en-tête affichera « — »; le reglage se fait via Parametres
    }
  }, [contextId]);

  const openAssignModal = useCallback(
    async (missionId: number) => {
      if (!contextId) return;
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
        setModalError(error instanceof Error ? error.message : "Chargement des chauffeurs impossible.");
      } finally {
        setModalPending(false);
      }
    },
    [contextId]
  );

  const openTransferModal = useCallback(
    async (missionId: number) => {
      if (!contextId) return;
      setTransferModalMissionId(missionId);
      setModalError(null);
      setModalPending(true);
      try {
        const payload = await getCompanyPartnershipsForTransfer({ contextId });
        const options = extractLabeledRows(
          payload,
          ["company_id", "target_company_id", "id"],
          ["company_name", "name", "label"]
        );
        setPartnerships(options);
        setSelectedPartnerId(options[0]?.id ?? null);
      } catch (error) {
        setModalError(error instanceof Error ? error.message : "Chargement des partenaires impossible.");
      } finally {
        setModalPending(false);
      }
    },
    [contextId]
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
        targetCompanyId: selectedPartnerId,
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
      setModalError(error instanceof Error ? error.message : "Transfert impossible.");
    } finally {
      setModalPending(false);
    }
  }, [activeContext?.context_id, contextId, refresh, selectedPartnerId, transferModalMissionId]);

  const runDispatchNow = useCallback(async () => {
    if (!contextId) return;
    setActionPending("dispatch");
    try {
      await runCompanyDispatch({ contextId, date });
      await refresh();
    } finally {
      setActionPending(null);
    }
  }, [contextId, date, refresh]);

  const runOptimizerNow = useCallback(async () => {
    if (!contextId) return;
    setActionPending("optimizer");
    try {
      await runCompanyOptimizer({ contextId, date });
      await refresh();
    } finally {
      setActionPending(null);
    }
  }, [contextId, date, refresh]);

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

  const scheduleRideNow = useCallback(
    async (missionId: number) => {
      if (!contextId) return;
      setMissionActionPendingId(missionId);
      try {
        const pickupAt = new Date(Date.now() + 15 * 60 * 1000).toISOString();
        await scheduleCompanyRide({
          contextId,
          missionId,
          payload: {
            pickup_at: pickupAt,
            timezone: "Europe/Zurich",
            note: "Reschedule from company rides list",
            force_recompute: true,
          },
        });
        await refresh();
      } finally {
        setMissionActionPendingId(null);
      }
    },
    [contextId, refresh]
  );

  useFocusEffect(
    useCallback(() => {
      void loadDispatchMode();
    }, [loadDispatchMode])
  );

  const goRideDetails = useCallback(
    (mission: CompanyDispatchMission) => {
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
    },
    [activeContext?.context_id, router]
  );

  return (
    <PermissionGuard permission="company:rides:read">
      <ScrollView
        style={rideStyles.scroll}
        contentContainerStyle={rideStyles.page}
        refreshControl={
          <RefreshControl
            refreshing={missionsQuery.isLoading}
            onRefresh={() => void refresh()}
            tintColor={E.BRAND}
          />
        }
      >
        <EnterpriseHeader
          date={date}
          mode={activeMode}
          realtimeStatus={realtimeStatus.status}
          trailing={activeContext?.context_id ? <CompanyInboxButton /> : null}
        />
        {missionsQuery.isLoading ? <Loader /> : null}
        <View style={rideStyles.searchBar}>
          <Ionicons name="search-outline" size={16} color={E.TEXT_MUTED} />
          <InputField
            value={search}
            onChangeText={setSearch}
            placeholder="Client, adresse ou chauffeur…"
            style={rideStyles.searchInput}
            placeholderTextColor={E.TEXT_MUTED}
          />
        </View>
        {activeMode !== "manual" ? (
          <View style={rideStyles.actionsRow}>
            <Button
              label={actionPending === "dispatch" ? "Exécution…" : "Lancer le dispatch"}
              variant="primary"
              onPress={() => void runDispatchNow()}
              disabled={!contextId || actionPending !== null || !canDispatchManage}
              style={rideStyles.actionBtn}
            />
            <Button
              label={actionPending === "optimizer" ? "Optimiseur…" : "Lancer l’optimiseur"}
              onPress={() => void runOptimizerNow()}
              disabled={!contextId || actionPending !== null || !canDispatchManage}
              style={rideStyles.actionBtn}
            />
          </View>
        ) : null}
        {params.filter?.toLowerCase() === "exceptions" ? (
          <Text style={rideStyles.exceptionsRouteHint} accessibilityRole="text">
            Vue « exceptions » : filtre local sur l’échéance (pas le comptage moteur). Les puces
            ci-dessous restent les statuts de mission.
          </Text>
        ) : null}
        <View
          style={rideStyles.tabsHeader}
          accessible={false}
          {...(Platform.OS === "web" ? ({ "data-active-tab": status } as object) : {})}
        >
          {RIDE_STATUS_FILTERS.map((item) => {
            const on = status === item.key;
            const c = filterCountByKey[item.key] ?? 0;
            return (
              <Pressable
                key={item.key}
                onPress={() => setStatus(item.key)}
                style={({ pressed }) => [
                  rideStyles.tabButton,
                  on && rideStyles.tabButtonActive,
                  on && tabActiveSurfaceShadow,
                  pressed && rideStyles.pressed,
                ]}
                accessibilityState={{ selected: on }}
                accessibilityLabel={`Filtrer : ${item.label} (${c})`}
                accessibilityHint={item.hint}
                testID={`ride-filter-tab-${item.key}`}
                {...(Platform.OS === "web" ? ({ "data-tour-id": `tab-${item.key}` } as object) : {})}
              >
                <Text
                  style={on ? rideStyles.tabLabelActive : rideStyles.tabLabel}
                  numberOfLines={1}
                  allowFontScaling
                >
                  {item.label}
                </Text>
                <View style={[rideStyles.tabBadge, on && rideStyles.tabBadgeActive]}>
                  <Text style={rideStyles.tabBadgeText}>{c}</Text>
                </View>
              </Pressable>
            );
          })}
        </View>
        <View style={rideStyles.listTop}>
          {filteredMissions.map((mission) => {
            const isExpanded = expandedMissionId === mission.mission_id;
            const thisBusy = missionActionPendingId === mission.mission_id;
            const completed = isDispatchCompleted(mission);
            const cancelled = isDispatchCancelled(mission);
            const showUrgent = isPickupSentinel(mission.scheduled_at);
            const unassigned = mission.driver_id == null;
            /** Sous l’en-tête : seulement Urgence (heure TBD) et Assigner (non assigné). « Détails » = dans le bloc déplié uniquement. */
            const timeSentinelAction =
              !completed && !cancelled && showUrgent ? (
                <EnterpriseRoundIconAction
                  icon="flash"
                  variant="urgent"
                  accessibilityLabel="Urgence"
                  onPress={() => void markUrgentNow(mission.mission_id)}
                  disabled={!contextId || thisBusy || !canUrgentRide}
                  showSpinner={thisBusy}
                  spinnerColor="#FFFFFF"
                />
              ) : undefined;
            return (
              <View key={mission.mission_id} style={rideStyles.rideCardWrapper}>
                <DispatchRideListCard
                  mission={mission}
                  expanded={isExpanded}
                  onToggleExpand={() =>
                    setExpandedMissionId((prev) => (prev === mission.mission_id ? null : mission.mission_id))
                  }
                  timeSentinelAction={timeSentinelAction}
                  onUnassignedPress={
                    !completed && !cancelled && unassigned ? () => void openAssignModal(mission.mission_id) : undefined
                  }
                  unassignedPressDisabled={!contextId || !canAssignRide}
                  footer={
                    isExpanded ? (
                      <EnterpriseFooterActionRow>
                        <EnterpriseActionChip
                          icon="open-outline"
                          label="Détails"
                          tone="details"
                          onPress={() => goRideDetails(mission)}
                        />
                        {!completed && !cancelled ? (
                          <>
                            {mission.driver_id != null ? (
                              <EnterpriseActionChip
                                icon="person-add-outline"
                                label="Réassigner"
                                onPress={() => void openAssignModal(mission.mission_id)}
                                disabled={!contextId || !canAssignRide}
                              />
                            ) : null}
                            <EnterpriseActionChip
                              icon="create-outline"
                              label="Éditer"
                              onPress={() => setEditMissionId(mission.mission_id)}
                              disabled={!contextId || !canEditRide}
                            />
                            <EnterpriseActionChip
                              icon="time-outline"
                              label={thisBusy ? "Planif…" : "Planifier"}
                              onPress={() => void scheduleRideNow(mission.mission_id)}
                              disabled={!contextId || thisBusy || !canScheduleRide}
                              showSpinner={thisBusy}
                              spinnerColor={E.BRAND}
                            />
                            <EnterpriseActionChip
                              icon="swap-horizontal-outline"
                              label="Transférer"
                              tone="transfer"
                              onPress={() => void openTransferModal(mission.mission_id)}
                              disabled={!contextId || !canTransferRide}
                            />
                            <EnterpriseActionChip
                              icon="close-circle-outline"
                              label={thisBusy ? "Annulation…" : "Annuler"}
                              tone="danger"
                              onPress={() => void cancelRideNow(mission.mission_id)}
                              disabled={!contextId || thisBusy || !canCancelRide}
                            />
                          </>
                        ) : null}
                      </EnterpriseFooterActionRow>
                    ) : null
                  }
                />
              </View>
            );
          })}
        </View>
        {!missionsQuery.isLoading && filteredMissions.length === 0 ? (
          <Text style={rideStyles.emptyHint}>Aucune course pour ce filtre.</Text>
        ) : null}
        {missionsQuery.error ? (
          <Text style={rideStyles.errorBlock}>
            {missionsQuery.error instanceof Error
              ? missionsQuery.error.message
              : "Erreur de chargement des courses."}
          </Text>
        ) : null}
      </ScrollView>
      <Modal
        visible={assignModalMissionId != null}
        title="Assigner un chauffeur"
        onClose={() => {
          if (!modalPending) setAssignModalMissionId(null);
        }}
      >
        {modalPending ? <Loader /> : null}
        {drivers.length === 0 && !modalPending ? (
          <Text style={rideStyles.mutedText}>Aucun chauffeur disponible.</Text>
        ) : null}
        {drivers.map((driver) => (
          <Pressable
            key={driver.id}
            onPress={() => setSelectedDriverId(driver.id)}
            style={[
              rideStyles.modalRow,
              selectedDriverId === driver.id ? rideStyles.modalRowSelected : rideStyles.modalRowNormal,
            ]}
          >
            <Text
              style={
                selectedDriverId === driver.id ? rideStyles.modalRowTextSelected : rideStyles.modalRowText
              }
            >
              {driver.label}
            </Text>
          </Pressable>
        ))}
        <Button
          label={modalPending ? "Assignation…" : "Confirmer"}
          variant="primary"
          onPress={() => void applyAssign()}
          disabled={modalPending || selectedDriverId == null}
        />
        {modalError ? <Text style={rideStyles.errorBlock}>{modalError}</Text> : null}
      </Modal>
      <TransferRideModal
        visible={transferModalMissionId != null}
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
      <RideCreateModal
        visible={createModalVisible}
        onClose={() => setCreateModalVisible(false)}
        onCreated={() => void refresh()}
      />
      <RideEditModal
        visible={editMissionId != null}
        missionId={editMissionId}
        isGuestMission
        initial={
          editMissionId == null
            ? null
            : {
                clientId: null,
                pickup: missions.find((item) => item.mission_id === editMissionId)?.pickup_label ?? "",
                dropoff: missions.find((item) => item.mission_id === editMissionId)?.dropoff_label ?? "",
                scheduledAt: missions.find((item) => item.mission_id === editMissionId)?.scheduled_at ?? null,
                notes: null,
              }
        }
        onClose={() => setEditMissionId(null)}
        onSaved={() => void refresh()}
      />
    </PermissionGuard>
  );
}
