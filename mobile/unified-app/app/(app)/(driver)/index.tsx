import { useCallback, useMemo, useState, type ComponentProps } from "react";
import { Pressable, RefreshControl, StyleSheet, Switch, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import {
  useDriverAvailabilityMutation,
  useDriverMissionsListFocusResync,
  useDriverMissionsQuery,
  useDriverRealtimeSync,
  useDriverStatusTransition,
} from "../../../src/features/driver/hooks";
import { getDriverStatusUx } from "../../../src/features/driver/statusDictionary";
import { useSession } from "../../../src/core/sessionProvider";
import type { DriverMission, DriverTransitionStatus } from "../../../src/features/driver/types";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import { DashboardMissionListSkeleton } from "../../../src/features/driver/components/DashboardMissionListSkeleton";
import { MissionCard } from "../../../src/features/driver/components/MissionCard";
import { ConfirmCompletionModal } from "../../../src/features/driver/components/ConfirmCompletionModal";
import { CancelJustificationModal } from "../../../src/features/driver/components/CancelJustificationModal";
import { ReleaseConfirmationModal } from "../../../src/features/driver/components/ReleaseConfirmationModal";
import { DriverStateBanners } from "../../../src/features/driver/components/DriverStateBanners";
import { filterNextMissionsOnly } from "../../../src/features/driver/domain/missionGrouping";
import { AppText, Screen, useAppViewport } from "../../../src/design/responsive";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";
import { createShadow } from "../../../src/styles/shadowStyles";

/** Aligné sur `dashboard.tsx` company : ombres / bordures / espacement homogènes. */
const dashboardSurfaceShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

/** Palette locale alignée sur `enterpriseOpsTheme` (mêmes valeurs que côté company dashboard). */
const C = {
  pageBg: E.BG,
  cardBg: E.CARD,
  text: E.TEXT,
  textMuted: E.TEXT_MUTED,
  textSub: E.TEXT_SEC,
  border: E.BORDER,
  brand: E.BRAND,
  /** Réf. tuiles KPI dashboard company (`kpiIconWrap`) — pas 10 % pour éviter un vert trop saturé. */
  kpiIconWell: "rgba(0, 121, 107, 0.08)",
  /** Pastille switch web alignée snapshot (`rgb(0, 150, 136)`). */
  switchThumbOn: "#009688",
} as const;

type KpiIconName = ComponentProps<typeof Ionicons>["name"];

type KpiRow = {
  key: string;
  label: string;
  icon: KpiIconName;
  value: string;
};

function selectActiveMission(missions: DriverMission[] | undefined): DriverMission | null {
  if (!Array.isArray(missions) || missions.length === 0) return null;
  const nextScope = filterNextMissionsOnly(missions);
  if (nextScope.length > 0) return nextScope[0] ?? null;
  const firstNonTerminal = missions.find((mission) => {
    const ux = getDriverStatusUx(typeof mission.status === "string" ? mission.status : null);
    return !ux.terminal;
  });
  return firstNonTerminal ?? missions[0] ?? null;
}

function isMissionStatusEqual(mission: DriverMission, ...candidates: string[]): boolean {
  const status = String(mission.status ?? "").toLowerCase();
  return candidates.includes(status);
}

function getScheduledEpoch(mission: DriverMission): number {
  const raw = (mission.scheduled_time ?? mission.scheduled_at) as unknown;
  if (typeof raw !== "string" || raw.length === 0) return Number.POSITIVE_INFINITY;
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
}

const SWISS_TZ = "Europe/Zurich";

/** Aperçu prochaine course : date + heure (Suisse) sur une ligne courte (aligné dashboard company). */
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

function conciseAddressSegment(s: string | null | undefined, maxLen = 48): string {
  const t = s?.trim() ?? "";
  if (!t) return "—";
  const head = t.split(",")[0]?.trim() || t;
  if (head.length <= maxLen) return head;
  return `${head.slice(0, Math.max(0, maxLen - 1))}…`;
}

function KpiTile({ row, onPress }: { row: KpiRow; onPress?: () => void }) {
  const inner = (
    <View style={styles.kpiTopRow}>
      <View style={styles.kpiIconWrap} accessibilityElementsHidden>
        <Ionicons name={row.icon} size={16} color={C.brand} />
      </View>
      <View style={styles.kpiTextCol}>
        <AppText variant="caption" style={styles.kpiLabel} numberOfLines={1}>
          {row.label}
        </AppText>
        <AppText variant="sectionTitle" style={styles.kpiValue} numberOfLines={1} adjustsFontSizeToFit>
          {row.value}
        </AppText>
      </View>
    </View>
  );
  if (onPress) {
    return (
      <Pressable
        onPress={onPress}
        accessibilityRole="button"
        accessibilityLabel={row.label}
        style={({ pressed }) => [styles.kpiStat, pressed && styles.kpiStatPressed]}
      >
        {inner}
      </Pressable>
    );
  }
  return (
    <View style={styles.kpiStat} accessibilityLabel={row.label}>
      {inner}
    </View>
  );
}

export default function DriverHomeScreen() {
  const router = useRouter();
  const { horizontalPadding } = useAppViewport();
  const { status: sessionStatus } = useSession();
  const missionsQuery = useDriverMissionsQuery();
  useDriverRealtimeSync();
  useDriverMissionsListFocusResync();
  const missions = useMemo(
    () => (Array.isArray(missionsQuery.data) ? (missionsQuery.data as DriverMission[]) : []),
    [missionsQuery.data]
  );
  const activeMission = selectActiveMission(missions);
  const availabilityMutation = useDriverAvailabilityMutation();
  const transitionMutation = useDriverStatusTransition();
  const [isAvailable, setIsAvailable] = useState(true);
  const [confirmCompletionOpen, setConfirmCompletionOpen] = useState(false);
  const [cancelMissionOpen, setCancelMissionOpen] = useState(false);
  const [releaseMissionOpen, setReleaseMissionOpen] = useState(false);
  const bootstrapPending = sessionStatus !== "ready" || missionsQuery.isLoading;

  const [pullRefreshing, setPullRefreshing] = useState(false);
  const onPullRefresh = useCallback(async () => {
    setPullRefreshing(true);
    try {
      await missionsQuery.refetch();
    } finally {
      setPullRefreshing(false);
    }
  }, [missionsQuery]);

  const { kpiRows, upcomingMissions } = useMemo(() => {
    let toDo = 0;
    let inProgress = 0;
    let completedToday = 0;

    const todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);
    const todayStartMs = todayStart.getTime();
    const tomorrowStartMs = todayStartMs + 24 * 60 * 60 * 1000;

    for (const m of missions) {
      if (isMissionStatusEqual(m, "assigned", "pending", "accepted", "awaiting_client_payment")) {
        toDo += 1;
      } else if (isMissionStatusEqual(m, "en_route", "in_progress", "arrived")) {
        inProgress += 1;
      } else if (isMissionStatusEqual(m, "completed", "return_completed")) {
        const finishedEpoch = getScheduledEpoch(m);
        if (
          Number.isFinite(finishedEpoch) &&
          finishedEpoch >= todayStartMs &&
          finishedEpoch < tomorrowStartMs
        ) {
          completedToday += 1;
        }
      }
    }

    const upcoming = missions
      .filter((m) => {
        const ux = getDriverStatusUx(typeof m.status === "string" ? m.status : null);
        if (ux.terminal) return false;
        if (activeMission && m.id === activeMission.id) return false;
        return true;
      })
      .sort((a, b) => getScheduledEpoch(a) - getScheduledEpoch(b))
      .slice(0, 3);

    const rows: KpiRow[] = [
      { key: "todo", label: "À effectuer", icon: "list-outline", value: String(toDo) },
      { key: "in_progress", label: "En cours", icon: "navigate-outline", value: String(inProgress) },
      { key: "completed", label: "Terminées", icon: "checkmark-done-outline", value: String(completedToday) },
      {
        key: "availability",
        label: "Disponibilité",
        icon: "sync-outline",
        value: isAvailable ? "ON" : "OFF",
      },
    ];

    return { kpiRows: rows, upcomingMissions: upcoming };
  }, [missions, activeMission, isAvailable]);

  const onOpenMission = (missionId: number) =>
    router.push({
      pathname: "/(app)/(driver)/missions/[missionId]",
      params: { missionId: String(missionId) },
    });

  const onAllMissions = () => router.push("/(app)/(driver)/missions");

  const onMissionTransitionFromDashboard = useCallback(
    (target: DriverTransitionStatus) => {
      if (!activeMission) return;
      if (target === "COMPLETED") {
        setConfirmCompletionOpen(true);
        return;
      }
      if (target === "CANCELLED") {
        setCancelMissionOpen(true);
        return;
      }
      transitionMutation.mutate({ missionId: activeMission.id, targetStatus: target });
    },
    [activeMission, transitionMutation]
  );

  /**
   * Bouton « Libérer » du `MissionCard` — ouvre la confirmation puis envoie
   * `CANCELLED` + `reason: "RELEASE"` (parité `operations-app:handleReleaseConfirm`).
   */
  const onMissionReleaseFromDashboard = useCallback(() => {
    if (!activeMission) return;
    setReleaseMissionOpen(true);
  }, [activeMission]);

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <>
          <Screen
          scroll
          backgroundColor={C.pageBg}
          withHorizontalPadding={false}
          contentContainerStyle={[
            styles.page,
            {
              backgroundColor: C.pageBg,
              paddingLeft: horizontalPadding,
              paddingRight: horizontalPadding,
            },
          ]}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          refreshControl={
            <RefreshControl
              refreshing={pullRefreshing}
              onRefresh={() => void onPullRefresh()}
              tintColor={C.brand}
              colors={[C.brand]}
            />
          }
        >
          <View style={styles.hero}>
            <View style={styles.heroIconWrap} accessibilityElementsHidden>
              <Ionicons name="speedometer-outline" size={28} color={C.brand} />
            </View>
            <View style={styles.heroText}>
              <AppText variant="screenTitle" style={styles.heroTitle}>
                Espace Driver
              </AppText>
            </View>
            <View style={styles.availabilityToggleColumn} accessibilityLabel="Disponibilité chauffeur">
              <View style={styles.availabilityIndicator}>
                <View
                  style={[
                    styles.availabilityDot,
                    { backgroundColor: isAvailable ? C.brand : C.textMuted },
                  ]}
                  accessibilityElementsHidden
                />
                <AppText variant="caption" style={styles.availabilityLabel} numberOfLines={1}>
                  {isAvailable ? "Disponible" : "Indisponible"}
                </AppText>
              </View>
              <Switch
                value={isAvailable}
                onValueChange={(next) => {
                  availabilityMutation.mutate(next, {
                    onSuccess: () => setIsAvailable(next),
                  });
                }}
                disabled={availabilityMutation.isPending}
                trackColor={{ false: "#CBD5E1", true: "rgba(0, 121, 107, 0.45)" }}
                thumbColor={isAvailable ? C.switchThumbOn : "#F1F5F9"}
                ios_backgroundColor="#CBD5E1"
                accessibilityRole="switch"
                accessibilityLabel={`Disponibilité chauffeur. Actuellement ${
                  isAvailable ? "disponible" : "indisponible"
                }. Basculer pour changer.`}
              />
            </View>
          </View>

          <DriverStateBanners />

          <View style={styles.kpiRow} accessibilityLabel="Indicateurs clés">
            {kpiRows.map((row) => (
              <KpiTile key={row.key} row={row} />
            ))}
          </View>

          {bootstrapPending ? (
            <View style={styles.dashboardSection}>
              <View style={styles.summaryHeaderRow}>
                <View style={styles.sectionIconWrap} accessibilityElementsHidden>
                  <Ionicons name="briefcase-outline" size={16} color={C.brand} />
                </View>
                <AppText variant="sectionTitle" style={styles.summaryTitle}>
                  Mission active
                </AppText>
              </View>
              <AppText variant="caption" style={styles.sectionHint}>
                Chargement de votre mission active…
              </AppText>
              <DashboardMissionListSkeleton />
            </View>
          ) : null}

          {missionsQuery.isError ? (
            <AppText variant="error">
              Erreur chargement missions : {(missionsQuery.error as Error)?.message ?? "Erreur"}
            </AppText>
          ) : null}

          {!bootstrapPending && activeMission ? (
            <View style={styles.missionActiveSection}>
              <View style={styles.missionActiveHeading}>
                <View style={styles.sectionIconWrap} accessibilityElementsHidden>
                  <Ionicons name="briefcase-outline" size={16} color={C.brand} />
                </View>
                <AppText variant="sectionTitle" style={styles.summaryTitle}>
                  Mission active
                </AppText>
              </View>
              <MissionCard
                mission={activeMission}
                pending={transitionMutation.isPending}
                onMissionTransition={onMissionTransitionFromDashboard}
                onMissionRelease={onMissionReleaseFromDashboard}
              />
            </View>
          ) : !bootstrapPending ? (
            <View style={styles.emptyMissionCard} accessibilityRole="text">
              <View style={styles.emptyIconWrap} accessibilityElementsHidden>
                <Ionicons name="car-outline" size={32} color={C.brand} />
              </View>
              <AppText variant="sectionTitle" style={styles.emptyTitle}>
                Aucune mission active
              </AppText>
              <AppText variant="bodyMuted" style={styles.emptySubtitle}>
                Les courses à venir ou en cours apparaîtront ici. Utilisez la barre du bas pour ouvrir
                les missions, les courses ou le chat.
              </AppText>
            </View>
          ) : null}

          {!bootstrapPending && upcomingMissions.length > 0 ? (
            <View style={styles.dashboardSection}>
              <View style={styles.summaryHeaderRow}>
                <View style={styles.sectionIconWrap} accessibilityElementsHidden>
                  <Ionicons name="time-outline" size={16} color={C.brand} />
                </View>
                <AppText variant="sectionTitle" style={styles.summaryTitle}>
                  Prochaines missions
                </AppText>
              </View>
              <View style={styles.sectionBody}>
                {upcomingMissions.map((m, index) => {
                  const scheduledRaw = (m.scheduled_time ?? m.scheduled_at) as string | null | undefined;
                  const clientName = typeof m.client_name === "string" && m.client_name.trim().length > 0
                    ? m.client_name.trim()
                    : "Invité";
                  return (
                    <Pressable
                      key={m.id}
                      onPress={() => onOpenMission(m.id)}
                      accessibilityRole="button"
                      accessibilityLabel={`Ouvrir mission ${m.id}`}
                      style={({ pressed }) => [
                        styles.missionBlock,
                        index < upcomingMissions.length - 1 && styles.missionBlockSep,
                        pressed && styles.missionBlockPressed,
                      ]}
                    >
                      <AppText variant="label" style={styles.missionWhen} numberOfLines={1}>
                        {formatNextCourseWhen(scheduledRaw)}
                      </AppText>
                      <AppText variant="label" style={styles.missionClientName} numberOfLines={1}>
                        {clientName}
                      </AppText>
                      <AppText variant="caption" style={styles.missionAddressLine} numberOfLines={2}>
                        <AppText variant="caption" style={styles.missionAddressKey}>
                          Départ :{" "}
                        </AppText>
                        {conciseAddressSegment((m.pickup_location as string | null | undefined) ?? null)}
                      </AppText>
                      <AppText variant="caption" style={styles.missionAddressLine} numberOfLines={2}>
                        <AppText variant="caption" style={styles.missionAddressKey}>
                          Arrivée :{" "}
                        </AppText>
                        {conciseAddressSegment((m.dropoff_location as string | null | undefined) ?? null)}
                      </AppText>
                    </Pressable>
                  );
                })}
                <Pressable
                  onPress={onAllMissions}
                  accessibilityRole="button"
                  accessibilityLabel="Voir toutes les missions"
                  style={({ pressed }) => [
                    styles.linkCtaRow,
                    styles.linkCtaRowFirst,
                    pressed && styles.linkCtaRowPressed,
                  ]}
                >
                  <AppText variant="label" style={styles.fleetCtaText}>
                    Voir toutes les missions
                  </AppText>
                  <Ionicons name="chevron-forward" size={18} color={C.brand} />
                </Pressable>
              </View>
            </View>
          ) : null}
        </Screen>
        <ConfirmCompletionModal
          visible={confirmCompletionOpen}
          missionId={activeMission?.id ?? null}
          pending={transitionMutation.isPending}
          onCancel={() => setConfirmCompletionOpen(false)}
          onConfirm={() => {
            if (!activeMission) return;
            transitionMutation.mutate({
              missionId: activeMission.id,
              targetStatus: "COMPLETED",
            });
            setConfirmCompletionOpen(false);
          }}
        />
        <CancelJustificationModal
          visible={cancelMissionOpen}
          pending={transitionMutation.isPending}
          onCancel={() => setCancelMissionOpen(false)}
          onConfirm={(reason) => {
            if (!activeMission) return;
            transitionMutation.mutate({
              missionId: activeMission.id,
              targetStatus: "CANCELLED",
              reason,
            });
            setCancelMissionOpen(false);
          }}
        />
        <ReleaseConfirmationModal
          visible={releaseMissionOpen}
          missionId={activeMission?.id ?? null}
          pending={transitionMutation.isPending}
          onCancel={() => setReleaseMissionOpen(false)}
          onConfirm={() => {
            if (!activeMission) return;
            transitionMutation.mutate({
              missionId: activeMission.id,
              targetStatus: "CANCELLED",
              reason: "RELEASE",
            });
            setReleaseMissionOpen(false);
          }}
        />
        </>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  /**
   * Miroir `dashboard.tsx` company + snapshot operations-app :
   * fond `#f4f7fc`, padding horizontal injecté via viewport (22 regular), gap vertical ~14–16.
   */
  page: {
    flexGrow: 1,
    paddingTop: 16,
    paddingBottom: 16,
    gap: 14,
  },
  hero: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 14,
  },
  heroIconWrap: {
    width: 52,
    height: 52,
    borderRadius: 16,
    backgroundColor: "rgba(10, 143, 122, 0.12)",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.12)",
  },
  heroText: {
    flex: 1,
    minWidth: 0,
    gap: 6,
  },
  /** Réf. titre hero snapshot : 22px / 27 lh, poids 700 (`r-color-1djweci`). */
  heroTitle: {
    color: C.text,
    fontSize: 22,
    lineHeight: 27,
    fontWeight: "700",
  },
  /** Colonne droite du hero : indicateur "Disponible/Indisponible" + Switch natif. */
  availabilityToggleColumn: {
    flexShrink: 0,
    alignItems: "flex-end",
    justifyContent: "flex-start",
    gap: 8,
    paddingTop: 6,
  },
  availabilityIndicator: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  availabilityDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  availabilityLabel: {
    color: C.textSub,
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.4,
    textTransform: "uppercase",
  },
  /** Mêmes tokens que `dashboard.tsx` company : tuile KPI. */
  kpiRow: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  kpiStat: {
    flexGrow: 1,
    minWidth: "40%",
    maxWidth: "100%",
    backgroundColor: C.cardBg,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 14,
    ...dashboardSurfaceShadow,
  },
  kpiStatPressed: { opacity: 0.88 },
  kpiTopRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  kpiIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: C.kpiIconWell,
    alignItems: "center",
    justifyContent: "center",
  },
  kpiTextCol: { flex: 1, minWidth: 0, justifyContent: "center" },
  kpiLabel: {
    color: C.textSub,
    fontWeight: "700",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    fontSize: 12,
  },
  kpiValue: {
    marginTop: 1,
    color: C.text,
    fontWeight: "800",
    lineHeight: 22,
  },
  /** Section blanche (mission active / prochaines missions). */
  dashboardSection: {
    backgroundColor: C.cardBg,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: C.border,
    padding: 16,
    ...dashboardSurfaceShadow,
  },
  sectionBody: {
    gap: 2,
    paddingTop: 0,
    paddingBottom: 2,
  },
  summaryHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 12,
  },
  sectionIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  summaryTitle: {
    color: C.text,
    fontSize: 16,
    fontWeight: "700",
    flex: 1,
    minWidth: 0,
  },
  /** Bloc titre « Mission active » au-dessus de la carte (réf. operations-app « Mission actuelle »). */
  missionActiveSection: {
    alignSelf: "stretch",
    gap: 10,
  },
  missionActiveHeading: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  sectionHint: {
    marginBottom: 8,
    color: C.textMuted,
  },
  /** Lignes mission (miroir des "Prochaines courses" company). */
  missionBlock: { gap: 3, paddingBottom: 10 },
  missionBlockSep: {
    marginBottom: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: C.border,
  },
  missionBlockPressed: { opacity: 0.88 },
  missionWhen: {
    color: C.brand,
    fontWeight: "800",
  },
  missionClientName: { color: C.text, fontWeight: "700", marginTop: 2 },
  missionAddressLine: { color: C.textSub, lineHeight: 16, fontWeight: "500", marginTop: 2 },
  missionAddressKey: { color: C.textMuted, fontWeight: "700" },
  /** Lien "Voir toutes les missions" (miroir company). */
  linkCtaRow: {
    marginTop: 8,
    minHeight: 44,
    paddingVertical: 10,
    paddingHorizontal: 2,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  linkCtaRowFirst: { marginTop: 10 },
  linkCtaRowPressed: { opacity: 0.88 },
  fleetCtaText: { color: C.brand, fontSize: 14, fontWeight: "800" },
  /**
   * État vide : même enveloppe que les tuiles KPI (`dashboardSurfaceShadow`)
   * + picto large dans un well vert léger (snapshot operations-app).
   */
  emptyMissionCard: {
    alignSelf: "stretch",
    alignItems: "center",
    backgroundColor: C.cardBg,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 16,
    paddingVertical: 24,
    paddingHorizontal: 20,
    gap: 10,
    ...dashboardSurfaceShadow,
  },
  emptyIconWrap: {
    width: 72,
    height: 72,
    borderRadius: 22,
    backgroundColor: C.kpiIconWell,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.12)",
  },
  emptyTitle: {
    color: C.text,
    textAlign: "center",
    fontSize: 18,
    lineHeight: 22,
    fontWeight: "600",
  },
  emptySubtitle: {
    textAlign: "center",
    maxWidth: 340,
    fontSize: 16,
    lineHeight: 24,
    fontWeight: "400",
    color: C.textMuted,
  },
});
