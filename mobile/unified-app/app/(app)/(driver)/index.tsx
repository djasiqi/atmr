import { useCallback, useEffect, useMemo, useState } from "react";
import { usePerfScreenReady } from "../../../src/core/observability/usePerfScreenReady";
import { InteractionManager, RefreshControl, StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import {
  useDriverAvailability,
  useDriverMissionsListFocusResync,
  useDriverMissionsQuery,
  useDriverTodayMissionsQuery,
  useDriverStatusTransition,
} from "../../../src/features/driver/hooks";
import { getDriverStatusUx } from "../../../src/features/driver/statusDictionary";
import { useSession } from "../../../src/core/sessionProvider";
import type { DriverMission, DriverTransitionStatus } from "../../../src/features/driver/types";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import { DashboardMissionListSkeleton } from "../../../src/features/driver/components/DashboardMissionListSkeleton";
import { ConfirmCompletionModal } from "../../../src/features/driver/components/ConfirmCompletionModal";
import { CancelJustificationModal } from "../../../src/features/driver/components/CancelJustificationModal";
import { ReleaseConfirmationModal } from "../../../src/features/driver/components/ReleaseConfirmationModal";
import { UnavailableConfirmationModal } from "../../../src/features/driver/components/UnavailableConfirmationModal";
import { getMissionClientDisplayName } from "../../../src/features/driver/domain/missionDisplay";
import { DriverStateBanners } from "../../../src/features/driver/components/DriverStateBanners";
import { DashboardMissionMap } from "../../../src/features/driver/components/DashboardMissionMap";
import { DashboardActiveMission } from "../../../src/features/driver/components/DashboardActiveMission";
import { DriverDashboardHeader } from "../../../src/features/driver/components/DriverDashboardHeader";
import { DriverIdleState } from "../../../src/features/driver/components/DriverIdleState";
import { filterNextMissionsOnly } from "../../../src/features/driver/domain/missionGrouping";
import { useMissionLayout } from "../../../src/features/driver/hooks/useMissionLayout";
import { AppText, Screen, useAppViewport } from "../../../src/design/responsive";
import { createShadow } from "../../../src/styles/shadowStyles";
import { DriverUpcomingMissions } from "../../../src/features/driver/components/DriverUpcomingMissions";
import {
  D,
  dashboardCardShadow,
} from "../../../src/features/driver/theme/driverDashboardTheme";

const dashboardSurfaceShadow = createShadow(dashboardCardShadow);

const C = {
  pageBg: D.pageBg,
  cardBg: D.cardBg,
  textMuted: D.textMuted,
  border: D.cardBorder,
  brand: D.brand,
} as const;

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

function getScheduledEpoch(mission: DriverMission): number {
  const raw = (mission.scheduled_time ?? mission.scheduled_at) as unknown;
  if (typeof raw !== "string" || raw.length === 0) return Number.POSITIVE_INFINITY;
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
}

export default function DriverHomeScreen() {
  const router = useRouter();
  const { horizontalPadding } = useAppViewport();
  const missionLayout = useMissionLayout();
  const { status: sessionStatus } = useSession();
  const missionsQuery = useDriverMissionsQuery();
  const todayMissionsQuery = useDriverTodayMissionsQuery();
  usePerfScreenReady(
    "driver.hub",
    "driver.hub.data_ready",
    missionsQuery.isSuccess || missionsQuery.isError
  );
  useDriverMissionsListFocusResync();
  const missions = useMemo(
    () => (Array.isArray(missionsQuery.data) ? (missionsQuery.data as DriverMission[]) : []),
    [missionsQuery.data]
  );
  const todayMissions = useMemo(
    () =>
      Array.isArray(todayMissionsQuery.data)
        ? (todayMissionsQuery.data as DriverMission[])
        : [],
    [todayMissionsQuery.data]
  );
  const activeMission = selectActiveMission(missions);
  const [deferredMinimap, setDeferredMinimap] = useState(false);
  useEffect(() => {
    setDeferredMinimap(false);
    if (!activeMission) return;
    const task = InteractionManager.runAfterInteractions(() => setDeferredMinimap(true));
    return () => task.cancel();
  }, [activeMission]);
  const dashboardMapHeight = useMemo(
    () => Math.min(missionLayout.mapHeight, 142),
    [missionLayout.mapHeight]
  );
  const {
    isAvailable,
    availabilityPending,
    unavailableConfirmOpen,
    requestToggleAvailability,
    confirmUnavailable,
    cancelUnavailableConfirm,
  } = useDriverAvailability();
  const transitionMutation = useDriverStatusTransition();
  const [confirmCompletionOpen, setConfirmCompletionOpen] = useState(false);
  const [cancelMissionOpen, setCancelMissionOpen] = useState(false);
  const [releaseMissionOpen, setReleaseMissionOpen] = useState(false);
  const bootstrapPending =
    sessionStatus !== "ready" ||
    (missionsQuery.isLoading && missionsQuery.data === undefined);

  const [pullRefreshing, setPullRefreshing] = useState(false);
  const onPullRefresh = useCallback(async () => {
    setPullRefreshing(true);
    try {
      await missionsQuery.refetch();
    } finally {
      setPullRefreshing(false);
    }
  }, [missionsQuery]);

  const upcomingMissions = useMemo(() => {
    return missions
      .filter((m) => {
        const ux = getDriverStatusUx(typeof m.status === "string" ? m.status : null);
        if (ux.terminal) return false;
        if (activeMission && m.id === activeMission.id) return false;
        return true;
      })
      .sort((a, b) => getScheduledEpoch(a) - getScheduledEpoch(b))
      .slice(0, 3);
  }, [missions, activeMission]);

  const onOpenMission = (missionId: number) =>
    router.push({
      pathname: "/(app)/(driver)/missions/[missionId]",
      params: { missionId: String(missionId) },
    });

  const onAllMissions = () => router.push("/(app)/(driver)/missions");
  const onChat = () => router.push("/(app)/(driver)/chat");

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
            pageTransition={false}
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
            <DriverDashboardHeader
              isAvailable={isAvailable}
              onToggleAvailability={requestToggleAvailability}
              availabilityPending={availabilityPending}
            />

            <DriverStateBanners />

            {bootstrapPending ? (
              <View style={styles.dashboardSection}>
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
                <DashboardActiveMission
                  mission={activeMission}
                  pending={transitionMutation.isPending}
                  onMissionTransition={onMissionTransitionFromDashboard}
                  onMissionRelease={onMissionReleaseFromDashboard}
                  onOpenDetails={() => onOpenMission(activeMission.id)}
                  onOpenChat={onChat}
                />
                {deferredMinimap ? (
                  <DashboardMissionMap mission={activeMission} height={dashboardMapHeight} />
                ) : (
                  <View style={{ height: dashboardMapHeight }} />
                )}
              </View>
            ) : !bootstrapPending ? (
              <DriverIdleState
                isAvailable={isAvailable}
                todayMissions={todayMissions}
              />
            ) : null}

            {!bootstrapPending ? (
              <DriverUpcomingMissions
                missions={upcomingMissions}
                onOpenMission={onOpenMission}
                onOpenAll={onAllMissions}
              />
            ) : null}
          </Screen>
          <ConfirmCompletionModal
            visible={confirmCompletionOpen}
            missionId={activeMission?.id ?? null}
            clientLabel={activeMission ? getMissionClientDisplayName(activeMission) : null}
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
          <UnavailableConfirmationModal
            visible={unavailableConfirmOpen}
            pending={availabilityPending}
            onCancel={cancelUnavailableConfirm}
            onConfirm={confirmUnavailable}
          />
        </>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    flexGrow: 1,
    paddingTop: 10,
    paddingBottom: 16,
    gap: 10,
  },
  dashboardSection: {
    backgroundColor: C.cardBg,
    borderRadius: D.controlRadius,
    borderWidth: 1,
    borderColor: C.border,
    padding: 16,
    ...dashboardSurfaceShadow,
  },
  sectionHint: {
    marginBottom: 8,
    color: C.textMuted,
  },
  missionActiveSection: {
    alignSelf: "stretch",
    gap: 12,
  },
});
