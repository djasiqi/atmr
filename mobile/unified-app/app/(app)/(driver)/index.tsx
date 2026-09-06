import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { useDriverFloatingTabScrollPadding } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import { DashboardMissionSlotSkeleton } from "../../../src/features/driver/components/DashboardMissionSlotSkeleton";
import { DashboardMapPlaceholder } from "../../../src/features/driver/components/DashboardMapPlaceholder";
import {
  DRIVER_DASHBOARD_MAP_HEIGHT,
  DRIVER_DASHBOARD_MISSION_SLOT_MIN,
  DRIVER_DASHBOARD_STATUS_TO_MISSION_GAP,
  resolveDriverDashboardPrimarySlot,
} from "../../../src/features/driver/components/driverDashboardShell";
import { ConfirmCompletionModal } from "../../../src/features/driver/components/ConfirmCompletionModal";
import { CancelJustificationModal } from "../../../src/features/driver/components/CancelJustificationModal";
import { ReleaseConfirmationModal } from "../../../src/features/driver/components/ReleaseConfirmationModal";
import { UnavailableConfirmationModal } from "../../../src/features/driver/components/UnavailableConfirmationModal";
import { getMissionClientDisplayName } from "../../../src/features/driver/domain/missionDisplay";
import { DriverStatusArea } from "../../../src/features/driver/components/DriverHubStatusLine";
import { measureDriverHubWindowEdge } from "../../../src/features/driver/components/driverHubLayoutMeasure";
import { useMissionLiveTrackingGuard } from "../../../src/features/driver/hooks/useMissionLiveTrackingGuard";
import { useTrackingAttentionState } from "../../../src/features/driver/hooks/useTrackingAttentionState";
import { requiresLiveTrackingPermission } from "../../../src/features/driver/services/missionLiveTrackingEligibility";
import { DashboardMissionMap } from "../../../src/features/driver/components/DashboardMissionMap";
import { DashboardActiveMission } from "../../../src/features/driver/components/DashboardActiveMission";
import { DriverDashboardHeader } from "../../../src/features/driver/components/DriverDashboardHeader";
import { DriverIdleState } from "../../../src/features/driver/components/DriverIdleState";
import { filterNextMissionsOnly } from "../../../src/features/driver/domain/missionGrouping";
import { useMissionLayout } from "../../../src/features/driver/hooks/useMissionLayout";
import { AppText, Screen, useAppViewport } from "../../../src/design/responsive";
import { DriverUpcomingMissions } from "../../../src/features/driver/components/DriverUpcomingMissions";
import { D } from "../../../src/features/driver/theme/driverDashboardTheme";

const C = {
  pageBg: D.pageBg,
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
  const scrollPad = useDriverFloatingTabScrollPadding();
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
    () => Math.min(missionLayout.mapHeight, DRIVER_DASHBOARD_MAP_HEIGHT),
    [missionLayout.mapHeight]
  );
  const bootstrapPending =
    sessionStatus !== "ready" ||
    (missionsQuery.isLoading && missionsQuery.data === undefined);
  const primarySlot = resolveDriverDashboardPrimarySlot({
    pending: bootstrapPending,
    hasActiveMission: Boolean(activeMission),
  });
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

  const [pullRefreshing, setPullRefreshing] = useState(false);
  const {
    trackingOnboarded,
    showPedagogicalPanel,
    onReadinessGateReady,
    dismissPedagogicalPanel,
  } = useTrackingAttentionState();
  const liveTrackingGuard = useMissionLiveTrackingGuard();

  const onOpenMission = useCallback(
    (missionId: number) =>
      router.push({
        pathname: "/(app)/(driver)/missions/[missionId]",
        params: { missionId: String(missionId) },
      }),
    [router]
  );

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

  const onAllMissions = () => router.push("/(app)/(driver)/missions");
  const onChat = () => router.push("/(app)/(driver)/chat");

  const runMissionTransition = useCallback(
    (target: DriverTransitionStatus) => {
      if (!activeMission) return;
      transitionMutation.mutate({ missionId: activeMission.id, targetStatus: target });
    },
    [activeMission, transitionMutation]
  );

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
      if (requiresLiveTrackingPermission(target)) {
        liveTrackingGuard.guardTransition({
          missionId: activeMission.id,
          target,
          onProceed: () => runMissionTransition(target),
        });
        return;
      }
      runMissionTransition(target);
    },
    [activeMission, liveTrackingGuard, runMissionTransition]
  );

  const missionSlotRef = useRef<View>(null);

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
            extraScrollBottomPadding={scrollPad}
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
              renderStatusLine={(idleLabel) => (
                <DriverStatusArea
                  idleLabel={idleLabel}
                  trackingNeedsAttention={showPedagogicalPanel}
                  trackingOnboarded={trackingOnboarded}
                  onTrackingReadyChange={onReadinessGateReady}
                  onDismissTracking={dismissPedagogicalPanel}
                />
              )}
            />

            {missionsQuery.isError ? (
              <AppText variant="error">
                Erreur chargement missions : {(missionsQuery.error as Error)?.message ?? "Erreur"}
              </AppText>
            ) : null}

            <View
              ref={missionSlotRef}
              style={styles.missionSlot}
              onLayout={() =>
                measureDriverHubWindowEdge(missionSlotRef.current, "missionTop")
              }
            >
              {primarySlot === "pending" ? (
                <DashboardMissionSlotSkeleton />
              ) : primarySlot === "mission" && activeMission ? (
                <DashboardActiveMission
                  mission={activeMission}
                  pending={transitionMutation.isPending}
                  onMissionTransition={onMissionTransitionFromDashboard}
                  onMissionRelease={onMissionReleaseFromDashboard}
                  onOpenDetails={() => onOpenMission(activeMission.id)}
                  onOpenChat={onChat}
                />
              ) : (
                <DriverIdleState
                  isAvailable={isAvailable}
                  todayMissions={todayMissions}
                />
              )}
            </View>

            <View style={[styles.mapSlot, { height: dashboardMapHeight }]}>
              {primarySlot === "mission" && activeMission && deferredMinimap ? (
                <DashboardMissionMap mission={activeMission} height={dashboardMapHeight} />
              ) : (
                <DashboardMapPlaceholder
                  height={dashboardMapHeight}
                  label={
                    primarySlot === "mission"
                      ? "Carte en cours…"
                      : "Localisation en cours…"
                  }
                />
              )}
            </View>

            {!bootstrapPending ? (
              <View style={styles.upcomingSlot}>
                <DriverUpcomingMissions
                  missions={upcomingMissions}
                  onOpenMission={onOpenMission}
                  onOpenAll={onAllMissions}
                />
              </View>
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
  },
  missionSlot: {
    alignSelf: "stretch",
    marginTop: DRIVER_DASHBOARD_STATUS_TO_MISSION_GAP,
    minHeight: DRIVER_DASHBOARD_MISSION_SLOT_MIN,
  },
  mapSlot: {
    alignSelf: "stretch",
    marginTop: 10,
  },
  upcomingSlot: {
    alignSelf: "stretch",
    marginTop: 10,
  },
});
