import { View } from "react-native";
import type { DashboardCompactMissionRow, DashboardCompactStat, DashboardDelayedMissionCard, DashboardLiveActivityItem, DashboardQuickAction } from "../../dashboard/companyDashboardViewModel";
import { DashboardLiveFeed } from "./DashboardLiveFeed";
import { DashboardMissionsStack } from "./DashboardMissionsStack";
import { DashboardQuickStrip } from "./DashboardQuickStrip";
import { DashboardStatusBar } from "./DashboardStatusBar";
import { DashboardStickyAlert, type DashboardAlertLine } from "./DashboardStickyAlert";
import { opsSurface } from "./dashboardMobileTokens";

export type DashboardOpsFeedProps = {
  stats: DashboardCompactStat[];
  quickActions: DashboardQuickAction[];
  liveActivity: DashboardLiveActivityItem[];
  delayedMission: DashboardDelayedMissionCard | null;
  alertLines: DashboardAlertLine[];
  errorText?: string | null;
  upcomingMissions: DashboardCompactMissionRow[];
  showEmptyMissions: boolean;
  onPressStat?: (key: DashboardCompactStat["key"]) => void;
  onQuickAction: (key: DashboardQuickAction["key"]) => void;
  onPressDelayed?: () => void;
  onPressAlerts?: () => void;
  onPressSeeAllActivity?: () => void;
  onPressMission?: (missionId: number) => void;
  onPressSeeAllMissions?: () => void;
  onSwipeUrgent?: () => void;
  onSwipeAssign?: () => void;
};

export function DashboardOpsFeed({
  stats,
  quickActions,
  liveActivity,
  delayedMission,
  alertLines,
  errorText,
  upcomingMissions,
  showEmptyMissions,
  onPressStat,
  onQuickAction,
  onPressDelayed,
  onPressAlerts,
  onPressSeeAllActivity,
  onPressMission,
  onPressSeeAllMissions,
  onSwipeUrgent,
  onSwipeAssign,
}: DashboardOpsFeedProps) {
  const showSticky =
    delayedMission != null || alertLines.length > 0 || Boolean(errorText?.trim());

  return (
    <View style={opsSurface.root} accessibilityLabel="Opérations du jour">
      <DashboardStatusBar stats={stats} onPressStat={onPressStat} />
      <View style={opsSurface.hairline} />
      <DashboardQuickStrip actions={quickActions} onPressAction={onQuickAction} />

      {showSticky ? (
        <>
          <DashboardStickyAlert
            delayedMission={delayedMission}
            alertLines={alertLines}
            errorText={errorText}
            onPressDelayed={onPressDelayed}
            onPressAlerts={onPressAlerts}
          />
          <View style={opsSurface.hairline} />
        </>
      ) : null}

      <DashboardLiveFeed items={liveActivity} onPressSeeAll={onPressSeeAllActivity} />

      {(liveActivity.length > 0 || showEmptyMissions || upcomingMissions.length > 0) && (
        <View style={opsSurface.hairline} />
      )}

      <DashboardMissionsStack
        missions={upcomingMissions}
        showEmpty={showEmptyMissions}
        onPressMission={onPressMission}
        onPressSeeAll={onPressSeeAllMissions}
        onSwipeUrgent={onSwipeUrgent}
        onSwipeAssign={onSwipeAssign}
      />
    </View>
  );
}
