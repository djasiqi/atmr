import { useCallback, useMemo } from "react";
import { StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { useCompanyDashboardScreenModel } from "../../../src/features/company/dashboard/useCompanyDashboardScreenModel";
import { CompanyFleetCockpit } from "../../../src/features/company/components/dashboard/CompanyFleetCockpit";
import { DayPickerSheet } from "../../../src/features/company/components/DayPickerSheet";
import { DispatchModeSheet } from "../../../src/features/company/components/DispatchModeSheet";
import type {
  DashboardCompactStat,
  DashboardQuickAction,
} from "../../../src/features/company/dashboard/companyDashboardViewModel";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";

export default function CompanyDashboardScreen() {
  const router = useRouter();
  const model = useCompanyDashboardScreenModel();
  const { uiModel } = model;

  const openRides = useCallback(
    (params?: Record<string, string>) => {
      router.push({
        pathname: "/(app)/(company)/rides",
        params,
      } as Parameters<typeof router.push>[0]);
    },
    [router]
  );

  const openMission = useCallback(
    (missionId: number) => {
      router.push({
        pathname: "/(app)/(company)/ride-details",
        params: { rideId: String(missionId) },
      } as Parameters<typeof router.push>[0]);
    },
    [router]
  );

  const onQuickAction = useCallback(
    (key: DashboardQuickAction["key"]) => {
      if (key === "create") openRides({ create: "1" });
      else if (key === "assign") openRides();
      else if (key === "urgent") openRides({ filter: "delayed" });
      else openRides();
    },
    [openRides]
  );

  const onStatPress = useCallback(
    (key: DashboardCompactStat["key"]) => {
      if (key === "delayed") openRides({ filter: "delayed" });
      else if (key === "in_progress") openRides({ status: "in_progress" });
      else openRides();
    },
    [openRides]
  );

  const alertLines = useMemo(
    () => uiModel.stickyAlertLines,
    [uiModel.stickyAlertLines]
  );

  const errorText =
    model.error && !model.isLikelyNetworkError && !model.isAuthFailure
      ? model.displayErrMsg
      : null;

  return (
    <PermissionGuard permission="company:dashboard:read">
      <View style={styles.root}>
        <CompanyFleetCockpit
          drivers={model.liveDrivers.drivers}
          missions={model.missions}
          date={model.selectedDate}
          headerMode={model.headerMode}
          realtimeStatus={model.realtime.transportStatus}
          realtimeDataFreshness={model.realtime.dataFreshness}
          realtimeLastEventAt={model.realtime.lastEventAt}
          canDispatchManage={model.canDispatchManage}
          refreshing={model.loading}
          onRefresh={() => void model.refreshAll()}
          onOpenDatePicker={() => model.setDateSheetOpen(true)}
          onOpenModePicker={model.canDispatchManage ? () => model.setModeSheetOpen(true) : undefined}
          onViewMission={openMission}
          onMessage={() =>
            router.push("/(app)/(company)/chat" as Parameters<typeof router.push>[0])
          }
          opsFeed={{
            stats: uiModel.compactStats,
            quickActions: uiModel.quickActions,
            liveActivity: uiModel.liveActivity,
            delayedMission: uiModel.delayedMission,
            alertLines,
            errorText,
            upcomingMissions: uiModel.upcomingMissions,
            showEmptyMissions: uiModel.showEmptyMissions,
            onPressStat: onStatPress,
            onQuickAction,
            onPressDelayed: uiModel.delayedMission
              ? () => openMission(uiModel.delayedMission!.missionId)
              : undefined,
            onPressAlerts: () => openRides({ filter: "delayed" }),
            onPressSeeAllActivity: () => openRides(),
            onPressMission: openMission,
            onPressSeeAllMissions: () => openRides(),
            onSwipeUrgent: () => openRides({ filter: "delayed" }),
            onSwipeAssign: () => openRides(),
          }}
        />
      </View>

      <DayPickerSheet
        visible={model.dateSheetOpen}
        selectedDate={model.selectedDate}
        onClose={() => model.setDateSheetOpen(false)}
        onSelectDate={(iso) => {
          model.setSelectedDate(iso);
          model.setDateSheetOpen(false);
        }}
      />
      <DispatchModeSheet
        visible={model.modeSheetOpen}
        mode={model.headerMode}
        onClose={() => model.setModeSheetOpen(false)}
        onSelectMode={(mode) => void model.applyDispatchModeFromSheet(mode)}
        switchingEnabled={model.canDispatchManage}
      />
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: E.BG,
  },
});
