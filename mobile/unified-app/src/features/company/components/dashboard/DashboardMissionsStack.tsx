import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardCompactMissionRow } from "../../dashboard/companyDashboardViewModel";
import { D } from "../../theme/companyDashboardTokens";
import { M, opsSurface } from "./dashboardMobileTokens";
import { SwipeableMissionRow } from "./SwipeableMissionRow";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  missions: DashboardCompactMissionRow[];
  showEmpty?: boolean;
  onPressMission?: (missionId: number) => void;
  onPressSeeAll?: () => void;
  onSwipeUrgent?: () => void;
  onSwipeAssign?: () => void;
};

export function DashboardMissionsStack({
  missions,
  showEmpty,
  onPressMission,
  onPressSeeAll,
  onSwipeUrgent,
  onSwipeAssign,
}: Props) {
  return (
    <View accessibilityLabel="Prochaines courses">
      <View style={opsSurface.sectionHead}>
        <AppText style={opsSurface.sectionTitle}>Courses</AppText>
        {onPressSeeAll ? (
          <Pressable onPress={onPressSeeAll} hitSlop={8} accessibilityRole="button">
            <AppText style={opsSurface.sectionLink}>Tout voir</AppText>
          </Pressable>
        ) : null}
      </View>

      {showEmpty ? (
        <View style={s.empty} accessibilityRole="text">
          <Ionicons name="car-outline" size={20} color={D.textMuted} />
          <AppText variant="caption" style={s.emptyText}>
            Aucune course planifiée aujourd’hui
          </AppText>
        </View>
      ) : (
        <View style={s.list}>
          {missions.map((m, index) => (
            <SwipeableMissionRow
              key={m.missionId}
              row={m}
              showSeparator={index < missions.length - 1}
              onPress={onPressMission ? () => onPressMission(m.missionId) : undefined}
              onSwipePrimary={onPressMission ? () => onPressMission(m.missionId) : undefined}
              onSwipeSecondary={
                m.status.tone === "delayed"
                  ? onSwipeUrgent
                  : m.status.tone === "assign"
                    ? onSwipeAssign
                    : onPressMission
                      ? () => onPressMission(m.missionId)
                      : undefined
              }
            />
          ))}
        </View>
      )}
    </View>
  );
}

const s = StyleSheet.create({
  list: { paddingBottom: 8 },
  empty: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 16,
    paddingHorizontal: M.padH,
  },
  emptyText: {
    color: D.textMuted,
    fontWeight: "600",
    fontSize: FONT_SIZE.px12,
  },
});
