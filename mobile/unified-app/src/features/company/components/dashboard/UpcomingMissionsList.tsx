import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardCompactMissionRow } from "../../dashboard/companyDashboardViewModel";
import { CompactMissionRow } from "./CompactMissionRow";
import { dashboardSharedStyles } from "./dashboardSharedStyles";
import { D } from "../../theme/companyDashboardTokens";

type Props = {
  missions: DashboardCompactMissionRow[];
  onPressMission?: (missionId: number) => void;
  onPressSeeAll?: () => void;
};

export function UpcomingMissionsList({ missions, onPressMission, onPressSeeAll }: Props) {
  if (missions.length === 0) return null;

  return (
    <View style={[dashboardSharedStyles.card, s.card]} accessibilityLabel="Prochaines courses">
      <View style={s.header}>
        <View style={s.titleRow}>
          <View style={s.iconWell}>
            <Ionicons name="time-outline" size={16} color={D.brandDark} />
          </View>
          <AppText variant="sectionTitle" style={dashboardSharedStyles.sectionTitle}>
            Prochaines courses
          </AppText>
        </View>
        {onPressSeeAll ? (
          <Pressable onPress={onPressSeeAll} accessibilityRole="button">
            <AppText variant="label" style={dashboardSharedStyles.sectionLink}>
              Tout voir
            </AppText>
          </Pressable>
        ) : null}
      </View>
      <View>
        {missions.map((m, index) => (
          <CompactMissionRow
            key={m.missionId}
            row={m}
            showSeparator={index < missions.length - 1}
            onPress={onPressMission ? () => onPressMission(m.missionId) : undefined}
            onAction={onPressMission ? () => onPressMission(m.missionId) : undefined}
          />
        ))}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  card: { padding: 16 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 8,
    gap: 8,
  },
  titleRow: { flexDirection: "row", alignItems: "center", gap: 8, flex: 1, minWidth: 0 },
  iconWell: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: D.brandSoft,
    alignItems: "center",
    justifyContent: "center",
  },
});
