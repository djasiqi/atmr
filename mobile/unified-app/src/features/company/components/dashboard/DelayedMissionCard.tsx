import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardDelayedMissionCard as DelayedCardModel } from "../../dashboard/companyDashboardViewModel";
import { D, dashboardCardShadowWeb } from "../../theme/companyDashboardTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  mission: DelayedCardModel;
  onPress?: () => void;
};

export function DelayedMissionCard({ mission, onPress }: Props) {
  return (
    <Pressable
      onPress={onPress}
      disabled={!onPress}
      style={({ pressed }) => [s.card, pressed && onPress && { opacity: 0.92 }]}
      accessibilityRole={onPress ? "button" : "text"}
      accessibilityLabel={`Mission en retard ${mission.delayMinutes} minutes`}
    >
      <View style={s.header}>
        <View style={s.badge}>
          <AppText variant="caption" style={s.badgeText}>
            Mission en retard
          </AppText>
        </View>
        <AppText variant="sectionTitle" style={s.delay}>
          {mission.delayMinutes} min
        </AppText>
      </View>
      <AppText variant="label" style={s.time}>
        {mission.scheduledTimeLabel}
      </AppText>
      <AppText variant="body" style={s.client} numberOfLines={1}>
        {mission.clientName}
      </AppText>
      <AppText variant="caption" style={s.route} numberOfLines={2}>
        {mission.routeLabel}
      </AppText>
      <View style={s.footerRow}>
        <AppText variant="caption" style={s.driver} numberOfLines={1}>
          Chauffeur : {mission.driverLabel}
        </AppText>
        {onPress ? (
          <Ionicons name="chevron-forward" size={20} color={D.danger} accessibilityElementsHidden />
        ) : null}
      </View>
    </Pressable>
  );
}

const s = StyleSheet.create({
  card: {
    padding: 16,
    borderRadius: D.radiusCard,
    backgroundColor: D.dangerSoft,
    borderWidth: 1,
    borderColor: D.dangerBorder,
    gap: 6,
    ...dashboardCardShadowWeb,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    backgroundColor: "rgba(239, 68, 68, 0.12)",
  },
  badgeText: { color: D.danger, fontWeight: "800", fontSize: FONT_SIZE.px11 },
  delay: { color: D.danger, fontWeight: "800" },
  time: { color: D.textSecondary, fontWeight: "700" },
  client: { color: D.text, fontWeight: "800" },
  route: { color: D.textSecondary, lineHeight: 18 },
  footerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
    marginTop: 2,
  },
  driver: { color: D.textMuted, flex: 1 },
});
