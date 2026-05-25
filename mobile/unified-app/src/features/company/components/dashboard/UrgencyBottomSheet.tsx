import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { EnterpriseBottomSheet } from "../EnterpriseBottomSheet";
import type { DashboardDelayedMissionCard } from "../../dashboard/companyDashboardViewModel";
import { D } from "../../theme/companyDashboardTokens";

type Props = {
  visible: boolean;
  mission: DashboardDelayedMissionCard | null;
  onClose: () => void;
  onViewMission?: () => void;
  onMessage?: () => void;
};

export function UrgencyBottomSheet({
  visible,
  mission,
  onClose,
  onViewMission,
  onMessage,
}: Props) {
  if (!mission) return null;

  return (
    <EnterpriseBottomSheet
      visible={visible}
      onClose={onClose}
      title="Urgence active"
      subtitle={`Retard ${mission.delayMinutes} min`}
    >
      <View style={s.body}>
        <View style={s.alertRow}>
          <Ionicons name="warning" size={22} color={D.danger} />
          <AppText variant="label" style={s.client} numberOfLines={1}>
            {mission.clientName}
          </AppText>
        </View>
        <AppText variant="caption" style={s.route} numberOfLines={2}>
          {mission.routeLabel}
        </AppText>
        <AppText variant="caption" style={s.driver}>
          Chauffeur : {mission.driverLabel}
        </AppText>
        <View style={s.actions}>
          {onMessage ? (
            <Pressable onPress={onMessage} style={s.chip} accessibilityRole="button">
              <Ionicons name="chatbubble-outline" size={16} color={D.text} />
              <AppText variant="caption">Message</AppText>
            </Pressable>
          ) : null}
          {onViewMission ? (
            <Pressable onPress={onViewMission} style={[s.chip, s.primary]} accessibilityRole="button">
              <AppText variant="caption" style={s.primaryText}>
                Voir mission
              </AppText>
              <Ionicons name="arrow-forward" size={16} color="#fff" />
            </Pressable>
          ) : null}
        </View>
      </View>
    </EnterpriseBottomSheet>
  );
}

const s = StyleSheet.create({
  body: { gap: 10, paddingBottom: 8 },
  alertRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  client: { color: D.danger, fontWeight: "800", flex: 1 },
  route: { color: D.textSecondary, lineHeight: 18 },
  driver: { color: D.textMuted },
  actions: { flexDirection: "row", gap: 8, marginTop: 4 },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: D.border,
    backgroundColor: D.cardBg,
  },
  primary: {
    flex: 1,
    justifyContent: "center",
    backgroundColor: D.brand,
    borderColor: D.brand,
  },
  primaryText: { color: "#fff", fontWeight: "700" },
});
