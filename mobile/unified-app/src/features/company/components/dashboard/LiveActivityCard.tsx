import { Pressable, StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardLiveActivityItem } from "../../dashboard/companyDashboardViewModel";
import { dashboardSharedStyles } from "./dashboardSharedStyles";
import { D } from "../../theme/companyDashboardTokens";

type Props = {
  items: DashboardLiveActivityItem[];
  onPressSeeAll?: () => void;
};

export function LiveActivityCard({ items, onPressSeeAll }: Props) {
  return (
    <View style={[dashboardSharedStyles.card, s.card]} accessibilityLabel="État opérationnel récent">
      <View style={s.header}>
        <AppText variant="sectionTitle" style={dashboardSharedStyles.sectionTitle}>
          État opérationnel
        </AppText>
        {onPressSeeAll ? (
          <Pressable onPress={onPressSeeAll} accessibilityRole="button" accessibilityLabel="Voir tout">
            <AppText variant="label" style={dashboardSharedStyles.sectionLink}>
              Voir tout
            </AppText>
          </Pressable>
        ) : null}
      </View>
      <View style={s.list}>
        {items.map((item, index) => (
          <View
            key={item.id}
            style={[s.row, index < items.length - 1 && s.rowSep]}
            accessibilityRole="text"
          >
            <View style={s.body}>
              <AppText
                variant="caption"
                style={[s.message, item.isDelayed && s.messageDanger]}
                numberOfLines={2}
              >
                {item.message}
              </AppText>
              {item.detail ? (
                <AppText variant="caption" style={s.detail} numberOfLines={1}>
                  {item.detail}
                </AppText>
              ) : null}
            </View>
            <AppText variant="caption" style={s.time}>
              {item.timeCaption || item.timeLabel}
            </AppText>
          </View>
        ))}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  card: { padding: 16, gap: 10 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  list: { gap: 0 },
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 10,
    paddingVertical: 8,
  },
  rowSep: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: D.border,
  },
  body: { flex: 1, gap: 2 },
  message: { color: D.text, fontWeight: "600", lineHeight: 18 },
  messageDanger: { color: D.danger },
  detail: { color: D.textMuted, fontWeight: "500", lineHeight: 14 },
  time: { color: D.textMuted, fontWeight: "600", flexShrink: 0, maxWidth: 88, textAlign: "right" },
});
