import { Pressable, ScrollView, StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardCompactStat } from "../../dashboard/companyDashboardViewModel";
import { dashboardSharedStyles } from "./dashboardSharedStyles";
import { D } from "../../theme/companyDashboardTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  stats: DashboardCompactStat[];
  onPressStat?: (key: DashboardCompactStat["key"]) => void;
};

export function StatusStatsCarousel({ stats, onPressStat }: Props) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={s.content}
      accessibilityLabel="Indicateurs opérationnels"
    >
      {stats.map((stat) => {
        const inner = (
          <>
            <AppText variant="sectionTitle" style={[s.value, { color: stat.accentColor }]}>
              {stat.value}
            </AppText>
            <AppText variant="caption" style={s.label} numberOfLines={1}>
              {stat.label}
            </AppText>
          </>
        );
        if (onPressStat) {
          return (
            <Pressable
              key={stat.key}
              onPress={() => onPressStat(stat.key)}
              style={({ pressed }) => [dashboardSharedStyles.card, s.tile, pressed && s.pressed]}
              accessibilityRole="button"
              accessibilityLabel={`${stat.label} ${stat.value}`}
            >
              {inner}
            </Pressable>
          );
        }
        return (
          <View key={stat.key} style={[dashboardSharedStyles.card, s.tile]} accessibilityLabel={`${stat.label} ${stat.value}`}>
            {inner}
          </View>
        );
      })}
    </ScrollView>
  );
}

const s = StyleSheet.create({
  content: { gap: 10, paddingVertical: 2 },
  tile: {
    minWidth: 108,
    paddingVertical: 14,
    paddingHorizontal: 14,
    gap: 4,
  },
  pressed: { opacity: 0.9 },
  value: { fontWeight: "800", fontSize: FONT_SIZE.px22 },
  label: { color: D.textSecondary, fontWeight: "700" },
});
