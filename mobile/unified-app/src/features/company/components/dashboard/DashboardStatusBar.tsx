import { Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import type { DashboardCompactStat } from "../../dashboard/companyDashboardViewModel";
import { D } from "../../theme/companyDashboardTokens";
import { M } from "./dashboardMobileTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  stats: DashboardCompactStat[];
  onPressStat?: (key: DashboardCompactStat["key"]) => void;
};

export function DashboardStatusBar({ stats, onPressStat }: Props) {
  const content = (
    <View style={s.row} accessibilityRole="summary">
      {stats.map((stat, index) => (
        <View key={stat.key} style={s.segment}>
          {index > 0 ? <View style={s.dotSep} /> : null}
          <View style={[s.statusDot, { backgroundColor: stat.accentColor }]} />
          <Text style={s.segmentText} numberOfLines={1}>
            <Text style={[s.value, { color: stat.accentColor }]}>{stat.value} </Text>
            {stat.label.toLowerCase()}
          </Text>
        </View>
      ))}
    </View>
  );

  if (!onPressStat) {
    return <View style={s.wrap}>{content}</View>;
  }

  return (
    <Pressable
      onPress={() => onPressStat("delayed")}
      style={({ pressed }) => [s.wrap, pressed && s.pressed]}
      accessibilityRole="button"
      accessibilityLabel="Indicateurs opérationnels, appuyer pour les retards"
    >
      {content}
      <Ionicons name="chevron-forward" size={16} color={D.textMuted} />
    </Pressable>
  );
}

const s = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: M.padH,
    paddingVertical: 10,
    gap: 6,
  },
  pressed: { opacity: 0.88 },
  row: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 4,
    minWidth: 0,
  },
  segment: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    maxWidth: "100%",
  },
  dotSep: {
    width: 3,
    height: 3,
    borderRadius: 2,
    backgroundColor: D.textMuted,
    marginHorizontal: 2,
    opacity: 0.5,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  segmentText: {
    color: D.textSecondary,
    fontWeight: "600",
    fontSize: FONT_SIZE.px12,
    lineHeight: 15,
  },
  value: {
    fontWeight: "800",
    fontSize: FONT_SIZE.px12,
  },
});
