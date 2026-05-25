import { Pressable, StyleSheet, Text, View } from "react-native";
import type { DashboardCompactStat } from "../../dashboard/companyDashboardViewModel";
import { D } from "../../theme/companyDashboardTokens";
import { fleetGlassPanel } from "../maps/fleetMapUiTokens";
import { FLEET_COCKPIT } from "./companyFleetCockpitLayout";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  stats: DashboardCompactStat[];
  top?: number;
  bottom?: number;
  horizontalInset?: number;
  onPressStat?: (key: DashboardCompactStat["key"]) => void;
};

/** Barre KPI unifiée — une seule capsule avec 4 stats. */
export function DashboardFloatingStatusBar({
  stats,
  top,
  bottom,
  horizontalInset = FLEET_COCKPIT.sideGutter,
  onPressStat,
}: Props) {
  const verticalAnchor =
    top != null
      ? ({ top } as const)
      : ({ bottom: bottom ?? 0 } as const);

  return (
    <View
      style={[s.row, verticalAnchor, { left: horizontalInset, right: horizontalInset }]}
      pointerEvents="box-none"
    >
      <View style={[fleetGlassPanel(s.pillGlass), s.pillUnified]}>
        {stats.map((stat, index) => (
          <Pressable
            key={stat.key}
            onPress={() => onPressStat?.(stat.key)}
            disabled={!onPressStat}
            style={({ pressed }) => [s.item, pressed && onPressStat && s.pressed]}
            accessibilityRole={onPressStat ? "button" : "text"}
            accessibilityLabel={`${stat.value} ${stat.label}`}
          >
            {index > 0 ? <View style={s.separator} /> : null}
            <View style={[s.dot, { backgroundColor: stat.accentColor }]} />
            <Text style={s.text} numberOfLines={1}>
              <Text style={[s.value, { color: stat.accentColor }]}>{stat.value}</Text>
              {" "}
              {stat.label.toLowerCase()}
            </Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  row: {
    position: "absolute",
    zIndex: 35,
    flexDirection: "column",
    alignItems: "center",
  },
  pillUnified: {
    width: "100%",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderRadius: 999,
    overflow: "hidden",
    minHeight: 34,
  },
  pillGlass: {
    borderRadius: 999,
    backgroundColor: "rgba(255, 255, 255, 0.82)",
    borderColor: "rgba(255, 255, 255, 0.7)",
  },
  item: {
    flex: 1,
    minWidth: 0,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    paddingHorizontal: 6,
    minHeight: 32,
  },
  separator: {
    width: 1,
    height: 14,
    marginRight: 6,
    backgroundColor: "rgba(148, 163, 184, 0.35)",
  },
  pressed: { opacity: 0.88, transform: [{ scale: 0.98 }] },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  text: {
    color: D.textSecondary,
    fontWeight: "600",
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
  },
  value: {
    fontWeight: "800",
    fontSize: FONT_SIZE.px11,
  },
});
