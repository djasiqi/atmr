import { StyleSheet, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { D } from "../theme/driverDashboardTheme";
import { DRIVER_DASHBOARD_MAP_HEIGHT } from "./driverDashboardShell";

type Props = {
  height?: number;
  label?: string;
};

/** Surface carte de même hauteur que le rendu Google (DRIVER-COLD-02). */
export function DashboardMapPlaceholder({
  height = DRIVER_DASHBOARD_MAP_HEIGHT,
  label = "Carte en cours…",
}: Props) {
  return (
    <View style={[s.wrap, { height }]} accessibilityLabel={label}>
      <AppText variant="caption" style={s.label}>
        {label}
      </AppText>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: {
    alignSelf: "stretch",
    borderRadius: 16,
    overflow: "hidden",
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    alignItems: "center",
    justifyContent: "center",
  },
  label: {
    color: D.textMuted,
    fontWeight: "600",
  },
});
