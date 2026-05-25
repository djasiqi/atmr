import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import { createShadow } from "../../../styles/shadowStyles";
import { D, dashboardSoftShadow } from "../theme/driverDashboardTheme";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

type Props = {
  prefix: string;
  distanceLabel: string;
  durationLabel: string;
  live?: boolean;
};

const badgeShadow = createShadow(dashboardSoftShadow);

export function MissionMapLiveBadge({ prefix, distanceLabel, durationLabel, live = false }: Props) {
  const hasMetrics = distanceLabel !== "—" || durationLabel !== "—";
  if (!hasMetrics) return null;

  return (
    <View style={styles.wrap} pointerEvents="none" accessibilityLabel={`${prefix} ${distanceLabel} ${durationLabel}`}>
      <View style={styles.badge}>
        {live ? (
          <View style={styles.liveDot} accessibilityElementsHidden />
        ) : (
          <Ionicons name="navigate-outline" size={11} color={D.brand} accessibilityElementsHidden />
        )}
        <View style={styles.textCol}>
          <AppText variant="caption" style={styles.prefix} numberOfLines={1}>
            {prefix}
          </AppText>
          <AppText variant="label" style={styles.metrics} numberOfLines={1}>
            {[distanceLabel, durationLabel].filter((v) => v !== "—").join(" · ")}
          </AppText>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: "absolute",
    left: 10,
    top: 10,
    zIndex: 4,
    maxWidth: "72%",
  },
  badge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(255,255,255,0.94)",
    borderWidth: 1,
    borderColor: D.cardBorder,
    borderRadius: 10,
    paddingVertical: 6,
    paddingHorizontal: 8,
    ...badgeShadow,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#22C55E",
  },
  textCol: {
    flexShrink: 1,
    gap: 1,
  },
  prefix: {
    color: D.textMuted,
    fontWeight: "700",
    fontSize: FONT_SIZE.px8,
    letterSpacing: 0.2,
    textTransform: "uppercase",
  },
  metrics: {
    color: D.text,
    fontWeight: "800",
    fontSize: FONT_SIZE.px12,
    lineHeight: 14,
  },
});
