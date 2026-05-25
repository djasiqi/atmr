import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { E } from "../../theme/enterpriseOpsTheme";
import { formatRideDistance, formatRideDuration } from "./rideRoutePreviewFormat";

type RideRouteStatsProps = {
  distanceMeters: number | null;
  durationSeconds: number | null;
  routeKind?: string;
};

export function RideRouteStats({
  distanceMeters,
  durationSeconds,
  routeKind = "Le plus rapide",
}: RideRouteStatsProps) {
  return (
    <View style={s.statsRow}>
      <View style={s.statItem}>
        <Ionicons name="trail-sign-outline" size={16} color={E.TEXT_SEC} />
        <View style={s.statTextCol}>
          <AppText variant="caption" style={s.statLabel}>Distance estimée</AppText>
          <AppText variant="caption" style={s.statValue}>{formatRideDistance(distanceMeters)}</AppText>
        </View>
      </View>
      <View style={s.statDivider} />
      <View style={s.statItem}>
        <Ionicons name="time-outline" size={16} color={E.TEXT_SEC} />
        <View style={s.statTextCol}>
          <AppText variant="caption" style={s.statLabel}>Temps estimé</AppText>
          <AppText variant="caption" style={s.statValue}>{formatRideDuration(durationSeconds)}</AppText>
        </View>
      </View>
      <View style={s.statDivider} />
      <View style={s.statItem}>
        <Ionicons name="git-network-outline" size={16} color={E.TEXT_SEC} />
        <View style={s.statTextCol}>
          <AppText variant="caption" style={s.statLabel}>Itinéraire</AppText>
          <AppText variant="caption" style={[s.statValue, s.statValueAccent]}>{routeKind}</AppText>
        </View>
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  statsRow: {
    flexDirection: "row" as const,
    alignItems: "stretch" as const,
    justifyContent: "space-between" as const,
    gap: 7,
    backgroundColor: "#FFFFFF",
    paddingVertical: 8,
    paddingHorizontal: 10,
  },
  statItem: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 6,
    flex: 1,
    minWidth: 0,
  },
  statTextCol: { flexShrink: 1, minWidth: 0 },
  statDivider: {
    width: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(148, 163, 184, 0.32)",
  },
  statLabel: {
    color: E.TEXT_MUTED,
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
  },
  statValue: {
    color: E.TEXT,
    fontWeight: "700" as const,
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    marginTop: 1,
  },
  statValueAccent: {
    color: E.BRAND_DARK,
  },
});
