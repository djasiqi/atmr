import { Platform, StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { CompanyDispatchMission } from "../../api/contracts";
import { conciseRouteSegment, resolveMissionUiStatus } from "../../dashboard/companyDashboardMissionUi";
import type { FleetMissionOverlay } from "./fleetMapMissionVisual";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  overlay: FleetMissionOverlay;
  mission?: CompanyDispatchMission | null;
};

export function MissionContextSnapshot({ overlay, mission }: Props) {
  const status = mission
    ? resolveMissionUiStatus(mission)
    : {
        label: "Mission active",
        tone: "in_progress" as const,
        barColor: overlay.routeStyle.color,
      };
  const pickupLabel = mission?.pickup_label;
  const dropoffLabel = mission?.dropoff_label;

  const eta = overlay.etaBadgeLabel ?? overlay.etaLabel;
  const route =
    pickupLabel || dropoffLabel
      ? `${conciseRouteSegment(pickupLabel, 28)} → ${conciseRouteSegment(dropoffLabel, 28)}`
      : null;

  return (
    <View style={s.card} pointerEvents="none" accessibilityLabel="Résumé mission active">
      <View style={[s.statusDot, { backgroundColor: status.barColor }]} />
      <View style={s.body}>
        <AppText variant="caption" style={s.title} numberOfLines={1}>
          {status.label}
          {eta ? ` · ${eta}` : ""}
        </AppText>
        {route ? (
          <AppText variant="caption" style={s.route} numberOfLines={1}>
            {route}
          </AppText>
        ) : null}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  card: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.94)",
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.35)",
    maxWidth: 320,
    ...Platform.select({
      ios: {
        shadowColor: "#0F172A",
        shadowOpacity: 0.1,
        shadowRadius: 8,
        shadowOffset: { width: 0, height: 3 },
      },
      android: { elevation: 4 },
      web: { boxShadow: "0 6px 16px rgba(15, 23, 42, 0.1)" },
    }),
  },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  body: { flex: 1, minWidth: 0 },
  title: { color: "#0F172A", fontWeight: "700", fontSize: FONT_SIZE.px12 },
  route: { color: "#64748B", marginTop: 2, fontSize: FONT_SIZE.px11 },
});
