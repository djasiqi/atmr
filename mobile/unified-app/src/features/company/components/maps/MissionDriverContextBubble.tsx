import { Platform, StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { CompanyDispatchMission } from "../../api/contracts";
import {
  conciseRouteSegment,
  resolveMissionUiStatus,
} from "../../dashboard/companyDashboardMissionUi";
import type { FleetMissionOverlay } from "./fleetMapMissionVisual";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  mission: CompanyDispatchMission;
  overlay?: FleetMissionOverlay | null;
  compact?: boolean;
};

/** Résumé mission compact — ancré visuellement au chauffeur (narration opérationnelle). */
export function MissionDriverContextBubble({ mission, overlay, compact = false }: Props) {
  const status = resolveMissionUiStatus(mission);
  const eta = overlay?.etaBadgeLabel ?? overlay?.etaLabel;
  const route = `${conciseRouteSegment(mission.pickup_label, compact ? 22 : 28)} → ${conciseRouteSegment(mission.dropoff_label, compact ? 22 : 28)}`;

  return (
    <View
      style={[s.card, compact && s.cardCompact, overlay?.isSelected && s.cardSelected]}
      pointerEvents="none"
      collapsable={false}
    >
      <View style={[s.statusDot, { backgroundColor: status.barColor }]} />
      <View style={s.body}>
        <AppText variant="caption" style={s.route} numberOfLines={1}>
          {route}
        </AppText>
        <AppText variant="caption" style={s.meta} numberOfLines={1}>
          {status.label}
          {eta ? ` · ${eta}` : ""}
        </AppText>
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  card: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
    backgroundColor: "rgba(255,255,255,0.96)",
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.4)",
    maxWidth: 240,
    ...Platform.select({
      ios: {
        shadowColor: "#0F172A",
        shadowOpacity: 0.14,
        shadowRadius: 6,
        shadowOffset: { width: 0, height: 2 },
      },
      android: { elevation: 5 },
      web: { boxShadow: "0 4px 14px rgba(15, 23, 42, 0.12)" },
    }),
  },
  cardCompact: {
    maxWidth: 200,
    paddingHorizontal: 8,
    paddingVertical: 5,
  },
  cardSelected: {
    borderColor: "rgba(0, 121, 107, 0.55)",
    borderWidth: 1.5,
  },
  statusDot: { width: 7, height: 7, borderRadius: 4 },
  body: { flex: 1, minWidth: 0 },
  route: { color: "#0F172A", fontWeight: "700", fontSize: FONT_SIZE.px11 },
  meta: { color: "#64748B", marginTop: 1, fontSize: FONT_SIZE.px10, fontWeight: "600" },
});
