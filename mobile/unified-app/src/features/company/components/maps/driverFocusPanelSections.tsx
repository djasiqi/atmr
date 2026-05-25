import { Linking, Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import type { CompanyDispatchMission } from "../../api/contracts";
import {
  conciseRouteSegment,
  formatEtaLabel,
  formatMissionScheduleTimeLabel,
} from "../../dashboard/companyDashboardMissionUi";
import type { FleetDriverMapItem } from "./fleetMapTypes";
import { FLEET_MAP_COLORS } from "./mapStatusTheme";
import { resolveDriverDisplayName } from "../../utils/companyDriverMapStatus";

type Props = {
  driver: FleetDriverMapItem;
  upcomingMissions?: CompanyDispatchMission[];
  onMessage?: () => void;
  onRecenter?: () => void;
  onViewMission?: (missionId: number) => void;
};

export function DriverFocusPanelSections({
  driver,
  upcomingMissions = [],
  onMessage,
  onRecenter,
  onViewMission,
}: Props) {
  const { enrichment } = driver;
  const mission = enrichment.linkedMission;
  const nextMission = upcomingMissions.find((m) => m.mission_id !== mission?.mission_id) ?? null;
  const delayMin = enrichment.delayMinutes ?? mission?.assignment_pickup_delay_minutes ?? 0;

  const handleCall = () => {
    const phone = enrichment.phone?.trim();
    if (phone) void Linking.openURL(`tel:${phone}`);
  };

  const handleNavigate = () => {
    const lat = mission?.dropoff_lat ?? mission?.pickup_lat;
    const lon = mission?.dropoff_lon ?? mission?.pickup_lon;
    if (lat != null && lon != null) {
      void Linking.openURL(`https://www.google.com/maps/dir/?api=1&destination=${lat},${lon}`);
    }
  };

  return (
    <View style={s.root}>
      <View style={s.block}>
        <AppText variant="caption" style={s.blockTitle}>
          État live
        </AppText>
        <AppText variant="body" style={s.value}>
          {enrichment.operationalStatus === "on_mission" ? "En route" : enrichment.operationalStatus}
        </AppText>
        {delayMin > 0 ? (
          <AppText variant="caption" style={s.warning}>
            Retard +{delayMin} min
          </AppText>
        ) : null}
      </View>

      {mission ? (
        <View style={s.block}>
          <AppText variant="caption" style={s.blockTitle}>
            Mission active
          </AppText>
          <AppText variant="body" style={s.value}>
            {mission.client_name ?? "Patient"}
          </AppText>
          <AppText variant="caption" style={s.muted}>
            {conciseRouteSegment(mission.pickup_label)} → {conciseRouteSegment(mission.dropoff_label)}
          </AppText>
          <AppText variant="caption" style={s.muted}>
            ETA {formatEtaLabel(mission)}
          </AppText>
        </View>
      ) : null}

      <View style={s.block}>
        <AppText variant="caption" style={s.blockTitle}>
          Véhicule
        </AppText>
        <AppText variant="body" style={s.value}>
          {enrichment.vehicleType ?? "Véhicule"} · {enrichment.licensePlate ?? "—"}
        </AppText>
      </View>

      <View style={s.block}>
        <AppText variant="caption" style={s.blockTitle}>
          Timeline
        </AppText>
        {mission ? (
          <AppText variant="caption" style={s.muted}>
            {formatMissionScheduleTimeLabel(mission.scheduled_at)} · course actuelle
          </AppText>
        ) : null}
        {nextMission ? (
          <AppText variant="caption" style={s.muted}>
            {formatMissionScheduleTimeLabel(nextMission.scheduled_at)} · prochaine ·{" "}
            {conciseRouteSegment(nextMission.pickup_label)}
          </AppText>
        ) : (
          <AppText variant="caption" style={s.muted}>
            Aucune mission suivante planifiée
          </AppText>
        )}
      </View>

      <View style={s.actions}>
        <ActionChip icon="call-outline" label="Appeler" onPress={handleCall} disabled={!enrichment.phone} />
        <ActionChip icon="chatbubble-outline" label="Message" onPress={onMessage} />
        <ActionChip icon="navigate-outline" label="Navigation" onPress={handleNavigate} />
        <ActionChip icon="locate-outline" label="Recentrer" onPress={onRecenter} />
        {mission && onViewMission ? (
          <ActionChip
            icon="open-outline"
            label="Mission"
            onPress={() => onViewMission(mission.mission_id)}
          />
        ) : null}
      </View>

      <AppText variant="caption" style={s.footer}>
        {resolveDriverDisplayName(driver)}
      </AppText>
    </View>
  );
}

function ActionChip({
  icon,
  label,
  onPress,
  disabled,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  onPress?: () => void;
  disabled?: boolean;
}) {
  return (
    <Pressable
      style={[s.chip, disabled && s.chipDisabled]}
      onPress={disabled ? undefined : onPress}
      accessibilityRole="button"
      accessibilityLabel={label}
    >
      <Ionicons name={icon} size={16} color={FLEET_MAP_COLORS.textPrimary} />
      <AppText variant="caption" style={s.chipLabel}>
        {label}
      </AppText>
    </Pressable>
  );
}

const s = StyleSheet.create({
  root: { gap: 10, paddingTop: 8 },
  block: { gap: 2 },
  blockTitle: { color: FLEET_MAP_COLORS.textMuted, textTransform: "uppercase", letterSpacing: 0.4 },
  value: { color: FLEET_MAP_COLORS.textPrimary, fontWeight: "600" },
  muted: { color: FLEET_MAP_COLORS.textMuted },
  warning: { color: "#DC2626", fontWeight: "600" },
  actions: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 4 },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: "rgba(15, 23, 42, 0.06)",
  },
  chipDisabled: { opacity: 0.4 },
  chipLabel: { color: FLEET_MAP_COLORS.textPrimary },
  footer: { color: FLEET_MAP_COLORS.textMuted, marginTop: 4 },
});
