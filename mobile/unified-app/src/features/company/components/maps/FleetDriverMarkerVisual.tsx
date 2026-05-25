import { Platform, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";

import {
  FLEET_STATUS_THEME,
  type FleetMarkerVariant,
  type FleetOperationalStatus,
} from "./mapStatusTheme";
import { FLEET_NATIVE_DRIVER_MARKER_SIZE_PX } from "./fleetLirieMarkerSizing";

export const FLEET_DRIVER_MARKER_BOX_PX = FLEET_NATIVE_DRIVER_MARKER_SIZE_PX;
export const FLEET_DRIVER_MARKER_BOX_SELECTED_PX = FLEET_NATIVE_DRIVER_MARKER_SIZE_PX;

export function resolveFleetDriverMarkerBoxPx(_selected = false): number {
  return FLEET_NATIVE_DRIVER_MARKER_SIZE_PX;
}

const DISC_PX = 30;

type Props = {
  status: FleetOperationalStatus;
  selected?: boolean;
  dimmed?: boolean;
};

function resolveDiscIcon(
  status: FleetOperationalStatus,
  variant: FleetMarkerVariant
): keyof typeof Ionicons.glyphMap {
  if (variant === "incident_triangle" || status === "incident") return "warning";
  if (status === "break") return "pause";
  return "car";
}

function glowColor(fill: string, strong = false): string {
  if (fill.startsWith("#") && fill.length === 7) {
    const r = parseInt(fill.slice(1, 3), 16);
    const g = parseInt(fill.slice(3, 5), 16);
    const b = parseInt(fill.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${strong ? 0.34 : 0.2})`;
  }
  return strong ? "rgba(0, 121, 107, 0.34)" : "rgba(0, 121, 107, 0.2)";
}

/** Fallback RN (Android) — glow discret, disque, icône, ombre douce. */
export function FleetDriverMarkerVisual({ status, selected = false, dimmed = false }: Props) {
  const theme = FLEET_STATUS_THEME[status];
  const variant = theme.markerVariant;
  const iconName = resolveDiscIcon(status, variant);
  const iconSize = status === "incident" ? 14 : 13;
  const showAlertDot = variant === "vehicle_alert" && status !== "incident";
  const liveGlow = theme.pulse === true;

  return (
    <View
      style={[
        s.box,
        { width: FLEET_DRIVER_MARKER_BOX_PX, height: FLEET_DRIVER_MARKER_BOX_PX },
        dimmed && s.dimmed,
      ]}
      pointerEvents="none"
      collapsable={false}
    >
      <View
        style={[
          s.glow,
          {
            width: DISC_PX + (liveGlow ? 8 : 4),
            height: DISC_PX + (liveGlow ? 8 : 4),
            borderRadius: (DISC_PX + (liveGlow ? 8 : 4)) / 2,
            backgroundColor: glowColor(theme.fill, liveGlow),
          },
          liveGlow && status === "on_mission" && s.glowMission,
        ]}
      />

      <View
        style={[
          s.disc,
          {
            width: DISC_PX,
            height: DISC_PX,
            borderRadius: DISC_PX / 2,
            backgroundColor: theme.fill,
          },
          selected && s.discSelected,
        ]}
      >
        <Ionicons name={iconName} size={iconSize} color="#FFFFFF" />
        {showAlertDot ? (
          <View style={[s.alertDot, { backgroundColor: theme.fill }]}>
            <Ionicons name="warning" size={7} color="#FFFFFF" />
          </View>
        ) : null}
      </View>
    </View>
  );
}

export function FleetLegendMarkerSwatch({
  color,
  variant,
}: {
  color: string;
  variant?: FleetMarkerVariant;
}) {
  const icon =
    variant === "incident_triangle" ? ("warning" as const) : ("car" as const);

  return (
    <View style={s.legendBox}>
      <View style={[s.legendDisc, { backgroundColor: color }]}>
        <Ionicons name={icon} size={8} color="#fff" />
      </View>
      {variant === "vehicle_alert" ? (
        <View style={[s.legendAlertDot, { backgroundColor: color }]}>
          <Ionicons name="warning" size={6} color="#fff" />
        </View>
      ) : null}
    </View>
  );
}

const discShadow = Platform.select({
  ios: {
    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 3 },
  },
  android: { elevation: 4 },
  default: {},
});

const s = StyleSheet.create({
  box: {
    alignItems: "center",
    justifyContent: "center",
    overflow: "visible",
  },
  dimmed: {
    opacity: 0.45,
  },
  glow: {
    position: "absolute",
  },
  glowMission: {
    borderWidth: 1.5,
    borderColor: "rgba(59, 130, 246, 0.35)",
  },
  disc: {
    alignItems: "center",
    justifyContent: "center",
    ...discShadow,
  },
  discSelected: {
    transform: [{ scale: 1.12 }],
  },
  alertDot: {
    position: "absolute",
    top: 0,
    right: 0,
    width: 11,
    height: 11,
    borderRadius: 6,
    alignItems: "center",
    justifyContent: "center",
  },
  legendBox: {
    width: 18,
    height: 18,
    alignItems: "center",
    justifyContent: "center",
  },
  legendDisc: {
    width: 14,
    height: 14,
    borderRadius: 7,
    alignItems: "center",
    justifyContent: "center",
  },
  legendAlertDot: {
    position: "absolute",
    top: 0,
    right: 0,
    width: 8,
    height: 8,
    borderRadius: 4,
    alignItems: "center",
    justifyContent: "center",
  },
});
