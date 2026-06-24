import { Platform, StyleSheet, Text, View } from "react-native";

import { resolveDriverMarkerInitials } from "./fleetMarkerIcons";
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

type Props = {
  status: FleetOperationalStatus;
  selected?: boolean;
  dimmed?: boolean;
  driverName?: string;
};

/** Fallback RN — cercle + initiales (parité web). */
export function FleetDriverMarkerVisual({
  status,
  selected = false,
  dimmed = false,
  driverName = "CH",
}: Props) {
  const theme = FLEET_STATUS_THEME[status];
  const size = FLEET_DRIVER_MARKER_BOX_PX;
  const initials = resolveDriverMarkerInitials(driverName);

  return (
    <View
      style={[
        s.box,
        { width: size, height: size },
        dimmed && s.dimmed,
      ]}
      pointerEvents="none"
      collapsable={false}
    >
      <View
        style={[
          s.disc,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: theme.fill,
            borderWidth: 2.5,
            borderColor: "#ffffff",
          },
          selected && s.discSelected,
        ]}
      >
        <Text style={s.initials}>{initials}</Text>
      </View>
    </View>
  );
}

export function FleetLegendMarkerSwatch({
  color,
  variant: _variant,
}: {
  color: string;
  variant?: FleetMarkerVariant;
}) {
  return (
    <View style={s.legendBox}>
      <View style={[s.legendDisc, { backgroundColor: color, borderWidth: 1.5, borderColor: "#fff" }]}>
        <Text style={s.legendInitials}>•</Text>
      </View>
    </View>
  );
}

const discShadow = Platform.select({
  ios: {
    shadowColor: "#000",
    shadowOpacity: 0.2,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },
  android: { elevation: 3 },
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
  disc: {
    alignItems: "center",
    justifyContent: "center",
    ...discShadow,
  },
  discSelected: {
    opacity: 1,
  },
  initials: {
    color: "#ffffff",
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0.2,
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
  legendInitials: {
    color: "#fff",
    fontSize: 8,
    fontWeight: "700",
    lineHeight: 10,
  },
});
