import { Platform, StyleSheet, Text, View } from "react-native";

import type { CompanyDriverLiveLocation } from "../../api/contracts";
import { driverFleetMarkerInitials, resolveFleetMarkerInitialsFromDisplayName } from "../../utils/companyDriverMapStatus";
import {
  FLEET_STATUS_THEME,
  type FleetMarkerVariant,
  type FleetOperationalStatus,
} from "./mapStatusTheme";
import { FLEET_DRIVER_MARKER_WEB_BASE_PX, FLEET_NATIVE_DRIVER_MARKER_SIZE_PX, LIRIE_ANDROID_DRIVER_MARKER_VIEW_PX } from "./fleetLirieMarkerSizing";

export const FLEET_DRIVER_MARKER_BOX_PX = FLEET_NATIVE_DRIVER_MARKER_SIZE_PX;
export const FLEET_DRIVER_MARKER_BOX_SELECTED_PX = FLEET_NATIVE_DRIVER_MARKER_SIZE_PX;

export function resolveFleetDriverMarkerBoxPx(_selected = false): number {
  return FLEET_NATIVE_DRIVER_MARKER_SIZE_PX;
}

type Props = {
  status: FleetOperationalStatus;
  selected?: boolean;
  dimmed?: boolean;
  driver?: CompanyDriverLiveLocation;
  driverName?: string;
  /** Android Marker enfant — taille réduite sans ombre (évite rognage carte). */
  compactForMap?: boolean;
};

/** Marqueur RN — cercle + initiales (typo système, parité web). */
export function FleetDriverMarkerVisual({
  status,
  selected = false,
  dimmed = false,
  driver,
  driverName = "CH",
  compactForMap = false,
}: Props) {
  const theme = FLEET_STATUS_THEME[status];
  const useCompact = compactForMap && Platform.OS === "android";
  const size = useCompact ? LIRIE_ANDROID_DRIVER_MARKER_VIEW_PX : FLEET_DRIVER_MARKER_BOX_PX;
  const borderWidth = useCompact ? 2 : 2.5;
  const initials = driver
    ? driverFleetMarkerInitials(driver)
    : resolveFleetMarkerInitialsFromDisplayName(driverName);
  const fontSize = Math.round(7.4 * (size / FLEET_DRIVER_MARKER_WEB_BASE_PX));

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
          useCompact ? s.discCompact : s.disc,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: theme.fill,
            borderWidth,
            borderColor: "#ffffff",
          },
          selected && s.discSelected,
        ]}
      >
        <Text style={[s.initials, { fontSize }]}>{initials}</Text>
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
  discCompact: {
    alignItems: "center",
    justifyContent: "center",
  },
  discSelected: {
    opacity: 1,
  },
  initials: {
    color: "#ffffff",
    fontWeight: "700",
    letterSpacing: 0.2,
    fontFamily: Platform.select({
      android: "sans-serif-medium",
      ios: "System",
      default: "System",
    }),
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
