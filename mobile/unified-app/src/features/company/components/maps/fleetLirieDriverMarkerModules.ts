import type { FleetOperationalStatus } from "./mapStatusTheme";

/**
 * Un seul require() par statut : Metro résout automatiquement
 * driver_lirie_*@2x.png et @3x.png selon PixelRatio (iOS et Android).
 * Ne pas require() explicitement les variantes @2x/@3x ni *mdpi* — le bundler les rejette.
 */
export type LirieDriverMarkerAssetKey =
  | "available"
  | "assigned"
  | "warning"
  | "critical"
  | "offline";

export const LIRIE_STATUS_TO_MARKER_ASSET: Record<FleetOperationalStatus, LirieDriverMarkerAssetKey> =
  {
    available: "available",
    on_mission: "assigned",
    break: "warning",
    delayed: "critical",
    incident: "critical",
    offline: "offline",
  };

const DRIVER_LIRIE_MARKER_ASSETS: Record<LirieDriverMarkerAssetKey, number> = {
  available: require("../../../../../assets/images/markers/driver_lirie_available.png"),
  assigned: require("../../../../../assets/images/markers/driver_lirie_assigned.png"),
  warning: require("../../../../../assets/images/markers/driver_lirie_warning.png"),
  critical: require("../../../../../assets/images/markers/driver_lirie_critical.png"),
  offline: require("../../../../../assets/images/markers/driver_lirie_offline.png"),
};

export function resolveLirieDriverMarkerModule(status: FleetOperationalStatus): number {
  const assetKey = LIRIE_STATUS_TO_MARKER_ASSET[status];
  return DRIVER_LIRIE_MARKER_ASSETS[assetKey];
}
