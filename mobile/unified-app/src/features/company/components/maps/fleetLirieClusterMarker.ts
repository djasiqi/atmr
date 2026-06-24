import { LIRIE_DRIVER_MARKER_DISPLAY_WIDTH_PX } from "./fleetLirieMarkerSizing";
import type { FleetDriverMapItem } from "./fleetMapTypes";
import type { FleetOperationalStatus } from "./mapStatusTheme";
import { resolveClusterMarkerSizePx } from "./fleetLirieMarkerSizing";

/** Statut le plus urgent parmi les chauffeurs du cluster (couleur du disque). */
const CLUSTER_STATUS_PRIORITY: FleetOperationalStatus[] = [
  "emergency",
  "incident",
  "delayed",
  "constrained",
  "busy",
  "assigned",
  "break",
  "available",
  "last_known",
  "offline",
];

export function pickClusterRepresentativeStatus(
  drivers: FleetDriverMapItem[]
): FleetOperationalStatus {
  for (const status of CLUSTER_STATUS_PRIORITY) {
    if (drivers.some((d) => d.enrichment.operationalStatus === status)) {
      return status;
    }
  }
  return "available";
}

export { resolveClusterMarkerSizePx };

export type FleetClusterCountBadgeLayout = {
  label: string;
  width: number;
  height: number;
};

/** @deprecated Clusters utilisent un seul disque web — conservé pour compat tests legacy. */
export function resolveFleetClusterCountBadgeLayout(count: number): FleetClusterCountBadgeLayout {
  const label = count > 99 ? "99+" : String(count);
  const height = 24;
  if (label.length <= 1) {
    return { label, width: height, height };
  }
  if (label.length === 2) {
    return { label, width: 28, height };
  }
  return { label, width: 32, height };
}

/** @deprecated */
export const FLEET_CLUSTER_COUNT_BADGE_ANCHOR = { x: -0.08, y: 0.68 } as const;

/** @deprecated */
export function resolveFleetClusterBadgeFontSize(label: string): number {
  if (label.length >= 3) return 9;
  return 11;
}

export type FleetClusterMarkerHostLayout = FleetClusterCountBadgeLayout & {
  iconW: number;
  iconH: number;
  hostW: number;
  hostH: number;
  fontSize: number;
};

/** @deprecated */
export function resolveFleetClusterMarkerHostLayout(count: number): FleetClusterMarkerHostLayout {
  const badge = resolveFleetClusterCountBadgeLayout(count);
  const iconW = LIRIE_DRIVER_MARKER_DISPLAY_WIDTH_PX;
  const iconH = iconW;
  const hostW = iconW + Math.round(badge.width * 0.72);
  const hostH = iconH;
  return {
    ...badge,
    iconW,
    iconH,
    hostW,
    hostH,
    fontSize: resolveFleetClusterBadgeFontSize(badge.label),
  };
}
