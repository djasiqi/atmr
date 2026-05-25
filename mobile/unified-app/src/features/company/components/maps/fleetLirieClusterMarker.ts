import { LIRIE_DRIVER_PIN_ASPECT } from "./fleetLirieDriverMarkerAssets";
import type { FleetDriverMapItem } from "./fleetMapTypes";
import type { FleetOperationalStatus } from "./mapStatusTheme";
import {
  LIRIE_CLUSTER_COUNT_BADGE_HEIGHT_PX,
  LIRIE_DRIVER_MARKER_DISPLAY_WIDTH_PX,
} from "./fleetLirieMarkerSizing";

/** Statut le plus urgent parmi les chauffeurs du cluster (icône représentative). */
const CLUSTER_STATUS_PRIORITY: FleetOperationalStatus[] = [
  "incident",
  "delayed",
  "on_mission",
  "break",
  "available",
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

export type FleetClusterCountBadgeLayout = {
  label: string;
  width: number;
  height: number;
};

/** Libellé + dimensions (cercle 1 chiffre, pilule 2+). */
export function resolveFleetClusterCountBadgeLayout(count: number): FleetClusterCountBadgeLayout {
  const label = count > 99 ? "99+" : String(count);
  const height = LIRIE_CLUSTER_COUNT_BADGE_HEIGHT_PX;
  if (label.length <= 1) {
    return { label, width: height, height };
  }
  if (label.length === 2) {
    return { label, width: 28, height };
  }
  return { label, width: 32, height };
}

/** Ancrage : y bas = pastille plus basse sur la carte (proche du corps de l’icône). */
export const FLEET_CLUSTER_COUNT_BADGE_ANCHOR = { x: -0.08, y: 0.68 } as const;

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

/** Dimensions du conteneur Marker (icône PNG + pastille à droite). */
export function resolveFleetClusterMarkerHostLayout(count: number): FleetClusterMarkerHostLayout {
  const badge = resolveFleetClusterCountBadgeLayout(count);
  const iconW = LIRIE_DRIVER_MARKER_DISPLAY_WIDTH_PX;
  const iconH = Math.round(iconW * LIRIE_DRIVER_PIN_ASPECT);
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
