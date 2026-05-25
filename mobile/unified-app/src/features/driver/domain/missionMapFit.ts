import type { MapLatLng } from "./missionMapCoordUtils";
import type { MissionMapRouteMode } from "./resolveMissionMapRoute";

export type MapEdgePadding = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

export type MissionMapFitRegion = {
  latitude: number;
  longitude: number;
  latitudeDelta: number;
  longitudeDelta: number;
};

/** Points à inclure dans le cadrage selon le statut / itinéraire actif. */
export function resolveMissionMapFitPoints(options: {
  mode: MissionMapRouteMode;
  driverCoord: MapLatLng | null;
  pickupCoord: MapLatLng | null;
  dropoffCoord: MapLatLng | null;
  routeCoordinates?: MapLatLng[] | null;
  /** Utiliser tous les points du tracé pour le bbox (recentrage). */
  useFullRoute?: boolean;
}): MapLatLng[] {
  const { mode, driverCoord, pickupCoord, dropoffCoord } = options;
  const routeCoords = options.routeCoordinates ?? [];
  const routeSample = options.useFullRoute
    ? routeCoords
    : sampleRouteCoordinates(routeCoords);

  let anchors: MapLatLng[] = [];

  switch (mode) {
    case "live_to_pickup":
      anchors = [driverCoord, pickupCoord].filter((p): p is MapLatLng => Boolean(p));
      break;
    case "live_to_dropoff":
      anchors = [driverCoord, dropoffCoord].filter((p): p is MapLatLng => Boolean(p));
      break;
    case "planned_full":
      anchors = [pickupCoord, dropoffCoord].filter((p): p is MapLatLng => Boolean(p));
      break;
    default:
      anchors = [pickupCoord ?? dropoffCoord ?? driverCoord].filter((p): p is MapLatLng =>
        Boolean(p)
      );
      break;
  }

  if (anchors.length === 0 && routeSample.length > 0) {
    return dedupeCoordinates(routeSample);
  }

  return dedupeCoordinates([...anchors, ...routeSample]);
}

/**
 * Région de cadrage avec marge autour du trajet.
 * Évite le zoom excessif sur les courtes distances (~400 m).
 */
export function computeMissionMapFitRegion(
  points: MapLatLng[],
  options?: {
    paddingFactor?: number;
    minLatitudeDelta?: number;
    minLongitudeDelta?: number;
  }
): MissionMapFitRegion | null {
  if (points.length === 0) return null;

  const paddingFactor = options?.paddingFactor ?? 2.55;
  const minLatDelta = options?.minLatitudeDelta ?? 0.026;
  const minLngDelta = options?.minLongitudeDelta ?? 0.026;

  const minLat = Math.min(...points.map((p) => p.latitude));
  const maxLat = Math.max(...points.map((p) => p.latitude));
  const minLng = Math.min(...points.map((p) => p.longitude));
  const maxLng = Math.max(...points.map((p) => p.longitude));

  const rawLatSpan = maxLat - minLat;
  const rawLngSpan = maxLng - minLng;

  const latitudeDelta = Math.max(minLatDelta, rawLatSpan * paddingFactor || minLatDelta);
  const longitudeDelta = Math.max(minLngDelta, rawLngSpan * paddingFactor || minLngDelta);

  return {
    latitude: (minLat + maxLat) / 2,
    longitude: (minLng + maxLng) / 2,
    latitudeDelta,
    longitudeDelta,
  };
}

/** Compense le badge (haut-gauche) et le bouton recentrage (bas-droit). */
export function applyMissionMapFitCenterBias(
  region: MissionMapFitRegion,
  padding: MapEdgePadding,
  mapHeight: number
): MissionMapFitRegion {
  const h = Math.max(120, mapHeight);
  const verticalPadDelta = (padding.top - padding.bottom) / h;
  const horizontalPadDelta = (padding.left - padding.right) / h;

  return {
    ...region,
    latitude: region.latitude - region.latitudeDelta * verticalPadDelta * 0.38,
    longitude: region.longitude + region.longitudeDelta * horizontalPadDelta * 0.28,
  };
}

/** Padding UI pour le calcul du biais de centrage. */
export function computeMissionMapFitEdgePadding(
  mapHeight: number,
  options?: { showRecenterControl?: boolean; reserveBadge?: boolean }
): MapEdgePadding {
  const reserveBadge = options?.reserveBadge ?? true;
  const showRecenter = options?.showRecenterControl ?? false;
  const h = Math.max(120, mapHeight);
  const sideInset = Math.round(h * 0.14);

  return {
    top: reserveBadge ? Math.min(58, Math.round(h * 0.38)) : sideInset,
    left: reserveBadge ? Math.max(18, sideInset) : sideInset,
    bottom: showRecenter ? Math.min(50, Math.round(h * 0.32)) : sideInset,
    right: showRecenter ? Math.max(50, sideInset) : sideInset,
  };
}

function dedupeCoordinates(points: MapLatLng[]): MapLatLng[] {
  const seen = new Set<string>();
  const out: MapLatLng[] = [];
  for (const p of points) {
    const key = `${p.latitude.toFixed(5)},${p.longitude.toFixed(5)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(p);
  }
  return out;
}

function sampleRouteCoordinates(coords: MapLatLng[], maxPoints = 14): MapLatLng[] {
  if (coords.length === 0) return [];
  if (coords.length <= maxPoints) return coords;
  const step = Math.max(1, Math.floor(coords.length / maxPoints));
  const out: MapLatLng[] = [];
  for (let i = 0; i < coords.length; i += step) {
    out.push(coords[i]);
  }
  const last = coords[coords.length - 1];
  const tail = out[out.length - 1];
  if (!tail || tail.latitude !== last.latitude || tail.longitude !== last.longitude) {
    out.push(last);
  }
  return out;
}
