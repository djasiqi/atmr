import type { MapLatLng } from "./missionMapCoordUtils";

export type DriverMapRegion = {
  latitude: number;
  longitude: number;
  latitudeDelta: number;
  longitudeDelta: number;
};

export type ColdStartCameraAction = "none" | "recenter";

export type ColdStartCameraDecision = {
  action: ColdStartCameraAction;
  consume: true;
};

const VISIBLE_INSET = 0.12;

export function isUsableMapRegion(region: DriverMapRegion | null | undefined): region is DriverMapRegion {
  if (!region) return false;
  const { latitude, longitude, latitudeDelta, longitudeDelta } = region;
  if (![latitude, longitude, latitudeDelta, longitudeDelta].every(Number.isFinite)) return false;
  if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return false;
  if (latitudeDelta <= 0 || longitudeDelta <= 0) return false;
  if (latitudeDelta > 80 || longitudeDelta > 80) return false;
  return true;
}

/** Le chauffeur est déjà visible dans la zone actuelle (avec une marge). */
export function isPointVisibleInRegion(point: MapLatLng, region: DriverMapRegion): boolean {
  if (!isUsableMapRegion(region)) return false;
  const latHalf = (region.latitudeDelta / 2) * (1 - VISIBLE_INSET);
  const lngHalf = (region.longitudeDelta / 2) * (1 - VISIBLE_INSET);
  return (
    Math.abs(point.latitude - region.latitude) <= latHalf &&
    Math.abs(point.longitude - region.longitude) <= lngHalf
  );
}

/**
 * Premier fix GNSS uniquement.
 * Après cet appel, `consume` est toujours true — plus de logique cold-start.
 */
export function resolveColdStartCameraAction(input: {
  consumed: boolean;
  gnssPoint: MapLatLng | null;
  currentRegion: DriverMapRegion | null;
  hadUsefulViewport: boolean;
}): ColdStartCameraDecision | { action: "none"; consume: false } {
  if (input.consumed) {
    return { action: "none", consume: true };
  }
  if (!input.gnssPoint) {
    return { action: "none", consume: false };
  }
  if (!input.hadUsefulViewport || !isUsableMapRegion(input.currentRegion)) {
    return { action: "recenter", consume: true };
  }
  if (isPointVisibleInRegion(input.gnssPoint, input.currentRegion)) {
    return { action: "none", consume: true };
  }
  return { action: "recenter", consume: true };
}
