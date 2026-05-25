/**
 * Hauteur carte mission (dp / px logiques) : **150** sur téléphone,
 * plafonds tablette / max calés sur les mêmes proportions qu’avant (180 → 260 → 340).
 */
export const MISSION_MAP_HEIGHT_PHONE = 150;
export const MISSION_MAP_HEIGHT_TABLET_CAP = Math.round((150 * 260) / 180);
export const MISSION_MAP_HEIGHT_MAX = Math.round((150 * 340) / 180);
/** Si aucune hauteur passée par le parent (hors `useMissionLayout`). */
export const MISSION_MAP_FALLBACK_HEIGHT = MISSION_MAP_HEIGHT_PHONE;

export { resolveGoogleMapsNativeApiKey, resolveGoogleMapsWebApiKey } from "../../../../config/googleMapsKeys";

export function toFiniteCoord(input: unknown): number | null {
  return typeof input === "number" && Number.isFinite(input) ? input : null;
}

export type LatLngLite = { latitude: number; longitude: number };

export type RegionLite = {
  latitude: number;
  longitude: number;
  latitudeDelta: number;
  longitudeDelta: number;
};

export function computeMissionRegion(
  pickup: LatLngLite | null,
  dropoff: LatLngLite | null,
  fallback: LatLngLite
): RegionLite {
  if (pickup && dropoff) {
    const minLat = Math.min(pickup.latitude, dropoff.latitude);
    const maxLat = Math.max(pickup.latitude, dropoff.latitude);
    const minLng = Math.min(pickup.longitude, dropoff.longitude);
    const maxLng = Math.max(pickup.longitude, dropoff.longitude);
    const pad = 2.55;
    const minDelta = 0.026;
    return {
      latitude: (minLat + maxLat) / 2,
      longitude: (minLng + maxLng) / 2,
      latitudeDelta: Math.max(minDelta, (maxLat - minLat) * pad || minDelta),
      longitudeDelta: Math.max(minDelta, (maxLng - minLng) * pad || minDelta),
    };
  }
  return {
    latitude: fallback.latitude,
    longitude: fallback.longitude,
    latitudeDelta: 0.045,
    longitudeDelta: 0.045,
  };
}
