import type { CompanyDriverLiveLocation } from "../api/contracts";

/** Seuil minimal de déplacement GPS avant publication UI (évite jitter stationnaire). */
export const FLEET_GPS_MIN_MOVE_METERS = 5;

const HEADING_EPSILON_DEG = 3;
const SPEED_EPSILON_KMH = 1;

const EARTH_RADIUS_M = 6_371_000;

export function haversineMeters(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.sqrt(a));
}

function headingDelta(a: number | null | undefined, b: number | null | undefined): number {
  if (a == null && b == null) return 0;
  if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) return Infinity;
  const diff = Math.abs(a - b) % 360;
  return diff > 180 ? 360 - diff : diff;
}

function speedDelta(a: number | null | undefined, b: number | null | undefined): number {
  if (a == null && b == null) return 0;
  if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) return Infinity;
  return Math.abs(a - b);
}

/** Ignore les micro-mouvements GPS sans changement métier visible. */
export function hasMeaningfulDriverLocationChange(
  current: CompanyDriverLiveLocation,
  incoming: CompanyDriverLiveLocation
): boolean {
  if (current.driver_id !== incoming.driver_id) return true;

  if (current.mission_id !== incoming.mission_id) return true;
  if (current.location_status !== incoming.location_status) return true;
  if (current.is_background !== incoming.is_background) return true;
  if (current.accepted_observability_only !== incoming.accepted_observability_only) return true;

  const movedMeters = haversineMeters(
    current.latitude,
    current.longitude,
    incoming.latitude,
    incoming.longitude
  );
  if (movedMeters >= FLEET_GPS_MIN_MOVE_METERS) return true;

  if (headingDelta(current.heading, incoming.heading) >= HEADING_EPSILON_DEG) return true;
  if (speedDelta(current.speed, incoming.speed) >= SPEED_EPSILON_KMH) return true;

  return false;
}

/** Fusionne en conservant la référence existante si aucun changement significatif. */
export function mergeDriverLocationPreserveReference(
  existing: CompanyDriverLiveLocation | undefined,
  incoming: CompanyDriverLiveLocation
): CompanyDriverLiveLocation {
  if (!existing) return incoming;
  if (!hasMeaningfulDriverLocationChange(existing, incoming)) return existing;
  return incoming;
}
