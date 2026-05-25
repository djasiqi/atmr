import { resolveDriverStatusForUx } from "../statusDictionary";
import type { DriverMission, DriverMissionStatus } from "../types";

const FALLBACK_SPEED_KMH = 32;

export type MissionRouteLeg =
  | { mode: "planned" }
  | { mode: "live"; destination: "pickup" | "dropoff" };

function firstPositiveNumber(...values: unknown[]): number | null {
  for (const candidate of values) {
    const value = typeof candidate === "number" ? candidate : Number(candidate);
    if (Number.isFinite(value) && value > 0) return value;
  }
  return null;
}

export type MissionCoord = { lat: number; lng: number };

export type MissionRouteMetrics = {
  distanceKm: number | null;
  durationMinutes: number | null;
};

/** Segment affiché selon le statut mission (planifié vs position chauffeur en direct). */
export function resolveLiveDriverOrigin(sources: {
  tracking: MissionCoord | null;
  etaDriver?: { lat: number | null; lon: number | null } | null;
  mission: MissionCoord | null;
}): MissionCoord | null {
  if (sources.tracking) return sources.tracking;
  const etaLat = sources.etaDriver?.lat;
  const etaLon = sources.etaDriver?.lon;
  if (etaLat != null && etaLon != null && Number.isFinite(etaLat) && Number.isFinite(etaLon)) {
    return { lat: etaLat, lng: etaLon };
  }
  return sources.mission;
}

export function mapLatLngToMissionCoord(point: { latitude: number; longitude: number } | null): MissionCoord | null {
  if (!point) return null;
  return { lat: point.latitude, lng: point.longitude };
}

export function resolveMissionRouteLeg(status: DriverMissionStatus | string | null | undefined): MissionRouteLeg {
  const key = resolveDriverStatusForUx(status);
  if (key === "EN_ROUTE") return { mode: "live", destination: "pickup" };
  // ARRIVED = patient en cours de montée, on garde la destination finale en live
  // pour éviter les écarts avec IN_PROGRESS lors du basculement de statut.
  if (key === "ARRIVED") return { mode: "live", destination: "dropoff" };
  if (key === "IN_PROGRESS") return { mode: "live", destination: "dropoff" };
  return { mode: "planned" };
}

export function estimateMissionRouteMetricsBetweenCoords(
  origin: MissionCoord,
  destination: MissionCoord
): MissionRouteMetrics {
  const km = haversineKm(origin, destination);
  if (km <= 0) return { distanceKm: null, durationMinutes: null };
  return {
    distanceKm: km,
    durationMinutes: Math.max(1, Math.round((km / FALLBACK_SPEED_KMH) * 60)),
  };
}

function haversineKm(a: MissionCoord, b: MissionCoord): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 6371 * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

export function missionCoordDistanceMeters(a: MissionCoord, b: MissionCoord): number {
  return haversineKm(a, b) * 1000;
}

export function extractMissionPickupCoord(mission: DriverMission): MissionCoord | null {
  const raw = mission as Record<string, unknown>;
  const lat = firstPositiveNumber(mission.pickup_lat, raw.pickup_lat);
  const lng = firstPositiveNumber(mission.pickup_lng, raw.pickup_lon, raw.pickup_lng);
  if (lat == null || lng == null) return null;
  return { lat, lng };
}

export function extractMissionDropoffCoord(mission: DriverMission): MissionCoord | null {
  const raw = mission as Record<string, unknown>;
  const lat = firstPositiveNumber(mission.dropoff_lat, raw.dropoff_lat);
  const lng = firstPositiveNumber(mission.dropoff_lng, raw.dropoff_lon, raw.dropoff_lng);
  if (lat == null || lng == null) return null;
  return { lat, lng };
}

/** Métriques trajet départ → arrivée depuis payload API (+ haversine si coords). */
export function resolveMissionRouteMetrics(mission: DriverMission): MissionRouteMetrics {
  const raw = mission as Record<string, unknown>;

  let distanceKm = firstPositiveNumber(
    mission.distance_km,
    mission.distanceKm,
    mission.route_distance_km,
    raw.distance_km,
    raw.route_distance_km,
    raw.estimated_distance_km,
    raw.travel_distance_km
  );

  const distanceMeters = firstPositiveNumber(
    raw.distance_meters,
    raw.estimated_distance_meters,
    raw.route_distance_meters,
    raw.travel_distance_meters
  );
  if (distanceKm == null && distanceMeters != null) {
    distanceKm = distanceMeters / 1000;
  }

  let durationMinutes = firstPositiveNumber(
    mission.estimated_duration_min,
    mission.duration_minutes,
    raw.duration_minutes,
    raw.duration_in_minutes,
    raw.route_duration_minutes,
    raw.travel_time_minutes
  );

  const durationSeconds = firstPositiveNumber(
    mission.duration_seconds,
    raw.duration_seconds,
    raw.estimated_duration_seconds,
    raw.route_duration_seconds,
    raw.travel_time_seconds
  );
  if (durationMinutes == null && durationSeconds != null) {
    durationMinutes = Math.max(1, Math.round(durationSeconds / 60));
  }

  const pickup = extractMissionPickupCoord(mission);
  const dropoff = extractMissionDropoffCoord(mission);
  if (pickup && dropoff) {
    const km = haversineKm(pickup, dropoff);
    if (km > 0) {
      if (distanceKm == null) distanceKm = km;
      if (durationMinutes == null) {
        durationMinutes = Math.max(1, Math.round((km / FALLBACK_SPEED_KMH) * 60));
      }
    }
  }

  return { distanceKm, durationMinutes };
}

export function formatMissionRouteDistanceKm(distanceKm: number | null | undefined): string {
  if (distanceKm == null || !Number.isFinite(distanceKm) || distanceKm <= 0) return "—";
  if (distanceKm >= 10) return `${Math.round(distanceKm)} km`;
  return `${distanceKm.toFixed(1)} km`;
}

export function formatMissionRouteDurationMinutes(minutes: number | null | undefined): string {
  if (minutes == null || !Number.isFinite(minutes) || minutes <= 0) return "—";
  return `${Math.round(minutes)} min`;
}

/** @deprecated Préférer resolveMissionRouteMetrics + formatMissionRouteDistanceKm */
export function compactMissionDistance(mission: DriverMission): string {
  return formatMissionRouteDistanceKm(resolveMissionRouteMetrics(mission).distanceKm);
}

/** @deprecated Préférer resolveMissionRouteMetrics + formatMissionRouteDurationMinutes */
export function compactMissionEta(mission: DriverMission): string {
  return formatMissionRouteDurationMinutes(resolveMissionRouteMetrics(mission).durationMinutes);
}
