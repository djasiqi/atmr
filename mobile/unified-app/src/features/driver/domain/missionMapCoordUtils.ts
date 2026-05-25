import type { DriverMission } from "../types";

export type MapLatLng = { latitude: number; longitude: number };

const SWISS_LAT_MIN = 45;
const SWISS_LAT_MAX = 48;
const SWISS_LNG_MIN = 5;
const SWISS_LNG_MAX = 11;

function isFiniteLatLng(lat: number, lng: number): boolean {
  return Number.isFinite(lat) && Number.isFinite(lng) && Math.abs(lat) <= 90 && Math.abs(lng) <= 180;
}

function isNearZeroPoint(lat: number, lng: number): boolean {
  return Math.abs(lat) < 0.0001 && Math.abs(lng) < 0.0001;
}

function isInSwissBounds(lat: number, lng: number): boolean {
  return lat >= SWISS_LAT_MIN && lat <= SWISS_LAT_MAX && lng >= SWISS_LNG_MIN && lng <= SWISS_LNG_MAX;
}

export function toFiniteCoordLoose(input: unknown): number | null {
  if (typeof input === "number" && Number.isFinite(input)) return input;
  if (typeof input === "string" && input.trim().length > 0) {
    const parsed = Number(input.replace(",", "."));
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

export function normalizeMissionMapPoint(rawLat: unknown, rawLng: unknown): MapLatLng | null {
  const lat = toFiniteCoordLoose(rawLat);
  const lng = toFiniteCoordLoose(rawLng);
  if (lat == null || lng == null) return null;
  if (!isFiniteLatLng(lat, lng) || isNearZeroPoint(lat, lng)) return null;
  if (!isInSwissBounds(lat, lng) && isInSwissBounds(lng, lat)) {
    return { latitude: lng, longitude: lat };
  }
  return { latitude: lat, longitude: lng };
}

export function missionCoordToMapPoint(coord: { lat: number; lng: number } | null): MapLatLng | null {
  if (!coord) return null;
  return normalizeMissionMapPoint(coord.lat, coord.lng);
}

function firstFiniteFromMission(raw: Record<string, unknown>, ...keys: string[]): unknown {
  for (const key of keys) {
    if (key in raw) return raw[key];
  }
  return null;
}

export type MissionMapCoordInput = {
  driverLat?: unknown;
  driverLng?: unknown;
  pickupLat?: unknown;
  pickupLng?: unknown;
  dropoffLat?: unknown;
  dropoffLng?: unknown;
  pickupLocation?: string | null;
  dropoffLocation?: string | null;
};

export function extractMissionMapCoordInput(mission: DriverMission): MissionMapCoordInput {
  const raw = mission as Record<string, unknown>;
  return {
    driverLat: firstFiniteFromMission(
      raw,
      "driver_lat",
      "driver_latitude",
      "current_lat",
      "current_latitude",
      "latitude"
    ),
    driverLng: firstFiniteFromMission(
      raw,
      "driver_lng",
      "driver_lon",
      "driver_longitude",
      "current_lng",
      "current_lon",
      "current_longitude",
      "longitude"
    ),
    pickupLat: raw.pickup_lat,
    pickupLng: firstFiniteFromMission(raw, "pickup_lng", "pickup_lon"),
    dropoffLat: raw.dropoff_lat,
    dropoffLng: firstFiniteFromMission(raw, "dropoff_lng", "dropoff_lon"),
    pickupLocation:
      typeof mission.pickup_location === "string" ? mission.pickup_location : null,
    dropoffLocation:
      typeof mission.dropoff_location === "string" ? mission.dropoff_location : null,
  };
}

export function resolveStaticMissionMapCoords(input: MissionMapCoordInput): {
  driverCoord: MapLatLng | null;
  pickupCoord: MapLatLng | null;
  dropoffCoord: MapLatLng | null;
} {
  return {
    driverCoord: normalizeMissionMapPoint(input.driverLat, input.driverLng),
    pickupCoord: normalizeMissionMapPoint(input.pickupLat, input.pickupLng),
    dropoffCoord: normalizeMissionMapPoint(input.dropoffLat, input.dropoffLng),
  };
}
