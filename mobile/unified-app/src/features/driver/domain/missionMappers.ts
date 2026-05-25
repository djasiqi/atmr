import { resolveDriverStatusForUx } from "../statusDictionary";
import type { DriverMission, DriverMissionDetail } from "../types";
import {
  clearArrivedAtPickupIfMilestoneIncompatible,
  hasArrivedAtPickupMilestone,
  markDriverArrivedAtPickupMilestone,
} from "./missionMilestoneOverlay";

function normalizeMaybeString(value: unknown): string | null | undefined {
  if (value === null || value === undefined) return value as null | undefined;
  if (typeof value === "string") return value;
  return String(value);
}

function statusUpperFromUnknown(value: unknown): string {
  return String(value ?? "")
    .trim()
    .toUpperCase();
}

/** API bookings : `pickup_lon` / `dropoff_lon` ; cartes RN : `pickup_lng` / `dropoff_lng`. */
function firstFiniteNumber(...values: unknown[]): number | undefined {
  for (const v of values) {
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v.trim().length > 0) {
      const n = Number(v);
      if (Number.isFinite(n)) return n;
    }
  }
  return undefined;
}

export function mapDriverMission(input: DriverMission): DriverMission {
  const id = typeof input.id === "number" ? input.id : null;
  const upper = statusUpperFromUnknown(input.status);

  if (id != null) {
    clearArrivedAtPickupIfMilestoneIncompatible(id, upper);
  }

  const raw = input as Record<string, unknown>;
  const pickupLng = firstFiniteNumber(input.pickup_lng, raw.pickup_lon);
  const dropoffLng = firstFiniteNumber(input.dropoff_lng, raw.dropoff_lon);
  const pickupLat = firstFiniteNumber(input.pickup_lat);
  const dropoffLat = firstFiniteNumber(input.dropoff_lat);

  const base: DriverMission = {
    ...input,
    pickup_location: normalizeMaybeString(input.pickup_location),
    dropoff_location: normalizeMaybeString(input.dropoff_location),
    scheduled_time: normalizeMaybeString(input.scheduled_time),
    updated_at: normalizeMaybeString(input.updated_at),
    client_name: normalizeMaybeString(input.client_name),
    ...(pickupLng != null ? { pickup_lng: pickupLng } : {}),
    ...(dropoffLng != null ? { dropoff_lng: dropoffLng } : {}),
    ...(pickupLat != null ? { pickup_lat: pickupLat } : {}),
    ...(dropoffLat != null ? { dropoff_lat: dropoffLat } : {}),
  };

  const milestone = statusUpperFromUnknown(raw.mission_milestone);
  if (id != null && milestone === "ARRIVED") {
    markDriverArrivedAtPickupMilestone(id);
  }

  if (id != null && upper === "EN_ROUTE" && hasArrivedAtPickupMilestone(id)) {
    return { ...base, status: resolveDriverStatusForUx("ARRIVED") };
  }

  return { ...base, status: resolveDriverStatusForUx(String(base.status ?? "")) };
}

export function mapDriverMissionDetail(input: DriverMissionDetail): DriverMissionDetail {
  return mapDriverMission(input);
}
