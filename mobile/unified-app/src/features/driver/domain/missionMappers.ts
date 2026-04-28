import { resolveDriverStatusForUx } from "../statusDictionary";
import type { DriverMission, DriverMissionDetail } from "../types";
import {
  clearArrivedAtPickupIfMilestoneIncompatible,
  hasArrivedAtPickupMilestone,
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

export function mapDriverMission(input: DriverMission): DriverMission {
  const id = typeof input.id === "number" ? input.id : null;
  const upper = statusUpperFromUnknown(input.status);

  if (id != null) {
    clearArrivedAtPickupIfMilestoneIncompatible(id, upper);
  }

  const base: DriverMission = {
    ...input,
    pickup_location: normalizeMaybeString(input.pickup_location),
    dropoff_location: normalizeMaybeString(input.dropoff_location),
    scheduled_time: normalizeMaybeString(input.scheduled_time),
    updated_at: normalizeMaybeString(input.updated_at),
    client_name: normalizeMaybeString(input.client_name),
  };

  if (id != null && upper === "EN_ROUTE" && hasArrivedAtPickupMilestone(id)) {
    return { ...base, status: resolveDriverStatusForUx("ARRIVED") };
  }

  return { ...base, status: resolveDriverStatusForUx(String(base.status ?? "")) };
}

export function mapDriverMissionDetail(input: DriverMissionDetail): DriverMissionDetail {
  return mapDriverMission(input);
}
