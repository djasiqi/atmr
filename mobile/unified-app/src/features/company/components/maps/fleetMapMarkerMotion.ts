import type { CompanyDriverLiveLocation } from "../../api/contracts";
import { haversineMeters } from "../../realtime/driverLiveLocationMerge";

export type FleetMapLatLng = { latitude: number; longitude: number };

export type FleetMarkerMotionPlan =
  | { mode: "snap" }
  | { mode: "animate"; durationMs: number };

export const DEFAULT_SNAP_DISTANCE_M = 250;
export const NOOP_DISTANCE_M = 1;
export const STALE_GAP_MS = 30_000;
export const MIN_DURATION_MS = 800;
export const MAX_DURATION_MS = 1_500;

const DURATION_MS_PER_METER = 3;

const SNAP_LOCATION_STATUSES = new Set<NonNullable<CompanyDriverLiveLocation["location_status"]>>([
  "stale",
  "offline",
  "last_known",
]);

function clampDurationMs(distanceM: number): number {
  const raw = MIN_DURATION_MS + distanceM * DURATION_MS_PER_METER;
  return Math.min(MAX_DURATION_MS, Math.max(MIN_DURATION_MS, Math.round(raw)));
}

function resolveRecordedAtMs(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export type ResolveFleetMarkerMotionPlanInput = {
  from?: FleetMapLatLng | null;
  to: FleetMapLatLng;
  previousRecordedAtMs?: number | null;
  nextRecordedAtMs?: number | null;
  locationStatus?: CompanyDriverLiveLocation["location_status"] | null;
  markerKeyChanged?: boolean;
  snapDistanceM?: number;
  noopDistanceM?: number;
};

export function resolveFleetMarkerMotionPlan(
  input: ResolveFleetMarkerMotionPlanInput
): FleetMarkerMotionPlan {
  const snapDistanceM = input.snapDistanceM ?? DEFAULT_SNAP_DISTANCE_M;
  const noopDistanceM = input.noopDistanceM ?? NOOP_DISTANCE_M;
  const from = input.from;

  if (!from) {
    return { mode: "snap" };
  }

  if (input.markerKeyChanged) {
    return { mode: "snap" };
  }

  const distanceM = haversineMeters(from.latitude, from.longitude, input.to.latitude, input.to.longitude);

  if (distanceM <= noopDistanceM) {
    return { mode: "snap" };
  }

  if (distanceM >= snapDistanceM) {
    return { mode: "snap" };
  }

  const prevRecorded = input.previousRecordedAtMs ?? null;
  const nextRecorded = input.nextRecordedAtMs ?? null;
  if (
    prevRecorded != null &&
    nextRecorded != null &&
    nextRecorded - prevRecorded >= STALE_GAP_MS
  ) {
    return { mode: "snap" };
  }

  if (input.locationStatus && SNAP_LOCATION_STATUSES.has(input.locationStatus)) {
    return { mode: "snap" };
  }

  return { mode: "animate", durationMs: clampDurationMs(distanceM) };
}

export function parseFleetMarkerRecordedAtMs(recordedAt: string | null | undefined): number | null {
  return resolveRecordedAtMs(recordedAt);
}

export type AnimatableFleetMarkerRef = {
  animateMarkerToCoordinate?: (coord: FleetMapLatLng, duration: number) => void;
} | null;

export function canAnimateFleetMarker(marker: AnimatableFleetMarkerRef): boolean {
  return typeof marker?.animateMarkerToCoordinate === "function";
}

export function animateFleetMarkerToCoordinate(
  marker: AnimatableFleetMarkerRef,
  coordinate: FleetMapLatLng,
  durationMs: number
): boolean {
  if (!canAnimateFleetMarker(marker)) return false;
  try {
    marker?.animateMarkerToCoordinate?.(coordinate, durationMs);
    return true;
  } catch {
    return false;
  }
}

/** Commit conditionné — évite qu'une animation obsolète écrase une cible plus récente. */
export function shouldApplyFleetMarkerCommit(seq: number, currentSeq: number): boolean {
  return seq === currentSeq;
}
