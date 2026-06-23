import type { CompanyDriverLiveLocation } from "../../api/contracts";
import {
  reportFleetMarkerAnimationSkipped,
  type FleetMarkerAnimationSkipReason,
} from "../../../../core/observability/fleetMapDiagnostics";
import { haversineMeters } from "../../realtime/driverLiveLocationMerge";
import { isValidMapCoord } from "./mapsIosNewArchSafeMode";

export type FleetMapLatLng = { latitude: number; longitude: number };

export function isValidFleetMapCoordinate(
  coord: FleetMapLatLng | null | undefined
): coord is FleetMapLatLng {
  return isValidMapCoord(coord?.latitude, coord?.longitude);
}

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

  if (!isValidFleetMapCoordinate(input.to)) {
    return { mode: "snap" };
  }

  if (!from) {
    return { mode: "snap" };
  }

  if (!isValidFleetMapCoordinate(from)) {
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

export type AnimateFleetMarkerInput = {
  marker: AnimatableFleetMarkerRef;
  from: FleetMapLatLng | null | undefined;
  to: FleetMapLatLng;
  durationMs: number;
  reportSkip?: boolean;
};

function reportAnimationSkip(
  reason: FleetMarkerAnimationSkipReason,
  extra?: Record<string, unknown>
): void {
  reportFleetMarkerAnimationSkipped(reason, extra);
}

export function animateFleetMarkerToCoordinate({
  marker,
  from,
  to,
  durationMs,
  reportSkip = true,
}: AnimateFleetMarkerInput): boolean {
  const skip = (reason: FleetMarkerAnimationSkipReason, extra?: Record<string, unknown>) => {
    if (reportSkip) {
      reportAnimationSkip(reason, extra);
    }
    return false;
  };

  if (!canAnimateFleetMarker(marker)) {
    return skip("marker_unavailable");
  }
  if (!from) {
    return skip("missing_previous");
  }
  if (!isValidFleetMapCoordinate(from)) {
    return skip("invalid_previous", { from });
  }
  if (!isValidFleetMapCoordinate(to)) {
    return skip("invalid_next", { to });
  }

  try {
    marker?.animateMarkerToCoordinate?.(to, durationMs);
    return true;
  } catch {
    return skip("marker_unavailable");
  }
}

/** Commit conditionné — évite qu'une animation obsolète écrase une cible plus récente. */
export function shouldApplyFleetMarkerCommit(seq: number, currentSeq: number): boolean {
  return seq === currentSeq;
}
