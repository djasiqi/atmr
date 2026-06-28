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

/** Parité web driverMarkerMotion.js — glide aligné sur l’intervalle GPS. */
export const FLEET_MARKER_MOTION_MIN_MS = 2_200;
export const FLEET_MARKER_MOTION_MAX_MS = 12_000;
export const FLEET_MARKER_MOTION_DEFAULT_MS = 7_500;
export const FLEET_MARKER_MOTION_DURATION_STRETCH = 1.42;

export const DEFAULT_SNAP_DISTANCE_M = 250;
export const NOOP_DISTANCE_M = 1;
/** Snap seulement si écart GPS très long (reconnexion / perte signal). */
export const STALE_RECORDED_GAP_MS = 120_000;

const SNAP_LOCATION_STATUSES = new Set<NonNullable<CompanyDriverLiveLocation["location_status"]>>([
  "stale",
  "offline",
  "last_known",
]);

function resolveRecordedAtMs(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** Courbe smoothstep — accélération / décélération (parité web). */
export function easeSmoothStep(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
}

export function interpolateFleetMarkerPosition(
  from: FleetMapLatLng,
  to: FleetMapLatLng,
  progress: number
): FleetMapLatLng {
  const t = easeSmoothStep(progress);
  return {
    latitude: from.latitude + (to.latitude - from.latitude) * t,
    longitude: from.longitude + (to.longitude - from.longitude) * t,
  };
}

/** Ajuste la durée selon la distance (courts trajets = glide plus long). */
export function resolveFleetMotionDurationFromDistance(durationMs: number, distanceM: number): number {
  if (!Number.isFinite(distanceM) || distanceM <= 0) return durationMs;
  if (distanceM < 12) return Math.max(durationMs, 3_000);
  if (distanceM > 180) {
    return Math.min(FLEET_MARKER_MOTION_MAX_MS, durationMs * 1.12);
  }
  return durationMs;
}

export function resolveFleetMarkerMotionDurationMs(
  previousRecordedAtMs: number | null,
  nextRecordedAtMs: number | null,
  lastMotionAtMs: number | null,
  distanceM: number,
  nowMs = Date.now()
): number {
  let base: number;
  if (
    previousRecordedAtMs != null &&
    nextRecordedAtMs != null &&
    nextRecordedAtMs > previousRecordedAtMs
  ) {
    const elapsed = nextRecordedAtMs - previousRecordedAtMs;
    const stretched = elapsed * FLEET_MARKER_MOTION_DURATION_STRETCH;
    base = Math.min(
      FLEET_MARKER_MOTION_MAX_MS,
      Math.max(FLEET_MARKER_MOTION_MIN_MS, stretched)
    );
  } else if (lastMotionAtMs != null && Number.isFinite(lastMotionAtMs)) {
    const elapsed = nowMs - lastMotionAtMs;
    if (elapsed > 0) {
      const stretched = elapsed * FLEET_MARKER_MOTION_DURATION_STRETCH;
      base = Math.min(
        FLEET_MARKER_MOTION_MAX_MS,
        Math.max(FLEET_MARKER_MOTION_MIN_MS, stretched)
      );
    } else {
      base = FLEET_MARKER_MOTION_MIN_MS;
    }
  } else {
    base = FLEET_MARKER_MOTION_DEFAULT_MS;
  }
  return resolveFleetMotionDurationFromDistance(base, distanceM);
}

export type ResolveFleetMarkerMotionPlanInput = {
  from?: FleetMapLatLng | null;
  to: FleetMapLatLng;
  previousRecordedAtMs?: number | null;
  nextRecordedAtMs?: number | null;
  lastMotionAtMs?: number | null;
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

  if (!from || !isValidFleetMapCoordinate(from)) {
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
    nextRecorded - prevRecorded >= STALE_RECORDED_GAP_MS
  ) {
    return { mode: "snap" };
  }

  if (input.locationStatus && SNAP_LOCATION_STATUSES.has(input.locationStatus)) {
    return { mode: "snap" };
  }

  const durationMs = resolveFleetMarkerMotionDurationMs(
    prevRecorded,
    nextRecorded,
    input.lastMotionAtMs ?? null,
    distanceM
  );

  return { mode: "animate", durationMs };
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

/** Animation native (iOS raster) — secours si interpolation JS indisponible. */
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
