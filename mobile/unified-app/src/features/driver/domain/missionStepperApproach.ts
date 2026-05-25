import type { DriverEtaSnapshot } from "../api";
import { resolveDriverStatusForUx } from "../statusDictionary";
import type { DriverMission } from "../types";
import { extractMissionMapCoordInput } from "./missionMapCoordUtils";
import { hasArrivedAtPickupMilestone } from "./missionMilestoneOverlay";
import {
  mapLatLngToMissionCoord,
  missionCoordDistanceMeters,
  resolveLiveDriverOrigin,
  type MissionCoord,
} from "./missionRouteMetrics";
import {
  clearAllStepperApproachBaselines,
  clearStepperApproachBaseline,
  resolveStepperApproachBaseline,
  resetStepperApproachBaseline,
  type StepperApproachSegment,
} from "./missionStepperApproachBaseline";

export {
  clearAllStepperApproachBaselines,
  resetStepperApproachBaseline,
} from "./missionStepperApproachBaseline";
export type { StepperApproachSegment } from "./missionStepperApproachBaseline";

const ARRIVAL_SNAP_METERS = 80;
const APPROACH_MAX_BEFORE_ARRIVE = 0.98;

export type ActiveStepperApproach = {
  segment: StepperApproachSegment;
  /** Index de l'étape de départ du segment (ex. Assignée = 0). */
  fromStepIndex: number;
  /** Index de l'étape cible (ex. Arrivé patient = 1). */
  targetStepIndex: number;
};

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function hasArrivedAtPickup(mission: DriverMission): boolean {
  const raw = mission as Record<string, unknown>;
  const milestone = String(raw.mission_milestone ?? "")
    .trim()
    .toUpperCase();
  if (milestone === "ARRIVED") return true;
  if (typeof mission.id === "number" && hasArrivedAtPickupMilestone(mission.id)) {
    return true;
  }
  return resolveDriverStatusForUx(mission.status) === "ARRIVED";
}

/** Ratio 0→1 : 0 = départ du segment, ~1 = arrivée au jalon suivant. */
export function computeStepperApproachProgress(
  currentDistanceMeters: number,
  baselineDistanceMeters: number
): number {
  if (!Number.isFinite(currentDistanceMeters) || currentDistanceMeters <= 0) {
    return APPROACH_MAX_BEFORE_ARRIVE;
  }
  const baseline = Math.max(baselineDistanceMeters, 1);
  if (currentDistanceMeters >= baseline) {
    return 0;
  }
  if (currentDistanceMeters <= ARRIVAL_SNAP_METERS) {
    return APPROACH_MAX_BEFORE_ARRIVE;
  }
  return clamp01(1 - currentDistanceMeters / baseline);
}

export function resolveStepperApproachProgressForMission(
  missionId: number,
  segment: StepperApproachSegment,
  currentDistanceMeters: number
): number {
  const baseline = resolveStepperApproachBaseline(missionId, segment, currentDistanceMeters);
  return computeStepperApproachProgress(currentDistanceMeters, baseline);
}

/** Segment actif avec remplissage GPS progressif, selon le statut mission. */
export function resolveActiveStepperApproach(mission: DriverMission): ActiveStepperApproach | null {
  const status = resolveDriverStatusForUx(mission.status);

  if (status === "EN_ROUTE" && !hasArrivedAtPickup(mission)) {
    return { segment: "pickup", fromStepIndex: 0, targetStepIndex: 1 };
  }
  if (status === "IN_PROGRESS") {
    return { segment: "dropoff", fromStepIndex: 2, targetStepIndex: 3 };
  }
  return null;
}

export function resolveStepperApproachDistance(
  driver: MissionCoord | null,
  target: MissionCoord | null
): number | null {
  if (!driver || !target) return null;
  const meters = missionCoordDistanceMeters(driver, target);
  return Number.isFinite(meters) && meters > 0 ? meters : null;
}

export function resolveDriverOriginForStepperApproach(
  mission: DriverMission,
  tracking: MissionCoord | null,
  etaSnapshot?: DriverEtaSnapshot | null
): MissionCoord | null {
  const mapInput = extractMissionMapCoordInput(mission);
  const missionDriver = mapLatLngToMissionCoord(
    mapInput.driverLat != null && mapInput.driverLng != null
      ? {
          latitude: Number(mapInput.driverLat),
          longitude: Number(mapInput.driverLng),
        }
      : null
  );
  return resolveLiveDriverOrigin({
    tracking,
    etaDriver: etaSnapshot
      ? { lat: etaSnapshot.driver_lat ?? null, lon: etaSnapshot.driver_lon ?? null }
      : null,
    mission: missionDriver,
  });
}

/** Compatibilité ascendante — pickup uniquement. */
export function clearPickupApproachForMission(missionId: number): void {
  clearStepperApproachBaseline(missionId, "pickup");
}

export function resetPickupApproachBaseline(missionId: number): void {
  resetStepperApproachBaseline(missionId, "pickup");
}
