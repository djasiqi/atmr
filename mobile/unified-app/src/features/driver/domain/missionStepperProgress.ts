import { hasArrivedAtPickupMilestone } from "./missionMilestoneOverlay";
import { resolveDriverStatusForUx } from "../statusDictionary";
import type { DriverMission } from "../types";

export type MissionStepperProgress = {
  /** Nombre d'étapes entièrement validées (coche verte). */
  completedCount: number;
  /** Étape en cours (anneau brand) — hors rapprochement GPS vers le patient. */
  activeIndex: number | null;
};

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function hasArrivedAtPickupMilestoneFromMission(mission: DriverMission): boolean {
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

/** Progression stepper dashboard selon statut mission + jalon « arrivé patient ». */
export function resolveMissionStepperProgress(mission: DriverMission): MissionStepperProgress {
  const status = resolveDriverStatusForUx(mission.status);
  const arrivedAtPickup = hasArrivedAtPickupMilestoneFromMission(mission);

  switch (status) {
    case "ASSIGNED":
      return { completedCount: 1, activeIndex: null };
    case "EN_ROUTE":
      if (arrivedAtPickup) {
        return { completedCount: 2, activeIndex: 2 };
      }
      return { completedCount: 1, activeIndex: null };
    case "ARRIVED":
      return { completedCount: 2, activeIndex: 2 };
    case "IN_PROGRESS":
      return { completedCount: 3, activeIndex: null };
    case "COMPLETED":
      return { completedCount: 4, activeIndex: null };
    default:
      return { completedCount: 0, activeIndex: null };
  }
}

/** Segment entre l'étape `fromIndex` et `fromIndex + 1`. */
export function isMissionStepperSegmentComplete(
  fromIndex: number,
  progress: MissionStepperProgress
): boolean {
  const nextIndex = fromIndex + 1;
  if (nextIndex < progress.completedCount) return true;
  return progress.activeIndex != null && nextIndex === progress.activeIndex;
}

/** Remplissage partiel 0→1 d'un segment (ex. rapprochement GPS). */
export function resolveMissionStepperSegmentFill(
  fromIndex: number,
  progress: MissionStepperProgress,
  partialFill: number | null
): number {
  if (isMissionStepperSegmentComplete(fromIndex, progress)) return 1;
  if (partialFill != null && progress.activeIndex == null) {
    const isPickupSegment = fromIndex === 0 && progress.completedCount === 1;
    const isDropoffSegment = fromIndex === 2 && progress.completedCount === 3;
    if (isPickupSegment || isDropoffSegment) {
      return clamp01(partialFill);
    }
  }
  if (progress.activeIndex != null && fromIndex + 1 === progress.activeIndex) {
    return 0;
  }
  return 0;
}

/** Demi-segment (gauche/droite d'une colonne) pour un remplissage continu. */
export function resolveMissionStepperHalfSegmentFill(
  totalFill: number,
  half: "first" | "second"
): number {
  const fill = clamp01(totalFill);
  if (half === "first") return clamp01(fill * 2);
  return clamp01((fill - 0.5) * 2);
}
