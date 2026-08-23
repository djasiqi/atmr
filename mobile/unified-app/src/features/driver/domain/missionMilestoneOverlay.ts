/**
 * Overlay ARRIVED côté mobile = optimistic UI uniquement (ARRIVED-SOT-2).
 *
 * La vérité durable vient du GET serveur :
 *   Booking EN_ROUTE + Assignment ARRIVED_PICKUP
 *   → status=arrived + mission_milestone=ARRIVED
 *
 * Le Set ne doit plus servir à restaurer après cold start / refresh.
 */
const arrivedPickupByMissionId = new Set<number>();

export function markDriverArrivedAtPickupMilestone(missionId: number): void {
  arrivedPickupByMissionId.add(missionId);
}

/** Rollback d’une transition optimiste `ARRIVED` si l’appel a échoué. */
export function unmarkDriverArrivedAtPickupMilestone(missionId: number): void {
  arrivedPickupByMissionId.delete(missionId);
}

function clearIfPresent(missionId: number): void {
  arrivedPickupByMissionId.delete(missionId);
}

export function hasArrivedAtPickupMilestone(missionId: number): boolean {
  return arrivedPickupByMissionId.has(missionId);
}

/** Quand l'API indique clairement qu'on a quitté l'étape en route, retirer l'overlay. */
export function clearArrivedAtPickupIfMilestoneIncompatible(
  missionId: number,
  statusUpper: string
): void {
  if (statusUpper === "EN_ROUTE" || statusUpper === "ARRIVED") {
    return;
  }
  clearIfPresent(missionId);
}

export function applyArrivedMilestoneFromStatusResponse(
  missionId: number,
  data: { mission_milestone?: unknown; status?: unknown } | null | undefined
): void {
  if (String(data?.mission_milestone ?? "").toUpperCase() === "ARRIVED") {
    markDriverArrivedAtPickupMilestone(missionId);
  }
  // Composition serveur : status=arrived (sans milestone explicite)
  if (String(data?.status ?? "").trim().toLowerCase() === "arrived") {
    markDriverArrivedAtPickupMilestone(missionId);
  }
}

export function resetDriverMilestoneOverlayForTests(): void {
  arrivedPickupByMissionId.clear();
}
