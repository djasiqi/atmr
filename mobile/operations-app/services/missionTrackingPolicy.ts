/**
 * P0 — Source unique de vérité : quels statuts mission activent le tracking mission renforcé.
 * Toute évolution métier (nouveaux statuts) doit passer par ce fichier uniquement.
 */

import { MissionStateManager, type MissionBarStatus } from "./missionState";

const TRACKING_ACTIVE_STATUSES: ReadonlySet<MissionBarStatus> = new Set([
  "ASSIGNED",
  "EN_ROUTE",
  "IN_PROGRESS",
]);

/**
 * Indique si le statut barre mission (MissionBarStatus) doit activer le mode tracking mission
 * (mission_live / native background renforcé), hors considération de permissions ou AppState.
 */
export function isMissionTrackingActiveStatus(status: MissionBarStatus): boolean {
  return TRACKING_ACTIVE_STATUSES.has(status);
}

/**
 * Mission chargée dans le manager ET statut éligible pour le tracking mission.
 * Aligné sur buildBgTrackingInputs (missionStatusEnabledForTracking + hasActiveMission).
 */
export function isMissionTrackingEligibleNow(): boolean {
  if (!MissionStateManager.isActive()) return false;
  const { currentStatus } = MissionStateManager.getState();
  return isMissionTrackingActiveStatus(currentStatus);
}
