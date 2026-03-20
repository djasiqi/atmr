/**
 * Contexte mission pour les payloads GPS : mission_live uniquement si booking actif (id connu).
 * Évite mission_live + mission_id null côté serveur.
 */

import { MissionStateManager } from "./missionState";

export type MissionTrackingContext = {
  missionId: number | null;
  mode: "mission_live" | "availability_presence";
};

export function resolveMissionContext(): MissionTrackingContext {
  const mission = MissionStateManager.getState().activeMission;
  const missionId = mission?.id ?? null;
  return {
    missionId,
    mode: missionId != null ? "mission_live" : "availability_presence",
  };
}
