import type { DriverMission } from "../types";

/**
 * Garde anti-régression par révision (P1 MISSION-STATE, M2).
 *
 * Le serveur attache à chaque mission une identité de lifecycle :
 *   - `assignment_id` : instance d'Assignment (nouveau chauffeur / redispatch
 *     ⇒ nouvel id ⇒ nouveau lifecycle légitime, on applique) ;
 *   - `mission_revision` : révision monotone incrémentée à chaque transition.
 *
 * Règle : un snapshot entrant STRICTEMENT plus ancien (même lifecycle,
 * révision inférieure) ne remplace JAMAIS l'état local. On ne devine pas via
 * le rang des statuts (retours, réaffectations et redispatchs rendraient ce
 * critère faux).
 */

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim().length > 0) {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

export function missionRevisionOf(mission: DriverMission | null | undefined): number | null {
  return toFiniteNumber(mission?.mission_revision);
}

export function missionAssignmentIdOf(
  mission: DriverMission | null | undefined
): number | null {
  return toFiniteNumber(mission?.assignment_id);
}

export type MissionMergeDecision = "apply" | "keep_local_stale_incoming";

export function decideMissionMerge(
  local: DriverMission | null | undefined,
  incoming: DriverMission
): MissionMergeDecision {
  if (!local || local.id !== incoming.id) return "apply";

  const localAssignment = missionAssignmentIdOf(local);
  const incomingAssignment = missionAssignmentIdOf(incoming);
  if (
    localAssignment != null &&
    incomingAssignment != null &&
    localAssignment !== incomingAssignment
  ) {
    // Nouveau lifecycle (réassignation / redispatch) : toujours appliquer.
    return "apply";
  }

  const localRevision = missionRevisionOf(local);
  const incomingRevision = missionRevisionOf(incoming);
  if (
    localRevision != null &&
    incomingRevision != null &&
    incomingRevision < localRevision
  ) {
    return "keep_local_stale_incoming";
  }

  return "apply";
}

export type GuardedMergeResult = {
  missions: DriverMission[];
  staleIgnoredCount: number;
};

/**
 * Remplacement complet de liste (polling) : l'appartenance à la liste vient
 * du serveur, mais chaque mission conserve l'état local si le snapshot
 * entrant est périmé.
 */
export function replaceMissionsGuarded(
  previous: DriverMission[] | undefined,
  incoming: DriverMission[]
): GuardedMergeResult {
  const localById = new Map<number, DriverMission>();
  (Array.isArray(previous) ? previous : []).forEach((mission) => {
    localById.set(mission.id, mission);
  });
  let staleIgnoredCount = 0;
  const missions = incoming.map((mission) => {
    const local = localById.get(mission.id);
    if (decideMissionMerge(local, mission) === "keep_local_stale_incoming") {
      staleIgnoredCount += 1;
      return local as DriverMission;
    }
    return mission;
  });
  return { missions, staleIgnoredCount };
}

/**
 * Fusion delta (reconcile incrémental) : les missions absentes du delta
 * restent inchangées ; celles présentes ne s'appliquent que si non périmées.
 */
export function mergeMissionsGuarded(
  previous: DriverMission[] | undefined,
  incoming: DriverMission[]
): GuardedMergeResult {
  const byId = new Map<number, DriverMission>();
  (Array.isArray(previous) ? previous : []).forEach((mission) => {
    byId.set(mission.id, mission);
  });
  let staleIgnoredCount = 0;
  incoming.forEach((mission) => {
    const local = byId.get(mission.id);
    if (decideMissionMerge(local, mission) === "keep_local_stale_incoming") {
      staleIgnoredCount += 1;
      return;
    }
    byId.set(mission.id, mission);
  });
  return { missions: Array.from(byId.values()), staleIgnoredCount };
}
