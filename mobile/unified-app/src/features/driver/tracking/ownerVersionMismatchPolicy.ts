/**
 * D5 — politique owner_version_mismatch (sans STOP natif direct).
 *
 * Un simple bump de missionContextVersion (même missionId) → L1 reconcile.
 * Changement de missionId avec FGS vivant → STOP owner-guarded puis START.
 */
export type NativeOwnerRef = {
  missionId: number | null;
  missionContextVersion: number;
};

export type OwnerMismatchDecision =
  | { action: "abort"; detail: string }
  | { action: "l1_reconcile"; detail: string }
  | { action: "owned_stop_then_start"; detail: string };

export function decideOwnerVersionMismatchAction(input: {
  platform: string;
  taskStarted: boolean;
  priorOwner: NativeOwnerRef | null | undefined;
  desiredOwner: NativeOwnerRef | null | undefined;
}): OwnerMismatchDecision {
  if (input.platform !== "android") {
    return { action: "abort", detail: "not_android" };
  }
  if (!input.taskStarted) {
    return { action: "abort", detail: "task_not_started" };
  }
  if (input.priorOwner == null || input.desiredOwner == null) {
    return { action: "abort", detail: "missing_owner" };
  }

  const prior = input.priorOwner;
  const desired = input.desiredOwner;
  const missionChanged = prior.missionId !== desired.missionId;
  const versionChanged = prior.missionContextVersion !== desired.missionContextVersion;

  if (!missionChanged && !versionChanged) {
    return { action: "abort", detail: "owners_equal" };
  }

  // Même mission, version différente : contexte déjà réécrit — pas d'Unregister.
  if (!missionChanged && versionChanged) {
    return {
      action: "l1_reconcile",
      detail: "same_mission_version_bump",
    };
  }

  // MissionId différent + FGS vivant : STOP ownership puis recovery START.
  return {
    action: "owned_stop_then_start",
    detail: "mission_id_changed",
  };
}
