import type {
  CompanyDispatchMission,
  CompanyDispatchMissionListResponse,
} from "../api/contracts";

function valuesDeepEqual(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (left == null || right == null) return left === right;
  if (typeof left !== "object" || typeof right !== "object") return false;
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
      return false;
    }
    return left.every((item, index) => valuesDeepEqual(item, right[index]));
  }
  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const keys = Object.keys(leftRecord);
  if (keys.length !== Object.keys(rightRecord).length) return false;
  return keys.every((key) => valuesDeepEqual(leftRecord[key], rightRecord[key]));
}

export function areDispatchMissionsContentEqual(
  previous: CompanyDispatchMission,
  next: CompanyDispatchMission
): boolean {
  return previous === next || valuesDeepEqual(previous, next);
}

/**
 * Conserve la référence des missions inchangées après un refetch / réconciliation.
 * Une mise à jour de #45711 ne crée pas 29 nouveaux objets.
 */
export function reconcileDispatchMissionList(
  previous: readonly CompanyDispatchMission[] | undefined,
  next: CompanyDispatchMission[]
): CompanyDispatchMission[] {
  if (!previous || previous.length === 0) return next;
  if (previous === next) return next;

  const previousById = new Map(previous.map((mission) => [mission.mission_id, mission]));
  let unchangedCount = 0;
  const reconciled = next.map((mission) => {
    const prior = previousById.get(mission.mission_id);
    if (prior && areDispatchMissionsContentEqual(prior, mission)) {
      unchangedCount += 1;
      return prior;
    }
    return mission;
  });

  if (
    unchangedCount === previous.length &&
    unchangedCount === next.length &&
    previous.every((mission, index) => mission === reconciled[index])
  ) {
    return previous as CompanyDispatchMission[];
  }
  return reconciled;
}

export function shareDispatchMissionsQueryData(
  oldData: CompanyDispatchMissionListResponse | undefined,
  newData: CompanyDispatchMissionListResponse
): CompanyDispatchMissionListResponse {
  if (!oldData) return newData;
  const missions = reconcileDispatchMissionList(oldData.missions, newData.missions);
  if (
    missions === oldData.missions &&
    oldData.context_id === newData.context_id &&
    oldData.total === newData.total &&
    oldData.loaded === newData.loaded &&
    oldData.is_complete === newData.is_complete &&
    oldData.pagination_error === newData.pagination_error &&
    oldData.next_page === newData.next_page
  ) {
    if (oldData.refreshed_at === newData.refreshed_at) return oldData;
    return { ...oldData, refreshed_at: newData.refreshed_at };
  }
  if (missions === newData.missions) return newData;
  return { ...newData, missions };
}
