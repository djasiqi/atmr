import type { QueryClient } from "@tanstack/react-query";
import type { CompanyDispatchMission, CompanyDispatchMissionListResponse } from "../api/contracts";

export type RideDetailOpenSnapshot = {
  missionId: number;
  mission: CompanyDispatchMission;
  openedAt: number;
};

/** Secondaire au cache RQ : 2 suffisent pour retour / back. */
const MAX_RIDE_DETAIL_SNAPSHOTS = 2;
const snapshots = new Map<number, RideDetailOpenSnapshot>();
const snapshotOrder: number[] = [];
let lastOpenStartedAt: number | null = null;
let lastOpenMissionId: number | null = null;

/** Mémorise le DTO liste juste avant la navigation Détails. */
export function rememberRideDetailSnapshot(mission: CompanyDispatchMission): void {
  const openedAt = Date.now();
  snapshots.set(mission.mission_id, {
    missionId: mission.mission_id,
    mission,
    openedAt,
  });
  lastOpenStartedAt = openedAt;
  lastOpenMissionId = mission.mission_id;
  const existing = snapshotOrder.indexOf(mission.mission_id);
  if (existing >= 0) snapshotOrder.splice(existing, 1);
  snapshotOrder.push(mission.mission_id);
  while (snapshotOrder.length > MAX_RIDE_DETAIL_SNAPSHOTS) {
    const evictId = snapshotOrder.shift();
    if (evictId != null && evictId !== lastOpenMissionId) snapshots.delete(evictId);
  }
}

export function peekRideDetailSnapshot(missionId: number): RideDetailOpenSnapshot | null {
  return snapshots.get(missionId) ?? null;
}

export function getRideDetailOpenStartedAt(missionId: number): number | null {
  if (lastOpenMissionId === missionId) return lastOpenStartedAt;
  return peekRideDetailSnapshot(missionId)?.openedAt ?? null;
}

export function resetRideDetailSnapshotsForTests(): void {
  snapshots.clear();
  snapshotOrder.length = 0;
  lastOpenStartedAt = null;
  lastOpenMissionId = null;
}

/** Repli : mission déjà présente dans le cache des journées Courses. */
export function findMissionInDispatchCache(
  queryClient: QueryClient,
  missionId: number
): CompanyDispatchMission | null {
  const entries = queryClient.getQueriesData<CompanyDispatchMissionListResponse>({
    predicate: (query) => {
      const key = query.queryKey;
      return Array.isArray(key) && key.includes("dispatch") && key.includes("missions");
    },
  });
  for (const [, data] of entries) {
    const found = data?.missions?.find((row) => row.mission_id === missionId);
    if (found) return found;
  }
  return null;
}
