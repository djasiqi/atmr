/**
 * DRIVER-RUNTIME-01C-B — snapshot missions asymétrique.
 * positive = 1 preuve suffit ; null = quorum post-READY obligatoire.
 */

export type MissionSnapshotQuerySlice = {
  networkReady: boolean;
  status: "pending" | "error" | "success";
  fetchStatus: "fetching" | "paused" | "idle";
  dataUpdatedAt: number;
  networkReadyAtMs: number;
};

export type DriverMissionSnapshot =
  | { status: "pending" }
  | { status: "resolved_mission"; missionId: number }
  | { status: "resolved_none" };

export type MissionSnapshotSource = {
  id: string;
  settledPostReady: boolean;
  missionId: number | null;
};

/** Une source est settled seulement après un fetch terminé post-SESSION_READY. */
export function isMissionSourceSettledPostReady(input: MissionSnapshotQuerySlice): boolean {
  if (!input.networkReady) return false;
  if (input.networkReadyAtMs <= 0) return false;
  if (input.status === "pending") return false;
  if (input.fetchStatus === "fetching") return false;
  if (input.dataUpdatedAt < input.networkReadyAtMs) return false;
  return input.status === "success" || input.status === "error";
}

/**
 * @deprecated Préférer `resolveDriverMissionSnapshot`. Conservé pour les tests
 * qui ne vérifient que le settled d’une source.
 */
export function resolveMissionSnapshotReady(input: MissionSnapshotQuerySlice): boolean {
  return isMissionSourceSettledPostReady(input);
}

export function resolveDriverMissionSnapshot(input: {
  networkReady: boolean;
  networkReadyGeneration: number;
  sources: MissionSnapshotSource[];
}): DriverMissionSnapshot {
  if (!input.networkReady || input.networkReadyGeneration <= 0) {
    return { status: "pending" };
  }
  for (const source of input.sources) {
    if (source.missionId != null && source.missionId > 0) {
      return { status: "resolved_mission", missionId: source.missionId };
    }
  }
  if (input.sources.length === 0) return { status: "pending" };
  if (input.sources.every((source) => source.settledPostReady)) {
    return { status: "resolved_none" };
  }
  return { status: "pending" };
}
