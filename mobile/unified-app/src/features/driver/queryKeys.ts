import { QueryClient } from "@tanstack/react-query";

export const driverQueryKeys = {
  missions: (contextId: string) => ["driver-missions", contextId] as const,
  missionDetail: (contextId: string, missionId: number) =>
    ["driver-mission-detail", contextId, missionId] as const,
  companyBookingsToday: (contextId: string) =>
    ["driver-company-bookings-today", contextId] as const,
  syncState: (contextId: string) => ["driver-sync-state", contextId] as const,
};

const MISSION_SCOPE_COALESCE_MS = 1500;
const lastMissionScopeInvalidateAt = new Map<string, number>();

function invalidateMissionScopeNow(
  queryClient: QueryClient,
  contextId: string,
  missionId?: number
): void {
  queryClient.invalidateQueries({ queryKey: driverQueryKeys.missions(contextId) });
  queryClient.invalidateQueries({
    queryKey: driverQueryKeys.companyBookingsToday(contextId),
  });
  queryClient.invalidateQueries({ queryKey: driverQueryKeys.syncState(contextId) });
  if (missionId) {
    queryClient.invalidateQueries({
      queryKey: driverQueryKeys.missionDetail(contextId, missionId),
    });
  }
}

/** DRIVER-RUNTIME-01B — une invalidation missions / company-bookings par fenêtre. */
export function invalidateDriverMissionScope(
  queryClient: QueryClient,
  contextId: string,
  missionId?: number
) {
  const now = Date.now();
  const last = lastMissionScopeInvalidateAt.get(contextId) ?? 0;
  if (now - last < MISSION_SCOPE_COALESCE_MS) {
    if (missionId) {
      queryClient.invalidateQueries({
        queryKey: driverQueryKeys.missionDetail(contextId, missionId),
      });
    }
    return;
  }
  lastMissionScopeInvalidateAt.set(contextId, now);
  invalidateMissionScopeNow(queryClient, contextId, missionId);
}

export function resetDriverMissionScopeInvalidationForTests(): void {
  lastMissionScopeInvalidateAt.clear();
}

// Backward-compatible alias while migrating call-sites.
export const invalidateDriverQueries = invalidateDriverMissionScope;

