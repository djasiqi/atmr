import { QueryClient } from "@tanstack/react-query";

export const driverQueryKeys = {
  missions: (contextId: string) => ["driver-missions", contextId] as const,
  missionDetail: (contextId: string, missionId: number) =>
    ["driver-mission-detail", contextId, missionId] as const,
  companyBookingsToday: (contextId: string) =>
    ["driver-company-bookings-today", contextId] as const,
  syncState: (contextId: string) => ["driver-sync-state", contextId] as const,
};

export function invalidateDriverMissionScope(
  queryClient: QueryClient,
  contextId: string,
  missionId?: number
) {
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

// Backward-compatible alias while migrating call-sites.
export const invalidateDriverQueries = invalidateDriverMissionScope;

