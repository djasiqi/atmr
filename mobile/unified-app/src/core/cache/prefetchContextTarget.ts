import { QueryClient } from "@tanstack/react-query";
import type { AuthContext } from "../contracts/auth";
import { contextScopedKey } from "./contextCache";
import { QUERY_STALE_TIME_MS } from "../queryStaleTimes";
import { queryCacheOptions } from "../queryCachePolicy";
import { driverQueryKeys } from "../../features/driver/queryKeys";
import { getDriverMissions } from "../../features/driver/api/driverHttp";
import { fetchHubUnreadCount } from "../../features/driver/messages/api";
import { companyQueryKeys } from "../../features/company/companyQueryKeys";
import {
  getDispatchMissions,
  getDriversLocationsSnapshot,
  getRealtimeDashboard,
} from "../../features/company/api/companyApi";

function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

/** Prefetch minimal avant switchContext API (non bloquant). */
export function prefetchContextTarget(queryClient: QueryClient, target: AuthContext): void {
  if (target.context_type === "driver" && target.context_id) {
    const companyId = target.company_id;
    void queryClient.prefetchQuery({
      queryKey: driverQueryKeys.missions(target.context_id),
      queryFn: getDriverMissions,
      staleTime: QUERY_STALE_TIME_MS.default,
    });
    if (typeof companyId === "number" && Number.isFinite(companyId)) {
      void queryClient.prefetchQuery({
        queryKey: ["driver", "message-hub", "unread", companyId],
        queryFn: () => fetchHubUnreadCount(companyId),
        staleTime: QUERY_STALE_TIME_MS.default,
      });
    }
    return;
  }

  if (target.context_type === "company" && target.context_id) {
    const date = todayIsoDate();
    void queryClient.prefetchQuery({
      queryKey: contextScopedKey(target.context_id, [
        ...companyQueryKeys.dashboard(target.context_id),
        date,
      ] as unknown[]),
      queryFn: () => getRealtimeDashboard({ contextId: target.context_id, date }),
      ...queryCacheOptions("operational"),
    });
    void queryClient.prefetchQuery({
      queryKey: contextScopedKey(
        target.context_id,
        [...companyQueryKeys.missions(target.context_id, date)] as unknown[]
      ),
      queryFn: () =>
        getDispatchMissions({
          contextId: target.context_id,
          date,
        }),
      ...queryCacheOptions("operational"),
    });
    // OPT-07A : unread / delays / optimizer hors chemin critique (après premier écran).
    void queryClient.prefetchQuery({
      queryKey: contextScopedKey(target.context_id, [
        ...companyQueryKeys.driversLocations(target.context_id),
      ] as unknown[]),
      queryFn: () => getDriversLocationsSnapshot({ contextId: target.context_id }),
      ...queryCacheOptions("realtime"),
    });
  }
}
