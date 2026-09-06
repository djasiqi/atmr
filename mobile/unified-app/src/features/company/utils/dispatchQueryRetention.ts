import { useEffect } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import { emitPerfKpi } from "../../../core/observability/perfKpi";
import { adjacentIsoDates } from "./prefetchAdjacentDispatchMissions";
import {
  dateFromMissionsQueryKey,
  isDispatchMissionsQueryKey,
} from "./dispatchMissionCachePatch";

/** J + J±1 sont toujours pinés. Au-delà : 2 dates LRU récemment vues. */
export const EXTRA_RECENT_DISPATCH_DAYS = 2;

/**
 * Filet ride-details : un DTO détail ≈ résumé liste (quelques Ko).
 * 40 inactifs ≈ 100–160 Ko — au-delà on évacue les plus anciens non observés.
 * Pas un « max 5 » arbitraire.
 */
export const MAX_INACTIVE_RIDE_DETAILS = 40;

export type DispatchCacheSurface = "rides" | "cockpit";

const surfaceDates = new Map<DispatchCacheSurface, string>();
const recentDates: string[] = [];

export function resetDispatchQueryRetentionForTests(): void {
  surfaceDates.clear();
  recentDates.length = 0;
}

export function rememberVisitedDispatchDate(date: string): void {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return;
  const index = recentDates.indexOf(date);
  if (index >= 0) recentDates.splice(index, 1);
  recentDates.unshift(date);
  if (recentDates.length > 16) recentDates.length = 16;
}

export function registerDispatchSurfaceDate(surface: DispatchCacheSurface, date: string): void {
  surfaceDates.set(surface, date);
  rememberVisitedDispatchDate(date);
}

export function pinnedDispatchDates(): string[] {
  return [...new Set(surfaceDates.values())];
}

export function datesToRetain(activeDates: readonly string[]): Set<string> {
  const keep = new Set<string>();
  for (const date of activeDates) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue;
    keep.add(date);
    const [prev, next] = adjacentIsoDates(date);
    keep.add(prev);
    keep.add(next);
  }
  let extras = 0;
  for (const date of recentDates) {
    if (keep.has(date)) continue;
    keep.add(date);
    extras += 1;
    if (extras >= EXTRA_RECENT_DISPATCH_DAYS) break;
  }
  return keep;
}

function isIsoDateKey(key: unknown): string | null {
  return dateFromMissionsQueryKey(key);
}

function isRideDetailsQueryKey(key: unknown, contextId: string): boolean {
  return (
    Array.isArray(key) &&
    key[0] === "ctx" &&
    key[1] === contextId &&
    key.includes("ride-details")
  );
}

function isDateScopedDispatchKey(key: unknown, contextId: string, kind: "dashboard" | "dispatch-delays"): boolean {
  return (
    Array.isArray(key) &&
    key[0] === "ctx" &&
    key[1] === contextId &&
    key.includes(kind) &&
    isIsoDateKey(key) != null
  );
}

export type DispatchCacheInventory = {
  mission_days: number;
  ride_details: number;
  dashboards: number;
  delay_days: number;
  retained_dates: string[];
  removed_dates: string[];
};

export function inventoryDispatchQueryCache(
  queryClient: QueryClient,
  contextId: string
): Omit<DispatchCacheInventory, "retained_dates" | "removed_dates"> {
  let missionDays = 0;
  let rideDetails = 0;
  let dashboards = 0;
  let delayDays = 0;
  for (const query of queryClient.getQueryCache().getAll()) {
    const key = query.queryKey;
    if (isDispatchMissionsQueryKey(key, contextId) && isIsoDateKey(key)) missionDays += 1;
    else if (isRideDetailsQueryKey(key, contextId)) rideDetails += 1;
    else if (isDateScopedDispatchKey(key, contextId, "dashboard")) dashboards += 1;
    else if (isDateScopedDispatchKey(key, contextId, "dispatch-delays")) delayDays += 1;
  }
  return {
    mission_days: missionDays,
    ride_details: rideDetails,
    dashboards,
    delay_days: delayDays,
  };
}

/**
 * Évacue les journées hors fenêtre utile. Ne touche jamais une query observée
 * (J ouvert / Cockpit ouvert). gcTime OPT-05 inchangé.
 */
export function pruneDispatchQueryCache(
  queryClient: QueryClient,
  contextId: string,
  activeDates: readonly string[] = pinnedDispatchDates()
): DispatchCacheInventory {
  const keep = datesToRetain(activeDates);
  const removedDates: string[] = [];

  queryClient.removeQueries({
    predicate: (query) => {
      if (query.getObserversCount() > 0) return false;
      const key = query.queryKey;
      const date = isIsoDateKey(key);
      if (!date) return false;
      const dayScoped =
        isDispatchMissionsQueryKey(key, contextId) ||
        isDateScopedDispatchKey(key, contextId, "dashboard") ||
        isDateScopedDispatchKey(key, contextId, "dispatch-delays");
      if (!dayScoped || keep.has(date)) return false;
      if (!removedDates.includes(date)) removedDates.push(date);
      return true;
    },
  });

  const inactiveDetails = queryClient.getQueryCache().getAll().filter((query) => {
    return (
      isRideDetailsQueryKey(query.queryKey, contextId) && query.getObserversCount() === 0
    );
  });
  if (inactiveDetails.length > MAX_INACTIVE_RIDE_DETAILS) {
    const overflow = inactiveDetails
      .slice()
      .sort((left, right) => left.state.dataUpdatedAt - right.state.dataUpdatedAt)
      .slice(0, inactiveDetails.length - MAX_INACTIVE_RIDE_DETAILS);
    for (const query of overflow) {
      queryClient.removeQueries({ queryKey: query.queryKey, exact: true });
    }
  }

  const counts = inventoryDispatchQueryCache(queryClient, contextId);
  const inventory: DispatchCacheInventory = {
    ...counts,
    retained_dates: [...keep].sort(),
    removed_dates: removedDates.sort(),
  };
  if (typeof __DEV__ !== "undefined" && __DEV__ && removedDates.length > 0) {
    emitPerfKpi("perf.context.cache_snapshot", {
      source: "dispatch.retention",
      cached_query_count: counts.mission_days + counts.ride_details + counts.dashboards + counts.delay_days,
      mission_days: counts.mission_days,
      ride_details: counts.ride_details,
      dashboards: counts.dashboards,
      delay_days: counts.delay_days,
      removed_day_count: removedDates.length,
    });
  }
  return inventory;
}

export function useRetainDispatchQueryCache(
  surface: DispatchCacheSurface,
  contextId: string | null,
  date: string
): void {
  const queryClient = useQueryClient();
  useEffect(() => {
    if (!contextId || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return;
    registerDispatchSurfaceDate(surface, date);
    pruneDispatchQueryCache(queryClient, contextId);
  }, [contextId, date, queryClient, surface]);
}
