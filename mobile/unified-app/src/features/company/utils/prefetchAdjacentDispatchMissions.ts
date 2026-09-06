import type { QueryClient } from "@tanstack/react-query";
import { contextScopedKey } from "../../../core/cache/contextCache";
import { queryCacheOptions } from "../../../core/queryCachePolicy";
import { getDispatchMissions } from "../api/companyApi";
import { companyQueryKeys } from "../companyQueryKeys";

const ISO_DATE_RE = /^(\d{4})-(\d{2})-(\d{2})$/;

/** Arithmétique calendaire sur une date ISO (`YYYY-MM-DD`), sans glisser d’heure locale. */
export function shiftIsoDate(isoDate: string, deltaDays: number): string {
  const match = ISO_DATE_RE.exec(isoDate.trim());
  if (!match) return isoDate;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const utc = new Date(Date.UTC(year, month - 1, day + deltaDays));
  return utc.toISOString().slice(0, 10);
}

export function adjacentIsoDates(isoDate: string): readonly [string, string] {
  return [shiftIsoDate(isoDate, -1), shiftIsoDate(isoDate, 1)];
}

export function dispatchMissionsQueryKey(contextId: string, date: string) {
  return contextScopedKey(contextId, [...companyQueryKeys.missions(contextId, date)] as unknown[]);
}

/**
 * Prefetch J-1 / J+1 après succès de J : **page 1 seulement** (OPT-04C).
 * La complétion 2..N n’a lieu que lorsque cette date devient J ouvert.
 * React Query déduplique les clés identiques et respecte `staleTime`.
 */
export function prefetchAdjacentDispatchMissions(
  queryClient: QueryClient,
  contextId: string,
  date: string
): void {
  for (const neighbor of adjacentIsoDates(date)) {
    void queryClient.prefetchQuery({
      queryKey: dispatchMissionsQueryKey(contextId, neighbor),
      queryFn: () => getDispatchMissions({ contextId, date: neighbor }),
      ...queryCacheOptions("adjacent"),
    });
  }
}
