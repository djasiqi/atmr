/**
 * OPT-05 — politique de cache React Query par nature de donnée.
 * staleTime = faut-il rafraîchir ?  gcTime = faut-il jeter le cache ?
 * Les secondes viennent de l’inventaire LIRIE (pas d’une valeur globale).
 */

export { QUERY_STALE_TIME_MS } from "./queryStaleTimes";

export type QueryCacheFamily =
  | "realtime"
  | "operational"
  | "adjacent"
  | "historical"
  | "detail"
  | "referential";

export type QueryCachePolicy = {
  staleTime: number;
  gcTime: number;
  refetchOnWindowFocus: boolean;
  refetchOnReconnect: boolean;
};

/**
 * Familles (audit OPT-04A + 04E) :
 * - realtime 10 s : présence / live — le socket GPS reste la source, RQ = filet
 * - operational 2 min : J / dashboard — les patchs 04E tiennent la fraîcheur
 * - adjacent 10 min : J±1 prefetch — ne pas perdre le cache à 30 s
 * - historical 15 min : journées hors voisinage
 * - detail 30 s : ride-details (verrouillé produit)
 * - referential 10 min : clients / config / factures
 */
export const QUERY_CACHE_POLICY: Record<QueryCacheFamily, QueryCachePolicy> = {
  realtime: {
    staleTime: 10_000,
    gcTime: 5 * 60_000,
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
  },
  operational: {
    staleTime: 2 * 60_000,
    gcTime: 30 * 60_000,
    refetchOnWindowFocus: true,
    refetchOnReconnect: false,
  },
  adjacent: {
    staleTime: 10 * 60_000,
    gcTime: 30 * 60_000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  },
  historical: {
    staleTime: 15 * 60_000,
    gcTime: 30 * 60_000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  },
  detail: {
    staleTime: 30_000,
    gcTime: 15 * 60_000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  },
  referential: {
    staleTime: 10 * 60_000,
    gcTime: 60 * 60_000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  },
};

/** gcTime défaut QueryClient : garder le cache pour une navigation rapide (≠ staleTime). */
export const DEFAULT_QUERY_GC_TIME_MS = 15 * 60_000;

const ISO_DATE_RE = /^(\d{4})-(\d{2})-(\d{2})$/;

export function todayIsoDate(now = new Date()): string {
  return now.toISOString().slice(0, 10);
}

export function isoDateDayDelta(date: string, today: string): number | null {
  if (!ISO_DATE_RE.test(date) || !ISO_DATE_RE.test(today)) return null;
  const left = Date.parse(`${date}T00:00:00.000Z`);
  const right = Date.parse(`${today}T00:00:00.000Z`);
  if (!Number.isFinite(left) || !Number.isFinite(right)) return null;
  return Math.round((left - right) / 86_400_000);
}

export function classifyDispatchDay(
  date: string,
  options?: { completeDay?: boolean; today?: string }
): QueryCacheFamily {
  if (options?.completeDay) return "operational";
  const delta = isoDateDayDelta(date, options?.today ?? todayIsoDate());
  if (delta === 0) return "operational";
  if (delta === 1 || delta === -1) return "adjacent";
  return "historical";
}

export function queryCacheOptions(family: QueryCacheFamily): QueryCachePolicy {
  return QUERY_CACHE_POLICY[family];
}

export function dispatchDayCacheOptions(
  date: string,
  options?: { completeDay?: boolean; today?: string }
): QueryCachePolicy {
  return queryCacheOptions(classifyDispatchDay(date, options));
}
