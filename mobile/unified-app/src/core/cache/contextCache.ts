import { QueryClient } from "@tanstack/react-query";
import { isFeatureEnabled } from "../featureFlags/registry";
import {
  countCtxScopedQueries,
  emitContextCacheSnapshot,
} from "../observability/perfKpi";

export function contextScopedKey(contextId: string | null, baseKey: unknown[]) {
  return ["ctx", contextId ?? "public", ...baseKey];
}

const MAX_PARKED_CONTEXTS = 5;
const parkedContextOrder: string[] = [];
/** Clés de queries marquées parkées (compatible toutes versions React Query). */
const parkedQueryKeyStrings = new Set<string>();

function queryKeyString(key: unknown): string {
  return JSON.stringify(key);
}

function isCtxScopedKey(key: unknown, contextId: string): boolean {
  return Array.isArray(key) && key[0] === "ctx" && key[1] === contextId;
}

function countParkedQueries(queryClient: QueryClient): number {
  return queryClient
    .getQueryCache()
    .getAll()
    .filter((q) => parkedQueryKeyStrings.has(queryKeyString(q.queryKey))).length;
}

function registerParkedContext(contextId: string): string | null {
  const idx = parkedContextOrder.indexOf(contextId);
  if (idx >= 0) parkedContextOrder.splice(idx, 1);
  parkedContextOrder.push(contextId);
  if (parkedContextOrder.length > MAX_PARKED_CONTEXTS) {
    return parkedContextOrder.shift() ?? null;
  }
  return null;
}

export function clearContextScopedCache(queryClient: QueryClient, contextId: string | null) {
  if (!contextId) return;
  queryClient.removeQueries({
    predicate: (query) => {
      if (!isCtxScopedKey(query.queryKey, contextId)) return false;
      parkedQueryKeyStrings.delete(queryKeyString(query.queryKey));
      return true;
    },
  });
  const idx = parkedContextOrder.indexOf(contextId);
  if (idx >= 0) parkedContextOrder.splice(idx, 1);
}

export function clearAllContextCache(queryClient: QueryClient) {
  queryClient.removeQueries({
    predicate: (query) => Array.isArray(query.queryKey) && query.queryKey[0] === "ctx",
  });
  parkedContextOrder.length = 0;
}

/** Marque les queries du contexte comme parkées (LRU max 2) au lieu de les supprimer. */
export function parkContextCache(queryClient: QueryClient, contextId: string | null) {
  if (!contextId) return;
  queryClient.getQueryCache().getAll().forEach((query) => {
    if (!isCtxScopedKey(query.queryKey, contextId)) return;
    parkedQueryKeyStrings.add(queryKeyString(query.queryKey));
  });
  const evictId = registerParkedContext(contextId);
  if (evictId) {
    clearContextScopedCache(queryClient, evictId);
  }
  emitContextCacheSnapshot({
    source: "contextCache.park",
    cached_context_count: parkedContextOrder.length,
    parked_query_count: countParkedQueries(queryClient),
    cached_query_count: countCtxScopedQueries(queryClient),
  });
}

export function hasRestorableContextCache(
  queryClient: QueryClient,
  contextId: string | null
): boolean {
  if (!contextId) return false;
  return queryClient.getQueryCache().getAll().some((query) => {
    if (!isCtxScopedKey(query.queryKey, contextId)) return false;
    return query.state.data !== undefined;
  });
}

export function restoreContextCache(queryClient: QueryClient, contextId: string | null): boolean {
  const cacheHit = hasRestorableContextCache(queryClient, contextId);
  if (contextId && parkedContextOrder.includes(contextId)) {
    registerParkedContext(contextId);
  }
  emitContextCacheSnapshot({
    source: "contextCache.restore",
    cached_context_count: parkedContextOrder.length,
    parked_query_count: countParkedQueries(queryClient),
    cached_query_count: countCtxScopedQueries(queryClient),
    cache_hit: cacheHit,
  });
  return cacheHit;
}

/** Nombre de contextes actuellement dans le LRU (tests + telemetry). */
export function getParkedContextCount(): number {
  return parkedContextOrder.length;
}

export function applyContextCachePolicyOnSwitch(
  queryClient: QueryClient,
  previousContextId: string | null
) {
  if (!previousContextId) return;
  if (isFeatureEnabled("mobile_context_cache_parking_enabled")) {
    parkContextCache(queryClient, previousContextId);
    return;
  }
  clearContextScopedCache(queryClient, previousContextId);
}
