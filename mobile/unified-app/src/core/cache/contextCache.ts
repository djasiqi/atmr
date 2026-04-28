import { QueryClient } from "@tanstack/react-query";

export function contextScopedKey(contextId: string | null, baseKey: unknown[]) {
  return ["ctx", contextId ?? "public", ...baseKey];
}

export function clearContextScopedCache(queryClient: QueryClient, contextId: string | null) {
  queryClient.removeQueries({
    predicate: (query) => {
      const key = query.queryKey;
      return Array.isArray(key) && key[0] === "ctx" && key[1] === (contextId ?? "public");
    },
  });
}

export function clearAllContextCache(queryClient: QueryClient) {
  queryClient.removeQueries({
    predicate: (query) => Array.isArray(query.queryKey) && query.queryKey[0] === "ctx",
  });
}
