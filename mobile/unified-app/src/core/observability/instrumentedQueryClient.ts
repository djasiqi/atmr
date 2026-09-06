import {
  QueryClient,
  QueryCache,
  MutationCache,
  type InvalidateQueryFilters,
  type InvalidateOptions,
} from "@tanstack/react-query";
import { QUERY_STALE_TIME_MS } from "../queryStaleTimes";
import { DEFAULT_QUERY_GC_TIME_MS } from "../queryCachePolicy";
import { recordReactQueryRefetch } from "./perfKpi";
import { traceInvalidateQueries } from "./perfInstrumentation";

export function createInstrumentedQueryClient(): QueryClient {
  const queryCache = new QueryCache({
    onSuccess: (_data, query) => {
      if (query.state.fetchStatus === "fetching" && query.state.dataUpdateCount > 1) {
        recordReactQueryRefetch(query.queryKey as unknown[], "query_success_refetch");
      }
    },
  });

  const mutationCache = new MutationCache();

  const client = new QueryClient({
    queryCache,
    mutationCache,
    defaultOptions: {
      queries: {
        staleTime: QUERY_STALE_TIME_MS.default,
        gcTime: DEFAULT_QUERY_GC_TIME_MS,
        retry: 1,
        refetchOnMount: "ifStale",
      },
      mutations: {
        retry: 1,
      },
    },
  });

  const originalInvalidate = client.invalidateQueries.bind(client);
  client.invalidateQueries = (
    filters?: InvalidateQueryFilters,
    options?: InvalidateOptions
  ) => {
    const queryKey =
      filters && typeof filters === "object" && "queryKey" in filters
        ? (filters as { queryKey?: unknown }).queryKey
        : filters;
    return traceInvalidateQueries(queryKey ?? "all", "query_client", () =>
      originalInvalidate(filters, options)
    );
  };

  return client;
}

/** Wrap refetch to attribute trigger for perf KPI. */
export function tracedRefetch(
  refetch: () => Promise<unknown>,
  queryKey: unknown[],
  trigger: string
): Promise<unknown> {
  recordReactQueryRefetch(queryKey, trigger);
  return refetch();
}
