import { useCallback, useEffect, useRef, useState } from "react";
import type { QueryClient } from "@tanstack/react-query";
import type { CompanyDispatchMissionListResponse } from "../api/contracts";
import { getDispatchMissions } from "../api/companyApi";
import { dispatchMissionsQueryKey } from "./prefetchAdjacentDispatchMissions";
import {
  applyDayPage,
  clearDayPaginationError,
  markDayPaginationError,
  shouldFetchNextDayPage,
} from "./dispatchDayPagination";

type UseCompleteOpenDispatchDayArgs = {
  enabled: boolean;
  contextId: string | null;
  date: string;
  queryClient: QueryClient;
  isQueryFetching: boolean;
};

/**
 * Complète J ouvert page après page (séquentiel). J±1 n’active pas ce hook.
 * Une réponse tardive est ignorée si la date / la génération a changé.
 */
export function useCompleteOpenDispatchDay({
  enabled,
  contextId,
  date,
  queryClient,
  isQueryFetching,
}: UseCompleteOpenDispatchDayArgs): { retryDayPagination: () => void } {
  const generationRef = useRef(0);
  const [retryNonce, setRetryNonce] = useState(0);

  const retryDayPagination = useCallback(() => {
    if (!contextId) return;
    queryClient.setQueryData<CompanyDispatchMissionListResponse>(
      dispatchMissionsQueryKey(contextId, date),
      (current) => (current ? clearDayPaginationError(current) : current)
    );
    setRetryNonce((value) => value + 1);
  }, [contextId, date, queryClient]);

  useEffect(() => {
    if (!enabled || !contextId || isQueryFetching) return undefined;
    const queryKey = dispatchMissionsQueryKey(contextId, date);
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    let cancelled = false;

    const run = async () => {
      while (!cancelled && generation === generationRef.current) {
        const current = queryClient.getQueryData<CompanyDispatchMissionListResponse>(queryKey);
        if (!current || !shouldFetchNextDayPage(current, true)) break;
        const page = current.next_page;
        try {
          const incoming = await getDispatchMissions({
            contextId,
            date,
            page,
            fetchReason: "pagination",
          });
          if (cancelled || generation !== generationRef.current) return;
          if (incoming.date && incoming.date !== date) return;
          queryClient.setQueryData<CompanyDispatchMissionListResponse>(queryKey, (previous) =>
            applyDayPage(previous, incoming)
          );
        } catch {
          if (cancelled || generation !== generationRef.current) return;
          queryClient.setQueryData<CompanyDispatchMissionListResponse>(queryKey, (previous) =>
            previous ? markDayPaginationError(previous) : previous
          );
          break;
        }
      }
    };

    void run();
    return () => {
      cancelled = true;
      generationRef.current += 1;
    };
  }, [contextId, date, enabled, isQueryFetching, queryClient, retryNonce]);

  return { retryDayPagination };
}
