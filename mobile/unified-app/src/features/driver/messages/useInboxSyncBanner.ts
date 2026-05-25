import type { UseQueryResult } from "@tanstack/react-query";
import { useDebouncedFlag } from "./useDebouncedFlag";
import type { MessageHubThread } from "./types";

type InboxQueryData = {
  threads: MessageHubThread[];
  unread_total: number;
  fromFallback?: boolean;
};

/**
 * Bannière « sync partielle » avec hystérésis : affichage retardé, masquage retardé.
 */
export function useInboxSyncBanner(
  query: UseQueryResult<InboxQueryData, Error>
): boolean {
  const wantsBanner =
    Boolean(query.data?.fromFallback) &&
    !query.isFetching &&
    !query.isLoading;

  const showBanner = useDebouncedFlag(wantsBanner, wantsBanner ? 1200 : 400);
  return showBanner;
}
