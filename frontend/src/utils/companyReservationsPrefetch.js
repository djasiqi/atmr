import { lirieKeys } from '../queryKeys/lirie';
import { fetchCompanyReservationsPaginated } from '../services/companyService';
import {
  DEFAULT_COMPANY_RESERVATIONS_LIST_SCOPE,
  buildDefaultCompanyReservationsFetchArgs,
} from './companyReservationsPage';
import { getAccessToken } from '../hooks/useAuthToken';
import { getActiveUser } from './webAuthSession';

export function canPrefetchCompanyReservations() {
  return Boolean(getAccessToken() || getActiveUser());
}

const PREFETCH_STALE_MS = 60_000;

export function shouldSkipCompanyReservationsPrefetch(queryState, now = Date.now()) {
  if (!queryState) return false;
  if (queryState.fetchStatus === 'fetching') return true;
  if (
    queryState.data
    && queryState.dataUpdatedAt
    && (now - queryState.dataUpdatedAt) < PREFETCH_STALE_MS
  ) {
    return true;
  }
  return false;
}

/** Démarre le fetch table (page 1, sans stats) dès que le token est connu. */
export function prefetchCompanyReservationsList(queryClient) {
  if (!queryClient?.prefetchQuery || !canPrefetchCompanyReservations()) return Promise.resolve();
  const queryKey = lirieKeys.companyReservationsPaginated(
    'me',
    DEFAULT_COMPANY_RESERVATIONS_LIST_SCOPE,
  );
  const state = queryClient.getQueryState?.(queryKey);
  if (shouldSkipCompanyReservationsPrefetch(state)) return Promise.resolve();
  return queryClient.prefetchQuery({
    queryKey,
    queryFn: () => fetchCompanyReservationsPaginated(buildDefaultCompanyReservationsFetchArgs()),
    staleTime: PREFETCH_STALE_MS,
  });
}
