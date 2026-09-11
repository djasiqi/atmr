/**
 * États d'affichage de la page Réservations entreprise.
 * La table ne dépend pas des KPI / top clients.
 */

export const DEFAULT_COMPANY_RESERVATIONS_LIST_SCOPE = {
  selectedDay: 'all',
  currentPage: 1,
  reservationsPerPage: 25,
  statusFilter: 'all',
  activeTab: 'all',
  searchTerm: '',
  sortOrder: 'desc',
};

export function buildDefaultCompanyReservationsFetchArgs() {
  return {
    page: 1,
    perPage: 25,
    sortOrder: 'desc',
    excludeCanceled: true,
    includeStats: false,
    includeTotal: false,
    fields: 'table',
  };
}

export function shouldShowReservationsSkeleton({ listLoading, hasListData }) {
  return Boolean(listLoading) && !hasListData;
}

export function formatReservationsResultsLabel({ listLoading, hasListData, total }) {
  if (listLoading && !hasListData) return 'Chargement…';
  const n = Number(total) || 0;
  return `${n} resultat${n !== 1 ? 's' : ''}`;
}

export function resolveReservationsStatsView({
  statsFromQuery,
  statsFromList,
  statsLoading,
  listLoading,
  hasListData,
}) {
  if (statsFromQuery) {
    return { stats: statsFromQuery, loading: false };
  }
  if (statsFromList) {
    return { stats: statsFromList, loading: false };
  }
  if (statsLoading || (listLoading && !hasListData)) {
    return { stats: null, loading: true };
  }
  return { stats: null, loading: false };
}
