import {
  shouldShowReservationsSkeleton,
  formatReservationsResultsLabel,
  resolveReservationsStatsView,
  DEFAULT_COMPANY_RESERVATIONS_LIST_SCOPE,
  buildDefaultCompanyReservationsFetchArgs,
} from '../companyReservationsPage';

describe('shouldShowReservationsSkeleton', () => {
  it('n’affiche pas le skeleton si le cache a déjà des lignes', () => {
    expect(shouldShowReservationsSkeleton({
      listLoading: true,
      hasListData: true,
    })).toBe(false);
  });

  it('affiche le skeleton seulement au premier chargement sans données', () => {
    expect(shouldShowReservationsSkeleton({
      listLoading: true,
      hasListData: false,
    })).toBe(true);
  });
});

describe('formatReservationsResultsLabel', () => {
  it('évite le faux « 0 résultats » pendant le chargement', () => {
    expect(formatReservationsResultsLabel({
      listLoading: true,
      hasListData: false,
      total: 0,
    })).toBe('Chargement…');
  });

  it('affiche le total réel une fois chargé', () => {
    expect(formatReservationsResultsLabel({
      listLoading: false,
      hasListData: true,
      total: 301,
    })).toBe('301 resultats');
  });
});

describe('resolveReservationsStatsView', () => {
  it('reste en chargement si les KPI ne sont pas prêts', () => {
    expect(resolveReservationsStatsView({
      statsFromQuery: null,
      statsFromList: null,
      statsLoading: true,
      listLoading: false,
      hasListData: true,
    })).toEqual({ stats: null, loading: true });
  });

  it('prend les stats dédiées dès qu’elles arrivent', () => {
    const stats = { total: 10, pending: 1, inProgress: 2, completed: 3, canceled: 0, revenue: 40 };
    expect(resolveReservationsStatsView({
      statsFromQuery: stats,
      statsFromList: null,
      statsLoading: false,
      listLoading: false,
      hasListData: true,
    })).toEqual({ stats, loading: false });
  });
});

describe('DEFAULT_COMPANY_RESERVATIONS_LIST_SCOPE', () => {
  it('correspond au premier écran (toutes, page 1, sans stats)', () => {
    expect(DEFAULT_COMPANY_RESERVATIONS_LIST_SCOPE).toEqual({
      selectedDay: 'all',
      currentPage: 1,
      reservationsPerPage: 25,
      statusFilter: 'all',
      activeTab: 'all',
      searchTerm: '',
      sortOrder: 'desc',
    });
    expect(buildDefaultCompanyReservationsFetchArgs()).toEqual({
      page: 1,
      perPage: 25,
      sortOrder: 'desc',
      excludeCanceled: true,
      includeStats: false,
      includeTotal: false,
      fields: 'table',
    });
  });
});
