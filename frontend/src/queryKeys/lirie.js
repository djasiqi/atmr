/**
 * TanStack Query keys — convention LIRIE (préfixe lirie, hash stable des filtres).
 * @see frontend/docs/QUERY_KEYS.md
 */

export const LIRIE_QK_PREFIX = 'lirie';

function stableSerialize(scope) {
  const keys = Object.keys(scope).sort();
  const sorted = {};
  keys.forEach((k) => {
    const v = scope[k];
    if (v !== undefined && v !== '') sorted[k] = v;
  });
  return JSON.stringify(sorted);
}

function shortHash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) {
    h = (h << 5) - h + s.charCodeAt(i);
    h |= 0;
  }
  return `h${(h >>> 0).toString(16).padStart(8, '0')}`;
}

export function listScopeHash(scope) {
  return shortHash(stableSerialize(scope));
}

export function invoiceFiltersHash(scope) {
  return shortHash(stableSerialize(scope));
}

export const lirieKeys = {
  company: () => [LIRIE_QK_PREFIX, 'company'],

  companyReservations: (day, scopeHash) => [
    LIRIE_QK_PREFIX,
    'company-reservations',
    day,
    scopeHash,
  ],

  /**
   * Liste serveur paginée (page Réservations entreprise) — `listScopeHash` des filtres.
   */
  companyReservationsPaginated: (companyId, scope) => [
    LIRIE_QK_PREFIX,
    'company-reservations-paginated',
    String(companyId),
    listScopeHash(scope),
  ],

  companyReservationsSummary: (day) => [
    LIRIE_QK_PREFIX,
    'company-reservations-summary',
    day,
  ],

  /**
   * Stats agrégées (KPI) réservations entreprise — clé SANS numéro de page
   * (les agrégats API ne dépendent pas de la pagination, cf. Lot 4 perf :
   * on évite de refetcher les stats à chaque changement de page).
   */
  companyReservationsStats: (companyId, scope) => [
    LIRIE_QK_PREFIX,
    'company-reservations-stats',
    String(companyId),
    listScopeHash(scope),
  ],

  assignedReservations: (day) => [LIRIE_QK_PREFIX, 'assigned-reservations', day],

  companyDrivers: () => [LIRIE_QK_PREFIX, 'company-drivers'],

  /**
   * Liste clients / institutions (page Clients entreprise, max 1000 côté API).
   */
  companyClients: (companyId) => [LIRIE_QK_PREFIX, 'company-clients', String(companyId ?? 'me')],

  companyDriverLocations: () => [LIRIE_QK_PREFIX, 'company-driver-locations'],

  dispatchDelays: (day) => [LIRIE_QK_PREFIX, 'dispatch-delays', day],

  dispatchRealtimeDashboard: (day) => [
    LIRIE_QK_PREFIX,
    'dispatch-realtime-dashboard',
    day,
  ],

  institutionOffers: () => [LIRIE_QK_PREFIX, 'institution-offers'],

  /** Cloche header entreprise — inbox in-app (24 h + non lues). */
  companyNotifications: (companyId) => [
    LIRIE_QK_PREFIX,
    'company-notifications',
    String(companyId ?? 'me'),
  ],

  dispatchMode: () => [LIRIE_QK_PREFIX, 'dispatch-mode'],

  /**
   * Bootstrap dashboard entreprise (Lot 3 perf) — scopé auth env × entreprise × jour
   * pour éviter toute fuite cross-tenant si plusieurs sessions (app/demo) partagent
   * le même QueryClient (voir clearTenantScopedClientCaches).
   */
  companyDashboardBootstrap: (authEnv, companyId, day) => [
    LIRIE_QK_PREFIX,
    'company-dashboard-bootstrap',
    String(authEnv ?? 'app'),
    String(companyId ?? 'me'),
    day ?? '__today__',
  ],

  /** @deprecated Utiliser companyInvoices — conservé si des imports legacy pointent ici. */
  invoices: (filtersHash) => [LIRIE_QK_PREFIX, 'invoices', filtersHash],

  /**
   * Registre des factures entreprise (GET paginé + stats) — scoping par `companyId` + hash filtres.
   */
  companyInvoices: (companyId, filtersHash) => [
    LIRIE_QK_PREFIX,
    'company-invoices',
    String(companyId),
    filtersHash,
  ],

  scopedCompany: (companyId) => [LIRIE_QK_PREFIX, 'company', companyId],
};

/**
 * Invalide les listes réservations entreprise : vue journée (dispatch) + page Réservations paginée.
 */
export function lirieInvalidateCompanyReservationLists(queryClient) {
  if (!queryClient?.invalidateQueries) return Promise.resolve();
  return Promise.all([
    queryClient.invalidateQueries({ queryKey: [LIRIE_QK_PREFIX, 'company-reservations'], exact: false }),
    queryClient.invalidateQueries({ queryKey: [LIRIE_QK_PREFIX, 'company-reservations-paginated'], exact: false }),
    queryClient.invalidateQueries({ queryKey: [LIRIE_QK_PREFIX, 'company-reservations-stats'], exact: false }),
  ]);
}
