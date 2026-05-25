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

  dispatchMode: () => [LIRIE_QK_PREFIX, 'dispatch-mode'],

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
  ]);
}
