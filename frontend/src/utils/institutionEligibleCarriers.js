/**
 * États de la modale « Ajouter un transporteur ».
 */

export const ELIGIBLE_CARRIERS_TIMEOUT_MS = 12_000;
export const ELIGIBLE_CARRIERS_STALE_MS = 5 * 60 * 1000;

export const ELIGIBLE_CARRIERS_COPY = {
  loading: 'Chargement des transporteurs…',
  empty: 'Aucun transporteur compatible disponible.',
  searchEmpty: 'Aucune entreprise ne correspond à votre recherche.',
  alreadyAdded: 'Tous les transporteurs compatibles sont déjà dans votre liste.',
  error: 'Impossible de charger les transporteurs.',
  forbidden: 'Vous n’avez pas l’autorisation de consulter les transporteurs.',
  retry: 'Réessayer',
};

export function resolveEligibleCarriersView({
  isPending,
  isError,
  error,
  availableCount = 0,
  notYetAddedCount = 0,
  catalogTotal = 0,
  search = '',
}) {
  if (isPending) return { kind: 'loading' };
  if (isError) {
    const status = error?.response?.status;
    if (status === 403) return { kind: 'forbidden' };
    return { kind: 'error' };
  }
  if (search.trim() && availableCount === 0) return { kind: 'search_empty' };
  if (notYetAddedCount === 0) {
    return { kind: catalogTotal > 0 ? 'already_added' : 'empty' };
  }
  return { kind: 'success' };
}

export function shouldRetryEligibleCarriers(failureCount, error) {
  const status = error?.response?.status;
  if (status === 401 || status === 403) return false;
  if (error?.code === 'ERR_CANCELED' || error?.name === 'CanceledError') return false;
  return failureCount < 1;
}
