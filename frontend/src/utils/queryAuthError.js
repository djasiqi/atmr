/**
 * Couche transversale : masquer les erreurs auth transitoires dans React Query
 * et identifier les erreurs à ne jamais rejouer automatiquement.
 */

import { isAuthRefreshInProgress, isMissingTokenErrorPayload } from './apiClient';

export const AUTH_TOKEN_NOT_FRESH = 'AUTH_TOKEN_NOT_FRESH';
export const AUTH_SESSION_EXPIRED = 'AUTH_SESSION_EXPIRED';

export const isFreshTokenRequiredError = (error) =>
  Boolean(error?.isFreshTokenRequired || error?.code === AUTH_TOKEN_NOT_FRESH);

export const isSessionExpiredError = (error) => {
  if (!error) return false;
  if (error?.isSessionExpired || error?.code === AUTH_SESSION_EXPIRED) {
    return true;
  }
  return isMissingTokenErrorPayload(error?.response?.data);
};

export const isRecoverableAuthError = (error) => {
  if (!error) return false;
  if (isAuthRefreshInProgress()) return true;
  if (isFreshTokenRequiredError(error)) return true;
  if (isSessionExpiredError(error)) return true;
  const status = error?.response?.status ?? error?.status;
  if (status === 401 && !isFreshTokenRequiredError(error)) {
    return isAuthRefreshInProgress();
  }
  return false;
};

/**
 * Quota API dépassé (429 `too_many_requests`) : chaque tentative supplémentaire consomme
 * le quota et retarde le déblocage de la fenêtre glissante côté serveur.
 */
export const isRateLimitError = (error) => {
  if (!error) return false;
  const status = error?.response?.status ?? error?.status;
  if (status === 429) return true;
  return error?.response?.data?.error === 'too_many_requests';
};

export const shouldShowQueryError = (error) => {
  if (!error) return false;
  if (isRecoverableAuthError(error)) return false;
  if (isSessionExpiredError(error)) return false;
  if (error?.meta?.suppressAuthError) return false;
  return true;
};

export const getQueryErrorMessage = (error) => {
  if (!shouldShowQueryError(error)) return '';
  if (isFreshTokenRequiredError(error)) {
    return 'Votre session nécessite une reconnexion.';
  }
  return (
    error?.response?.data?.error ||
    error?.response?.data?.message ||
    error?.message ||
    'Une erreur est survenue.'
  );
};
