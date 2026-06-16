/**
 * Couche transversale : masquer les erreurs auth transitoires dans React Query.
 */

import { isAuthRefreshInProgress } from './apiClient';

export const AUTH_TOKEN_NOT_FRESH = 'AUTH_TOKEN_NOT_FRESH';
export const AUTH_SESSION_EXPIRED = 'AUTH_SESSION_EXPIRED';

export const isFreshTokenRequiredError = (error) =>
  Boolean(error?.isFreshTokenRequired || error?.code === AUTH_TOKEN_NOT_FRESH);

export const isRecoverableAuthError = (error) => {
  if (!error) return false;
  if (isAuthRefreshInProgress()) return true;
  if (isFreshTokenRequiredError(error)) return true;
  const status = error?.response?.status ?? error?.status;
  if (status === 401 && !isFreshTokenRequiredError(error)) {
    return isAuthRefreshInProgress();
  }
  return false;
};

export const shouldShowQueryError = (error) => {
  if (!error) return false;
  if (isRecoverableAuthError(error)) return false;
  if (error?.meta?.suppressAuthError) return false;
  return true;
};

export const getQueryErrorMessage = (error) => {
  if (!shouldShowQueryError(error)) return '';
  if (isFreshTokenRequiredError(error)) {
    return 'Cette action nécessite une confirmation de mot de passe.';
  }
  return (
    error?.response?.data?.error ||
    error?.response?.data?.message ||
    error?.message ||
    'Une erreur est survenue.'
  );
};
