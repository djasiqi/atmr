/**
 * Garantit un JWT access utilisable pour le handshake Socket.IO / appels sensibles.
 * Si `getAccessToken()` est null alors que la session UI est encore active
 * (app_user / cookies), tente un refresh avant d'abandonner.
 */

import { getAccessToken } from '../hooks/useAuthToken';
import { hasActiveSession } from './webAuthSession';

let inFlight = null;

/**
 * @returns {Promise<string|null>} access token utilisable, ou null
 */
export async function ensureUsableAccessToken() {
  const existing = getAccessToken();
  if (existing) {
    return existing;
  }
  if (!hasActiveSession()) {
    return null;
  }
  if (inFlight) {
    return inFlight;
  }
  inFlight = (async () => {
    try {
      const { refreshSessionTokens } = await import('./apiClient');
      await refreshSessionTokens();
    } catch {
      return null;
    }
    return getAccessToken();
  })().finally(() => {
    inFlight = null;
  });
  return inFlight;
}

/** Réinitialise l'état interne (tests uniquement). */
export function resetEnsureUsableAccessTokenForTests() {
  inFlight = null;
}
