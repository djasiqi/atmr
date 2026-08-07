/**
 * Garantit une session web prête pour Socket.IO / appels sensibles.
 * En cookie-only HttpOnly : un refresh HTTP 200 suffit (pas de JWT en JS).
 */

import { getAccessToken } from '../hooks/useAuthToken';
import { hasActiveSession } from './webAuthSession';

let inFlight = null;

/**
 * Refresh si besoin ; succès cookie-only = true même sans JWT lisible en JS.
 * @returns {Promise<boolean>}
 */
export async function ensureWebAuthReady() {
  if (getAccessToken()) {
    return true;
  }
  if (!hasActiveSession()) {
    return false;
  }
  if (inFlight) {
    return inFlight;
  }
  inFlight = (async () => {
    try {
      const { refreshSessionTokens } = await import('./apiClient');
      await refreshSessionTokens();
      return true;
    } catch (error) {
      const status = error?.response?.status;
      const terminal = status === 401 || status === 400 || status === 403;
      if (terminal) {
        try {
          const { expireCurrentWebSession } = await import('./apiClient');
          expireCurrentWebSession({ reason: 'session_expired' });
        } catch (_) {
          // ignore
        }
      }
      return false;
    }
  })().finally(() => {
    inFlight = null;
  });
  return inFlight;
}

/**
 * @returns {Promise<string|null>} JWT JS si présent (mobile/legacy), sinon null après refresh cookie-only OK
 */
export async function ensureUsableAccessToken() {
  const existing = getAccessToken();
  if (existing) {
    return existing;
  }
  const ready = await ensureWebAuthReady();
  if (!ready) {
    return null;
  }
  return getAccessToken();
}

/** Réinitialise l'état interne (tests uniquement). */
export function resetEnsureUsableAccessTokenForTests() {
  inFlight = null;
}
