/**
 * Renouvellement proactif du JWT tant que l'utilisateur travaille.
 * Évite l'expiration à 1 h pendant une session active (clics, scroll, navigation…).
 */

import { refreshSessionTokens } from './apiClient';
import { isSessionIdleWarningActive } from './deferredSessionLogout';
import { isExplicitLogoutInProgress, isLoginSessionInProgress } from './sessionLogoutState';
import { hasActiveSession } from './webAuthSession';
import {
  isUserRecentlyActive,
  onUserActivity,
  SESSION_WORKING_LOOKBACK_MS,
} from './userActivityTracker';

/** Intervalle entre deux tentatives de refresh (access token ≈ 1 h). */
export const SESSION_KEEPALIVE_INTERVAL_MS = 45 * 60 * 1000;

/**
 * Écart minimum entre deux refresh réussis non forcés.
 * Doit rester proche de l'intervalle keep-alive : un refresh trop tôt après
 * login remplace le JWT « fresh » (fresh=True) par un access token non-fresh,
 * ce qui casse les actions protégées (ex. PUT /companies/me).
 */
export const MIN_REFRESH_GAP_MS = SESSION_KEEPALIVE_INTERVAL_MS;

/** Backoff après un échec transitoire (réseau / 5xx / 429). */
export const REFRESH_FAILURE_BACKOFF_MS = 30 * 1000;

let lastRefreshSuccessAt = 0;
let lastRefreshFailureAt = 0;
let intervalId = null;
let activityUnsub = null;
let keepAliveStarted = false;
let keepAliveSuspended = false;

export function suspendSessionKeepAlive() {
  keepAliveSuspended = true;
}

/** Marque la session comme venant d'être établie / renouvelée (évite un refresh immédiat). */
export function noteAuthTokensRenewed() {
  lastRefreshSuccessAt = Date.now();
  lastRefreshFailureAt = 0;
}

export function resumeSessionKeepAlive() {
  keepAliveSuspended = false;
  // Après login / reprise de session : conserver le token fresh jusqu'au prochain cycle.
  noteAuthTokensRenewed();
}

/**
 * Classe une erreur de refresh : terminale (auth morte) vs transitoire (réseau / surcharge).
 * @returns {'terminal_failure' | 'transient_failure'}
 */
export function classifyRefreshFailure(error) {
  const status = error?.response?.status;
  if (status == null && (error?.code === 'ECONNABORTED' || error?.message === 'Network Error' || !error?.response)) {
    return 'transient_failure';
  }
  if (status === 429 || (status != null && status >= 500)) {
    return 'transient_failure';
  }
  if (status === 401 || status === 400 || status === 403) {
    return 'terminal_failure';
  }
  // Pas de réponse HTTP typique → transitoire
  if (status == null) {
    return 'transient_failure';
  }
  return 'terminal_failure';
}

/**
 * @returns {Promise<{ status: 'refreshed' | 'terminal_failure' | 'transient_failure' | 'skipped', error?: unknown }>}
 */
export async function tryRefreshSessionIfNeeded({ force = false } = {}) {
  if (
    keepAliveSuspended ||
    isExplicitLogoutInProgress() ||
    isLoginSessionInProgress()
  ) {
    return { status: 'skipped' };
  }
  if (!hasActiveSession()) {
    return { status: 'skipped' };
  }
  if (!force && isSessionIdleWarningActive()) {
    return { status: 'skipped' };
  }
  if (!force && !isUserRecentlyActive(SESSION_WORKING_LOOKBACK_MS)) {
    return { status: 'skipped' };
  }

  const now = Date.now();
  if (!force && now - lastRefreshSuccessAt < MIN_REFRESH_GAP_MS) {
    return { status: 'skipped' };
  }
  if (!force && lastRefreshFailureAt > 0 && now - lastRefreshFailureAt < REFRESH_FAILURE_BACKOFF_MS) {
    return { status: 'skipped' };
  }

  try {
    await refreshSessionTokens();
    if (isExplicitLogoutInProgress()) {
      return { status: 'skipped' };
    }
    noteAuthTokensRenewed();
    return { status: 'refreshed' };
  } catch (error) {
    const kind = classifyRefreshFailure(error);
    lastRefreshFailureAt = Date.now();
    return { status: kind, error };
  }
}

export function startSessionKeepAlive() {
  if (keepAliveStarted || typeof window === 'undefined') {
    return () => {};
  }
  keepAliveStarted = true;
  // Évite un refresh au premier clic si une session existante est déjà active.
  noteAuthTokensRenewed();

  intervalId = setInterval(() => {
    void tryRefreshSessionIfNeeded();
  }, SESSION_KEEPALIVE_INTERVAL_MS);

  activityUnsub = onUserActivity(() => {
    void tryRefreshSessionIfNeeded();
  });

  return () => {
    keepAliveStarted = false;
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
    if (activityUnsub) {
      activityUnsub();
      activityUnsub = null;
    }
  };
}

export function resetSessionKeepAliveForTests() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
  if (activityUnsub) {
    activityUnsub();
    activityUnsub = null;
  }
  keepAliveStarted = false;
  keepAliveSuspended = false;
  lastRefreshSuccessAt = 0;
  lastRefreshFailureAt = 0;
}
