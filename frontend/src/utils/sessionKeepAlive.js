/**
 * Renouvellement proactif du JWT / cookies tant que l'utilisateur travaille.
 * Scheduler basé sur access_expires_at (métadonnée), pas sur decode JWT HttpOnly.
 */

import { refreshSessionTokens, expireCurrentWebSession } from './apiClient';
import {
  getStoredAccessExpiresAtMs,
  isAccessExpired,
  isAccessNearExpiry,
} from './accessExpiry';
import { isSessionIdleWarningActive } from './deferredSessionLogout';
import { isExplicitLogoutInProgress, isLoginSessionInProgress } from './sessionLogoutState';
import { hasActiveSession } from './webAuthSession';
import {
  getMsSinceLastUserActivity,
  onSessionResume,
  onUserActivity,
  SESSION_WORKING_LOOKBACK_MS,
  isUserRecentlyActive,
} from './userActivityTracker';

/** Intervalle filet de sécurité entre deux tentatives. */
export const SESSION_KEEPALIVE_INTERVAL_MS = 45 * 60 * 1000;

/**
 * Écart minimum entre deux refresh réussis non motivés par l'expiration.
 * Évite de remplacer trop tôt un JWT fresh=True après login.
 */
export const MIN_REFRESH_GAP_MS = SESSION_KEEPALIVE_INTERVAL_MS;

/** Backoff après un échec transitoire (réseau / 5xx / 429). */
export const REFRESH_FAILURE_BACKOFF_MS = 30 * 1000;

/** Refresh anticipé avant expiration access. */
export const ACCESS_REFRESH_AHEAD_MS = 5 * 60 * 1000;

/** Marge d'horloge. */
export const CLOCK_SKEW_MS = 30 * 1000;

const EXPIRY_FORCE_REASONS = new Set([
  'access_expiring',
  'access_expired',
  'visibility_resume',
  'socket_recovery',
]);

let lastRefreshSuccessAt = 0;
let lastRefreshFailureAt = 0;
let intervalId = null;
let scheduledRefreshTimerId = null;
let activityUnsub = null;
let resumeUnsub = null;
let keepAliveStarted = false;
let keepAliveSuspended = false;
let resumeRefreshInFlight = false;

export function suspendSessionKeepAlive() {
  keepAliveSuspended = true;
  cancelScheduledRefresh();
}

/** Marque la session comme venant d'être établie / renouvelée (évite un refresh immédiat). */
export function noteAuthTokensRenewed() {
  lastRefreshSuccessAt = Date.now();
  lastRefreshFailureAt = 0;
}

export function resumeSessionKeepAlive() {
  keepAliveSuspended = false;
  // Après login / reprise de session réelle : conserver le token fresh.
  noteAuthTokensRenewed();
  scheduleRefreshFromExp();
}

export function cancelScheduledRefresh() {
  if (scheduledRefreshTimerId != null) {
    clearTimeout(scheduledRefreshTimerId);
    scheduledRefreshTimerId = null;
  }
}

/**
 * Programme un refresh à T−5 min − skew depuis access_expires_at stocké.
 */
export function scheduleRefreshFromExp() {
  cancelScheduledRefresh();
  if (typeof window === 'undefined' || keepAliveSuspended) {
    return;
  }
  const expiresAt = getStoredAccessExpiresAtMs();
  if (expiresAt == null) {
    return;
  }
  const now = Date.now();
  const delay = Math.max(0, expiresAt - now - ACCESS_REFRESH_AHEAD_MS - CLOCK_SKEW_MS);
  scheduledRefreshTimerId = setTimeout(() => {
    scheduledRefreshTimerId = null;
    void tryRefreshSessionIfNeeded({
      force: true,
      reason: isAccessExpired() ? 'access_expired' : 'access_expiring',
    });
  }, delay);
}

/**
 * Refresh forcé si access expiré ou proche (≤ 5 min).
 * @returns {Promise<{ status: string, error?: unknown }|null>}
 */
export async function refreshIfNearExpiry({ reason = 'access_expiring' } = {}) {
  if (isAccessExpired()) {
    return tryRefreshSessionIfNeeded({ force: true, reason: 'access_expired' });
  }
  if (isAccessNearExpiry(ACCESS_REFRESH_AHEAD_MS + CLOCK_SKEW_MS)) {
    return tryRefreshSessionIfNeeded({ force: true, reason });
  }
  return { status: 'skipped' };
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
  if (status == null) {
    return 'transient_failure';
  }
  return 'terminal_failure';
}

/**
 * @param {{ force?: boolean, reason?: string }} [options]
 * @returns {Promise<{ status: 'refreshed' | 'terminal_failure' | 'transient_failure' | 'skipped', error?: unknown }>}
 */
export async function tryRefreshSessionIfNeeded({
  force = false,
  reason = 'interval_fallback',
} = {}) {
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

  const bypassGap = Boolean(force) || EXPIRY_FORCE_REASONS.has(reason);

  // Idle warning souverain : ne pas prolonger silencieusement.
  if (isSessionIdleWarningActive() && reason !== 'idle_stay_connected') {
    return { status: 'skipped' };
  }

  if (!force && !isUserRecentlyActive(SESSION_WORKING_LOOKBACK_MS)) {
    return { status: 'skipped' };
  }

  // Même en force « expiry », respecter l'idle 8 h (sauf stay_connected explicite).
  if (
    force &&
    reason !== 'idle_stay_connected' &&
    getMsSinceLastUserActivity() >= SESSION_WORKING_LOOKBACK_MS
  ) {
    return { status: 'skipped' };
  }

  const now = Date.now();
  if (!bypassGap && now - lastRefreshSuccessAt < MIN_REFRESH_GAP_MS) {
    return { status: 'skipped' };
  }
  if (
    !force &&
    lastRefreshFailureAt > 0 &&
    now - lastRefreshFailureAt < REFRESH_FAILURE_BACKOFF_MS
  ) {
    return { status: 'skipped' };
  }
  // Backoff conservé sauf recovery / expiry contrôlée
  if (
    force &&
    !EXPIRY_FORCE_REASONS.has(reason) &&
    reason !== 'socket_recovery' &&
    reason !== 'idle_stay_connected' &&
    lastRefreshFailureAt > 0 &&
    now - lastRefreshFailureAt < REFRESH_FAILURE_BACKOFF_MS
  ) {
    return { status: 'skipped' };
  }

  try {
    await refreshSessionTokens();
    if (isExplicitLogoutInProgress()) {
      return { status: 'skipped' };
    }
    noteAuthTokensRenewed();
    scheduleRefreshFromExp();
    try {
      const {
        getCompanySocketStatusSnapshot,
        retryCompanySocket,
      } = await import('../services/companySocket');
      const status = getCompanySocketStatusSnapshot();
      if (!status?.connected) {
        retryCompanySocket();
      }
    } catch (_) {
      // Socket optionnel / non chargé
    }
    return { status: 'refreshed' };
  } catch (error) {
    const kind = classifyRefreshFailure(error);
    lastRefreshFailureAt = Date.now();
    if (kind === 'terminal_failure') {
      expireCurrentWebSession({ reason: 'session_expired' });
    }
    return { status: kind, error };
  }
}

async function handleSessionResume() {
  if (resumeRefreshInFlight || keepAliveSuspended) {
    return;
  }
  resumeRefreshInFlight = true;
  try {
    if (getMsSinceLastUserActivity() >= SESSION_WORKING_LOOKBACK_MS) {
      // Idle souverain : laisser deferredSessionLogout gérer warning/logout.
      try {
        const { ensureIdleGuardEvaluated } = await import('./deferredSessionLogout');
        if (typeof ensureIdleGuardEvaluated === 'function') {
          ensureIdleGuardEvaluated();
        }
      } catch (_) {
        // ignore
      }
      return;
    }
    await refreshIfNearExpiry({ reason: 'visibility_resume' });
  } finally {
    resumeRefreshInFlight = false;
  }
}

export function startSessionKeepAlive() {
  if (keepAliveStarted || typeof window === 'undefined') {
    return () => {};
  }
  keepAliveStarted = true;
  // P0-A : ne PAS appeler noteAuthTokensRenewed() ici (reload ≠ renouvellement).
  scheduleRefreshFromExp();

  intervalId = setInterval(() => {
    void tryRefreshSessionIfNeeded({ reason: 'interval_fallback' });
  }, SESSION_KEEPALIVE_INTERVAL_MS);

  activityUnsub = onUserActivity(() => {
    void tryRefreshSessionIfNeeded({ reason: 'interval_fallback' });
  });

  resumeUnsub = onSessionResume(() => {
    void handleSessionResume();
  });

  return () => {
    keepAliveStarted = false;
    cancelScheduledRefresh();
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
    if (activityUnsub) {
      activityUnsub();
      activityUnsub = null;
    }
    if (resumeUnsub) {
      resumeUnsub();
      resumeUnsub = null;
    }
  };
}

export function resetSessionKeepAliveForTests() {
  cancelScheduledRefresh();
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
  if (activityUnsub) {
    activityUnsub();
    activityUnsub = null;
  }
  if (resumeUnsub) {
    resumeUnsub();
    resumeUnsub = null;
  }
  keepAliveStarted = false;
  keepAliveSuspended = false;
  lastRefreshSuccessAt = 0;
  lastRefreshFailureAt = 0;
  resumeRefreshInFlight = false;
}
