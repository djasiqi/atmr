/**
 * Renouvellement proactif du JWT tant que l'utilisateur travaille.
 * Évite l'expiration à 1 h pendant une session active (clics, scroll, navigation…).
 */

import { refreshSessionTokens } from './apiClient';
import { cancelDeferredLogout } from './deferredSessionLogout';
import { isExplicitLogoutInProgress, isLoginSessionInProgress } from './sessionLogoutState';
import { hasActiveSession } from './webAuthSession';
import {
  isUserRecentlyActive,
  onUserActivity,
  SESSION_WORKING_LOOKBACK_MS,
} from './userActivityTracker';

/** Intervalle entre deux tentatives de refresh (access token ≈ 1 h). */
export const SESSION_KEEPALIVE_INTERVAL_MS = 45 * 60 * 1000;

const MIN_REFRESH_GAP_MS = 5 * 60 * 1000;

let lastRefreshAttemptAt = 0;
let intervalId = null;
let activityUnsub = null;
let keepAliveStarted = false;
let keepAliveSuspended = false;

export function suspendSessionKeepAlive() {
  keepAliveSuspended = true;
}

export function resumeSessionKeepAlive() {
  keepAliveSuspended = false;
}

export async function tryRefreshSessionIfNeeded({ force = false } = {}) {
  if (keepAliveSuspended || isExplicitLogoutInProgress()) {
    return false;
  }
  if (!hasActiveSession()) {
    return false;
  }
  if (!force && !isUserRecentlyActive(SESSION_WORKING_LOOKBACK_MS)) {
    return false;
  }

  const now = Date.now();
  if (!force && now - lastRefreshAttemptAt < MIN_REFRESH_GAP_MS) {
    return false;
  }

  lastRefreshAttemptAt = now;
  try {
    await refreshSessionTokens();
    if (isExplicitLogoutInProgress()) {
      return false;
    }
    cancelDeferredLogout();
    return true;
  } catch {
    return false;
  }
}

export function startSessionKeepAlive() {
  if (keepAliveStarted || typeof window === 'undefined') {
    return () => {};
  }
  keepAliveStarted = true;

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
  lastRefreshAttemptAt = 0;
}
