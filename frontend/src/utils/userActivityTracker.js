/**
 * Suivi léger de l'activité utilisateur (souris, clavier, scroll, navigation).
 * Activité locale vs distante (autre onglet) pour la garde d'inactivité bancaire.
 */

import { getActivePublicId, getAuthEnv } from './webAuthSession';

const THROTTLE_MS = 1000;

/** Écriture cross-onglet : au plus une fois toutes les 5 s. */
const CROSS_TAB_WRITE_THROTTLE_MS = 5 * 1000;

const FUTURE_SKEW_MS = 5 * 1000;

/** Fenêtre courte : interaction en cours. */
export const USER_ACTIVE_WINDOW_MS = 30 * 1000;

/** Fenêtre longue : l'utilisateur est encore « en session de travail » (keep-alive JWT). */
export const SESSION_WORKING_LOOKBACK_MS = 55 * 60 * 1000;

export const ACTIVITY_STORAGE_KEY_PREFIX = 'lirie_last_user_activity';

let lastActivityAt = Date.now();
let throttleTimeout = null;
let lastCrossTabWriteAt = 0;
let trackingStarted = false;
const activitySubscribers = new Set();

export const getUserActivityStorageKey = (
  env = getAuthEnv(),
  publicId = getActivePublicId(env)
) => `${ACTIVITY_STORAGE_KEY_PREFIX}:${env}:${publicId || 'anonymous'}`;

const notifyActivity = (source = 'local') => {
  activitySubscribers.forEach((listener) => {
    try {
      listener(lastActivityAt, { source });
    } catch (_) {
      // ignore
    }
  });
};

const parseRemoteActivityTimestamp = (rawValue) => {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  const now = Date.now();
  if (parsed > now + FUTURE_SKEW_MS) {
    return null;
  }
  return Math.min(parsed, now);
};

const writeCrossTabActivity = (timestamp) => {
  if (typeof window === 'undefined') {
    return;
  }
  const now = Date.now();
  // lastCrossTabWriteAt === 0 : jamais écrit (évite le faux positif quand Date.now()===0 sous fake timers)
  if (lastCrossTabWriteAt > 0 && now - lastCrossTabWriteAt < CROSS_TAB_WRITE_THROTTLE_MS) {
    return;
  }
  lastCrossTabWriteAt = now || 1;
  try {
    localStorage.setItem(getUserActivityStorageKey(), String(timestamp));
  } catch (_) {
    // ignore (mode privé, quota…)
  }
};

export const recordUserActivity = () => {
  if (throttleTimeout) {
    return;
  }
  lastActivityAt = Date.now();
  notifyActivity('local');
  writeCrossTabActivity(lastActivityAt);
  throttleTimeout = setTimeout(() => {
    throttleTimeout = null;
  }, THROTTLE_MS);
};

/** Applique une activité distante (tests / storage). */
export const applyRemoteUserActivity = (rawTimestamp) => {
  const safeTimestamp = parseRemoteActivityTimestamp(rawTimestamp);
  if (safeTimestamp == null) {
    return false;
  }
  if (safeTimestamp <= lastActivityAt) {
    return false;
  }
  lastActivityAt = safeTimestamp;
  notifyActivity('remote');
  return true;
};

export const getLastUserActivityAt = () => lastActivityAt;

export const getMsSinceLastUserActivity = () => Date.now() - lastActivityAt;

/** Vrai si une interaction récente a eu lieu (clic, scroll, frappe, navigation…). */
export const isUserRecentlyActive = (windowMs = USER_ACTIVE_WINDOW_MS) =>
  getMsSinceLastUserActivity() < windowMs;

export const onUserActivity = (listener) => {
  activitySubscribers.add(listener);
  return () => activitySubscribers.delete(listener);
};

const onStorageActivity = (event) => {
  if (!event || event.storageArea !== localStorage) {
    return;
  }
  const expectedKey = getUserActivityStorageKey();
  if (event.key !== expectedKey) {
    return;
  }
  applyRemoteUserActivity(event.newValue);
};

export const startUserActivityTracking = () => {
  if (trackingStarted || typeof window === 'undefined') {
    return () => {};
  }
  trackingStarted = true;

  // Interactions volontaires uniquement (pas mousemove : empêche la détection d'inactivité).
  const options = { passive: true, capture: false };
  const events = ['click', 'keydown', 'touchstart', 'scroll', 'pointerdown'];
  events.forEach((eventName) => {
    window.addEventListener(eventName, recordUserActivity, options);
  });
  window.addEventListener('storage', onStorageActivity);

  return () => {
    events.forEach((eventName) => {
      window.removeEventListener(eventName, recordUserActivity, options);
    });
    window.removeEventListener('storage', onStorageActivity);
    trackingStarted = false;
    if (throttleTimeout) {
      clearTimeout(throttleTimeout);
      throttleTimeout = null;
    }
  };
};

/** Réinitialise l'état interne (tests uniquement). */
export const resetUserActivityTrackerForTests = () => {
  lastActivityAt = Date.now();
  lastCrossTabWriteAt = 0;
  if (throttleTimeout) {
    clearTimeout(throttleTimeout);
    throttleTimeout = null;
  }
  trackingStarted = false;
  activitySubscribers.clear();
};
