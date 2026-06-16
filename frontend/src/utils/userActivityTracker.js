/**
 * Suivi léger de l'activité utilisateur (souris, clavier, scroll, navigation).
 * Utilisé pour différer la déconnexion lorsqu'une reconnexion JWT est requise.
 */

const THROTTLE_MS = 1000;

/** Fenêtre courte : interaction en cours (pause du compte à rebours). */
export const USER_ACTIVE_WINDOW_MS = 30 * 1000;

/** Fenêtre longue : l'utilisateur est encore « en session de travail » (keep-alive JWT). */
export const SESSION_WORKING_LOOKBACK_MS = 55 * 60 * 1000;

let lastActivityAt = Date.now();
let throttleTimeout = null;
let trackingStarted = false;
const activitySubscribers = new Set();

const notifyActivity = () => {
  activitySubscribers.forEach((listener) => {
    try {
      listener(lastActivityAt);
    } catch (_) {
      // ignore
    }
  });
};

export const recordUserActivity = () => {
  if (throttleTimeout) {
    return;
  }
  lastActivityAt = Date.now();
  notifyActivity();
  throttleTimeout = setTimeout(() => {
    throttleTimeout = null;
  }, THROTTLE_MS);
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

  return () => {
    events.forEach((eventName) => {
      window.removeEventListener(eventName, recordUserActivity, options);
    });
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
  if (throttleTimeout) {
    clearTimeout(throttleTimeout);
    throttleTimeout = null;
  }
  trackingStarted = false;
  activitySubscribers.clear();
};
