/**
 * Déconnexion différée lorsqu'une reconnexion JWT est requise :
 * - utilisateur actif → pas de déconnexion immédiate ;
 * - inactivité → compte à rebours visible → déconnexion + redirection login.
 */

import { toast } from 'sonner';
import {
  isUserRecentlyActive,
  onUserActivity,
  USER_ACTIVE_WINDOW_MS,
} from './userActivityTracker';

export const SESSION_REAUTH_REQUIRED_EVENT = 'session-reauth-required';

/** Durée du compte à rebours avant déconnexion (utilisateur inactif). */
export const SESSION_LOGOUT_COUNTDOWN_MS = 60 * 1000;

const TOAST_ID = 'deferred-session-logout';
const POLL_MS = 1000;

/** idle | waiting_activity | counting */
let phase = 'idle';
let pollTimer = null;
let countdownTimer = null;
let countdownEndsAt = null;
let unsubscribeActivity = null;
let initialized = false;
let pendingSchedule = false;
let pendingScheduleOptions = {};

const clearTimers = () => {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (countdownTimer) {
    clearInterval(countdownTimer);
    countdownTimer = null;
  }
};

const dismissToast = () => {
  toast.dismiss(TOAST_ID);
};

const showWaitingToast = () => {
  toast.warning(
    'Votre session nécessite une reconnexion. La déconnexion aura lieu après une période d\'inactivité.',
    { id: TOAST_ID, duration: Infinity }
  );
};

const showCountdownToast = (secondsLeft) => {
  toast.warning(
    `Session expirée. Déconnexion automatique dans ${secondsLeft} s…`,
    { id: TOAST_ID, duration: Infinity }
  );
};

const executeLogout = () => {
  cancelDeferredLogout();
  import('./apiClient')
    .then(({ logoutUser }) => logoutUser({ preserveNext: true }))
    .catch(() => {
      window.location.href = '/login';
    });
};

const startCountdown = () => {
  if (phase === 'counting') {
    return;
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  phase = 'counting';
  countdownEndsAt = Date.now() + SESSION_LOGOUT_COUNTDOWN_MS;

  const tick = () => {
    if (isUserRecentlyActive(USER_ACTIVE_WINDOW_MS)) {
      phase = 'waiting_activity';
      countdownEndsAt = null;
      if (countdownTimer) {
        clearInterval(countdownTimer);
        countdownTimer = null;
      }
      showWaitingToast();
      return;
    }

    const remainingMs = (countdownEndsAt || 0) - Date.now();
    if (remainingMs <= 0) {
      executeLogout();
      return;
    }
    showCountdownToast(Math.max(1, Math.ceil(remainingMs / 1000)));
  };

  tick();
  countdownTimer = setInterval(tick, POLL_MS);
};

const ensurePollLoop = () => {
  if (pollTimer) {
    return;
  }
  pollTimer = setInterval(() => {
    if (phase === 'idle') {
      return;
    }
    if (phase === 'waiting_activity' && !isUserRecentlyActive(USER_ACTIVE_WINDOW_MS)) {
      startCountdown();
    }
  }, POLL_MS);
};

export const scheduleDeferredLogout = ({ silentUntilIdle = false } = {}) => {
  if (phase !== 'idle') {
    return;
  }

  phase = 'waiting_activity';
  if (!silentUntilIdle) {
    showWaitingToast();
  }

  unsubscribeActivity = onUserActivity(() => {
    if (phase === 'counting') {
      phase = 'waiting_activity';
      countdownEndsAt = null;
      if (countdownTimer) {
        clearInterval(countdownTimer);
        countdownTimer = null;
      }
      showWaitingToast();
      ensurePollLoop();
    }
  });

  ensurePollLoop();

  if (!isUserRecentlyActive(USER_ACTIVE_WINDOW_MS)) {
    startCountdown();
  }
};

export const cancelDeferredLogout = () => {
  phase = 'idle';
  countdownEndsAt = null;
  clearTimers();
  dismissToast();
  if (unsubscribeActivity) {
    unsubscribeActivity();
    unsubscribeActivity = null;
  }
};

/** Réinitialise l'état interne (tests uniquement). */
export const resetDeferredSessionLogoutForTests = () => {
  cancelDeferredLogout();
  initialized = false;
  pendingSchedule = false;
  pendingScheduleOptions = {};
};

export const initDeferredSessionLogout = () => {
  if (typeof window === 'undefined' || initialized) {
    return () => {};
  }
  initialized = true;

  const onReauthRequired = (event) => {
    scheduleDeferredLogout(event?.detail || {});
  };
  const onAuthChanged = () => cancelDeferredLogout();

  window.addEventListener(SESSION_REAUTH_REQUIRED_EVENT, onReauthRequired);
  window.addEventListener('auth-changed', onAuthChanged);

  if (pendingSchedule) {
    pendingSchedule = false;
    scheduleDeferredLogout(pendingScheduleOptions);
    pendingScheduleOptions = {};
  }

  return () => {
    initialized = false;
    window.removeEventListener(SESSION_REAUTH_REQUIRED_EVENT, onReauthRequired);
    window.removeEventListener('auth-changed', onAuthChanged);
    cancelDeferredLogout();
  };
};

export const notifySessionReauthRequired = (options = {}) => {
  if (typeof window === 'undefined') {
    return;
  }
  if (!initialized) {
    pendingSchedule = true;
    pendingScheduleOptions = options;
    return;
  }
  scheduleDeferredLogout(options);
};
