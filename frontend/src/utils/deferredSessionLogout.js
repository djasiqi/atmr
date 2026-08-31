/**
 * Garde d'inactivité bancaire :
 * ACTIVE → IDLE_WARNING (préavis 2 min) → Rester connecté / Se déconnecter / timeout.
 * Activité locale (clic/frappe) pendant le préavis → prolongement auto (refresh).
 * SESSION_INVALID est gérée ailleurs (logout immédiat).
 */

import React from 'react';
import { toast } from 'sonner';
import SessionIdleWarningToast, {
  sessionIdleWarningToastStyles,
} from '../components/auth/SessionIdleWarningToast';
import { hasActiveSession, isInstitutionWebSession } from './webAuthSession';
import {
  getMsSinceLastUserActivity,
  onUserActivity,
  recordUserActivity,
  SESSION_WORKING_LOOKBACK_MS,
} from './userActivityTracker';

/** Idle institution : 15 min (backend 900 s). Clic / scroll / frappe réarment. */
export const INSTITUTION_IDLE_TIMEOUT_MS = 15 * 60 * 1000;

/** Seuil d'inactivité avant préavis (company / défaut = lookback 8 h). */
export const SESSION_IDLE_TIMEOUT_MS = SESSION_WORKING_LOOKBACK_MS;

/** Durée du compte à rebours avant déconnexion. */
export const SESSION_IDLE_WARNING_MS = 2 * 60 * 1000;

/** Durée idle totale avant logout (institution 15 min, sinon 8 h + préavis). */
export const getSessionIdleTimeoutMs = () =>
  isInstitutionWebSession() ? INSTITUTION_IDLE_TIMEOUT_MS : SESSION_IDLE_TIMEOUT_MS;

/**
 * Seuil d'affichage du warning.
 * Institution : T−2 min (13 min) puis logout à T15.
 * Autres rôles : inchangé (warning après 8 h, logout +2 min).
 */
export const getSessionIdleWarningAtMs = () =>
  isInstitutionWebSession()
    ? Math.max(0, INSTITUTION_IDLE_TIMEOUT_MS - SESSION_IDLE_WARNING_MS)
    : SESSION_IDLE_TIMEOUT_MS;

/** @deprecated Utiliser SESSION_IDLE_WARNING_MS */
export const SESSION_LOGOUT_COUNTDOWN_MS = SESSION_IDLE_WARNING_MS;

export const SESSION_REAUTH_REQUIRED_EVENT = 'session-reauth-required';

const IDLE_WARNING_TOAST_ID = 'session-idle-warning';
const TRANSIENT_TOAST_ID = 'session-idle-transient';
const POLL_MS = 1000;

/** STOPPED | ACTIVE | IDLE_WARNING | RENEWING */
let phase = 'STOPPED';
let pollTimer = null;
let countdownTimer = null;
let warningEndsAt = null;
let unsubscribeActivity = null;
let initialized = false;

const clearCountdownTimer = () => {
  if (countdownTimer) {
    clearInterval(countdownTimer);
    countdownTimer = null;
  }
};

const clearPollTimer = () => {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
};

const dismissIdleToast = () => {
  toast.dismiss(IDLE_WARNING_TOAST_ID);
};

const WARNING_TOTAL_SECONDS = Math.round(SESSION_IDLE_WARNING_MS / 1000);

const showIdleWarningToast = (secondsLeft, { renewing = false } = {}) => {
  toast.custom(
    () =>
      React.createElement(SessionIdleWarningToast, {
        secondsLeft,
        totalSeconds: WARNING_TOTAL_SECONDS,
        renewing,
        onStay: () => {
          if (!renewing) {
            return handleStayConnected();
          }
          return undefined;
        },
        onLogout: () => {
          if (!renewing) {
            return handleUserLogout();
          }
          return undefined;
        },
      }),
    {
      id: IDLE_WARNING_TOAST_ID,
      duration: Infinity,
      dismissible: false,
      className: sessionIdleWarningToastStyles.sonnerHost,
      unstyled: true,
    }
  );
};

const executeLogout = (options) => {
  stopSessionIdleGuard();
  import('./apiClient')
    .then(({ logoutUser }) => logoutUser(options))
    .catch(() => {
      import('./authNavigation').then(({ requestAuthNavigate }) => {
        requestAuthNavigate('/login', { replace: true });
      });
    });
};

const handleUserLogout = () => {
  executeLogout({
    immediate: true,
    reason: 'user_logout',
    preserveNext: false,
  });
};

const handleIdleTimeout = () => {
  executeLogout({
    immediate: true,
    reason: 'idle_timeout',
    preserveNext: true,
  });
};

const restoreStayWarning = () => {
  if (phase === 'RENEWING') {
    phase = 'IDLE_WARNING';
  }
  toast.error('Impossible de prolonger la session.', {
    id: TRANSIENT_TOAST_ID,
    duration: 4000,
  });
  const secondsLeft = Math.max(1, Math.ceil(((warningEndsAt || 0) - Date.now()) / 1000));
  showIdleWarningToast(secondsLeft, { renewing: false });
};

const handleStayConnected = async () => {
  if (phase !== 'IDLE_WARNING') {
    return;
  }
  phase = 'RENEWING';
  const remainingMs = (warningEndsAt || 0) - Date.now();
  showIdleWarningToast(Math.max(1, Math.ceil(remainingMs / 1000)), { renewing: true });

  try {
    const { tryRefreshSessionIfNeeded } = await import('./sessionKeepAlive');
    const result = await tryRefreshSessionIfNeeded({
      force: true,
      reason: 'idle_stay_connected',
    });

    if (result?.status === 'terminal_failure') {
      executeLogout({
        immediate: true,
        reason: 'session_expired',
        preserveNext: true,
      });
      return;
    }

    if (result?.status !== 'refreshed' && result?.status !== 'skipped') {
      restoreStayWarning();
      return;
    }

    const { postInteractiveSessionActivity } = await import(
      './sessionActivityHeartbeat'
    );
    const activity = await postInteractiveSessionActivity();

    if (activity?.status === 'ok') {
      recordUserActivity();
      resolveIdleWarning();
      return;
    }

    if (activity?.status === 'terminal_failure') {
      executeLogout({
        immediate: true,
        reason: 'session_expired',
        preserveNext: true,
      });
      return;
    }

    restoreStayWarning();
  } catch (_) {
    restoreStayWarning();
  }
};

const tickWarning = () => {
  if (phase !== 'IDLE_WARNING' && phase !== 'RENEWING') {
    return;
  }
  const remainingMs = (warningEndsAt || 0) - Date.now();
  if (remainingMs <= 0) {
    handleIdleTimeout();
    return;
  }
  if (phase === 'IDLE_WARNING') {
    showIdleWarningToast(Math.max(1, Math.ceil(remainingMs / 1000)), { renewing: false });
  }
};

const enterIdleWarning = () => {
  if (phase === 'IDLE_WARNING' || phase === 'RENEWING') {
    return;
  }
  phase = 'IDLE_WARNING';
  warningEndsAt = Date.now() + SESSION_IDLE_WARNING_MS;
  tickWarning();
  clearCountdownTimer();
  countdownTimer = setInterval(tickWarning, POLL_MS);
};

export const resolveIdleWarning = () => {
  if (phase !== 'IDLE_WARNING' && phase !== 'RENEWING') {
    return;
  }
  clearCountdownTimer();
  warningEndsAt = null;
  dismissIdleToast();
  toast.dismiss(TRANSIENT_TOAST_ID);
  if (phase !== 'STOPPED') {
    phase = 'ACTIVE';
  }
};

export const isSessionIdleWarningActive = () =>
  phase === 'IDLE_WARNING' || phase === 'RENEWING';

/** Évalue immédiatement le seuil idle (ex. après resume focus/visibility). */
export const ensureIdleGuardEvaluated = () => {
  if (phase !== 'ACTIVE') {
    return;
  }
  if (getMsSinceLastUserActivity() >= getSessionIdleWarningAtMs()) {
    enterIdleWarning();
  }
};

const ensurePollLoop = () => {
  if (pollTimer || phase === 'STOPPED') {
    return;
  }
  pollTimer = setInterval(() => {
    if (phase !== 'ACTIVE') {
      return;
    }
    if (getMsSinceLastUserActivity() >= getSessionIdleWarningAtMs()) {
      enterIdleWarning();
    }
  }, POLL_MS);
};

export const startSessionIdleGuard = () => {
  if (typeof window === 'undefined') {
    return;
  }
  if (phase === 'STOPPED') {
    phase = 'ACTIVE';
  } else if (phase === 'ACTIVE' || phase === 'IDLE_WARNING' || phase === 'RENEWING') {
    // déjà en cours
  } else {
    phase = 'ACTIVE';
  }

  if (!unsubscribeActivity) {
    unsubscribeActivity = onUserActivity((_ts, meta = {}) => {
      if (phase !== 'IDLE_WARNING' && phase !== 'RENEWING') {
        return;
      }
      // Autre onglet déjà actif → ferme le préavis sans refresh local.
      if (meta.source === 'remote') {
        resolveIdleWarning();
        return;
      }
      // Clic / frappe / focus pendant le préavis → prolongement (refresh JWT).
      if (meta.source === 'local' && phase === 'IDLE_WARNING') {
        void handleStayConnected();
      }
    });
  }

  ensurePollLoop();

  if (phase === 'ACTIVE' && getMsSinceLastUserActivity() >= getSessionIdleWarningAtMs()) {
    enterIdleWarning();
  }
};

export const stopSessionIdleGuard = () => {
  phase = 'STOPPED';
  warningEndsAt = null;
  clearCountdownTimer();
  clearPollTimer();
  dismissIdleToast();
  toast.dismiss(TRANSIENT_TOAST_ID);
  if (unsubscribeActivity) {
    unsubscribeActivity();
    unsubscribeActivity = null;
  }
};

/** Alias déprécié : ne stoppe plus la garde, ferme seulement le préavis. */
export const cancelDeferredLogout = () => {
  resolveIdleWarning();
};

const syncGuardWithAuth = () => {
  if (hasActiveSession()) {
    recordUserActivity();
    startSessionIdleGuard();
  } else {
    stopSessionIdleGuard();
  }
};

/** @deprecated Plus de déconnexion différée post-401 ; no-op. */
export const scheduleDeferredLogout = () => {};

/** @deprecated Plus de déconnexion différée post-401 ; no-op. */
export const notifySessionReauthRequired = () => {};

/** Réinitialise l'état interne (tests uniquement). */
export const resetDeferredSessionLogoutForTests = () => {
  stopSessionIdleGuard();
  initialized = false;
};

export const initDeferredSessionLogout = () => {
  if (typeof window === 'undefined' || initialized) {
    return () => {};
  }
  initialized = true;

  const onAuthChanged = () => syncGuardWithAuth();
  window.addEventListener('auth-changed', onAuthChanged);

  // Reload : session déjà présente sans nouvel auth-changed.
  syncGuardWithAuth();

  return () => {
    initialized = false;
    window.removeEventListener('auth-changed', onAuthChanged);
    stopSessionIdleGuard();
  };
};
