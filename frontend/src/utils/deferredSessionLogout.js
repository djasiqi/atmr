/**
 * Garde d'inactivité bancaire :
 * ACTIVE → IDLE_WARNING (préavis 60s) → Rester connecté / Se déconnecter / timeout.
 * SESSION_INVALID est gérée ailleurs (logout immédiat).
 */

import { toast } from 'sonner';
import { hasActiveSession } from './webAuthSession';
import {
  getMsSinceLastUserActivity,
  onUserActivity,
  recordUserActivity,
  SESSION_WORKING_LOOKBACK_MS,
} from './userActivityTracker';

/** Seuil d'inactivité avant préavis (= lookback keep-alive). */
export const SESSION_IDLE_TIMEOUT_MS = SESSION_WORKING_LOOKBACK_MS;

/** Durée du compte à rebours avant déconnexion. */
export const SESSION_IDLE_WARNING_MS = 60 * 1000;

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

const idleWarningMessage = (secondsLeft) =>
  `Votre session expirera dans ${secondsLeft} secondes en raison de votre inactivité.`;

const showIdleWarningToast = (secondsLeft, { renewing = false } = {}) => {
  toast.warning(idleWarningMessage(secondsLeft), {
    id: IDLE_WARNING_TOAST_ID,
    duration: Infinity,
    closeButton: false,
    dismissible: false,
    action: {
      label: renewing ? 'Renouvellement…' : 'Rester connecté',
      onClick: (event) => {
        event?.preventDefault?.();
        if (renewing) {
          return undefined;
        }
        return handleStayConnected();
      },
    },
    cancel: {
      label: 'Se déconnecter',
      onClick: () => {
        if (renewing) {
          return undefined;
        }
        return handleUserLogout();
      },
    },
  });
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

const handleStayConnected = async () => {
  if (phase !== 'IDLE_WARNING') {
    return;
  }
  phase = 'RENEWING';
  const remainingMs = (warningEndsAt || 0) - Date.now();
  showIdleWarningToast(Math.max(1, Math.ceil(remainingMs / 1000)), { renewing: true });

  try {
    const { tryRefreshSessionIfNeeded } = await import('./sessionKeepAlive');
    const result = await tryRefreshSessionIfNeeded({ force: true });

    if (result?.status === 'refreshed') {
      recordUserActivity();
      resolveIdleWarning();
      return;
    }

    if (result?.status === 'terminal_failure') {
      executeLogout({
        immediate: true,
        reason: 'session_expired',
        preserveNext: true,
      });
      return;
    }

    // transient / skipped
    if (phase === 'RENEWING') {
      phase = 'IDLE_WARNING';
    }
    toast.error('Impossible de prolonger la session.', {
      id: TRANSIENT_TOAST_ID,
      duration: 4000,
    });
    const secondsLeft = Math.max(1, Math.ceil(((warningEndsAt || 0) - Date.now()) / 1000));
    showIdleWarningToast(secondsLeft, { renewing: false });
  } catch (_) {
    if (phase === 'RENEWING') {
      phase = 'IDLE_WARNING';
    }
    toast.error('Impossible de prolonger la session.', {
      id: TRANSIENT_TOAST_ID,
      duration: 4000,
    });
    const secondsLeft = Math.max(1, Math.ceil(((warningEndsAt || 0) - Date.now()) / 1000));
    showIdleWarningToast(secondsLeft, { renewing: false });
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

const ensurePollLoop = () => {
  if (pollTimer || phase === 'STOPPED') {
    return;
  }
  pollTimer = setInterval(() => {
    if (phase !== 'ACTIVE') {
      return;
    }
    if (getMsSinceLastUserActivity() >= SESSION_IDLE_TIMEOUT_MS) {
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
      if (meta.source === 'remote' && (phase === 'IDLE_WARNING' || phase === 'RENEWING')) {
        resolveIdleWarning();
      }
    });
  }

  ensurePollLoop();

  if (phase === 'ACTIVE' && getMsSinceLastUserActivity() >= SESSION_IDLE_TIMEOUT_MS) {
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
