/**
 * Garde-fou contre la reconnexion automatique lors d'une déconnexion volontaire.
 * Le keep-alive et le bootstrap /auth/me ne doivent pas rétablir la session
 * pendant ou juste après un logout explicite.
 */

export const EXPLICIT_LOGOUT_SESSION_KEY = 'lirie_explicit_logout';

/** Signal multi-onglets (localStorage) — visible par tous les onglets. */
export const AUTH_LOGOUT_AT_KEY = 'lirie_auth_logout_at';

/** Raison affichée sur /login (sessionStorage, onglet courant). */
export const AUTH_LOGOUT_REASON_KEY = 'lirie_auth_logout_reason';

export const markCrossTabLogout = () => {
  try {
    localStorage.setItem(AUTH_LOGOUT_AT_KEY, String(Date.now()));
  } catch (_) {
    // ignore
  }
};

export const clearCrossTabLogoutMarker = () => {
  try {
    localStorage.removeItem(AUTH_LOGOUT_AT_KEY);
  } catch (_) {
    // ignore
  }
};

export const setAuthLogoutReason = (reason) => {
  if (!reason) return;
  try {
    sessionStorage.setItem(AUTH_LOGOUT_REASON_KEY, reason);
  } catch (_) {
    // ignore
  }
};

export const consumeAuthLogoutReason = () => {
  try {
    const reason = sessionStorage.getItem(AUTH_LOGOUT_REASON_KEY);
    if (reason) {
      sessionStorage.removeItem(AUTH_LOGOUT_REASON_KEY);
    }
    return reason;
  } catch (_) {
    return null;
  }
};

let explicitLogoutInProgress = false;

export const beginExplicitLogout = () => {
  explicitLogoutInProgress = true;
  try {
    sessionStorage.setItem(EXPLICIT_LOGOUT_SESSION_KEY, String(Date.now()));
  } catch (_) {
    // ignore (mode privé, quota…)
  }
};

export const isExplicitLogoutInProgress = () => explicitLogoutInProgress;

export const endExplicitLogout = () => {
  explicitLogoutInProgress = false;
};

export const hasRecentExplicitLogout = () => {
  try {
    return Boolean(sessionStorage.getItem(EXPLICIT_LOGOUT_SESSION_KEY));
  } catch (_) {
    return false;
  }
};

export const clearExplicitLogoutMarker = () => {
  try {
    sessionStorage.removeItem(EXPLICIT_LOGOUT_SESSION_KEY);
  } catch (_) {
    // ignore
  }
};

let loginSessionInProgress = false;

/** Bloque les déconnexions automatiques pendant l'échange de session au login. */
export const beginLoginSession = () => {
  loginSessionInProgress = true;
};

export const endLoginSession = () => {
  loginSessionInProgress = false;
};

export const isLoginSessionInProgress = () => loginSessionInProgress;
