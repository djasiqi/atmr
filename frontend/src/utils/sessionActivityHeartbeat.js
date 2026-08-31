/**
 * Client du heartbeat institution canonique POST /auth/session-activity.
 * Réarme web_session.last_interactive_activity_at sur le SID du JWT courant.
 * Ne crée pas de session, ne change pas les timeouts.
 */

import apiClient from './apiClient';

const SESSION_ACTIVITY_PATH = '/auth/session-activity';

const isTerminalActivityError = (error) => {
  const status = error?.response?.status;
  const payload = error?.response?.data || {};
  const code = payload.error_code || payload.error;
  return (
    status === 401 ||
    code === 'session_revoked' ||
    code === 'session_expired' ||
    code === 'idle_timeout'
  );
};

/**
 * Enregistre une activité interactive serveur (même SID).
 * @returns {Promise<{status: string, updated?: boolean, sid?: string|null, error?: unknown}>}
 */
export async function postInteractiveSessionActivity() {
  try {
    const response = await apiClient.post(
      SESSION_ACTIVITY_PATH,
      {},
      { skipAuthRedirect: true }
    );
    const data = response?.data || {};
    if (data.ok) {
      return {
        status: 'ok',
        updated: Boolean(data.updated),
        sid: data.sid || null,
      };
    }
    return { status: 'transient_failure' };
  } catch (error) {
    if (isTerminalActivityError(error)) {
      return { status: 'terminal_failure', error };
    }
    return { status: 'transient_failure', error };
  }
}
