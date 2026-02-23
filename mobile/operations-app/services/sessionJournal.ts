/**
 * P0.1 – Journal de session interne (app chauffeur)
 *
 * Reason codes pour diagnostiquer logout/401/refresh en environnement mobile instable.
 * Envoyé en header X-Session-Diag à l'API pour corrélation backend.
 */

import AsyncStorage from "@react-native-async-storage/async-storage";

/** P0.2+: REFRESH_WAIT = une requête attend un refresh déjà en cours. */
/** P0.3+: SOCKET_* = états socket pour observabilité (jamais de logout sur disconnect). */
/** P0.4+: FOREGROUND_RESYNC_* = resync au retour au premier plan. */
export type SessionEvent =
  | "APP_START"
  | "LOGIN_SUCCESS"
  | "TOKEN_STORED"
  | "ACCESS_EXPIRED"
  | "REFRESH_START"
  | "REFRESH_SUCCESS"
  | "REFRESH_FAIL"
  | "REFRESH_WAIT"
  | "API_401"
  | "SOCKET_CONNECT"
  | "SOCKET_DISCONNECT"
  | "SOCKET_CONNECTING"
  | "SOCKET_CONNECTED"
  | "SOCKET_RECONNECT_SUCCESS"
  | "NETWORK_CHANGE"
  | "APP_BACKGROUND"
  | "APP_FOREGROUND"
  | "FOREGROUND_RESYNC_START"
  | "FOREGROUND_RESYNC_SUCCESS"
  | "FOREGROUND_RESYNC_FAIL"
  | "ENTERPRISE_APP_FOREGROUND"
  | "LOGOUT_TRIGGERED"
  | "DEVICE_ID_ERROR" /** R3: device_id non créé (storage), socket continue sans extras, pas de logout */
  | `SOCKET_DISCONNECTED:${string}`
  | `SOCKET_RECONNECT_ATTEMPT:${number}`
  | `SOCKET_CONNECT_ERROR:${string}`
  | "SOCKET_AUTH_REFRESH_ATTEMPT"
  | "SOCKET_AUTH_REFRESH_SUCCESS"
  | `SOCKET_RECONNECT_BACKOFF:${number}`
  | "SOCKET_AUTH_RECOVERY_EXHAUSTED"
  | "SOCKET_RECONNECT_FAILED"; /** P2.1.2: enterprise 10 tentatives épuisées */

export const SESSION_JOURNAL_KEYS = {
  LAST_EVENT: "session_journal_last_event",
  LAST_AT: "session_journal_last_at",
} as const;

type Listener = (event: SessionEvent, at: number) => void;
const listeners: Listener[] = [];

let lastEvent: SessionEvent | null = null;
let lastAt: number = 0;

/**
 * Enregistre un événement de session et notifie les listeners.
 * Persiste en mémoire (sync) + AsyncStorage (RN) ou localStorage (web) en fire-and-forget.
 */
export function pushSessionEvent(event: SessionEvent): void {
  const at = Date.now();
  lastEvent = event;
  lastAt = at;
  try {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(SESSION_JOURNAL_KEYS.LAST_EVENT, event);
      localStorage.setItem(SESSION_JOURNAL_KEYS.LAST_AT, String(at));
    }
  } catch (_e) {
    // ignore (SSR / storage indisponible)
  }
  AsyncStorage.setItem(SESSION_JOURNAL_KEYS.LAST_EVENT, event).catch(() => {});
  AsyncStorage.setItem(SESSION_JOURNAL_KEYS.LAST_AT, String(at)).catch(() => {});
  listeners.forEach((cb) => cb(event, at));
}

/**
 * Retourne le dernier événement et son timestamp (sync, mémoire ou localStorage).
 */
export function getLastSessionEvent(): { event: SessionEvent; at: number } | null {
  if (lastEvent && lastAt) return { event: lastEvent, at: lastAt };
  try {
    if (typeof localStorage !== "undefined") {
      const e = localStorage.getItem(SESSION_JOURNAL_KEYS.LAST_EVENT) as SessionEvent | null;
      const t = localStorage.getItem(SESSION_JOURNAL_KEYS.LAST_AT);
      if (e && t) {
        const at = Number(t);
        lastEvent = e;
        lastAt = at;
        return { event: e, at };
      }
    }
  } catch (_e) {
    // ignore
  }
  return null;
}

/**
 * Charge le dernier événement depuis AsyncStorage (RN, après cold start).
 * À appeler au montage du menu debug pour afficher le dernier reason post-redémarrage.
 */
export async function getLastSessionEventFromStorage(): Promise<{
  event: SessionEvent;
  at: number;
} | null> {
  try {
    const [e, t] = await Promise.all([
      AsyncStorage.getItem(SESSION_JOURNAL_KEYS.LAST_EVENT),
      AsyncStorage.getItem(SESSION_JOURNAL_KEYS.LAST_AT),
    ]);
    if (e && t) {
      const at = Number(t);
      lastEvent = e as SessionEvent;
      lastAt = at;
      return { event: e as SessionEvent, at };
    }
  } catch (_e) {
    // ignore
  }
  return null;
}

/** P0.3+ Observabilité : suffixe état socket "S:ONLINE" | "S:RECONN" | "S:OFF" (mis à jour par le manager socket). */
let connectionStateSuffix: "ONLINE" | "RECONN" | "OFF" | null = null;
export function setConnectionStateSuffix(s: "ONLINE" | "RECONN" | "OFF" | null): void {
  connectionStateSuffix = s;
}
export function getConnectionStateSuffix(): "ONLINE" | "RECONN" | "OFF" | null {
  return connectionStateSuffix;
}

/** Reset pour tests uniquement (évite fuite d'état entre tests). */
export function _testingReset(): void {
  lastEvent = null;
  lastAt = 0;
  connectionStateSuffix = null;
}

/**
 * Valeur à envoyer en header X-Session-Diag (dernier reason + timestamp court).
 * Format: "EVENT|ts" ou "EVENT|ts|S:ONLINE" si connectionStateSuffix est défini (P0.3+).
 */
export function getSessionDiagHeaderValue(): string | null {
  const last = getLastSessionEvent();
  if (!last) return null;
  const base = `${last.event}|${last.at}`;
  if (connectionStateSuffix) return `${base}|S:${connectionStateSuffix}`;
  return base;
}

/**
 * Abonnement pour mettre à jour l'UI (ex. menu debug).
 */
export function subscribeSessionJournal(cb: Listener): () => void {
  listeners.push(cb);
  return () => {
    const i = listeners.indexOf(cb);
    if (i >= 0) listeners.splice(i, 1);
  };
}
