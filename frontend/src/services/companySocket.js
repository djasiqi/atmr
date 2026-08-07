import { io } from 'socket.io-client';
import { getAccessToken } from '../hooks/useAuthToken';
import { SOCKET_CONFIG, SOCKET_PATH, getSocketTransports, isDevelopmentLocalhost } from '../config/socketConfig';
import {
  getCompanySocketMetricsSnapshot,
  isCompanySocketMetricsDebugEnabled,
  recordConnectReady,
  recordConnectStarted,
  recordReconnectAttempt,
  setActiveListenerCount,
  trackListenerCountForWatchdog,
  warnIfTooManyListeners,
} from './companySocketMetrics';
import { companyReconnectCircuitBreaker } from './reconnectCircuitBreaker';
import { bindUrgentAlertSoundListeners } from '../utils/urgentAlertSound';
import { setSubscribedCursor } from '../utils/companyRealtimeSequenceGate';
import {
  labelForConnectError,
  labelForDisabled,
  labelForDisconnectReason,
  labelForMissingToken,
  labelForOffline,
  labelForReconnectFailed,
} from './socketStatusReasons';

const isDevelopment = process.env.NODE_ENV === 'development';

let socket = null;
let connectPromise = null;
/** Incrémenté à chaque create/destroy pour invalider callbacks async obsolètes. */
let socketGenerationId = 0;

/** Registre multi-abonnés : event -> Set<callback> */
const eventSubscribers = new Map();
/** Pont unique socket.io -> fan-out Set */
const socketBridgeListeners = new Map();

let currentCompanyId = null;
let lastJoinCompanyEmitId = null;
let networkListenersSetup = false;
let onOnlineHandler = null;
let onOfflineHandler = null;

/** Dernier état notifié (badge / hooks). */
let lastSocketStatus = {
  connected: false,
  reconnecting: false,
  reasonCode: null,
  reasonLabel: null,
};

/** Évite une tempête de refresh sur AUTH_REQUIRED / token manquant. */
let authRecoveryPromise = null;

export const COMPANY_SOCKET_STATE_EVENT = 'company_socket_state';

function notifyCompanySocketState(detail) {
  const next = {
    connected: false,
    reconnecting: false,
    reasonCode: null,
    reasonLabel: null,
    ...lastSocketStatus,
    ...detail,
  };
  if (next.connected) {
    next.reasonCode = null;
    next.reasonLabel = null;
  }
  lastSocketStatus = {
    connected: Boolean(next.connected),
    reconnecting: Boolean(next.reconnecting),
    reasonCode: next.reasonCode || null,
    reasonLabel: next.reasonLabel || null,
  };
  if (typeof window === 'undefined') return;
  window.dispatchEvent(
    new CustomEvent(COMPANY_SOCKET_STATE_EVENT, {
      detail: { ...lastSocketStatus },
    })
  );
}

export function getCompanySocketStatusSnapshot() {
  return { ...lastSocketStatus };
}

function nextSocketGeneration() {
  socketGenerationId += 1;
  return socketGenerationId;
}

function isSocketGenerationActive(generation) {
  return generation === socketGenerationId && generation > 0;
}

function recordReconnectFailure() {
  const enteredCooldown = companyReconnectCircuitBreaker.recordFailure(socket);
  if (enteredCooldown && isCompanySocketMetricsDebugEnabled()) {
    // eslint-disable-next-line no-console
    console.warn('[CompanySocket] circuit breaker: cooldown reconnect actif');
  }
}

function isReconnectCooldownActive() {
  return companyReconnectCircuitBreaker.isCooldownActive();
}

function syncListenerMetrics() {
  const byEvent = {};
  let total = 0;
  eventSubscribers.forEach((set, event) => {
    byEvent[event] = set.size;
    total += set.size;
    warnIfTooManyListeners(event, set.size);
  });
  setActiveListenerCount(total, byEvent);
  trackListenerCountForWatchdog(total);
}

function fanOutEvent(event, payload) {
  const subs = eventSubscribers.get(event);
  if (!subs || subs.size === 0) return;
  subs.forEach((callback) => {
    try {
      callback(payload);
    } catch (err) {
      console.error(JSON.stringify({
        event: 'socket_listener_error',
        socket_event: event,
        error: err?.message || String(err),
        timestamp: new Date().toISOString(),
      }));
    }
  });
}

function ensureSocketBridgeListener(event) {
  if (!socket || socketBridgeListeners.has(event)) return;
  const bridge = (payload) => fanOutEvent(event, payload);
  socket.on(event, bridge);
  socketBridgeListeners.set(event, bridge);
}

function removeSocketBridgeListener(event) {
  if (!socket) return;
  const bridge = socketBridgeListeners.get(event);
  if (bridge) {
    socket.off(event, bridge);
    socketBridgeListeners.delete(event);
  }
}

function teardownNetworkListeners() {
  if (!networkListenersSetup || typeof window === 'undefined') return;
  if (onOnlineHandler) {
    window.removeEventListener('online', onOnlineHandler);
    onOnlineHandler = null;
  }
  if (onOfflineHandler) {
    window.removeEventListener('offline', onOfflineHandler);
    onOfflineHandler = null;
  }
  networkListenersSetup = false;
}

function setupNetworkListeners(generation) {
  if (networkListenersSetup || typeof window === 'undefined') return;

  onOnlineHandler = () => {
    if (!isSocketGenerationActive(generation)) return;
    if (isReconnectCooldownActive()) return;
    if (socket && !socket.connected) {
      socket.connect();
    }
  };

  onOfflineHandler = () => {
    if (!isSocketGenerationActive(generation)) return;
    const { reasonCode, reasonLabel } = labelForOffline();
    notifyCompanySocketState({
      connected: false,
      reconnecting: false,
      reasonCode,
      reasonLabel,
    });
  };

  window.addEventListener('online', onOnlineHandler);
  window.addEventListener('offline', onOfflineHandler);
  networkListenersSetup = true;
}

function disposeCompanySocketInstance() {
  nextSocketGeneration();
  teardownNetworkListeners();

  if (socket) {
    socketBridgeListeners.forEach((bridge, event) => {
      try {
        socket.off(event, bridge);
      } catch {}
    });
    socketBridgeListeners.clear();
    try {
      socket.removeAllListeners();
    } catch {}
    try {
      socket.disconnect();
    } catch {}
    socket = null;
  }

  notifyCompanySocketState({
    connected: false,
    reconnecting: false,
    reasonCode: 'DISPOSED',
    reasonLabel: 'Connexion temps réel arrêtée',
  });
}

const getSocketUrl = () => {
  const isProduction = process.env.NODE_ENV === 'production';
  const isDevLocalhost = isDevelopmentLocalhost();
  const metricsDebug = isCompanySocketMetricsDebugEnabled();

  if (!isProduction && isDevLocalhost) {
    if (typeof window !== 'undefined' && window.location) {
      return window.location.origin;
    }
    return process.env.REACT_APP_SOCKET_URL || 'http://127.0.0.1:5000';
  }

  if (metricsDebug) {
    // eslint-disable-next-line no-console
    console.debug('[CompanySocket] getSocketUrl context', {
      isProduction,
      isDevLocalhost,
      NODE_ENV: process.env.NODE_ENV,
    });
  }

  const socketUrl = process.env.REACT_APP_SOCKET_URL;
  if (socketUrl && socketUrl.startsWith('http')) {
    try {
      const url = new URL(socketUrl);
      if (isProduction && !socketUrl.startsWith('https://')) {
        throw new Error('REACT_APP_SOCKET_URL invalide en production');
      }
      return url.origin;
    } catch (e) {
      console.error('Invalid REACT_APP_SOCKET_URL:', socketUrl, e);
    }
  }

  const baseUrl =
    process.env.REACT_APP_API_BASE_URL ||
    process.env.REACT_APP_API_URL;
  if (baseUrl && baseUrl.startsWith('http')) {
    try {
      const url = new URL(baseUrl);
      if (isProduction && !baseUrl.startsWith('https://')) {
        throw new Error('REACT_APP_API_BASE_URL invalide en production');
      }
      return url.origin;
    } catch (e) {
      console.error('Invalid API URL:', baseUrl, e);
    }
  }

  if (isProduction) {
    const errorMsg =
      'REACT_APP_SOCKET_URL ou REACT_APP_API_BASE_URL est requis en production.';
    console.error(errorMsg);
    throw new Error(errorMsg);
  }

  return 'http://127.0.0.1:5000';
};

if (process.env.NODE_ENV === 'production' && typeof window !== 'undefined') {
  if (!process.env.REACT_APP_SOCKET_URL && !process.env.REACT_APP_API_BASE_URL) {
    console.error(
      '[Socket.IO] Variables d\'environnement manquantes en production: ' +
      'REACT_APP_SOCKET_URL ou REACT_APP_API_BASE_URL doit être défini.'
    );
  }
}

async function recoverAuthAndReconnect(targetSocket, generation) {
  if (authRecoveryPromise) {
    return authRecoveryPromise;
  }
  authRecoveryPromise = (async () => {
    try {
      // Cookie-only web : refresh HTTP 200 suffit (pas de JWT lisible en JS).
      const { ensureWebAuthReady } = await import('../utils/ensureUsableAccessToken');
      const ready = await ensureWebAuthReady();
      if (!ready || !isSocketGenerationActive(generation)) {
        return false;
      }
      if (targetSocket?.io?.opts) {
        targetSocket.io.opts.reconnection = true;
      }
      if (targetSocket && !targetSocket.connected) {
        targetSocket.connect();
      } else if (!targetSocket) {
        retryCompanySocket();
      }
      return true;
    } catch {
      return false;
    } finally {
      authRecoveryPromise = null;
    }
  })();
  return authRecoveryPromise;
}

function buildSocketOptions() {
  const jitterDelay = Math.random() * 100;
  const jitterMax = Math.random() * 500;
  const bearerToken = getAccessToken();

  return {
    ...SOCKET_CONFIG,
    path: SOCKET_PATH,
    reconnectionDelay: SOCKET_CONFIG.reconnectionDelay + jitterDelay,
    reconnectionDelayMax: SOCKET_CONFIG.reconnectionDelayMax + jitterMax,
    transports: getSocketTransports(),
    auth: (cb) => {
      // Web cookie-only : cb({}) est normal — le cookie HttpOnly part via withCredentials.
      // Bearer JS uniquement pour mobile bridge / legacy.
      const token = getAccessToken();
      if (token) {
        cb({ token });
      } else {
        cb({});
      }
    },
    // Dev-only metadata sent by Socket.IO's query string. This makes it easy to
    // identify whether the browser even starts the Engine.IO handshake.
    query: isDevelopment
      ? {
          client: 'company-web',
          has_token: bearerToken ? '1' : '0',
        }
      : undefined,
  };
}

function attachSocketHandlers(targetSocket, generation, { onFirstConnectError } = {}) {
  bindUrgentAlertSoundListeners(targetSocket);
  targetSocket.io.on('reconnect_attempt', () => {
    if (!isSocketGenerationActive(generation)) return;
    if (isReconnectCooldownActive()) return;
    recordReconnectAttempt();
    notifyCompanySocketState({ connected: false, reconnecting: true });
  });

  targetSocket.io.on('reconnect_failed', () => {
    if (!isSocketGenerationActive(generation)) return;
    recordReconnectFailure();
    const { reasonCode, reasonLabel } = labelForReconnectFailed();
    notifyCompanySocketState({
      connected: false,
      reconnecting: false,
      reasonCode,
      reasonLabel,
    });
  });

  targetSocket.on('connect', () => {
    if (!isSocketGenerationActive(generation)) return;
    companyReconnectCircuitBreaker.recordSuccess();
    recordConnectReady();
    notifyCompanySocketState({
      connected: true,
      reconnecting: false,
      reasonCode: null,
      reasonLabel: null,
    });

    if (currentCompanyId) {
      try {
        targetSocket.emit('join_company', { company_id: currentCompanyId });
      } catch {}
    }

    setupNetworkListeners(generation);
  });

  targetSocket.on('joined_company', (payload) => {
    if (!isSocketGenerationActive(generation)) return;
    const companyId = payload?.company_id ?? currentCompanyId;
    // `subscribed_cursor` : null = Redis dégradé au moment du join (PR1 dashboard
    // reliability) — voir backend/sockets/chat.py::handle_join_company.
    setSubscribedCursor(companyId, payload?.subscribed_cursor ?? null);
  });

  targetSocket.on('reconnect', () => {
    if (!isSocketGenerationActive(generation)) return;
    if (currentCompanyId) {
      try {
        targetSocket.emit('join_company', { company_id: currentCompanyId });
        targetSocket.emit('get_driver_locations');
        if (typeof window !== 'undefined') {
          window.dispatchEvent(
            new CustomEvent('company_socket_reconnected', {
              detail: { companyId: currentCompanyId, at: Date.now() },
            })
          );
        }
      } catch (err) {
        console.error(JSON.stringify({
          event: 'socket_rejoin_error',
          error: err?.message || String(err),
          timestamp: new Date().toISOString(),
        }));
      }
    }
  });

  targetSocket.on('disconnect', (reason) => {
    if (!isSocketGenerationActive(generation)) return;
    const mapped = labelForDisconnectReason(reason);
    notifyCompanySocketState({
      connected: false,
      reconnecting: false,
      reasonCode: mapped.reasonCode,
      reasonLabel: mapped.reasonLabel,
    });

    const hasLocalToken = !!getAccessToken();
    if (reason === 'io server disconnect' || (reason === 'unauthorized' && hasLocalToken)) {
      if (targetSocket.io?.opts) {
        targetSocket.io.opts.reconnection = false;
      }
    }
  });

  targetSocket.on('connect_error', (err) => {
    if (!isSocketGenerationActive(generation)) return;
    recordReconnectFailure();
    const msg = err?.message || err?.description || String(err || '');
    const mapped = labelForConnectError(msg);
    notifyCompanySocketState({
      connected: false,
      reconnecting: false,
      reasonCode: mapped.reasonCode,
      reasonLabel: mapped.reasonLabel,
    });
    if (typeof onFirstConnectError === 'function') {
      onFirstConnectError(err);
    }
    if (typeof window !== 'undefined') {
      // On ne dispatch `socket_connection_rejected` (qui déclenche un toast utilisateur)
      // que pour les erreurs identifiables comme un refus d'auth/accès côté backend.
      // Les erreurs de transport (websocket error, xhr poll error, timeout, transport
      // close, ECONNRESET…) doivent rester silencieuses : Socket.IO reconnecte tout seul
      // et le circuit breaker remontera l'état via COMPANY_SOCKET_STATE_EVENT en cas
      // d'échec persistant. Sans ce filtre, un simple hoquet réseau produisait une
      // cascade de toasts "auth refusée".
      const authCodes = [
        'AUTH_REQUIRED',
        'AUTH_INVALID',
        'TOKEN_EXPIRED',
        'AUTH_FORBIDDEN',
        'COMPANY_NOT_FOUND',
        'DRIVER_OR_COMPANY_NOT_FOUND',
        'RATE_LIMIT',
      ];
      const isAuthLike = authCodes.some((c) => msg.includes(c));
      const recoverableAuth = ['AUTH_REQUIRED', 'AUTH_INVALID', 'TOKEN_EXPIRED'].some((c) =>
        msg.includes(c)
      );
      if (recoverableAuth) {
        // Ne pas dispatcher `socket_connection_rejected` tout de suite :
        // le dashboard company logout sur AUTH_REQUIRED, ce qui empêche le refresh JWT.
        void recoverAuthAndReconnect(targetSocket, generation).then((ok) => {
          if (ok || !isSocketGenerationActive(generation) || typeof window === 'undefined') {
            return;
          }
          window.dispatchEvent(
            new CustomEvent('socket_connection_rejected', {
              detail: {
                message: msg,
                code: mapped.reasonCode || msg,
                reasonLabel: mapped.reasonLabel,
                originalError: err,
              },
            })
          );
        });
        return;
      }
      if (isAuthLike) {
        window.dispatchEvent(
          new CustomEvent('socket_connection_rejected', {
            detail: {
              message: msg,
              code: mapped.reasonCode || msg,
              reasonLabel: mapped.reasonLabel,
              originalError: err,
            },
          })
        );
      } else {
        // Événement séparé pour la télémétrie / debugging sans déclencher de toast utilisateur.
        window.dispatchEvent(
          new CustomEvent('socket_connect_transport_error', {
            detail: {
              message: msg,
              code: mapped.reasonCode || msg,
              reasonLabel: mapped.reasonLabel,
              originalError: err,
            },
          })
        );
      }
    }
  });

  targetSocket.on('unauthorized', (err) => {
    if (!isSocketGenerationActive(generation)) return;
    const reason = err?.reason || err?.data?.reason;
    if (reason !== 'token_expired' && reason !== 'AUTH_REQUIRED') return;

    // Refresh JWT puis reconnecte (évite handshake auth: {} → AUTH_REQUIRED en boucle).
    void recoverAuthAndReconnect(targetSocket, generation);
  });

  eventSubscribers.forEach((_, event) => {
    ensureSocketBridgeListener(event);
  });
}

export function getCompanySocket() {
  const socketEnabled = process.env.REACT_APP_SOCKET_ENABLED !== 'false';
  if (!socketEnabled) {
    const { reasonCode, reasonLabel } = labelForDisabled();
    notifyCompanySocketState({
      connected: false,
      reconnecting: false,
      reasonCode,
      reasonLabel,
    });
    return null;
  }

  if (socket && socket.connected) return socket;

  // Web cookie-only : absence de JWT en localStorage est normale.
  // Le handshake part avec withCredentials (cookie HttpOnly).
  // AUTH_REQUIRED client uniquement si aucune session UI récupérable.
  if (!getAccessToken()) {
    try {
      const { hasActiveSession } = require('../utils/webAuthSession');
      if (!hasActiveSession()) {
        const { reasonCode, reasonLabel } = labelForMissingToken();
        notifyCompanySocketState({
          connected: false,
          reconnecting: false,
          reasonCode,
          reasonLabel,
        });
        return socket || null;
      }
    } catch (_) {
      // Continuer le handshake cookie-only
    }
  }

  if (typeof navigator !== 'undefined' && navigator.onLine === false) {
    const { reasonCode, reasonLabel } = labelForOffline();
    notifyCompanySocketState({
      connected: false,
      reconnecting: false,
      reasonCode,
      reasonLabel,
    });
  }

  if (!connectPromise) {
    recordConnectStarted();

    connectPromise = new Promise((resolve, reject) => {
      let settled = false;
      const settleResolve = (value) => {
        if (settled) return;
        settled = true;
        resolve(value);
      };
      const settleReject = (error) => {
        if (settled) return;
        settled = true;
        connectPromise = null;
        reject(error);
      };

      try {
        if (socket) {
          disposeCompanySocketInstance();
        }

        const activeGeneration = nextSocketGeneration();
        const socketOptions = buildSocketOptions();
        const socketUrl = getSocketUrl();
        if (isDevelopment) {
          // eslint-disable-next-line no-console
          console.info('[CompanySocket] creating socket', {
            socketUrl,
            path: socketOptions.path,
            transports: socketOptions.transports,
            hasAccessToken: Boolean(getAccessToken()),
          });
        }
        socket = io(socketUrl, socketOptions);
        attachSocketHandlers(socket, activeGeneration, {
          onFirstConnectError: settleReject,
        });

        const onInitialConnect = () => {
          if (!isSocketGenerationActive(activeGeneration)) {
            socket.off('connect', onInitialConnect);
            return;
          }
          socket.off('connect', onInitialConnect);
          settleResolve(socket);
        };

        socket.on('connect', onInitialConnect);
        if (!socket.connected) {
          socket.connect();
        }

        if (socket.connected) {
          socket.off('connect', onInitialConnect);
          settleResolve(socket);
        }
      } catch (e) {
        connectPromise = null;
        reject(e);
      }
    });
  }

  return socket || null;
}

export async function ensureCompanySocket() {
  const existing = getCompanySocket();
  if (existing && existing.connected) return existing;
  if (!connectPromise) return null;
  try {
    return await connectPromise;
  } catch {
    return null;
  }
}

export async function joinCompanyRoom(companyId) {
  const s = await ensureCompanySocket();
  if (!s) return;
  currentCompanyId = companyId;
  if (lastJoinCompanyEmitId === companyId) {
    return;
  }
  lastJoinCompanyEmitId = companyId;
  try {
    s.emit('join_company', { company_id: companyId });
  } catch {}
  try {
    s.emit('join_company_room', { company_id: companyId });
  } catch {}
}

export async function leaveCompanyRoom(companyId) {
  const s = await ensureCompanySocket();
  if (!s) return;
  try {
    s.emit('leave_company_room', { company_id: companyId });
  } catch {}
  currentCompanyId = null;
  lastJoinCompanyEmitId = null;
}

function subscribeToEvent(event, callback) {
  if (!eventSubscribers.has(event)) {
    eventSubscribers.set(event, new Set());
  }
  const subs = eventSubscribers.get(event);
  subs.add(callback);
  syncListenerMetrics();

  const s = getCompanySocket();
  if (s) {
    ensureSocketBridgeListener(event);
  }

  return () => unsubscribeFromEvent(event, callback);
}

function unsubscribeFromEvent(event, callback) {
  const subs = eventSubscribers.get(event);
  if (!subs) return;
  if (callback) {
    subs.delete(callback);
  } else {
    subs.clear();
  }
  if (subs.size === 0) {
    eventSubscribers.delete(event);
    removeSocketBridgeListener(event);
  }
  syncListenerMetrics();
}

export async function onDriverLocationUpdate(callback) {
  return subscribeToEvent('driver_location_update', callback);
}

export async function offDriverLocationUpdate(callback) {
  unsubscribeFromEvent('driver_location_update', callback);
}

export async function on(event, callback) {
  return subscribeToEvent(event, callback);
}

export async function once(event, callback) {
  const s = await ensureCompanySocket();
  if (!s) return undefined;
  const generation = socketGenerationId;
  const wrapper = (...args) => {
    if (!isSocketGenerationActive(generation)) return;
    s.off(event, wrapper);
    callback(...args);
  };
  s.once(event, wrapper);
  return () => s.off(event, wrapper);
}

export async function off(event, callback) {
  unsubscribeFromEvent(event, callback);
}

export async function waitUntilConnected(timeoutMs = 10000) {
  const s = await ensureCompanySocket();
  if (!s) return null;
  if (s.connected) return s;

  const generation = socketGenerationId;

  return new Promise((resolve) => {
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(result);
    };

    const timer = setTimeout(() => {
      finish(s.connected && isSocketGenerationActive(generation) ? s : null);
    }, timeoutMs);

    const onConnect = () => {
      if (!isSocketGenerationActive(generation)) {
        finish(null);
        return;
      }
      finish(s);
    };

    const cleanup = () => {
      clearTimeout(timer);
      s.off('connect', onConnect);
    };

    s.once('connect', onConnect);
  });
}

export function getSocketId() {
  return socket?.id || null;
}

export function getCompanySocketMetrics() {
  return getCompanySocketMetricsSnapshot();
}

export function disconnectCompanySocket() {
  try {
    connectPromise = null;
    currentCompanyId = null;
    lastJoinCompanyEmitId = null;
    companyReconnectCircuitBreaker.reset(socket);
    disposeCompanySocketInstance();
    eventSubscribers.clear();
    syncListenerMetrics();
  } catch (err) {
    console.error(JSON.stringify({
      event: 'socket_disconnect_error',
      error: err?.message || String(err),
      timestamp: new Date().toISOString(),
    }));
  }
}

/** Relance manuelle depuis le badge (après échec / session rétablie). */
export function retryCompanySocket() {
  try {
    if (socket && !socket.connected) {
      if (socket.io?.opts) {
        socket.io.opts.reconnection = true;
      }
      socket.connect();
      notifyCompanySocketState({
        connected: false,
        reconnecting: true,
        reasonCode: 'RETRY',
        reasonLabel: 'Nouvelle tentative de connexion…',
      });
      return socket;
    }
    disconnectCompanySocket();
    return getCompanySocket();
  } catch (err) {
    const mapped = labelForConnectError(err?.message || String(err));
    notifyCompanySocketState({
      connected: false,
      reconnecting: false,
      reasonCode: mapped.reasonCode,
      reasonLabel: mapped.reasonLabel,
    });
    return null;
  }
}
