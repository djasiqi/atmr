import { io } from 'socket.io-client';
import { getAccessToken } from '../hooks/useAuthToken';

let socket = null;
let connectPromise = null;
const listeners = new Map(); // event -> callback

// En mode développement (localhost:3000), utiliser le proxy (plus fiable sur Windows/Docker)
const isDevelopmentLocalhost =
  typeof window !== 'undefined' && window.location && /localhost:3000$/i.test(window.location.host);

const API_URL = (() => {
  // En dev avec localhost:3000, utiliser le proxy
  if (isDevelopmentLocalhost) {
    return window.location.origin; // localhost:3000 -> via proxy
  }

  const baseUrl = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '/api';
  // If it's a full URL, extract the origin
  if (baseUrl.startsWith('http')) {
    try {
      const url = new URL(baseUrl);
      return url.origin;
    } catch (e) {
      console.error('Invalid API URL:', baseUrl);
      return window.location.origin;
    }
  }
  // If it's a relative path like "/api", use window.location.origin
  return window.location.origin;
})();

function buildSocketOptions() {
  // En dev via proxy: utiliser uniquement polling (plus fiable avec webpack-dev-server)
  // En prod: laisser Socket.IO gérer (WS prioritaire, fallback polling)
  const isDev =
    typeof window !== 'undefined' &&
    window.location &&
    /localhost:3000$/i.test(window.location.host);

  const base = {
    path: '/socket.io',
    // 🔒 Auth dynamique : sera rappelé à chaque (re)connexion
    auth: (cb) => {
      const token = getAccessToken();
      cb({ token });
    },
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 20000,
    forceNew: false,
    withCredentials: true,
    // En dev: polling uniquement (WebSocket upgrade échoue via proxy webpack-dev-server)
    transports: isDev ? ['polling'] : ['polling', 'websocket'],
  };
  return base;
}

export function getCompanySocket() {
  if (socket && socket.connected) return socket;

  if (!connectPromise) {
    connectPromise = new Promise((resolve, reject) => {
      try {
        // token lu dynamiquement via buildSocketOptions.auth()
        // eslint-disable-next-line no-console
        console.log('[CompanySocket] Connexion à:', API_URL);
        socket = io(API_URL, buildSocketOptions());

        socket.on('connect', () => {
          // eslint-disable-next-line no-console
          console.log('✅ WebSocket connecté (company)', socket.id);
          resolve(socket);
        });

        socket.on('disconnect', (reason) => {
          // eslint-disable-next-line no-console
          console.log('🔌 WebSocket déconnecté:', reason);
          connectPromise = null;
        });

        socket.on('connect_error', (err) => {
          console.error('⛔ Erreur de connexion WebSocket:', err?.message || err);
          connectPromise = null;
          reject(err);
        });

        socket.on('unauthorized', (err) => {
          console.error('⛔ Unauthorized WebSocket:', err);
        });
      } catch (e) {
        console.error('❌ Socket init error:', e);
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
  return connectPromise;
}

// ✅ Rejoindre une room d’entreprise (legacy no-op: le backend joint déjà la room à la connexion côté 'company')
export async function joinCompanyRoom(companyId) {
  const s = await ensureCompanySocket();
  if (!s) return;
  // Optionnel: si un handler existe côté serveur
  s.emit('join_company_room', { company_id: companyId });
}

// ✅ Quitter la room (optionnel si le serveur expose un handler)
export async function leaveCompanyRoom(companyId) {
  const s = await ensureCompanySocket();
  if (!s) return;
  s.emit('leave_company_room', { company_id: companyId });
}

// ✅ Écouter les mises à jour de localisation des chauffeurs
export async function onDriverLocationUpdate(callback) {
  const s = await ensureCompanySocket();
  if (!s) return;
  // Remplace l’éventuel listener existant pour éviter les doublons
  const evt = 'driver_location';
  const prev = listeners.get(evt);
  if (prev) s.off(evt, prev);
  s.on(evt, callback);
  listeners.set(evt, callback);
}

// ✅ Arrêter d’écouter les mises à jour
export async function offDriverLocationUpdate() {
  const s = await ensureCompanySocket();
  if (!s) return;
  const evt = 'driver_location';
  const prev = listeners.get(evt);
  if (prev) {
    s.off(evt, prev);
    listeners.delete(evt);
  }
}

// 🔧 Utilitaires génériques d'abonnement (évite la multiplication de helpers spécifiques)
export async function on(event, callback) {
  const s = await ensureCompanySocket();
  if (!s) return;
  const prev = listeners.get(event);
  if (prev) s.off(event, prev);
  s.on(event, callback);
  listeners.set(event, callback);
}

export async function once(event, callback) {
  const s = await ensureCompanySocket();
  if (!s) return;
  s.once(event, callback);
}

export async function off(event) {
  const s = await ensureCompanySocket();
  if (!s) return;
  const prev = listeners.get(event);
  if (prev) {
    s.off(event, prev);
    listeners.delete(event);
  }
}

export async function waitUntilConnected(timeoutMs = 10000) {
  const start = Date.now();
  let s = await ensureCompanySocket();
  while (s && !s.connected && Date.now() - start < timeoutMs) {
    await new Promise((r) => setTimeout(r, 100));
    s = socket;
  }
  return s?.connected ? s : null;
}

export function getSocketId() {
  return socket?.id || null;
}

// ✅ Fermeture propre (ex. au logout)
export function disconnectCompanySocket() {
  try {
    listeners.forEach((cb, evt) => {
      socket?.off(evt, cb);
    });
    listeners.clear();
    connectPromise = null;
    if (socket) {
      socket.disconnect();
      socket = null;
    }
  } catch {}
}
