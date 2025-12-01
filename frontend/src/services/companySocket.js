import { io } from 'socket.io-client';
import { getAccessToken } from '../hooks/useAuthToken';

let socket = null;
let connectPromise = null;
const listeners = new Map(); // event -> callback
let currentCompanyId = null; // company to (re)join on connect/reconnect
let lastHeartbeat = Date.now(); // Track last heartbeat timestamp
let heartbeatInterval = null; // Interval for sending pings
const pingTimestamps = new Map(); // Track ping timestamps for latency metrics
let networkListenersSetup = false; // Track if network listeners are set up

// En mode développement (localhost:3000), utiliser le proxy (plus fiable sur Windows/Docker)
const isDevelopmentLocalhost =
  typeof window !== 'undefined' && window.location && /localhost:3000$/i.test(window.location.host);

const API_URL = (() => {
  if (isDevelopmentLocalhost) {
    return 'http://127.0.0.1:5000';
  }

  const baseUrl =
    process.env.REACT_APP_SOCKET_URL ||
    process.env.REACT_APP_API_BASE_URL ||
    process.env.REACT_APP_API_URL;
  if (baseUrl && baseUrl.startsWith('http')) {
    try {
      const url = new URL(baseUrl);
      return url.origin;
    } catch (e) {
      console.error('Invalid SOCKET URL:', baseUrl);
      return window.location.origin;
    }
  }

  return window.location.origin;
})();

function buildSocketOptions() {
  // ✅ Jitter anti-storm: ajouter variation aléatoire pour éviter reconnexions simultanées
  const jitterDelay = Math.random() * 100; // 0-100ms
  const jitterMax = Math.random() * 500; // 0-500ms
  
  const base = {
    path: '/socket.io',
    // 🔒 Auth dynamique : sera rappelé à chaque (re)connexion
    auth: (cb) => {
      const token = getAccessToken();
      cb({ token });
    },
    reconnection: true,
    reconnectionAttempts: Infinity,   // ✅ Changé de 5 à Infinity pour reconnexion infinie
    reconnectionDelay: 1000 + jitterDelay,  // ✅ Jitter: 1000-1100ms
    reconnectionDelayMax: 10000 + jitterMax,  // ✅ Jitter: 10000-10500ms
    timeout: 20000,
    forceNew: false,
    withCredentials: true,
    transports: isDevelopmentLocalhost ? ['websocket', 'polling'] : ['websocket', 'polling'],
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
          // ✅ Logs structurés
          console.log(JSON.stringify({
            event: 'socket_connect',
            socket_id: socket.id,
            timestamp: new Date().toISOString(),
            user_type: 'company'
          }));
          
          // Rejoindre automatiquement la room entreprise si connue
          if (currentCompanyId) {
            try {
              socket.emit('join_company', { company_id: currentCompanyId });
            } catch {}
          }
          
          // ✅ Heartbeat : envoyer ping toutes les 30s et écouter pong
          lastHeartbeat = Date.now();
          
          // Écouter les pong du serveur avec tracking latence
          socket.on('pong', (data) => {
            const pingTime = pingTimestamps.get('last_ping');
            if (pingTime) {
              const latency = Date.now() - pingTime;
              
              // ✅ Métriques latence
              console.log(JSON.stringify({
                event: 'heartbeat_pong',
                latency_ms: latency,
                timestamp: data?.timestamp,
                received_at: new Date().toISOString()
              }));
              
              // Warning si latence élevée
              if (latency > 500) {
                console.warn(JSON.stringify({
                  event: 'heartbeat_high_latency',
                  latency_ms: latency,
                  threshold_ms: 500,
                  timestamp: new Date().toISOString()
                }));
              }
              
              pingTimestamps.delete('last_ping');
            }
            lastHeartbeat = Date.now();
          });
          
          // Envoyer des ping toutes les 30s
          if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
          }
          heartbeatInterval = setInterval(() => {
            if (socket && socket.connected) {
              const timeSinceLastHeartbeat = Date.now() - lastHeartbeat;
              // Si pas de pong depuis >60s, forcer reconnexion
              if (timeSinceLastHeartbeat > 60000) {
                console.warn(JSON.stringify({
                  event: 'heartbeat_timeout',
                  time_since_last_ms: timeSinceLastHeartbeat,
                  threshold_ms: 60000,
                  action: 'force_reconnect',
                  timestamp: new Date().toISOString()
                }));
                socket.disconnect();
                socket.connect();
                lastHeartbeat = Date.now();
              } else {
                socket.emit('ping');
                // ✅ Track timestamp avant ping
                pingTimestamps.set('last_ping', Date.now());
                console.log(JSON.stringify({
                  event: 'heartbeat_ping',
                  timestamp: new Date().toISOString()
                }));
              }
            }
          }, 30000); // Toutes les 30s
          
          // ✅ Détection réseau (online/offline)
          if (!networkListenersSetup && typeof window !== 'undefined') {
            window.addEventListener('online', () => {
              console.log(JSON.stringify({
                event: 'network_online',
                timestamp: new Date().toISOString()
              }));
              if (socket && !socket.connected) {
                console.log(JSON.stringify({
                  event: 'socket_reconnect_triggered',
                  reason: 'network_online',
                  timestamp: new Date().toISOString()
                }));
                socket.connect();
              }
            });
            
            window.addEventListener('offline', () => {
              console.log(JSON.stringify({
                event: 'network_offline',
                timestamp: new Date().toISOString()
              }));
            });
            
            networkListenersSetup = true;
          }
          
          resolve(socket);
        });

        // ✅ Handler reconnect : rejoin automatiquement les rooms et relancer heartbeat
        socket.on('reconnect', (attempt) => {
          // ✅ Logs structurés
          console.log(JSON.stringify({
            event: 'socket_reconnect',
            attempt: attempt,
            timestamp: new Date().toISOString()
          }));
          lastHeartbeat = Date.now();
          
          // Rejoin automatique des rooms
          if (currentCompanyId) {
            try {
              socket.emit('join_company', { company_id: currentCompanyId });
              console.log(JSON.stringify({
                event: 'socket_rejoin_room',
                room: 'company',
                company_id: currentCompanyId,
                timestamp: new Date().toISOString()
              }));
            } catch (err) {
              console.error(JSON.stringify({
                event: 'socket_rejoin_error',
                error: err?.message || String(err),
                timestamp: new Date().toISOString()
              }));
            }
          }
          
          // Relancer heartbeat si pas déjà actif
          if (!heartbeatInterval) {
            heartbeatInterval = setInterval(() => {
              if (socket && socket.connected) {
                const timeSinceLastHeartbeat = Date.now() - lastHeartbeat;
                if (timeSinceLastHeartbeat > 60000) {
                  console.warn(JSON.stringify({
                    event: 'heartbeat_timeout',
                    time_since_last_ms: timeSinceLastHeartbeat,
                    action: 'force_reconnect',
                    timestamp: new Date().toISOString()
                  }));
                  socket.disconnect();
                  socket.connect();
                  lastHeartbeat = Date.now();
                } else {
                  socket.emit('ping');
                  pingTimestamps.set('last_ping', Date.now());
                  console.log(JSON.stringify({
                    event: 'heartbeat_ping',
                    timestamp: new Date().toISOString()
                  }));
                }
              }
            }, 30000);
          }
        });

        socket.on('disconnect', (reason) => {
          // ✅ Logs structurés
          console.log(JSON.stringify({
            event: 'socket_disconnect',
            reason: reason,
            timestamp: new Date().toISOString()
          }));
          connectPromise = null;
          // Nettoyer l'interval heartbeat
          if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
            heartbeatInterval = null;
          }
        });

        socket.on('connect_error', (err) => {
          console.error(JSON.stringify({
            event: 'socket_connect_error',
            error: err?.message || String(err),
            timestamp: new Date().toISOString()
          }));
          connectPromise = null;
          reject(err);
        });

        socket.on('unauthorized', (err) => {
          console.error(JSON.stringify({
            event: 'socket_unauthorized',
            error: err?.error || String(err),
            timestamp: new Date().toISOString()
          }));
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
  currentCompanyId = companyId;
  // Compat: certains backends écoutent join_company, d'autres join_company_room
  try {
    s.emit('join_company', { company_id: companyId });
  } catch {}
  try {
    s.emit('join_company_room', { company_id: companyId });
  } catch {}
}

// ✅ Quitter la room (optionnel si le serveur expose un handler)
export async function leaveCompanyRoom(companyId) {
  const s = await ensureCompanySocket();
  if (!s) return;
  try {
    s.emit('leave_company_room', { company_id: companyId });
  } catch {}
  currentCompanyId = null;
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
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
      heartbeatInterval = null;
    }
    pingTimestamps.clear(); // Nettoyer timestamps
    if (socket) {
      socket.disconnect();
      socket = null;
    }
    
    // ✅ Nettoyer listeners réseau (optionnel, mais propre)
    if (networkListenersSetup && typeof window !== 'undefined') {
      // Les listeners online/offline peuvent rester, mais on pourrait les nettoyer si nécessaire
      networkListenersSetup = false;
    }
    
    console.log(JSON.stringify({
      event: 'socket_disconnect_cleanup',
      timestamp: new Date().toISOString()
    }));
  } catch (err) {
    console.error(JSON.stringify({
      event: 'socket_disconnect_error',
      error: err?.message || String(err),
      timestamp: new Date().toISOString()
    }));
  }
}
