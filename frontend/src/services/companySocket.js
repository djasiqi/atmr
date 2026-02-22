import { io } from 'socket.io-client';
import { getAccessToken } from '../hooks/useAuthToken';
import { SOCKET_CONFIG, SOCKET_PATH, getSocketTransports, isDevelopmentLocalhost } from '../config/socketConfig';

let socket = null;
let connectPromise = null;
const listeners = new Map(); // event -> callback
let currentCompanyId = null; // company to (re)join on connect/reconnect
// ✅ Heartbeat géré nativement par Socket.IO (ping_interval=25s, ping_timeout=60s)
// Suppression du heartbeat applicatif redondant
let networkListenersSetup = false; // Track if network listeners are set up

// ✅ isDevelopmentLocalhost importé depuis socketConfig.js

// ✅ Socket.IO doit utiliser le proxy en développement pour éviter les problèmes CORS/cookies
// Le proxy est configuré dans setupProxy.js pour /socket.io
const getSocketUrl = () => {
  const isProduction = process.env.NODE_ENV === 'production';
  
  // ✅ Détecter dynamiquement si on est en développement localhost
  const isDevLocalhost = isDevelopmentLocalhost();
  
  // ✅ PRIORITÉ 1: En développement localhost, TOUJOURS utiliser le proxy
  // (même si REACT_APP_SOCKET_URL est définie, pour éviter les problèmes CORS/cookies)
  if (!isProduction && isDevLocalhost) {
    // ✅ Utiliser le proxy relatif pour Socket.IO (via setupProxy.js)
    // Cela permet d'envoyer les cookies httpOnly correctement
    if (typeof window !== 'undefined' && window.location) {
      // Utiliser le proxy relatif pour Socket.IO
      const proxyUrl = window.location.origin;
      // eslint-disable-next-line no-console
      console.log('[CompanySocket] Mode développement détecté, utilisation du proxy:', proxyUrl);
      return proxyUrl;
    }
    // Fallback si window n'est pas disponible
    const devSocketUrl = process.env.REACT_APP_SOCKET_URL || 'http://127.0.0.1:5000';
    // eslint-disable-next-line no-console
    console.warn('[CompanySocket] window.location non disponible, fallback:', devSocketUrl);
    return devSocketUrl;
  }
  
  // ✅ DEBUG: Log toutes les valeurs pour comprendre le comportement
  // eslint-disable-next-line no-console
  console.log('[CompanySocket] getSocketUrl() - Debug:', {
    isProduction,
    isDevelopmentLocalhost: isDevLocalhost,
    NODE_ENV: process.env.NODE_ENV,
    REACT_APP_SOCKET_URL: process.env.REACT_APP_SOCKET_URL,
    REACT_APP_API_BASE_URL: process.env.REACT_APP_API_BASE_URL,
    REACT_APP_API_URL: process.env.REACT_APP_API_URL,
    windowLocation: typeof window !== 'undefined' && window.location ? window.location.host : 'N/A',
    windowOrigin: typeof window !== 'undefined' && window.location ? window.location.origin : 'N/A',
  });
  
  // ✅ PRIORITÉ 2: Variable d'environnement dédiée pour Socket.IO (production uniquement)
  const socketUrl = process.env.REACT_APP_SOCKET_URL;
  if (socketUrl && socketUrl.startsWith('http')) {
    // eslint-disable-next-line no-console
    console.log('[CompanySocket] REACT_APP_SOCKET_URL trouvée:', socketUrl);
    try {
      const url = new URL(socketUrl);
      // Validation en production : doit être HTTPS
      if (isProduction && !socketUrl.startsWith('https://')) {
        console.error(
          'REACT_APP_SOCKET_URL doit commencer par "https://" en production. ' +
          `Valeur actuelle: ${socketUrl}`
        );
        throw new Error('REACT_APP_SOCKET_URL invalide en production');
      }
      return url.origin;
    } catch (e) {
      console.error('Invalid REACT_APP_SOCKET_URL:', socketUrl, e);
    }
  }

  // ✅ PRIORITÉ 3: Variables d'environnement API (fallback)
  const baseUrl =
    process.env.REACT_APP_API_BASE_URL ||
    process.env.REACT_APP_API_URL;
  if (baseUrl && baseUrl.startsWith('http')) {
    // eslint-disable-next-line no-console
    console.log('[CompanySocket] REACT_APP_API_BASE_URL/API_URL trouvée:', baseUrl);
    try {
      const url = new URL(baseUrl);
      // Validation en production : doit être HTTPS
      if (isProduction && !baseUrl.startsWith('https://')) {
        console.error(
          'REACT_APP_API_BASE_URL doit commencer par "https://" en production. ' +
          `Valeur actuelle: ${baseUrl}`
        );
        throw new Error('REACT_APP_API_BASE_URL invalide en production');
      }
      return url.origin;
    } catch (e) {
      console.error('Invalid API URL:', baseUrl, e);
    }
  }
  
  // Log pour déboguer si on n'utilise pas le proxy
  if (!isProduction && !isDevLocalhost) {
    // eslint-disable-next-line no-console
    console.log('[CompanySocket] Mode développement mais isDevelopmentLocalhost=false', {
      isDevelopmentLocalhost: isDevLocalhost,
      windowLocation: typeof window !== 'undefined' && window.location ? window.location.host : 'N/A',
    });
  }

  // ✅ PRIORITÉ 4: Production - erreur si aucune variable n'est définie
  if (isProduction) {
    const errorMsg =
      'REACT_APP_SOCKET_URL ou REACT_APP_API_BASE_URL est requis en production. ' +
      'Définissez au moins une de ces variables d\'environnement avec une URL HTTPS valide.';
    console.error(errorMsg);
    throw new Error(errorMsg);
  }

  // Fallback pour développement (si NODE_ENV n'est pas 'production')
  return 'http://127.0.0.1:5000';
};

// ✅ Validation au démarrage en production
if (process.env.NODE_ENV === 'production' && typeof window !== 'undefined') {
  if (!process.env.REACT_APP_SOCKET_URL && !process.env.REACT_APP_API_BASE_URL) {
    console.error(
      '[Socket.IO] ⚠️ Variables d\'environnement manquantes en production: ' +
      'REACT_APP_SOCKET_URL ou REACT_APP_API_BASE_URL doit être défini.'
    );
  }
}

// ✅ SOCKET_PATH importé depuis socketConfig.js

function buildSocketOptions() {
  // ✅ Jitter anti-storm: ajouter variation aléatoire pour éviter reconnexions simultanées
  const jitterDelay = Math.random() * 100; // 0-100ms
  const jitterMax = Math.random() * 500; // 0-500ms
  
  return {
    ...SOCKET_CONFIG,
    // ✅ Override path pour utiliser SOCKET_PATH (peut être différent de SOCKET_CONFIG.path)
    path: SOCKET_PATH,
    // ✅ Appliquer jitter aux délais de reconnexion
    reconnectionDelay: SOCKET_CONFIG.reconnectionDelay + jitterDelay,
    reconnectionDelayMax: SOCKET_CONFIG.reconnectionDelayMax + jitterMax,
    // ✅ Utiliser transports adaptés (polling d'abord en dev localhost)
    transports: getSocketTransports(),
    // 🔒 Auth dynamique : sera rappelé à chaque (re)connexion
    // ✅ Support cookies httpOnly : si pas de token dans localStorage, utiliser uniquement les cookies
    auth: (cb) => {
      const token = getAccessToken();
      // ✅ Si token disponible (mode mobile), l'envoyer dans le payload
      // ✅ Sinon, envoyer objet vide pour utiliser uniquement les cookies httpOnly (mode web)
      if (token) {
        cb({ token });
      } else {
        // Mode cookies httpOnly : ne pas envoyer de payload auth, le serveur lira les cookies
        cb({});
      }
    },
  };
}

export function getCompanySocket() {
  // ✅ Feature flag : désactiver Socket.IO si SOCKET_ENABLED=false
  const socketEnabled = process.env.REACT_APP_SOCKET_ENABLED !== 'false';
  if (!socketEnabled) {
    console.log('[CompanySocket] Socket.IO désactivé (REACT_APP_SOCKET_ENABLED=false)');
    return null;
  }
  
  if (socket && socket.connected) return socket;

  if (!connectPromise) {
    connectPromise = new Promise((resolve, reject) => {
      try {
        // token lu dynamiquement via buildSocketOptions.auth()
        const socketOptions = buildSocketOptions();
        // ✅ Obtenir l'URL Socket.IO dynamiquement (pour utiliser le proxy en dev)
        const socketUrl = getSocketUrl();
        // eslint-disable-next-line no-console
        console.log('[CompanySocket] Connexion à:', socketUrl, 'avec path:', socketOptions.path);
        socket = io(socketUrl, socketOptions);

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
          
          // ✅ Heartbeat géré nativement par Socket.IO (ping_interval=25s, ping_timeout=60s)
          // Le heartbeat applicatif a été supprimé car redondant
          
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

        // ✅ Handler reconnect : rejoin automatiquement les rooms
        socket.on('reconnect', (attempt) => {
          // ✅ Logs structurés
          console.log(JSON.stringify({
            event: 'socket_reconnect',
            attempt: attempt,
            timestamp: new Date().toISOString()
          }));
          
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
          
          // ✅ Heartbeat géré nativement par Socket.IO, pas besoin de relancer
        });

        socket.on('disconnect', (reason) => {
          // ✅ Logs structurés
          console.log(JSON.stringify({
            event: 'socket_disconnect',
            reason: reason,
            timestamp: new Date().toISOString()
          }));
          connectPromise = null;
          
          // ✅ Ne pas reconnecter si serveur a déconnecté volontairement
          // NOTE:
          // - `unauthorized` arrive souvent quand un vieux `authToken` expiré traîne en localStorage.
          //   Dans ce cas, on veut **pouvoir** se reconnecter (souvent via cookies httpOnly).
          // - On stoppe donc la reconnexion uniquement si on a encore un token local à envoyer.
          const hasLocalToken = !!getAccessToken();
          if (reason === 'io server disconnect' || (reason === 'unauthorized' && hasLocalToken)) {
            console.log(JSON.stringify({
              event: 'socket_disconnect_stop_reconnect',
              reason: reason,
              timestamp: new Date().toISOString()
            }));
            // Désactiver la reconnexion automatique
            socket.io.reconnect = false;
            return; // Pas de reconnexion auto
          }
          
          // ✅ Heartbeat géré nativement par Socket.IO, pas de nettoyage nécessaire
        });

        socket.on('connect_error', (err) => {
          const msg = err?.message || err?.description || (err?.data && err.data.message) || String(err || '');
          console.error(JSON.stringify({
            event: 'socket_connect_error',
            error: msg,
            timestamp: new Date().toISOString()
          }));
          connectPromise = null;
          // Backend envoie maintenant des codes via ConnectionRefusedError (AUTH_REQUIRED, TOKEN_EXPIRED, etc.)
          // L’app peut écouter socket_connection_rejected pour afficher un toast / logout.
          if (typeof window !== 'undefined') {
            window.dispatchEvent(
              new CustomEvent('socket_connection_rejected', {
                detail: { message: msg, code: msg, originalError: err },
              })
            );
          }
          reject(err);
        });

        socket.on('unauthorized', (err) => {
          console.error(JSON.stringify({
            event: 'socket_unauthorized',
            error: err?.error || String(err),
            timestamp: new Date().toISOString()
          }));

          // ✅ Auto-récupération: si token expiré, on le purge pour basculer sur cookies httpOnly
          // et on autorise une reconnexion.
          const reason = err?.reason || err?.data?.reason;
          if (reason === 'token_expired') {
            try {
              localStorage.removeItem('authToken');
              localStorage.removeItem('refreshToken');
            } catch {}
            try {
              socket.io.reconnect = true;
              // Relancer une connexion (Socket.IO gère le backoff)
              socket.connect();
            } catch {}
          }
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
    // ✅ Heartbeat géré nativement par Socket.IO, pas de nettoyage nécessaire
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

