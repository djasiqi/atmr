// services/institutionSocket.js
/**
 * ÉTAPE 6: Service Socket.IO pour le portail Institution
 * 
 * Gère la connexion temps réel et les événements:
 * - request_sent
 * - offer_accepted
 * - request_converted
 * - booking_status_updated
 * - request_cancelled
 * - booking_cancelled
 */

import { io } from 'socket.io-client';
import { SOCKET_CONFIG, SOCKET_PATH, getSocketTransports, isDevelopmentLocalhost } from '../config/socketConfig';

let socket = null;
let connectPromise = null;
const listeners = new Map();
let currentInstitutionId = null;

/**
 * Obtient l'URL Socket.IO (réutilise la logique de companySocket)
 */
const getSocketUrl = () => {
  const isProduction = process.env.NODE_ENV === 'production';
  const isDevLocalhost = isDevelopmentLocalhost();
  
  if (!isProduction && isDevLocalhost) {
    if (typeof window !== 'undefined' && window.location) {
      return window.location.origin;
    }
    return process.env.REACT_APP_SOCKET_URL || 'http://127.0.0.1:5000';
  }
  
  const socketUrl = process.env.REACT_APP_SOCKET_URL;
  if (socketUrl && socketUrl.startsWith('http')) {
    try {
      const url = new URL(socketUrl);
      return url.origin;
    } catch (e) {
      console.error('Invalid REACT_APP_SOCKET_URL:', socketUrl, e);
    }
  }

  const baseUrl = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL;
  if (baseUrl && baseUrl.startsWith('http')) {
    try {
      const url = new URL(baseUrl);
      return url.origin;
    } catch (e) {
      console.error('Invalid API URL:', baseUrl, e);
    }
  }
  
  if (isProduction) {
    throw new Error('REACT_APP_SOCKET_URL or REACT_APP_API_BASE_URL required in production');
  }

  return 'http://127.0.0.1:5000';
};

/**
 * Construit les options Socket.IO
 */
function buildSocketOptions() {
  const jitterDelay = Math.random() * 100;
  const jitterMax = Math.random() * 500;
  
  return {
    ...SOCKET_CONFIG,
    path: SOCKET_PATH,
    reconnectionDelay: SOCKET_CONFIG.reconnectionDelay + jitterDelay,
    reconnectionDelayMax: SOCKET_CONFIG.reconnectionDelayMax + jitterMax,
    transports: getSocketTransports(),
    // Auth via cookies httpOnly
    auth: (cb) => {
      cb({});
    },
  };
}

/**
 * Obtient ou crée la connexion Socket.IO institution
 */
export function getInstitutionSocket() {
  const socketEnabled = process.env.REACT_APP_SOCKET_ENABLED !== 'false';
  if (!socketEnabled) {
    console.log('[InstitutionSocket] Socket.IO désactivé');
    return null;
  }
  
  if (socket && socket.connected) return socket;

  if (!connectPromise) {
    connectPromise = new Promise((resolve, reject) => {
      try {
        const socketOptions = buildSocketOptions();
        const socketUrl = getSocketUrl();
        
        console.log('[InstitutionSocket] Connexion à:', socketUrl);
        socket = io(socketUrl, socketOptions);

        socket.on('connect', () => {
          console.log('[InstitutionSocket] Connecté, socket_id:', socket.id);
          
          // Rejoindre la room institution
          if (currentInstitutionId) {
            socket.emit('join_institution', { institution_id: currentInstitutionId });
          }
          
          resolve(socket);
        });

        socket.on('reconnect', (attempt) => {
          console.log('[InstitutionSocket] Reconnecté après', attempt, 'tentatives');
          
          if (currentInstitutionId) {
            socket.emit('join_institution', { institution_id: currentInstitutionId });
          }
        });

        socket.on('disconnect', (reason) => {
          console.log('[InstitutionSocket] Déconnecté:', reason);
          connectPromise = null;
        });

        socket.on('connect_error', (err) => {
          console.error('[InstitutionSocket] Erreur connexion:', err?.message || err);
          connectPromise = null;
          reject(err);
        });

      } catch (e) {
        console.error('[InstitutionSocket] Erreur init:', e);
        connectPromise = null;
        reject(e);
      }
    });
  }
  
  return socket || null;
}

/**
 * S'assure que la socket est connectée
 */
export async function ensureInstitutionSocket() {
  const existing = getInstitutionSocket();
  if (existing && existing.connected) return existing;
  if (!connectPromise) return null;
  return connectPromise;
}

/**
 * Rejoint la room de l'institution
 */
export async function joinInstitutionRoom(institutionId) {
  const s = await ensureInstitutionSocket();
  if (!s) return;
  currentInstitutionId = institutionId;
  s.emit('join_institution', { institution_id: institutionId });
}

/**
 * Quitte la room de l'institution
 */
export async function leaveInstitutionRoom(institutionId) {
  const s = await ensureInstitutionSocket();
  if (!s) return;
  s.emit('leave_institution_room', { institution_id: institutionId });
  currentInstitutionId = null;
}

/**
 * Écoute un événement
 */
export async function on(event, callback) {
  const s = await ensureInstitutionSocket();
  if (!s) return;
  const prev = listeners.get(event);
  if (prev) s.off(event, prev);
  s.on(event, callback);
  listeners.set(event, callback);
}

/**
 * Arrête d'écouter un événement
 */
export async function off(event) {
  const s = await ensureInstitutionSocket();
  if (!s) return;
  const prev = listeners.get(event);
  if (prev) {
    s.off(event, prev);
    listeners.delete(event);
  }
}

/**
 * Déconnecte proprement
 */
export function disconnectInstitutionSocket() {
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
    console.log('[InstitutionSocket] Déconnexion propre');
  } catch (err) {
    console.error('[InstitutionSocket] Erreur déconnexion:', err);
  }
}

// ============================================================================
// Événements spécifiques Institution (helpers)
// ============================================================================

/**
 * Écoute l'événement request_sent
 */
export async function onRequestSent(callback) {
  await on('request_sent', callback);
}

/**
 * Écoute l'événement offer_accepted
 */
export async function onOfferAccepted(callback) {
  await on('offer_accepted', callback);
}

/**
 * Écoute l'événement request_converted
 */
export async function onRequestConverted(callback) {
  await on('request_converted', callback);
}

/**
 * Écoute l'événement booking_status_updated
 */
export async function onBookingStatusUpdated(callback) {
  await on('booking_status_updated', callback);
}

/**
 * Écoute l'événement request_cancelled
 */
export async function onRequestCancelled(callback) {
  await on('request_cancelled', callback);
}

/**
 * Écoute l'événement booking_cancelled
 */
export async function onBookingCancelled(callback) {
  await on('booking_cancelled', callback);
}

const institutionSocketApi = {
  getInstitutionSocket,
  ensureInstitutionSocket,
  joinInstitutionRoom,
  leaveInstitutionRoom,
  on,
  off,
  disconnectInstitutionSocket,
  onRequestSent,
  onOfferAccepted,
  onRequestConverted,
  onBookingStatusUpdated,
  onRequestCancelled,
  onBookingCancelled,
};

export default institutionSocketApi;
