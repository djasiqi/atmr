/**
 * Socket.IO — portail client (room `client_<public_id>` côté serveur).
 * Écoute `client_booking_updated` (jalons transport : acceptation, assignation, en route).
 */

import { io } from 'socket.io-client';
import { getAccessToken } from '../hooks/useAuthToken';
import { SOCKET_CONFIG, SOCKET_PATH, getSocketTransports, isDevelopmentLocalhost } from '../config/socketConfig';

let socket = null;
let connectPromise = null;

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
      return new URL(socketUrl).origin;
    } catch (e) {
      console.error('Invalid REACT_APP_SOCKET_URL:', socketUrl, e);
    }
  }

  const baseUrl = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL;
  if (baseUrl && baseUrl.startsWith('http')) {
    try {
      return new URL(baseUrl).origin;
    } catch (e) {
      console.error('Invalid API URL:', baseUrl, e);
    }
  }

  if (isProduction) {
    throw new Error('REACT_APP_SOCKET_URL or REACT_APP_API_BASE_URL required in production');
  }

  return 'http://127.0.0.1:5000';
};

function buildSocketOptions() {
  const jitterDelay = Math.random() * 100;
  const jitterMax = Math.random() * 500;

  return {
    ...SOCKET_CONFIG,
    path: SOCKET_PATH,
    reconnectionDelay: SOCKET_CONFIG.reconnectionDelay + jitterDelay,
    reconnectionDelayMax: SOCKET_CONFIG.reconnectionDelayMax + jitterMax,
    transports: getSocketTransports(),
    auth: (cb) => {
      const token = getAccessToken();
      if (token) {
        cb({ token });
      } else {
        cb({});
      }
    },
  };
}

export function getClientPortalSocket() {
  const socketEnabled = process.env.REACT_APP_SOCKET_ENABLED !== 'false';
  if (!socketEnabled) {
    return null;
  }

  if (socket && socket.connected) return socket;

  if (!connectPromise) {
    connectPromise = new Promise((resolve, reject) => {
      try {
        const socketOptions = buildSocketOptions();
        const socketUrl = getSocketUrl();
        socket = io(socketUrl, socketOptions);

        socket.on('connect', () => {
          resolve(socket);
        });

        socket.on('disconnect', () => {
          connectPromise = null;
        });

        socket.on('connect_error', (err) => {
          console.warn('[ClientPortalSocket] connect_error', err?.message || err);
          connectPromise = null;
          reject(err);
        });
      } catch (e) {
        connectPromise = null;
        reject(e);
      }
    });
  }

  return socket || null;
}

export async function ensureClientPortalSocket() {
  getClientPortalSocket();
  if (socket?.connected) return socket;
  if (!connectPromise) return null;
  try {
    return await connectPromise;
  } catch {
    return null;
  }
}

export function disconnectClientPortalSocket() {
  try {
    socket?.removeAllListeners();
    socket?.disconnect();
  } catch {
    /* ignore */
  }
  socket = null;
  connectPromise = null;
}
