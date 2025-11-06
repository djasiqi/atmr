// frontend/src/utils/apiClient.js
import axios from 'axios';
import { getRefreshToken } from '../hooks/useAuthToken';

let baseApiRest = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '/api/v1';

let socketTarget = process.env.REACT_APP_SOCKET_URL || '/socket.io';

// En dev (CRA sur localhost:3000), on force le proxy '/api' pour éviter le CORS
try {
  if (
    typeof window !== 'undefined' &&
    window.location &&
    /localhost:3000$/i.test(window.location.host)
  ) {
    // En dev, utiliser explicitement /api/v1 pour s'aligner avec le backend versionné
    baseApiRest = '/api/v1';
    socketTarget = '/socket.io';
  }
} catch (_) {
  // no-op
}

const apiRest = axios.create({
  baseURL: baseApiRest,
  headers: {
    'Content-Type': 'application/json; charset=utf-8',
    Accept: 'application/json; charset=utf-8',
  },
  withCredentials: false,
  timeout: 30000,
  responseType: 'json',
  responseEncoding: 'utf8',
});

export const apiSocket = axios.create({
  baseURL: socketTarget,
  timeout: 30000,
});

const addAuthHeader = (cfg = {}) => {
  if (!cfg.headers) {
    cfg.headers = {};
  }

  const token = localStorage.getItem('authToken');
  const hasAuthHeader = cfg.headers.Authorization;

  if (cfg.baseURL && cfg.baseURL.endsWith('/')) {
    cfg.baseURL = cfg.baseURL.slice(0, -1);
  }

  if (
    token &&
    !hasAuthHeader &&
    !cfg.url?.includes('/auth/refresh-token') &&
    !(cfg.headers['X-Skip-Auth'] || cfg.headers['x-skip-auth'])
  ) {
    cfg.headers.Authorization = `Bearer ${token}`;
  }

  return cfg;
};

apiRest.interceptors.request.use(addAuthHeader);
apiSocket.interceptors.request.use(addAuthHeader);

export const apiClient = apiRest;

// ✅ Flag pour éviter boucle infinie refresh
let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

export const cleanLocalSession = () => {
  localStorage.removeItem('authToken');
  localStorage.removeItem('refreshToken');
  localStorage.removeItem('user');
  localStorage.removeItem('public_id');
};

export const logoutUser = async (options = { redirect: true }) => {
  try {
    await apiClient.delete('/shadow-mode/session', {
      baseURL: '/api',
      skipAuthRedirect: true,
    });
  } catch (error) {
    console.warn(
      '⚠️ Impossible de désactiver le Shadow Mode lors de la déconnexion:',
      error?.response?.data || error?.message || error
    );
  } finally {
    cleanLocalSession();

    if (options?.redirect !== false) {
      window.location.href = '/login';
    }
  }
};

apiClient.interceptors.response.use(
  (res) => res,
  async (error) => {
    const status = error?.response?.status;
    const cfg = error?.config || {};

    // Message sympa pour 429 (limiter)
    if (status === 429) {
      console.warn('Vous avez effectué trop de requêtes. Merci de patienter un peu.');
    }

    // ✅ Gestion 401 avec refresh automatique
    if (status === 401 && !cfg.skipAuthRedirect) {
      const refreshToken = getRefreshToken();

      // Si pas de refresh token ou déjà en train de refresh une requête /auth/refresh-token
      if (!refreshToken || cfg.url?.includes('/auth/refresh-token')) {
        logoutUser();
        return Promise.reject(error);
      }

      // Si déjà en train de refresh, mettre en queue
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then((token) => {
            cfg.headers.Authorization = `Bearer ${token}`;
            return apiClient(cfg); // Retry requête originale
          })
          .catch((err) => {
            return Promise.reject(err);
          });
      }

      // Premier 401 → tenter refresh
      isRefreshing = true;

      try {
        const response = await apiClient.post(
          '/auth/refresh-token',
          {},
          {
            headers: {
              Authorization: `Bearer ${refreshToken}`,
            },
            skipAuthRedirect: true, // Éviter boucle
          }
        );

        const newToken = response.data.access_token;
        localStorage.setItem('authToken', newToken);

        // Process queued requests
        processQueue(null, newToken);

        // Retry requête originale avec nouveau token
        cfg.headers.Authorization = `Bearer ${newToken}`;
        // ⚡ Marquer que c'est un retry après refresh réussi pour éviter logs d'erreur
        cfg._retryAfterRefresh = true;
        // ⚡ Supprimer l'erreur de la config pour éviter les logs Axios
        delete cfg._isRetry;
        const retryResponse = await apiClient(cfg);
        // ✅ Refresh réussi → retourner la réponse réussie (pas l'erreur 401 initiale)
        return retryResponse;
      } catch (refreshError) {
        processQueue(refreshError, null);
        logoutUser();
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    // Pas de fallback automatique vers /api/v1: on reste sur la vérité du backend
    return Promise.reject(error);
  }
);

export default apiClient;
