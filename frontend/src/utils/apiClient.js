// frontend/src/utils/apiClient.js
import axios from 'axios';
import { getRefreshToken } from '../hooks/useAuthToken';

let apiBase = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '/api';

// En dev (CRA sur localhost:3000), on force le proxy '/api' pour éviter le CORS
try {
  if (
    typeof window !== 'undefined' &&
    window.location &&
    /localhost:3000$/i.test(window.location.host)
  ) {
    // En dev, reste strictement sur /api pour s'aligner avec le backend (pas de /v1 implicite)
    apiBase = '/api';
  }
} catch (_) {
  // no-op
}

const apiClient = axios.create({
  baseURL: apiBase,
  headers: {
    'Content-Type': 'application/json; charset=utf-8',
    Accept: 'application/json; charset=utf-8',
  },
  withCredentials: false, // ❌ pas de cookies, on est en JWT header
  timeout: 30000,
  // ✅ Force UTF-8 encoding pour toutes les réponses
  responseType: 'json',
  responseEncoding: 'utf8',
});

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

export const logoutUser = () => {
  localStorage.removeItem('authToken');
  localStorage.removeItem('refreshToken');
  localStorage.removeItem('user');
  localStorage.removeItem('public_id');
  window.location.href = '/login';
};

apiClient.interceptors.request.use((cfg) => {
  // normalise baseURL (évite //api si jamais apiBase finit par /)
  if (cfg.baseURL && cfg.baseURL.endsWith('/')) {
    cfg.baseURL = cfg.baseURL.slice(0, -1);
  }

  // NE PAS remplacer Authorization si déjà présent (cas du refresh token)
  const hasAuthHeader = cfg.headers && cfg.headers.Authorization;
  const token = localStorage.getItem('authToken');

  // N'ajoute pas Authorization si :
  // - Le header Authorization est déjà défini (refresh token)
  // - On a un opt-out explicite
  // - C'est une requête vers /auth/refresh-token
  if (
    !hasAuthHeader &&
    token &&
    !cfg.url?.includes('/auth/refresh-token') &&
    !(cfg.headers && (cfg.headers['X-Skip-Auth'] || cfg.headers['x-skip-auth']))
  ) {
    cfg.headers.Authorization = `Bearer ${token}`;
  }
  return cfg;
});

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

        // Retry requête originale
        cfg.headers.Authorization = `Bearer ${newToken}`;
        return apiClient(cfg);
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
