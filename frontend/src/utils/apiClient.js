// frontend/src/utils/apiClient.js
import axios from 'axios';
// ✅ P1-1: getRefreshToken n'est plus utilisé (cookies httpOnly)
// import { getRefreshToken } from '../hooks/useAuthToken';
import {
  getCompanyScopedAccessToken,
  getAuthEnv as getSessionAuthEnv,
  getEnvAccessToken,
  getEnvRefreshToken,
  getEnvUser,
  removeLegacyGlobalTokens,
  setAuthEnv as setSessionAuthEnv,
} from './webAuthSession';
import { notifySessionReauthRequired } from './deferredSessionLogout';
import { requestAuthNavigate } from './authNavigation';
import {
  beginExplicitLogout,
  endExplicitLogout,
  isExplicitLogoutInProgress,
  isLoginSessionInProgress,
} from './sessionLogoutState';
import {
  isUserRecentlyActive,
  SESSION_WORKING_LOOKBACK_MS,
} from './userActivityTracker';

let baseApiRest = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '/api/v1';

let socketTarget = process.env.REACT_APP_SOCKET_URL || '/socket.io';

const AUTH_ENV_STORAGE_KEY = 'lirie_auth_env';
const APP_ENV_KEY = 'app';
const DEMO_ENV_KEY = 'demo';
const LOCAL_UNIFIED_HOST_REGEX = /^(localhost|127\.0\.0\.1)$/i;
const PROD_UNIFIED_HOST_REGEX = /(^|\.)lirie\.ch$/i;
// Dev: /api/app (setupProxy rewrite). Prod: /api/v1 (api/app inexistant)
const API_BASES = {
  gateway: '/api/gateway',
  app: process.env.NODE_ENV === 'development' ? '/api/app' : '/api/v1',
  demo: '/api/demo',
};

const _isCompanyRoutePath = (url = '') => {
  const targetUrl = String(url || '');
  return (
    targetUrl.startsWith('/companies/') ||
    targetUrl.startsWith('/company_dispatch') ||
    targetUrl.startsWith('/company-settings')
  );
};

const parseOptionalBoolean = (value) => {
  if (value == null) return null;
  const normalized = String(value).trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  return null;
};

const isLocalhostDev3000 = () => {
  try {
    return (
      typeof window !== 'undefined' &&
      window.location &&
      /^(localhost|127\.0\.0\.1):3000$/i.test(window.location.host)
    );
  } catch (_) {
    return false;
  }
};

const isGatewayAuthRoute = (url = '') => {
  const targetUrl = String(url || '');
  return (
    targetUrl.startsWith('/auth/login') ||
    targetUrl.startsWith('/auth/context') ||
    targetUrl.startsWith('/auth/csrf-token')
  );
};

const isUnifiedGatewayHost = () => {
  try {
    const forcedUnifiedMode = parseOptionalBoolean(process.env.REACT_APP_UNIFIED_LOGIN_MODE);
    if (forcedUnifiedMode !== null) {
      return forcedUnifiedMode;
    }

    const hostname = window?.location?.hostname || '';
    const port = window?.location?.port || '';
    const isLocalUnifiedDevHost =
      LOCAL_UNIFIED_HOST_REGEX.test(hostname) && port === '3000';
    const isProdUnifiedHost = PROD_UNIFIED_HOST_REGEX.test(hostname);

    return Boolean(
      typeof window !== 'undefined' &&
      window.location &&
      (isLocalUnifiedDevHost || isProdUnifiedHost)
    );
  } catch (_) {
    return false;
  }
};

export const getCurrentAuthEnv = () => {
  return getSessionAuthEnv();
};

export const setCurrentAuthEnv = (env) => {
  return setSessionAuthEnv(env);
};

const getStorageTokenByEnv = (env) =>
  getEnvAccessToken(env, { allowLegacy: false });

const _getStorageRefreshByEnv = (env) =>
  getEnvRefreshToken(env, { allowLegacy: false });

const getStoredRoleFromEnv = (env) => {
  try {
    const parsed = getEnvUser(env);
    return String(parsed?.role || '')
      .trim()
      .toLowerCase();
  } catch (_) {
    return '';
  }
};

const getScopedCompanyAccessToken = (env) => {
  if (env === DEMO_ENV_KEY) {
    // En demo, utiliser strictement le token demo pour éviter les mélanges app<->demo.
    return getStorageTokenByEnv(DEMO_ENV_KEY);
  }
  return getCompanyScopedAccessToken(APP_ENV_KEY) || getStorageTokenByEnv(APP_ENV_KEY);
};

const resolveApiBase = (url) => {
  if (!isUnifiedGatewayHost()) {
    return baseApiRest;
  }

  if (isGatewayAuthRoute(url)) {
    return API_BASES.gateway;
  }

  const env = getCurrentAuthEnv();
  return env === DEMO_ENV_KEY ? API_BASES.demo : API_BASES.app;
};

// En dev (CRA sur localhost:3000), on force le proxy '/api' pour éviter le CORS
try {
  if (isLocalhostDev3000()) {
    // En dev, utiliser explicitement /api/v1 pour s'aligner avec le backend versionné
    // Même si une variable d'env API existe, on force le proxy local pour que le mode
    // unifié (gateway/app/demo) passe par setupProxy et évite les blocages CSRF cross-origin.
    baseApiRest = '/api/v1';
    socketTarget = '/socket.io';
  }
} catch (_) {
  // no-op
}

/** Profil client, listes de réservations, etc. peuvent dépasser 30s (charge serveur, réseau, requêtes parallèles). */
const DEFAULT_REST_TIMEOUT_MS = 60_000;

const apiRest = axios.create({
  baseURL: baseApiRest,
  headers: {
    'Content-Type': 'application/json; charset=utf-8',
    Accept: 'application/json; charset=utf-8',
  },
  withCredentials: true, // ✅ Activer les cookies httpOnly pour l'authentification
  timeout: DEFAULT_REST_TIMEOUT_MS,
  responseType: 'json',
  responseEncoding: 'utf8',
});

export const apiSocket = axios.create({
  baseURL: socketTarget,
  timeout: 30000,
});

// ✅ Gestion du token CSRF pour les requêtes mutantes
let csrfToken = null;
let csrfTokenExpiry = null;

const getCsrfToken = async (baseURLForCsrf = baseApiRest) => {
  // Vérifier si le token est encore valide
  if (csrfToken && csrfTokenExpiry && Date.now() < csrfTokenExpiry) {
    return csrfToken;
  }

  try {
    // Récupérer un nouveau token CSRF en utilisant axios directement pour éviter les dépendances circulaires
    const response = await axios.get(`${baseURLForCsrf}/auth/csrf-token`, {
      withCredentials: true,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    });
    csrfToken = response.data.csrf_token;
    const ttl = response.data.ttl || 3600; // TTL en secondes
    csrfTokenExpiry = Date.now() + (ttl * 1000) - 60000; // Expirer 1 minute avant pour éviter les problèmes
    return csrfToken;
  } catch (error) {
    console.warn('⚠️ Impossible de récupérer le token CSRF:', error);
    return null;
  }
};

const addAuthHeader = async (cfg = {}) => {
  if (!cfg.headers) {
    cfg.headers = {};
  }

  const isRelativeAppPath =
    typeof cfg.url === 'string' &&
    !cfg.url.startsWith('/api/') &&
    !cfg.url.startsWith('http');
  const hasDefaultApiBase =
    !cfg.baseURL ||
    cfg.baseURL === baseApiRest ||
    cfg.baseURL === process.env.REACT_APP_API_BASE_URL ||
    cfg.baseURL === process.env.REACT_APP_API_URL;

  if (
    isUnifiedGatewayHost() &&
    !cfg.skipEnvRouting &&
    isRelativeAppPath &&
    hasDefaultApiBase
  ) {
    cfg.baseURL = resolveApiBase(cfg.url);
  }

  if (cfg.baseURL && cfg.baseURL.endsWith('/')) {
    cfg.baseURL = cfg.baseURL.slice(0, -1);
  }

  if (isUnifiedGatewayHost() && !cfg.headers.Authorization) {
    const explicitEnv =
      cfg._targetEnv === DEMO_ENV_KEY || cfg._targetEnv === APP_ENV_KEY
        ? cfg._targetEnv
        : null;
    const env =
      explicitEnv ||
      (cfg.baseURL === API_BASES.demo
        ? DEMO_ENV_KEY
        : cfg.baseURL === API_BASES.app
          ? APP_ENV_KEY
          : getCurrentAuthEnv());
    cfg._targetEnv = env;
    const role = getStoredRoleFromEnv(env);
    const envToken = getStorageTokenByEnv(env);
    const legacyCrossEnvToken =
      process.env.REACT_APP_ENABLE_LEGACY_GLOBAL_TOKEN_FALLBACK === 'true' && env === APP_ENV_KEY
        ? getEnvAccessToken(APP_ENV_KEY, { allowLegacy: true })
        : null;
    const companyScopedToken =
      role === 'company' || role === 'admin' ? getScopedCompanyAccessToken(env) : null;
    // Pour baseURL === /api/demo : jamais authToken/refreshToken (legacy app)
    const token =
      env === DEMO_ENV_KEY
        ? envToken || companyScopedToken
        : envToken || companyScopedToken || legacyCrossEnvToken;
    if (token) {
      cfg.headers.Authorization = `Bearer ${token}`;
    }
  }

  // ✅ Ajouter le token CSRF pour les requêtes mutantes (POST, PUT, DELETE, PATCH)
  const method = cfg.method?.toUpperCase() || (cfg.url ? 'GET' : 'GET');
  if (cfg.skipCsrf) {
    return cfg;
  }

  // Le login/context gateway est un endpoint d'entrée public: ne pas précharger un CSRF app/demo.
  if (
    isUnifiedGatewayHost() &&
    cfg.baseURL === API_BASES.gateway &&
    isGatewayAuthRoute(cfg.url)
  ) {
    return cfg;
  }

  if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
    const csrfBase = isUnifiedGatewayHost()
      ? cfg._targetEnv === DEMO_ENV_KEY
        ? API_BASES.demo
        : API_BASES.app
      : baseApiRest;
    const csrf = await getCsrfToken(csrfBase);
    if (csrf) {
      cfg.headers['X-CSRF-Token'] = csrf;
    }
  }

  return cfg;
};

// ✅ Garde anti-régression dashboard company : company_dispatch/* => uniquement company_access_token (jamais driver_access_token)
const COMPANY_ACCESS_TOKEN_KEY = 'company_access_token';

apiRest.interceptors.request.use((config) => {
  const base = config.baseURL ?? '';
  const url = config.url ?? '';
  const fullUrl = typeof url === 'string' && url.startsWith('http') ? url : `${base}${url}`;

  if (fullUrl.includes('/company_dispatch')) {
    delete config.headers.Authorization;
    delete config.headers.authorization;
    if (config.headers.common) {
      delete config.headers.common.Authorization;
      delete config.headers.common.authorization;
    }
    const env = getCurrentAuthEnv();
    let companyToken = getScopedCompanyAccessToken(env);

    if (!companyToken) {
      const role = getStoredRoleFromEnv(env);
      const envToken = getStorageTokenByEnv(env);
      if ((role === 'company' || role === 'admin') && envToken) {
        companyToken = envToken;
        try {
          if (env === APP_ENV_KEY) {
            localStorage.setItem(COMPANY_ACCESS_TOKEN_KEY, envToken);
          }
        } catch (_) {
          // no-op
        }
      }
    }
    if (companyToken) {
      config.headers.Authorization = `Bearer ${companyToken}`;
      return config;
    }
    // ✅ Fail fast : impossible de réintroduire le bug 401 sur company_dispatch
    return Promise.reject(
      new Error('Company dispatch called without company_access_token')
    );
  }
  return config;
});

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

const FRESH_TOKEN_REQUIRED_MESSAGE =
  'Cette action nécessite une reconnexion récente. Veuillez vous reconnecter pour continuer.';

export const AUTH_TOKEN_NOT_FRESH = 'AUTH_TOKEN_NOT_FRESH';

export const isAuthRefreshInProgress = () => isRefreshing;

const waitForAuthRefreshIdle = async (timeoutMs = 5000) => {
  const started = Date.now();
  while (isAuthRefreshInProgress()) {
    if (Date.now() - started > timeoutMs) {
      break;
    }
    await new Promise((resolve) => {
      setTimeout(resolve, 50);
    });
  }
};

const isFreshTokenErrorPayload = (errorData = {}) => {
  const errorMsg = (errorData.msg || errorData.error || errorData.message || '').toLowerCase();
  return (
    errorMsg.includes('fresh') ||
    errorMsg.includes('frais') ||
    errorMsg.includes('fresh token required') ||
    errorMsg.includes('token must be fresh') ||
    errorMsg.includes('only fresh tokens') ||
    errorMsg.includes("n'est pas frais") ||
    errorMsg.includes('token n\'est pas frais')
  );
};

const requestDeferredSessionLogout = (cfg = {}) => {
  if (cfg.skipFreshTokenLogout || cfg.skipAuthRedirect || isLoginSessionInProgress()) {
    return;
  }

  void (async () => {
    try {
      const { tryRefreshSessionIfNeeded } = await import('./sessionKeepAlive');
      if (await tryRefreshSessionIfNeeded({ force: true })) {
        return;
      }
    } catch (_) {
      // ignore
    }

    notifySessionReauthRequired({
      silentUntilIdle: isUserRecentlyActive(SESSION_WORKING_LOOKBACK_MS),
    });
  })();
};

/** Renouvelle access + refresh token (cookies httpOnly + miroir localStorage). */
export async function refreshSessionTokens(targetEnv = getCurrentAuthEnv()) {
  if (isRefreshing) {
    // Pendant un logout explicite, ne pas attendre un refresh en cours (risque de blocage infini).
    if (isExplicitLogoutInProgress()) {
      return false;
    }
    return new Promise((resolve, reject) => {
      failedQueue.push({
        resolve: () => resolve(true),
        reject,
      });
    });
  }

  isRefreshing = true;
  try {
    const refreshToken = getEnvRefreshToken(targetEnv, { allowLegacy: true });
    const refreshBase = isUnifiedGatewayHost()
      ? targetEnv === DEMO_ENV_KEY
        ? API_BASES.demo
        : API_BASES.app
      : undefined;

    const refreshResponse = await apiClient.post(
      '/auth/refresh-token',
      refreshToken ? { refresh_token: refreshToken } : {},
      {
        skipAuthRedirect: true,
        _targetEnv: targetEnv,
        ...(refreshBase ? { baseURL: refreshBase } : {}),
      }
    );

    const refreshed = refreshResponse?.data || {};
    const nextAccessToken = refreshed.access_token || refreshed.token;
    const nextRefreshToken = refreshed.refresh_token;
    if (!isExplicitLogoutInProgress()) {
      if (nextAccessToken) {
        if (targetEnv === DEMO_ENV_KEY) {
          localStorage.setItem('demo_access_token', nextAccessToken);
        } else {
          localStorage.setItem('app_access_token', nextAccessToken);
        }
      }
      if (nextRefreshToken) {
        if (targetEnv === DEMO_ENV_KEY) {
          localStorage.setItem('demo_refresh_token', nextRefreshToken);
        } else {
          localStorage.setItem('app_refresh_token', nextRefreshToken);
        }
      }
      removeLegacyGlobalTokens();
    }
    processQueue(null, null);
    return true;
  } catch (error) {
    processQueue(error, null);
    throw error;
  } finally {
    isRefreshing = false;
  }
};

const rejectFreshTokenRequired = (error, cfg = {}) => {
  requestDeferredSessionLogout(cfg);
  return Promise.reject({
    ...error,
    code: AUTH_TOKEN_NOT_FRESH,
    isFreshTokenRequired: true,
    message: FRESH_TOKEN_REQUIRED_MESSAGE,
  });
};

export const cleanLocalSession = () => {
  localStorage.removeItem(AUTH_ENV_STORAGE_KEY);
  localStorage.removeItem('app_user');
  localStorage.removeItem('app_public_id');
  localStorage.removeItem('app_access_token');
  localStorage.removeItem('app_refresh_token');
  localStorage.removeItem('demo_user');
  localStorage.removeItem('demo_public_id');
  localStorage.removeItem('demo_access_token');
  localStorage.removeItem('demo_refresh_token');
  // Legacy
  localStorage.removeItem('user');
  localStorage.removeItem('public_id');
  localStorage.removeItem('authToken');
  localStorage.removeItem('refreshToken');
  // Company (snake_case + ancien camelCase pendant migration)
  localStorage.removeItem('company_user');
  localStorage.removeItem('company_public_id');
  localStorage.removeItem('company_access_token');
  localStorage.removeItem('company_refresh_token');
  localStorage.removeItem('company_authToken');
  localStorage.removeItem('company_refreshToken');
  // Driver
  localStorage.removeItem('driver_user');
  localStorage.removeItem('driver_public_id');
  localStorage.removeItem('driver_access_token');
  localStorage.removeItem('driver_refresh_token');
  localStorage.removeItem('driver_authToken');
  localStorage.removeItem('driver_refreshToken');
  // Institution
  localStorage.removeItem('institution_user');
  localStorage.removeItem('institution_public_id');
  localStorage.removeItem('institution_access_token');
  localStorage.removeItem('institution_refresh_token');
  if (apiRest.defaults?.headers?.common) {
    delete apiRest.defaults.headers.common.Authorization;
    delete apiRest.defaults.headers.common.authorization;
  }
};

const LOGOUT_SERVER_TIMEOUT_MS = 8000;

const withTimeout = (promise, timeoutMs, label = 'operation') =>
  Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => {
        reject(new Error(`${label} timeout after ${timeoutMs}ms`));
      }, timeoutMs);
    }),
  ]);

export const logoutUser = async (options = {}) => {
  const { redirect = true, preserveNext = false } = options;

  await waitForAuthRefreshIdle();
  beginExplicitLogout();

  try {
    try {
      const { cancelDeferredLogout } = await import('./deferredSessionLogout');
      cancelDeferredLogout();
    } catch (_) {
      // ignore
    }

    try {
      const { suspendSessionKeepAlive } = await import('./sessionKeepAlive');
      suspendSessionKeepAlive();
    } catch (_) {
      // ignore
    }

    const postLogout = () =>
      apiClient.post('/auth/logout', {}, { skipAuthRedirect: true });

    const serverCleanup = (async () => {
      try {
        // Révoque les tokens côté serveur et supprime les cookies httpOnly.
        // Sans cet appel, auth-changed relance /auth/me avec les cookies encore valides
        // et reconnecte l'utilisateur immédiatement après le nettoyage local.
        await postLogout();
      } catch (error) {
        // Ne pas relancer refreshSessionTokens ici : peut bloquer indéfiniment si un refresh
        // est déjà en cours. Le nettoyage local dans finally reste la source de vérité UI.
        console.warn(
          '⚠️ Déconnexion serveur impossible (session locale nettoyée quand même):',
          error?.response?.data || error?.message || error
        );
      }
    })();

    try {
      await withTimeout(serverCleanup, LOGOUT_SERVER_TIMEOUT_MS, 'logout server cleanup');
    } catch (error) {
      console.warn(
        '⚠️ Déconnexion serveur interrompue (session locale nettoyée quand même):',
        error?.message || error
      );
    }
  } finally {
    csrfToken = null;
    csrfTokenExpiry = null;
    cleanLocalSession();

    try {
      const { disconnectCompanySocket } = await import('../services/companySocket');
      disconnectCompanySocket();
    } catch (_) {
      // ignore
    }

    // Vider le cache React Query pour éviter les données stale entre comptes
    try {
      const { queryClient } = require('../App');
      queryClient.clear();
    } catch (_) {
      // Silencieux si App pas encore chargé
    }

    window.dispatchEvent(new Event('auth-changed'));
    endExplicitLogout();

    if (redirect !== false) {
      const path = `${window.location.pathname}${window.location.search || ''}`;
      const isAlreadyLogin = window.location.pathname === '/login';
      const loginPath =
        preserveNext && !isAlreadyLogin && path && path !== '/'
          ? `/login?next=${encodeURIComponent(path)}`
          : '/login';
      requestAuthNavigate(loginPath, { replace: true });
    }
  }
};

apiClient.interceptors.response.use(
  (res) => res,
  async (error) => {
    const status = error?.response?.status;
    const cfg = error?.config || {};
    const requestUrl = String(cfg.url || '');
    const isAuthEndpoint =
      requestUrl.includes('/auth/login') ||
      requestUrl.includes('/auth/refresh-token') ||
      requestUrl.includes('/auth/refresh') ||
      requestUrl.includes('/auth/logout') ||
      requestUrl.includes('/auth/passwordless/');
    // Aucun fallback cross-env : requêtes demo ne retentent jamais sur /api/app

    // Message sympa pour 429 (limiter)
    if (status === 429) {
      console.warn('Vous avez effectué trop de requêtes. Merci de patienter un peu.');
    }

    // ✅ Gestion 401 avec refresh automatique
    if (status === 401 && !cfg.skipAuthRedirect && !isAuthEndpoint) {
      // Token non-fresh : déconnexion + redirection login (sauf opt-out explicite)
      if (isFreshTokenErrorPayload(error?.response?.data)) {
        return rejectFreshTokenRequired(error, cfg);
      }
      
      // Si déjà en train de refresh une requête /auth/refresh-token, éviter boucle
      if (requestUrl.includes('/auth/refresh-token')) {
        requestDeferredSessionLogout(cfg);
        return Promise.reject(error);
      }

      // Si déjà en train de refresh, mettre en queue
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then(() => {
            // ✅ P1-1: Les nouveaux tokens sont dans les cookies
            // Pas besoin d'ajouter Authorization header
            return apiClient(cfg); // Retry requête originale
          })
          .catch((err) => {
            return Promise.reject(err);
          });
      }

      // Premier 401 → tenter refresh
      isRefreshing = true;

      try {
        const targetEnv = cfg._targetEnv || getCurrentAuthEnv();
        await refreshSessionTokens(targetEnv);

        // ✅ P1-1: Retry requête originale
        cfg._retryAfterRefresh = true;
        delete cfg._isRetry;
        try {
          const retryResponse = await apiClient(cfg);
          return retryResponse;
        } catch (retryError) {
          throw retryError;
        }
      } catch (refreshError) {
        processQueue(refreshError, null);

        if (process.env.NODE_ENV === 'production') {
          import('@sentry/react')
            .then((Sentry) => {
              Sentry.captureException(refreshError, {
                tags: { auth_flow: 'refresh_token_failed' },
                extra: {
                  requestUrl: cfg?.url,
                  requestMethod: cfg?.method,
                },
              });
            })
            .catch(() => {});
        }

        if (isFreshTokenErrorPayload(refreshError?.response?.data)) {
          return rejectFreshTokenRequired(refreshError, cfg);
        }
        // Ne pas faire de logout global si l'erreur est cross-env (requête demo avec session app ou inversement)
        const requestEnv = cfg.baseURL === API_BASES.demo ? DEMO_ENV_KEY : cfg.baseURL === API_BASES.app ? APP_ENV_KEY : null;
        const currentEnv = getCurrentAuthEnv();
        const isCrossEnvError = requestEnv && requestEnv !== currentEnv;
        if (!isCrossEnvError) {
          requestDeferredSessionLogout(cfg);
        }
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    if (status === 401 && isFreshTokenErrorPayload(error?.response?.data)) {
      return rejectFreshTokenRequired(error, cfg);
    }
    // Pas de fallback automatique vers /api/v1: on reste sur la vérité du backend
    return Promise.reject(error);
  }
);

export default apiClient;
