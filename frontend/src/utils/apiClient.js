// frontend/src/utils/apiClient.js
import axios from 'axios';
// ✅ P1-1: getRefreshToken n'est plus utilisé (cookies httpOnly)
// import { getRefreshToken } from '../hooks/useAuthToken';

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
  withCredentials: true, // ✅ Activer les cookies httpOnly pour l'authentification
  timeout: 30000,
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

const getCsrfToken = async () => {
  // Vérifier si le token est encore valide
  if (csrfToken && csrfTokenExpiry && Date.now() < csrfTokenExpiry) {
    return csrfToken;
  }

  try {
    // Récupérer un nouveau token CSRF en utilisant axios directement pour éviter les dépendances circulaires
    const response = await axios.get(`${baseApiRest}/auth/csrf-token`, {
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

  if (cfg.baseURL && cfg.baseURL.endsWith('/')) {
    cfg.baseURL = cfg.baseURL.slice(0, -1);
  }

  // ✅ P1-1: Standardisation sur cookies httpOnly uniquement
  // Les tokens sont stockés dans des cookies httpOnly définis par le backend
  // Les cookies sont envoyés automatiquement avec withCredentials: true
  // On ne doit PAS envoyer le header Authorization car le backend lit automatiquement les cookies
  // Exception: uniquement pour /auth/refresh-token qui peut nécessiter le refresh_token dans le header
  // mais pour le web, le backend lit le refresh_token depuis les cookies automatiquement

  // ✅ Ajouter le token CSRF pour les requêtes mutantes (POST, PUT, DELETE, PATCH)
  const method = cfg.method?.toUpperCase() || (cfg.url ? 'GET' : 'GET');
  if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
    const csrf = await getCsrfToken();
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

  if (fullUrl.includes('/company_dispatch/')) {
    if (process.env.NODE_ENV === 'development') {
      console.warn('delays/live called', new Error().stack);
    }
    delete config.headers.Authorization;
    delete config.headers.authorization;
    if (config.headers.common) {
      delete config.headers.common.Authorization;
      delete config.headers.common.authorization;
    }
    const companyToken = localStorage.getItem(COMPANY_ACCESS_TOKEN_KEY)
      || localStorage.getItem('company_authToken'); // fallback migration snake_case
    if (companyToken) {
      config.headers.Authorization = `Bearer ${companyToken}`;
      return config;
    }
    if (process.env.NODE_ENV === 'development') {
      console.warn('[company_dispatch] blocked: missing company_access_token', { url: fullUrl });
    }
    return Promise.reject(new Error('Missing company_access_token for company_dispatch'));
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

export const cleanLocalSession = () => {
  localStorage.removeItem('user');
  localStorage.removeItem('public_id');
  localStorage.removeItem('authToken');
  localStorage.removeItem('refreshToken');
  // Séparation company / driver (snake_case + ancien camelCase pendant migration)
  localStorage.removeItem('company_user');
  localStorage.removeItem('company_public_id');
  localStorage.removeItem('company_access_token');
  localStorage.removeItem('company_refresh_token');
  localStorage.removeItem('company_authToken');
  localStorage.removeItem('company_refreshToken');
  localStorage.removeItem('driver_user');
  localStorage.removeItem('driver_public_id');
  localStorage.removeItem('driver_access_token');
  localStorage.removeItem('driver_refresh_token');
  localStorage.removeItem('driver_authToken');
  localStorage.removeItem('driver_refreshToken');
  if (apiRest.defaults?.headers?.common) {
    delete apiRest.defaults.headers.common.Authorization;
    delete apiRest.defaults.headers.common.authorization;
  }
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

    // ✅ Log uniquement en développement
    if (process.env.NODE_ENV === 'development') {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:interceptor:entry',message:'Response interceptor triggered',data:{status,url:cfg.url,method:cfg.method,has_response_data:!!error?.response?.data,response_data_keys:error?.response?.data ? Object.keys(error?.response?.data) : []},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
    }

    // Message sympa pour 429 (limiter)
    if (status === 429) {
      console.warn('Vous avez effectué trop de requêtes. Merci de patienter un peu.');
    }

    // ✅ Gestion 401 avec refresh automatique
    if (status === 401 && !cfg.skipAuthRedirect) {
      // ✅ Log uniquement en développement
      if (process.env.NODE_ENV === 'development') {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:401_detected',message:'401 error detected, checking if fresh token required',data:{url:cfg.url,error_data:error?.response?.data,error_msg:error?.response?.data?.msg,error_error:error?.response?.data?.error,error_message:error?.response?.data?.message},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
      }
      
      // ✅ Détecter AVANT le refresh si c'est un problème de token fresh
      const errorData = error?.response?.data || {};
      const errorMsg = (errorData.msg || errorData.error || errorData.message || '').toLowerCase();
      const isFreshTokenRequired = 
        errorMsg.includes('fresh') || 
        errorMsg.includes('frais') || // Français
        errorMsg.includes('fresh token required') ||
        errorMsg.includes('token must be fresh') ||
        errorMsg.includes('only fresh tokens') ||
        errorMsg.includes("n'est pas frais") || // Français: "n'est pas frais"
        errorMsg.includes('token n\'est pas frais'); // Français: "token n'est pas frais"
      
      // ✅ Log uniquement en développement
      if (process.env.NODE_ENV === 'development') {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:check_fresh_before_refresh',message:'Checking if fresh token required before refresh',data:{url:cfg.url,error_msg:errorMsg,is_fresh_token_required:isFreshTokenRequired},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
      }
      
      // Si c'est un problème de token fresh, ne pas tenter le refresh mais retourner l'erreur avec le flag
      if (isFreshTokenRequired) {
        // ✅ Log uniquement en développement
        if (process.env.NODE_ENV === 'development') {
          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:fresh_token_detected_before_refresh',message:'Fresh token required detected before refresh, rejecting with flag',data:{url:cfg.url},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
          // #endregion
        }
        return Promise.reject({
          ...error,
          isFreshTokenRequired: true,
          message: 'Cette action nécessite une reconnexion récente. Veuillez vous reconnecter pour continuer.',
        });
      }
      
      // ✅ P1-1: Pour le web, le refresh_token est dans les cookies httpOnly
      // Le backend lit automatiquement le refresh_token depuis les cookies
      // Pas besoin de récupérer depuis localStorage
      
      // Si déjà en train de refresh une requête /auth/refresh-token, éviter boucle
      if (cfg.url?.includes('/auth/refresh-token')) {
        logoutUser();
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
        // ✅ P1-1: Le refresh_token est dans les cookies httpOnly
        // Le backend lit automatiquement le refresh_token depuis les cookies
        // Pas besoin d'envoyer dans le header Authorization
        await apiClient.post(
          '/auth/refresh-token',
          {},
          {
            skipAuthRedirect: true, // Éviter boucle
          }
        );

        // ✅ P1-1: Les nouveaux tokens sont dans les cookies httpOnly
        // Pas besoin de les stocker dans localStorage
        // Le backend a déjà mis à jour les cookies

        // Process queued requests (pas besoin de token, les cookies sont automatiques)
        processQueue(null, null);

        // ✅ P1-1: Retry requête originale
        // Les nouveaux cookies sont automatiquement envoyés avec withCredentials: true
        // Pas besoin d'ajouter Authorization header
        // ⚡ Marquer que c'est un retry après refresh réussi pour éviter logs d'erreur
        cfg._retryAfterRefresh = true;
        // ⚡ Supprimer l'erreur de la config pour éviter les logs Axios
        delete cfg._isRetry;
        // ✅ Log uniquement en développement
        if (process.env.NODE_ENV === 'development') {
          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:retry_after_refresh',message:'Retrying request after successful refresh',data:{url:cfg.url,using_cookies:true},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
          // #endregion
        }
        try {
          const retryResponse = await apiClient(cfg);
          // ✅ Log uniquement en développement
          if (process.env.NODE_ENV === 'development') {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:retry_success',message:'Retry after refresh succeeded',data:{url:cfg.url,status:retryResponse?.status},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
            // #endregion
          }
          // ✅ Refresh réussi → retourner la réponse réussie (pas l'erreur 401 initiale)
          return retryResponse;
        } catch (retryError) {
          // ✅ Log uniquement en développement
          if (process.env.NODE_ENV === 'development') {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:retry_failed',message:'Retry after refresh failed',data:{url:cfg.url,status:retryError?.response?.status,error_data:retryError?.response?.data,error_msg:retryError?.response?.data?.msg,error_error:retryError?.response?.data?.error,error_message:retryError?.response?.data?.message},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
            // #endregion
          }
          // Si le retry échoue aussi, propager l'erreur
          throw retryError;
        }
      } catch (refreshError) {
        processQueue(refreshError, null);
        
        // ✅ Détecter si l'erreur est due à un token non-fresh requis
        const errorData = refreshError?.response?.data || {};
        const errorMsg = (errorData.msg || errorData.error || errorData.message || '').toLowerCase();
        const isFreshTokenRequired = 
          errorMsg.includes('fresh') || 
          errorMsg.includes('frais') || // Français
          errorMsg.includes('fresh token required') ||
          errorMsg.includes('token must be fresh') ||
          errorMsg.includes('only fresh tokens') ||
          errorMsg.includes("n'est pas frais") || // Français: "n'est pas frais"
          errorMsg.includes('token n\'est pas frais'); // Français: "token n'est pas frais"
        
        // ✅ Log uniquement en développement
        if (process.env.NODE_ENV === 'development') {
          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:refresh_error_catch',message:'Refresh error caught',data:{status:refreshError?.response?.status,error_data:errorData,error_msg:errorMsg,is_fresh_token_required:isFreshTokenRequired,all_error_keys:Object.keys(errorData)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
          // #endregion
        }
        
        // Si c'est un problème de token fresh, ne pas déconnecter mais laisser le composant gérer l'erreur
        if (isFreshTokenRequired) {
          // ✅ Log uniquement en développement
          if (process.env.NODE_ENV === 'development') {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:fresh_token_detected',message:'Fresh token required detected, rejecting with flag',data:{url:cfg.url},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
            // #endregion
          }
          return Promise.reject({
            ...refreshError,
            isFreshTokenRequired: true,
            message: 'Cette action nécessite une reconnexion récente. Veuillez vous reconnecter pour continuer.',
          });
        }
        
        // ✅ Log uniquement en développement
        if (process.env.NODE_ENV === 'development') {
          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:logout_triggered',message:'Logging out user after refresh error',data:{url:cfg.url},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
          // #endregion
        }
        logoutUser();
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    // ✅ Détecter aussi dans l'erreur initiale si c'est un problème de token fresh
    const errorData = error?.response?.data || {};
    const errorMsg = (errorData.msg || errorData.error || errorData.message || '').toLowerCase();
    const isFreshTokenRequired = 
      errorMsg.includes('fresh') || 
      errorMsg.includes('frais') || // Français
      errorMsg.includes('fresh token required') ||
      errorMsg.includes('token must be fresh') ||
      errorMsg.includes('only fresh tokens') ||
      errorMsg.includes("n'est pas frais") || // Français: "n'est pas frais"
      errorMsg.includes('token n\'est pas frais'); // Français: "token n'est pas frais"
    
    // ✅ Log uniquement en développement
    if (process.env.NODE_ENV === 'development') {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:check_fresh_token_initial',message:'Checking if initial error is fresh token required',data:{status,url:cfg.url,error_data:errorData,error_msg:errorMsg,is_fresh_token_required:isFreshTokenRequired,all_error_keys:Object.keys(errorData)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
    }
    
    if (isFreshTokenRequired && status === 401) {
      // ✅ Log uniquement en développement
      if (process.env.NODE_ENV === 'development') {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:fresh_token_initial_detected',message:'Fresh token required detected in initial error',data:{url:cfg.url},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
        // #endregion
      }
      return Promise.reject({
        ...error,
        isFreshTokenRequired: true,
        message: 'Cette action nécessite une reconnexion récente. Veuillez vous reconnecter pour continuer.',
      });
    }

    // ✅ Log uniquement en développement
    if (process.env.NODE_ENV === 'development') {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'apiClient.js:rejecting_error',message:'Rejecting error without special handling',data:{status,url:cfg.url,error_data:errorData},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
      // #endregion
    }
    // Pas de fallback automatique vers /api/v1: on reste sur la vérité du backend
    return Promise.reject(error);
  }
);

export default apiClient;
