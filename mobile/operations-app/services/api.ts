// services/api.ts
import Constants from "expo-constants";
import { Platform } from "react-native";
import * as Application from "expo-application";
import axios, { isAxiosError } from "axios";
import { getLogger } from "@/utils/logger";
import {
  secureStorage,
  asyncStorage,
  commitSessionTokensAtomically,
} from "./storage";

const log = getLogger("Api");
import {
  AuthInvalidError,
  AuthNotReadyError,
  getAuthFailureReason,
  isPublicEndpoint,
  reportAuthNotReadyMetric,
  shouldLogoutFromRefreshFailure,
} from "@/services/authGuards";
import { logAuthEvent, beginRefreshCycle } from "@/services/authLogging";
import { notifyAuthNotReady, isAuthReadySync } from "@/services/authSync";
import { invokeForceLogoutDriver } from "@/services/authController";
import {
  getSessionDiagHeaderValue,
  pushSessionEvent,
} from "@/services/sessionJournal";
import { sendIngestEvent } from "@/src/config/telemetry";
import { buildAuthNamespace } from "@/services/storage/keys";

// Helper pour les logs de debug (dev uniquement, ingest désactivable via EXPO_PUBLIC_DISABLE_INGEST)
const debugLog = (data: Record<string, unknown>) => {
  if (__DEV__) {
    try {
      sendIngestEvent(data);
    } catch {
      // ignore
    }
  }
};

// --- Config base URL (inclut /api pour matcher le backend) ---
const expoExtra = Constants.expoConfig?.extra || {};
const ENV_API_URL = process.env.EXPO_PUBLIC_API_URL;
const PROD_API_URL = ENV_API_URL;
const DEV_API_URL = expoExtra.devApiUrl || expoExtra.publicApiUrl;
const RAW_APP_VARIANT = String(expoExtra.APP_VARIANT || process.env.APP_VARIANT || "prod");
const NATIVE_APP_ID = Application.applicationId || "";
const IS_NATIVE_PROD_APP_ID = NATIVE_APP_ID === "ch.liri.operations";
const APP_VARIANT = IS_NATIVE_PROD_APP_ID ? "prod" : RAW_APP_VARIANT;

const getDevHost = (): string => {
  // Sur le web, toujours utiliser localhost (le navigateur tourne sur la même machine)
  if (Platform.OS === 'web') {
    return "localhost";
  }
  
  // Sur mobile (iOS/Android), détecter l'IP locale de la machine de développement
  const legacyHost = (Constants as any)?.manifest?.debuggerHost?.split(":")[0]; // Expo < 49
  const newHost = (Constants as any)?.expoConfig?.hostUri?.split(":")[0]; // Expo 49+
  const detectedHost = newHost || legacyHost;
  if (
    !detectedHost ||
    detectedHost === "localhost" ||
    detectedHost === "127.0.0.1"
  ) {
    return "172.20.10.2"; // ← IP locale pour appareils mobiles
  }
  return detectedHost;
};

const ENV_PORT = process.env.EXPO_PUBLIC_BACKEND_PORT;
const PORT = ENV_PORT || expoExtra.backendPort || "5000";
const PROD_API_FALLBACK = "https://api.lirie.ch";

// En développement, forcer l'utilisation de getDevHost() pour éviter de pointer vers la production
// Surtout important sur le web où on doit utiliser l'hôte local
const getDevBaseURL = () => {
  if (Platform.OS === 'web') {
    // Sur le web : utiliser 127.0.0.1 (évite les problèmes IPv6/localhost sur Windows + Docker)
    return `http://127.0.0.1:${PORT}`;
  }
  // Sur mobile, utiliser getDevHost() pour détecter l'IP locale
  return `http://${getDevHost()}:${PORT}`;
};

// Détecter le mode développement de manière plus fiable
// En web bundled, __DEV__ peut être false même en développement local
// On vérifie aussi l'origine de la page (localhost) et l'environnement d'exécution
const isDevelopment = () => {
  // En runtime Metro/Expo dev, __DEV__ est la source de vérité.
  // APP_VARIANT peut rester "prod" pour des raisons de build native.
  if (__DEV__) {
    return true;
  }
  if (APP_VARIANT === "prod") {
    return false;
  }
  
  // En web, vérifier si on est sur localhost (développement local)
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    const hostname = window.location?.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname?.startsWith('192.168.') || hostname?.startsWith('10.0.') || hostname?.startsWith('172.16.')) {
      return true;
    }
  }
  
  // Vérifier l'environnement d'exécution Expo
  const executionEnv = Constants.executionEnvironment;
  if (executionEnv === 'bare' || executionEnv === 'standalone') {
    // En standalone, on est probablement en production
    return false;
  }
  
  // Si on a une URL de développement explicite différente de la prod, on est probablement en dev
  if (DEV_API_URL && DEV_API_URL !== PROD_API_URL && !DEV_API_URL.includes('api.lirie.ch')) {
    return true;
  }
  
  return false;
};

const validateProdApiUrl = (value: unknown): string | null => {
  if (typeof value !== "string" || !value.trim()) {
    log.error("invalid prod api url: missing EXPO_PUBLIC_API_URL", {});
    return null;
  }
  const normalized = value.trim().replace(/\/$/, "");
  if (!normalized.startsWith("https://")) {
    log.error("invalid prod api url: must be https", { value: normalized });
    return null;
  }
  if (normalized.includes("localhost") || normalized.includes("127.0.0.1")) {
    log.error("invalid prod api url: localhost forbidden", { value: normalized });
    return null;
  }
  return normalized;
};

// Déterminer l'URL de base à utiliser
const getBaseURL = () => {
  const isDev = isDevelopment();
  
  if (isDev) {
    // Switch explicite : forcer prod en dev (téléphone réel, tunnel, backend local inaccessible)
    if (process.env.EXPO_PUBLIC_USE_PROD_IN_DEV === '1') {
      return validateProdApiUrl(PROD_API_URL) ?? PROD_API_FALLBACK;
    }
    // En développement, sur le web, toujours utiliser localhost (ignorer DEV_API_URL si défini à la prod)
    if (Platform.OS === 'web') {
      return getDevBaseURL();
    }
    // Sur mobile, utiliser DEV_API_URL si défini et différent de la prod, sinon getDevBaseURL()
    if (DEV_API_URL && DEV_API_URL !== PROD_API_URL && !DEV_API_URL.includes('api.lirie.ch')) {
      return DEV_API_URL;
    }
    return getDevBaseURL();
  }
  
  // En production, utiliser PROD_API_URL
  return validateProdApiUrl(PROD_API_URL) ?? PROD_API_FALLBACK;
};

export const baseURL = `${getBaseURL().replace(/\/$/, "")}/api/v1`;

// --- Debug: afficher la configuration résolue au démarrage (visible dans Metro/JS Inspector) ---
try {
  // On loggue une seule fois au chargement du module
  // Evite d'imprimer des secrets: uniquement des URLs/ports publics
  // Visible dans: Dev Menu -> Open JS Inspector ou les logs Metro
  // Astuce: cherche "[API] baseURL" dans les logs
  // On garde le log aussi en prod pour diagnostic, mais il ne contient pas de secrets
  // Désactive si besoin en commentant la ligne ci-dessous
  log.info("base url configured", {
    baseURL,
    ENV_API_URL,
    DEV_API_URL,
    PROD_API_URL,
    ENV_PORT,
    PORT,
    APP_VARIANT: (Constants.expoConfig as any)?.extra?.APP_VARIANT,
    APP_VARIANT_RESOLVED: APP_VARIANT,
    NATIVE_APP_ID,
    IS_NATIVE_PROD_APP_ID,
  });
} catch {
  // ignore
}

// ✅ P2-2: CSRF optionnel — driver = Bearer only. CSRF = cookie-session browser, pas requis pour Bearer API.
let csrfToken: string | null = null;
let csrfTokenPromise: Promise<string | null> | null = null;

/**
 * ✅ P2-2: Récupère le token CSRF depuis le backend (optionnel pour mobile).
 * Le token est mis en cache et réutilisé pour toutes les requêtes mutantes.
 */
async function fetchCSRFToken(): Promise<string | null> {
  // Si déjà en cours de récupération, attendre la promesse existante
  if (csrfTokenPromise) {
    return csrfTokenPromise;
  }

  // Si déjà en cache, retourner immédiatement
  if (csrfToken) {
    return csrfToken;
  }

  // Créer une nouvelle promesse pour récupérer le token
  csrfTokenPromise = (async () => {
    try {
      // ✅ P2-2: Récupérer le token CSRF depuis l'endpoint backend
      // Sur web: requête "simple" (fetch sans en-têtes custom) pour éviter preflight OPTIONS → GET jamais envoyé (preuve logs Docker)
      const url = `${baseURL}/auth/csrf-token`;
      log.info("csrf token fetch started", { url });

      let token: string | null = null;
      if (typeof fetch !== "undefined") {
        // RN/Expo: fetch est plus robuste au boot que certains adapters Axios récents.
        // Web: on garde "credentials: omit" pour éviter un preflight inutile.
        const res = await fetch(url, {
          method: "GET",
          credentials: Platform.OS === "web" ? "omit" : "same-origin",
        });
        if (res.ok) {
          const data = await res.json();
          token = data?.csrf_token ?? null;
        }
      } else {
        const response = await api.get("/auth/csrf-token", {
          withCredentials: Platform.OS === "web",
        });
        token = response.data?.csrf_token ?? null;
      }

      if (token) {
        csrfToken = token;
        log.success("csrf token fetched");
      } else {
        log.warn("csrf token not received from backend");
      }

      return token;
    } catch (error) {
      // ✅ P2-2: CSRF optionnel — driver = Bearer only ; ne pas considérer l'absence de CSRF comme panne API.
      if (isAxiosError(error)) {
        const isNetworkError =
          error.code === "ERR_NETWORK" ||
          error.message === "Network Error" ||
          error.response == null;
        if (isNetworkError) {
          log.warn("csrf unavailable, bearer auth only", {});
        } else {
          log.warn("csrf fetch error (optional)", {
            status: error.response?.status,
            url: error.config?.url,
          });
        }
      } else {
        log.warn("csrf unavailable, continuing with bearer only", {});
      }
      csrfToken = null;
      return null;
    } finally {
      csrfTokenPromise = null; // Réinitialiser pour permettre un nouveau fetch si nécessaire
    }
  })();

  return csrfTokenPromise;
}

/**
 * ✅ Initialise le token CSRF (à appeler après le login ou au démarrage)
 */
export const initializeCSRFToken = async (): Promise<void> => {
  try {
    await fetchCSRFToken();
  } catch (error) {
    log.warn("csrf init error", { error });
  }
};

/**
 * ✅ P2-2: Invalide le token CSRF (utile lors du logout ou changement de session).
 */
export const invalidateCSRFToken = () => {
  csrfToken = null;
  csrfTokenPromise = null;
};

// --- instance axios ---
export const api = axios.create({
  baseURL,
  timeout: 30000,
  // ✅ Web driver (8081): pas de cookies — auth via Authorization Bearer uniquement (évite ERR_NETWORK / CORS).
  // Native: withCredentials true si besoin cookies ailleurs.
  withCredentials: Platform.OS !== "web",
  headers: {
    "Content-Type": "application/json",
    // ✅ X-Requested-With seulement sur mobile (pas sur web pour éviter les problèmes CORS)
    ...(Platform.OS !== "web" ? { "X-Requested-With": "Expo" } : {}),
  },
});

// Log de la configuration des headers au démarrage (dev uniquement)
if (__DEV__) {
  const hasXRequestedWith = Platform.OS !== "web";
  log.debug("api instance created", {
    platform: Platform.OS,
    hasXRequestedWith,
    baseURL,
  });
  debugLog({
    location: "api.ts:axios.create",
    message: "API instance created",
    data: {
      platform: Platform.OS,
      hasXRequestedWith,
      baseURL,
    },
    timestamp: Date.now(),
    sessionId: "debug-session",
    runId: "run1",
    hypothesisId: "A",
  });
}

// ⚡ Phase 3 : Cache partagé pour l'intercepteur request
// Réduit les lectures SecureStore répétées lors de requêtes simultanées
let interceptorTokenCache: string | null = null;
let interceptorTokenCacheTime = 0;
const INTERCEPTOR_CACHE_TTL = 30000; // 30 secondes

// ⚡ Phase 4 : Métriques de performance pour l'intercepteur (dev uniquement)
let interceptorCacheHitCount = 0;
let interceptorCacheMissCount = 0;
let interceptorRequestCount = 0;
const INTERCEPTOR_METRICS_LOG_INTERVAL = 100; // Log toutes les 100 requêtes

/**
 * Invalide le cache de l'intercepteur (utile lors du logout ou changement de token)
 */
export const invalidateInterceptorCache = () => {
  interceptorTokenCache = null;
  interceptorTokenCacheTime = 0;
  resetAuthNotReadyDedupe();
  if (__DEV__) {
    interceptorCacheHitCount = 0;
    interceptorCacheMissCount = 0;
    interceptorRequestCount = 0;
  }
};

/** Guard anti-race : dedupe AuthNotReadyError (missing_access_token) par endpoint pour éviter 5 popups si l'utilisateur clique 5 fois. */
const AUTH_NOT_READY_DEDUPE_MS = 3000;
let lastAuthNotReadyKey: string | null = null;
let lastAuthNotReadyTime = 0;

/** À appeler après clearAll / logout pour réinitialiser le dedupe. */
export const resetAuthNotReadyDedupe = () => {
  lastAuthNotReadyKey = null;
  lastAuthNotReadyTime = 0;
};

// --- Authorization bearer + Device ID ---
api.interceptors.request.use(
  async (config) => {
    // ✅ Garde anti-régression : l'app chauffeur n'utilise PAS company_dispatch (réservé COMPANY/ADMIN).
    // ETA / retard : utiliser uniquement GET /driver/me/bookings/eta.
    const url = config.url ?? "";
    if (url.includes("company_dispatch")) {
      const msg =
        "Driver app must not call company_dispatch. Use /driver/me/bookings/eta only.";
      if (__DEV__) {
        log.error("driver app must not call company_dispatch", { url, msg });
      }
      throw new Error(msg);
    }

    // ✅ CORRECTION #1 : Guard d'initialisation
    // Attendre que l'auth soit prête avant de permettre les requêtes (sauf login)
    const isLoginRequest =
      config.url === "/auth/login" || config.url?.endsWith("/auth/login");
    const isPublic = isPublicEndpoint(config.url, "driver");
    if (!isPublic) {
      try {
        const { waitForAuthReady } = await import("@/services/authSync");
        await waitForAuthReady({
          timeoutMs: 5000,
          reasonOnTimeout: "auth_bootstrapping",
        });
      } catch (error) {
        // ✅ P1 (strict): ne jamais envoyer une requête protégée sans auth prête
        if (__DEV__) {
          log.warn("auth not ready timeout, request rejected", { url: config.url });
        }
        reportAuthNotReadyMetric({
          kind: "driver",
          reason: "auth_bootstrapping",
          url: config.url,
        });
        throw new AuthNotReadyError({
          kind: "driver",
          reason: "auth_bootstrapping",
          url: config.url,
        });
      }
    }
    
    // ✅ P2-2: Ajouter le token CSRF pour les requêtes mutantes
    const isMutatingMethod = ["POST", "PUT", "DELETE", "PATCH"].includes(
      config.method?.toUpperCase() || ""
    );
    
    if (isMutatingMethod) {
      let tokenToUse: string | null = null;

      // Sur web, essayer de récupérer le token du cookie d'abord
      if (Platform.OS === "web" && typeof document !== "undefined") {
        try {
          const cookieToken = document.cookie
            .split("; ")
            .find((row) => row.startsWith("csrf_token="))
            ?.split("=")[1];
          
          if (cookieToken) {
            log.success("csrf token from document.cookie");
            tokenToUse = cookieToken;
          } else {
            log.info("csrf not in document.cookie (may be httpOnly)");
          }
        } catch (error) {
          log.warn("csrf cookie read error", { error });
        }
      }

      // Fallback: utiliser le token en cache (récupéré via initializeCSRFToken)
      if (!tokenToUse && csrfToken) {
        log.success("using cached csrf token");
        tokenToUse = csrfToken;
      }

      // Si toujours pas de token, essayer de le récupérer maintenant (dernière chance)
      if (!tokenToUse) {
        log.info("no csrf token, fetching");
        const token = await fetchCSRFToken();
        if (token) {
          tokenToUse = token;
        }
      }

      // Ajouter le token à la requête si disponible
      if (tokenToUse) {
        config.headers["X-CSRF-Token"] = tokenToUse;
        log.success("csrf token added to request", { method: config.method, url: config.url });
      } else {
        log.warn("no csrf token for request", { method: config.method, url: config.url });
      }
    }
    
    const now = Date.now();

    // ⚡ OPTIMISATION : Utiliser le cache si disponible et valide
    // Évite les lectures SecureStore répétées lors de requêtes simultanées
    let token = interceptorTokenCache;
    if (!token || now - interceptorTokenCacheTime >= INTERCEPTOR_CACHE_TTL) {
      // Cache invalide ou expiré → lire depuis SecureStore (qui utilise son propre cache)
      token = await secureStorage.getAccessToken();
      interceptorTokenCache = token;
      interceptorTokenCacheTime = now;
      if (__DEV__) {
        interceptorCacheMissCount++;
      }
    } else {
      if (__DEV__) {
        interceptorCacheHitCount++;
      }
    }

    // ⚡ Phase 4 : Métriques de performance (dev uniquement)
    if (__DEV__) {
      interceptorRequestCount++;
      if (interceptorRequestCount % INTERCEPTOR_METRICS_LOG_INTERVAL === 0) {
        const totalRequests =
          interceptorCacheHitCount + interceptorCacheMissCount;
        const cacheHitRate =
          totalRequests > 0
            ? (interceptorCacheHitCount / totalRequests) * 100
            : 0;
        log.info("interceptor performance", {
          cacheHitRate: cacheHitRate.toFixed(1),
          hits: interceptorCacheHitCount,
          misses: interceptorCacheMissCount,
          total: interceptorRequestCount,
        });
      }
    }

    const logData = {
      location: "api.ts:interceptor:request",
      message: "interceptor request entry",
      data: {
        url: config.url,
        method: config.method,
        isLoginRequest,
        baseURL: config.baseURL,
        hasToken: !!token,
        isAuthReady: isAuthReadySync(),
        platform: Platform.OS,
        withCredentials: config.withCredentials,
        headers: {
          "X-Requested-With": config.headers?.["X-Requested-With"] || config.headers?.get?.("X-Requested-With"),
          "Content-Type": config.headers?.["Content-Type"] || config.headers?.get?.("Content-Type"),
        },
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      runId: "run1",
      hypothesisId: "A",
    };
    if (__DEV__) {
      log.debug("auth state (non-sensitive)", {
        hasToken: !!token,
        isAuthReady: isAuthReadySync(),
        url: config.url,
        ts: Date.now(),
      });
    }
    log.debug("driver api interceptor", logData);
    debugLog(logData);

    // ✅ Ne pas ajouter le token pour les requêtes de login/refresh
    // ✅ IMPORTANT: ne pas écraser un Authorization explicite (ex: appels enterprise via `api`)
    const hasExplicitAuthHeader =
      Boolean((config.headers as any)?.Authorization) ||
      Boolean((config.headers as any)?.authorization);
    if (token && !isLoginRequest && !hasExplicitAuthHeader) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // ✅ P1 (strict): requête protégée => Authorization obligatoire
    // Si un Authorization explicite a été fourni, on n'impose pas le token driver.
    // ✅ Guard anti-race : dedupe par endpoint + 2–3 s pour éviter 5 popups si l'utilisateur clique 5 fois.
    if (!isPublic && !hasExplicitAuthHeader && !token) {
      // Option 1 — Assert soft en dev : attraper invariant cassé (isAuthReady mais pas de token)
      if (__DEV__ && isAuthReadySync()) {
        log.warn("isAuthReady but no token (invariant broken)", {});
      }
      const urlKey = config.url ?? "";
      const now = Date.now();
      const isDedupe =
        urlKey === lastAuthNotReadyKey &&
        now - lastAuthNotReadyTime < AUTH_NOT_READY_DEDUPE_MS;
      if (!isDedupe) {
        lastAuthNotReadyKey = urlKey;
        lastAuthNotReadyTime = now;
      }
      if (__DEV__ && !isDedupe) {
        log.warn("auth not ready, request rejected", { url: config.url });
      }
      if (!isDedupe) {
        reportAuthNotReadyMetric({
          kind: "driver",
          reason: "missing_access_token",
          url: config.url,
        });
        logAuthEvent("AUTH_ACCESS_ABSENT", {
          kind: "driver",
          reason: "missing_access_token",
          url: config.url?.slice(0, 100),
        });
      }
      throw new AuthNotReadyError({
        kind: "driver",
        reason: "missing_access_token",
        url: config.url,
        silentDedupe: isDedupe,
      });
    }

    // ✅ Envoyer X-Device-ID (stable) pour tracking des sessions / refresh tokens
    // (ne doit pas dépendre de l'usage préalable du mode enterprise)
    if (Platform.OS !== "web") {
      try {
        const deviceId = await asyncStorage.getOrCreateDeviceId();
        config.headers["X-Device-ID"] = deviceId;
      } catch (e) {
        if (__DEV__) {
          log.warn("x-device-id generation failed", { error: e });
        }
      }
    }

    // ✅ P0.1: X-Session-Diag (dernier reason code) pour corrélation backend
    const sessionDiag = getSessionDiagHeaderValue();
    if (sessionDiag) {
      config.headers["X-Session-Diag"] = sessionDiag;
    }

    return config;
  },
  (error) => Promise.reject(error)
);

// --- Gestion refresh token automatique sur 401 ---
// Flag pour éviter plusieurs refresh simultanés
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (value: string) => void;
  reject: (error: any) => void;
}> = [];

const processQueue = (error: any, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else if (token) {
      prom.resolve(token);
    } else {
      prom.reject(new Error("Token manquant après refresh"));
    }
  });
  failedQueue = [];
};

const driverRefreshPromiseBySessionKey = new Map<string, Promise<string>>();

async function getDriverSessionKey(): Promise<string> {
  const token = await secureStorage.getAccessToken();
  const userPublicId = (await secureStorage.getUserPublicId()) ?? "unknown";
  let tenantId: string | number | null = null;
  let sessionId: string | null = null;

  if (token) {
    try {
      const payload = JSON.parse(atob(token.split(".")[1]));
      tenantId = payload?.company_id ?? null;
      sessionId = payload?.session_id ?? null;
    } catch {
      // no-op: fallback namespace stays stable on user id
    }
  }

  return buildAuthNamespace({
    role: "driver",
    userId: userPublicId,
    tenantId,
    sessionId,
  });
}

export async function runDriverRefreshSingleflight(
  sessionKey: string,
  source:
    | "api_interceptor"
    | "api_401"
    | "foreground_resume"
    | "foreground_resync"
    | "proactive_refresh"
    | "socket_reconnect"
    | "socket_connect_error"
    | "socket_unauthorized"
    | "bootstrap"
    | "boot_restore"
    | "profile_refresh"
    | "manual_action"
): Promise<string> {
  const key = sessionKey || "driver:unknown:none:none";
  const existing = driverRefreshPromiseBySessionKey.get(key);
  if (existing) return existing;

  const promise = performDriverRefresh(source).finally(() => {
    driverRefreshPromiseBySessionKey.delete(key);
  });
  driverRefreshPromiseBySessionKey.set(key, promise);
  return promise;
}

async function performDriverRefresh(
  source:
    | "api_interceptor"
    | "api_401"
    | "foreground_resume"
    | "foreground_resync"
    | "proactive_refresh"
    | "socket_reconnect"
    | "socket_connect_error"
    | "socket_unauthorized"
    | "bootstrap"
    | "boot_restore"
    | "profile_refresh"
    | "manual_action"
): Promise<string> {
  const normalizedTriggerSource =
    source === "foreground_resync"
      ? "foreground_resume"
      : source === "api_401"
        ? "api_interceptor"
        : source === "socket_connect_error" || source === "socket_unauthorized"
          ? "socket_reconnect"
          : source === "boot_restore" || source === "profile_refresh"
            ? "manual_action"
            : source;
  const refreshToken = await secureStorage.getRefreshToken();
  if (!refreshToken) {
    throw new Error("Pas de refresh token disponible");
  }

  // ✅ Isolation driver/enterprise : ne jamais envoyer un token entreprise à l'endpoint driver
  const { isDriverRefreshToken } = await import("@/utils/jwtAudience");
  if (!isDriverRefreshToken(refreshToken)) {
    log.warn("driver refresh token has wrong audience (enterprise token?), clearing");
    await invokeForceLogoutDriver({
      reason: "refresh_invalid",
      severity: "AUTH_HARD_FAILURE",
      source: "driver",
      trigger_source: normalizedTriggerSource,
    });
    throw new AuthInvalidError({ route: "driver", reason: "refresh_invalid" });
  }

  const refreshResponse = await refreshAccessToken(refreshToken);
  const newAccessToken = refreshResponse.access_token;

  await commitSessionTokensAtomically({
    scope: "driver",
    accessToken: newAccessToken,
    refreshToken: refreshResponse.refresh_token,
    trigger_source: normalizedTriggerSource,
  });

  // Mettre à jour le cache de l'intercepteur pour cohérence immédiate
  interceptorTokenCache = newAccessToken;
  interceptorTokenCacheTime = Date.now();

  return newAccessToken;
}

export async function refreshDriverTokenSingleflight(): Promise<string> {
  return runDriverRefreshSingleflight(
    await getDriverSessionKey(),
    "manual_action"
  );
}

// Interceptor response avec refresh automatique
api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const originalRequest = error.config;

    // ✅ CORRECTION (audit 34-38):
    // - 403 est souvent un "forbidden" fonctionnel (rôle/droits) et ne doit pas déclencher un refresh.
    // - On ne tente donc un refresh automatique que sur 401.
    //
    // Exception: si la requête de refresh elle-même échoue (401/403), on nettoie la session.
    if (originalRequest?.url?.includes("/auth/refresh-token")) {
      const refreshStatus = error.response?.status;
      const isNetworkError = !error.response;
      const serverReason = error.response?.data?.reason;

      if (refreshStatus === 401) {
        const decision = shouldLogoutFromRefreshFailure(error, "refresh_endpoint");
        // Politique prudente: 401 non structuré => pas de logout immédiat.
        if (decision.reason === "unknown_refresh_401") {
          logAuthEvent("AUTH_REFRESH_FAIL_SOFT", {
            route: "driver",
            status: refreshStatus,
            reason: decision.reason,
            refresh_attempted: true,
            outcome: "retry_later",
            source: "refresh_endpoint",
          });
          processQueue(error, null);
          return Promise.reject(error);
        }
        pushSessionEvent("REFRESH_FAIL");
        const reason = serverReason === "refresh_expired" ? "refresh_expired" : "refresh_invalid";
        logAuthEvent("AUTH_REFRESH_FAIL_HARD", {
          route: "driver",
          status: refreshStatus,
          reason,
          refresh_attempted: true,
          outcome: "logout",
          source: "refresh_endpoint",
        });
        processQueue(new AuthInvalidError({ route: "driver", reason }), null);
        await invokeForceLogoutDriver({
          reason,
          severity: "AUTH_HARD_FAILURE",
          source: "driver",
          trigger_source: "api_interceptor",
        });
        return Promise.reject(new AuthInvalidError({ route: "driver", reason }));
      } else if (refreshStatus === 403) {
        if (serverReason === "account_disabled") {
          pushSessionEvent("REFRESH_FAIL");
          logAuthEvent("AUTH_REFRESH_FAIL_HARD", {
            route: "driver",
            status: 403,
            refresh_attempted: true,
            outcome: "logout",
            source: "refresh_endpoint",
            server_reason: serverReason,
          });
          processQueue(new AuthInvalidError({ route: "driver", reason: "account_disabled" }), null);
          await invokeForceLogoutDriver({
            reason: "account_disabled",
            severity: "AUTH_HARD_FAILURE",
            source: "driver",
            trigger_source: "api_interceptor",
          });
          return Promise.reject(new AuthInvalidError({ route: "driver", reason: "account_disabled" }));
        }
        log.warn("refresh token 403 without account_disabled, treating as transient", {
          serverError: error.response?.data?.error,
        });
        processQueue(error, null);
        return Promise.reject(error);
      } else if (isNetworkError) {
        log.warn("refresh token network error, user stays connected", {});
        return Promise.reject(error);
      } else {
        // Autres erreurs → ne pas déconnecter non plus
        log.warn("refresh token server error, user stays connected", { status: refreshStatus });
        return Promise.reject(error);
      }
    }

    // ✅ Ne pas traiter 401 sur /auth/login : mauvais identifiants, pas session expirée
    const isLoginRequest =
      originalRequest?.url === "/auth/login" ||
      originalRequest?.url?.endsWith("/auth/login");
    const isAuthError = error.response?.status === 401;
    if (isAuthError && !originalRequest._retry && !isLoginRequest) {
      // ✅ P0.1: 401 = access token expiré/invalide → journal avant refresh
      pushSessionEvent("API_401");

      // ✅ P0.2: Si déjà en train de refresh, attendre le même (singleflight) puis rejouer
      if (isRefreshing) {
        pushSessionEvent("REFRESH_WAIT");
        logAuthEvent("AUTH_401_HANDLING", {
          route: "driver",
          refresh_attempted: true,
          outcome: "wait_inflight",
          queue_count: failedQueue.length + 1,
        });
        if (__DEV__) {
          log.info("refresh wait, request queued", { queueCount: failedQueue.length + 1 });
        }
        return new Promise<string>((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then((token) => {
            originalRequest.headers.Authorization = `Bearer ${token}`;
            return api(originalRequest);
          })
          .catch((err) => Promise.reject(err));
      }

      // Premier 401 → tenter refresh (jamais logout sans avoir tenté refresh)
      originalRequest._retry = true;
      isRefreshing = true;
      pushSessionEvent("REFRESH_START");
      const refreshCycleId = beginRefreshCycle("driver");
      logAuthEvent("AUTH_REFRESH_START", { route: "driver", trigger: "api_401", refresh_cycle_id: refreshCycleId });

      try {
        const sessionKey = await getDriverSessionKey();
        const newAccessToken = await runDriverRefreshSingleflight(
          sessionKey,
          "api_interceptor"
        );
        pushSessionEvent("REFRESH_SUCCESS");
        logAuthEvent("AUTH_REFRESH_SUCCESS", { route: "driver" });
        if (__DEV__) {
          log.success("token refreshed, cache updated");
        }

        // Traiter la queue
        processQueue(null, newAccessToken);

        // Rejouer la requête originale
        originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
        return api(originalRequest);
      } catch (refreshError: any) {
        // ✅ Log détaillé pour diagnostic
        const refreshStatus = refreshError?.response?.status;
        const refreshData = refreshError?.response?.data;
        const isNetworkError = !refreshError?.response; // Pas de réponse = erreur réseau
        
        log.warn("refresh token failed", {
          status: refreshStatus || "network",
          data: refreshData || refreshError?.message || refreshError,
        });
        
        if (refreshStatus === 401) {
          const decision = shouldLogoutFromRefreshFailure(
            refreshError,
            "refresh_endpoint"
          );
          if (!decision.shouldLogout) {
            logAuthEvent("AUTH_REFRESH_FAIL_SOFT", {
              route: "driver",
              status: 401,
              refresh_attempted: true,
              outcome: "retry_later",
              reason: decision.reason,
              source: "api_401_catch",
            });
            processQueue(refreshError, null);
            return Promise.reject(refreshError);
          }
          pushSessionEvent("REFRESH_FAIL");
          const reason =
            getAuthFailureReason(refreshError) === "refresh_expired"
              ? "refresh_expired"
              : "refresh_invalid";
          processQueue(new AuthInvalidError({ route: "driver", reason }), null);
          await invokeForceLogoutDriver({
            reason,
            severity: "AUTH_HARD_FAILURE",
            source: "driver",
            trigger_source: "api_interceptor",
          });
          return Promise.reject(new AuthInvalidError({ route: "driver", reason }));
        } else if (isNetworkError) {
          logAuthEvent("AUTH_REFRESH_FAIL", {
            route: "driver",
            status: "network",
            refresh_attempted: true,
            outcome: "retry_later",
          });
          log.warn("refresh token network error, user stays connected", {});
          processQueue(refreshError, null);
          return Promise.reject(refreshError);
        } else {
          logAuthEvent("AUTH_REFRESH_FAIL", {
            route: "driver",
            status: refreshStatus ?? "unknown",
            refresh_attempted: true,
            outcome: "retry_later",
          });
          log.warn("refresh token server error, user stays connected", { status: refreshStatus });
          processQueue(refreshError, null);
          return Promise.reject(refreshError);
        }
      } finally {
        isRefreshing = false;
      }
    }
    
    // ✅ 403 = forbidden fonctionnel → ne pas retry automatiquement (audit 34-38)
    if (error.response?.status === 403 && !originalRequest._retry) {
      log.error("access denied (403)", {
        url: originalRequest.url,
        data: error.response?.data || error.message,
      });
      // Si c'est un 403, ne pas retry automatiquement.
      return Promise.reject(error);
    }

    // Autres erreurs → log et rejeter
    if (isAxiosError(error)) {
      const code = (error as any)?.code;
      log.warn("api error", {
        url: error.config?.url,
        method: error.config?.method,
        status: error.response?.status,
        data: error.response?.data,
        code,
        message: error.message,
      });
      if (code === "ECONNABORTED") {
        log.warn("api timeout (30s), check network/server latency", {});
      }
      if (code === "ERR_NETWORK" && Platform.OS === "web") {
        const origin = typeof window !== "undefined" ? window.location.origin : "?";
        log.warn("err_network on web, check backend and cors", {
          baseUrl: baseURL.replace("/api/v1", ""),
          origin,
        })
        const fullURL = (originalRequest?.baseURL || "") + (originalRequest?.url || "");
        sendIngestEvent({
          location: "api.ts:response_interceptor",
          message: "ERR_NETWORK",
          data: { fullURL, method: originalRequest?.method, code, baseURL: originalRequest?.baseURL, url: originalRequest?.url },
          timestamp: Date.now(),
          sessionId: "debug-session",
          runId: "run1",
          hypothesisId: "H3",
        });
      }
    }
    return Promise.reject(error);
  }
);

// ========= Types =========
export type User = {
  id: number;
  public_id: string;
  first_name: string;
  last_name: string;
  email: string;
  phone?: string;
};

export type Driver = {
  id: number;
  user_id: number;
  username: string;
  first_name: string;
  last_name: string;
  phone: string;
  photo: string;
  company_id: number;
  company_name: string;
  is_active: boolean;
  is_available: boolean;
  driver_type?: "REGULAR" | "EMERGENCY";
  vehicle_assigned: string;
  brand: string;
  license_plate: string;
  driver_photo?: string;
  latitude: number | null;
  longitude: number | null;
  user: {
    id: number;
    username: string;
    email: string;
    role: string;
    public_id: string;
  };
  company: { id: number; name: string };
  // Champs HR / profil étendu (optionnels, renvoyés par le backend)
  contract_type?: string;
  nationality?: string;
  avs_number?: string;
  employment_start_date?: string;
  employment_end_date?: string;
  weekly_hours?: number | null;
  license_categories?: string[];
  license_valid_until?: string;
  medical_valid_until?: string;
  trainings?: Array<{ name: string; valid_until?: string }>;
  emergency_contact_name?: string;
  emergency_contact_phone?: string;
};

export const registerPushToken = async (payload: {
  token: string;
  driverId: number;
  device_id?: string;
  deviceId?: string;
  platform?: "ios" | "android";
  provider?: "expo" | "fcm";
}) => {
  const device_id =
    payload.device_id ||
    payload.deviceId ||
    (Platform.OS !== "web" ? await asyncStorage.getOrCreateDeviceId() : undefined);
  const platform =
    payload.platform ||
    (Platform.OS === "ios" || Platform.OS === "android" ? Platform.OS : undefined);
  const provider =
    payload.provider ||
    (payload.token.startsWith("ExponentPushToken") ? "expo" : "fcm");

  const res = await api.post("/driver/save-push-token", {
    token: payload.token,
    driverId: payload.driverId,
    device_id,
    platform,
    provider,
  });
  return res.data;
};

export type AuthResponse = {
  message: string;
  token: string; // <-- access token (renommé "token" par le backend)
  refresh_token?: string; // <-- refresh token (optionnel, mais toujours présent en pratique)
  user: {
    id: number;
    public_id: string;
    username: string;
    email: string;
    role: string;
    force_password_change: boolean;
  };
};

// ========== Auth ==========
export const loginDriver = async (
  email: string,
  password: string
): Promise<AuthResponse> => {
  try {
    const fullURL = `${baseURL}/auth/login`;
    const logData = {
      location: "api.ts:loginDriver",
      message: "loginDriver entry",
      data: {
        baseURL,
        fullURL,
        hasEmail: Boolean(email),
        hasPassword: Boolean(password),
        platform: Platform.OS,
        isDev: __DEV__,
        executionEnv: Constants.executionEnvironment,
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      runId: "run1",
      hypothesisId: "D",
    };
    log.debug("loginDriver entry", logData);
    debugLog(logData);
    const requestHeaders = {
      "X-Requested-With": api.defaults.headers.common["X-Requested-With"] || api.defaults.headers["X-Requested-With"] || "not set",
      "Content-Type": api.defaults.headers.common["Content-Type"] || api.defaults.headers["Content-Type"],
      platform: Platform.OS,
    };
    const preRequestLog = {
      location: "api.ts:loginDriver",
      message: "loginDriver before request",
      data: {
        url: "/auth/login",
        headers: requestHeaders,
        platform: Platform.OS,
        willSendXRequestedWith: Platform.OS !== "web",
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      runId: "run1",
      hypothesisId: "A",
    };
    log.debug("loginDriver before request", preRequestLog);
    debugLog(preRequestLog);
    // Appel principal via Axios
    const response = await api.post<AuthResponse>("/auth/login", {
      email,
      password,
    });
    const successLogData = {
      location: "api.ts:loginDriver",
      message: "loginDriver success",
      data: {
        status: response.status,
        hasToken: !!response.data?.token,
        hasRefreshToken: !!response.data?.refresh_token,
        responseKeys: response.data ? Object.keys(response.data) : [],
        responseData: response.data ? {
          hasMessage: !!response.data.message,
          hasUser: !!response.data.user,
          hasToken: !!response.data.token,
          hasRefreshToken: !!response.data.refresh_token,
          tokenLength: response.data.token ? response.data.token.length : 0,
          refreshTokenLength: response.data.refresh_token ? response.data.refresh_token.length : 0,
        } : null,
        responseHeaders: {
          "x-requested-with": response.headers?.["x-requested-with"] || response.headers?.["X-Requested-With"],
        },
        platform: Platform.OS,
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      runId: "run1",
      hypothesisId: "B",
    };
    log.debug("loginDriver success", successLogData);
    debugLog(successLogData);
    const data = response.data;
    
    const storageLog = {
      location: "api.ts:loginDriver",
      message: "loginDriver storing tokens",
      data: {
        hasToken: !!data?.token,
        hasRefreshToken: !!data?.refresh_token,
        willStoreToken: !!data?.token,
        willStoreRefreshToken: !!data?.refresh_token,
        tokenValue: data?.token ? `${data.token.substring(0, 20)}...` : null,
        platform: Platform.OS,
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      runId: "run1",
      hypothesisId: "C",
    };
    log.debug("loginDriver storing tokens", storageLog);
    debugLog(storageLog);
    
    // ✅ Stocker le token d'accès dans AsyncStorage
    if (data?.token) {
      const beforeStoreLog = {
        location: "api.ts:loginDriver",
        message: "before setAccessToken",
        data: { hasToken: !!data.token, tokenLength: data.token.length },
        timestamp: Date.now(),
        sessionId: "debug-session",
        runId: "run1",
        hypothesisId: "D",
      };
      debugLog(beforeStoreLog);
      await secureStorage.setAccessToken(data.token);
      // ✅ Forcer l'intercepteur à relire le token à la prochaine requête (évite cache null)
      invalidateInterceptorCache();
      const afterStoreLog = {
        location: "api.ts:loginDriver",
        message: "after setAccessToken",
        data: { stored: true },
        timestamp: Date.now(),
        sessionId: "debug-session",
        runId: "run1",
        hypothesisId: "D",
      };
      debugLog(afterStoreLog);
    }
    
    // ✅ Stocker le refresh_token dans SecureStore (sécurisé)
    if (data?.refresh_token) {
      await secureStorage.setRefreshToken(data.refresh_token);
    }
    
    // ✅ Stocker le public_id pour auto-login (optionnel)
    if (data?.user?.public_id) {
      await secureStorage.setUserPublicId(data.user.public_id);
    }
    
    return data;
  } catch (err: unknown) {
    // Repli: si "Network Error", retenter immédiatement via fetch (RN), même endpoint
    const isAxiosNetErr =
      isAxiosError(err) &&
      (err as any)?.code === "ERR_NETWORK" || (isAxiosError(err) && err.message?.toLowerCase().includes("network error"));

    if (!isAxiosNetErr) {
      throw err;
    }

    try {
      // Timeout 30s via AbortController
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30000);
      const res = await fetch(`${baseURL}/auth/login`, {
        method: "POST",
        // ✅ Sur le web, activer credentials pour envoyer/recevoir les cookies httpOnly
        credentials: Platform.OS === "web" ? "include" : "omit",
        headers: {
          "Content-Type": "application/json",
          // ✅ X-Requested-With seulement sur mobile (pas sur web pour éviter les problèmes CORS)
          ...(Platform.OS !== "web" ? { "X-Requested-With": "Expo" } : {}),
        },
        body: JSON.stringify({ email, password }),
        signal: controller.signal,
      });
      clearTimeout(timeout);

      const text = await res.text();
      if (!res.ok) {
        throw new Error(`Login via fetch a échoué (${res.status}): ${text}`);
      }
      const data = JSON.parse(text) as AuthResponse;
      
      // ✅ Stocker les tokens dans SecureStore (même logique que pour Axios)
      if (data?.token) {
        await secureStorage.setAccessToken(data.token);
      }
      if (data?.refresh_token) {
        await secureStorage.setRefreshToken(data.refresh_token);
      }
      if (data?.user?.public_id) {
        await secureStorage.setUserPublicId(data.user.public_id);
      }
      
      return data;
    } catch (fallbackError) {
      log.warn("login fallback (fetch) failed", { error: fallbackError });
      throw err; // renvoyer l'erreur Axios originale pour la logique appelante
    }
  }
};

export const fetchUserInfo = async (): Promise<{
  id: number;
  public_id: string;
  username: string;
  email: string;
  role: string;
}> => {
  const res = await api.get("/auth/me");
  return res.data;
};

// ========== Refresh Token ==========
export type RefreshTokenResponse = {
  access_token: string;
  refresh_token: string; // ✅ Toujours présent pour mobile (backend garantit via X-Requested-With: Expo)
  user: {
    public_id: string;
    role: string;
    company_id?: number;
    driver_id?: number;
  };
};

export const refreshAccessToken = async (
  refreshToken: string
): Promise<RefreshTokenResponse> => {
  const response = await api.post<RefreshTokenResponse>("/auth/refresh-token", {
    refresh_token: refreshToken,
  });
  return response.data;
};

// ========== Driver ==========
export const fetchDriverProfile = async (): Promise<Driver> => {
  const res = await api.get<{ profile: Driver }>("/driver/me/profile");
  return res.data.profile;
};

export interface DriverProfilePayload {
  vehicle_assigned?: string;
  brand?: string;
  license_plate?: string;
  phone?: string;
}

export const updateDriverProfile = async (
  payload: DriverProfilePayload
): Promise<Driver> => {
  const res = await api.put<{ profile: Driver; message: string }>(
    "/driver/me/profile",
    payload
  );
  return res.data.profile;
};

export type UpdatePhotoResponse = { profile: Driver; message: string };

export const updateDriverPhoto = async (
  photo: string
): Promise<UpdatePhotoResponse> => {
  const response = await api.put<UpdatePhotoResponse>("/driver/me/photo", {
    photo,
  });
  return response.data;
};

export const updateDriverAvailability = async (
  is_available: boolean
): Promise<{ message: string }> => {
  const response = await api.put<{ message: string }>(
    "/driver/me/availability",
    { is_available }
  );
  return response.data;
};

export interface SwitchToEnterpriseResponse {
  token: string;
  refresh_token: string;
  user: {
    public_id: string;
    email: string;
    first_name?: string;
    last_name?: string;
  };
  company: {
    id: number;
    name: string;
  };
}

export const switchToEnterpriseToken = async (): Promise<SwitchToEnterpriseResponse> => {
  const logData = {
    location: "api.ts:switchToEnterpriseToken",
    message: "switchToEnterpriseToken entry",
    data: {
      url: "/driver/me/switch-to-enterprise",
      platform: Platform.OS,
    },
    timestamp: Date.now(),
    sessionId: "debug-session",
    runId: "run1",
    hypothesisId: "A",
  };
  log.debug("switchToEnterpriseToken entry", logData);
  debugLog(logData);
  try {
    const response = await api.post<SwitchToEnterpriseResponse>(
      "/driver/me/switch-to-enterprise"
    );
    const successLogData = {
      location: "api.ts:switchToEnterpriseToken",
      message: "switchToEnterpriseToken success",
      data: {
        status: response.status,
        hasToken: !!response.data?.token,
        hasError: !!(response.data as any)?.error,
        errorMessage: (response.data as any)?.error,
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      runId: "run1",
      hypothesisId: "B",
    };
    log.success("switchToEnterpriseToken success", successLogData);
    debugLog(successLogData);
    return response.data;
  } catch (error: any) {
    const errorLogData = {
      location: "api.ts:switchToEnterpriseToken",
      message: "switchToEnterpriseToken error",
      data: {
        errorType: error?.constructor?.name || typeof error,
        status: error?.response?.status,
        statusText: error?.response?.statusText,
        errorMessage: error?.message,
        responseData: error?.response?.data,
        hasResponse: !!error?.response,
      },
      timestamp: Date.now(),
      sessionId: "debug-session",
      runId: "run1",
      hypothesisId: "E",
    };
    log.error("switchToEnterpriseToken error", errorLogData);
    debugLog(errorLogData);
    throw error;
  }
};

// ========== Localisation ==========
export interface DriverLocationPayload {
  latitude?: number;
  longitude?: number;
  lat?: number;
  lon?: number;
  speed?: number;
  speed_mps?: number;
  heading?: number;
  accuracy?: number;
  accuracy_m?: number;
  timestamp?: number | string;
  location_mode?: "mission_live" | "availability_presence" | "passive_last_known";
  recorded_at?: string;
  sent_at?: string;
  is_background?: boolean;
  mission_id?: number | null;
  /** Identité logique du point (queue) — retry HTTP réutilise le même id */
  location_event_id?: string;
  device_status?: {
    battery_level?: number;
    low_power_mode?: boolean;
  };
}
type UpdateLocationResp = { ok?: boolean; source?: string; message?: string };

export const updateDriverLocation = async (
  payload: DriverLocationPayload
): Promise<UpdateLocationResp> => {
  const latitude = payload.latitude ?? payload.lat;
  const longitude = payload.longitude ?? payload.lon;
  if (typeof latitude !== "number" || typeof longitude !== "number") {
    throw new Error("Latitude et longitude doivent être numériques");
  }
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    throw new Error("Coordonnées invalides (NaN/Infinity)");
  }
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
    throw new Error("Coordonnées hors bornes");
  }

  const recordedAt =
    payload.recorded_at ??
    (typeof payload.timestamp === "number"
      ? new Date(payload.timestamp).toISOString()
      : payload.timestamp || new Date().toISOString());

  const body: Record<string, unknown> = {
    latitude,
    longitude,
    lat: latitude,
    lon: longitude,
    speed: payload.speed_mps ?? payload.speed ?? 0,
    speed_mps: payload.speed_mps ?? payload.speed ?? 0,
    heading: payload.heading ?? 0,
    accuracy: payload.accuracy_m ?? payload.accuracy ?? 0,
    accuracy_m: payload.accuracy_m ?? payload.accuracy ?? 0,
    ts: recordedAt,
    recorded_at: recordedAt,
    sent_at: payload.sent_at ?? new Date().toISOString(),
    location_mode: payload.location_mode ?? "mission_live",
    is_background: payload.is_background ?? false,
    mission_id: payload.mission_id ?? null,
  };
  if (payload.device_status) {
    body.device_status = payload.device_status;
  }
  if (payload.location_event_id) {
    body.location_event_id = payload.location_event_id;
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (payload.location_event_id) {
    headers["X-Location-Event-Id"] = payload.location_event_id;
  }

  try {
    // ✅ FIX invalid_json: sérialiser explicitement pour éviter body vide/malformé
    // (React Native / axios / background peut parfois perdre le body sur PUT)
    const response = await api.put<UpdateLocationResp>(
      "/driver/me/location",
      JSON.stringify(body),
      {
        headers,
      }
    );
    return response.data ?? {};
  } catch (error: unknown) {
    if (isAxiosError(error)) {
      // Supprimer les erreurs 401/403/404 car elles sont attendues si l'utilisateur n'est pas un chauffeur
      const status = error.response?.status;
      if (status === 401 || status === 403 || status === 404) {
        log.debug("updateDriverLocation access denied (not driver)", { status });
        // Retourner une réponse vide au lieu de lancer une erreur
        return { ok: false, message: "Accès non autorisé" };
      }
      const msg =
        typeof error.response?.data === "string"
          ? error.response.data
          : ((error.response?.data as any)?.message ?? error.message);
      throw new Error(msg);
    }
    if (error instanceof Error) throw error;
    throw new Error("Erreur inconnue lors de la mise à jour de la position");
  }
};

export const updateDriverLocationLegacy = async (
  latitude: number,
  longitude: number
) => updateDriverLocation({ latitude, longitude });

export interface DriverRouteResponse {
  polyline_encoded?: string;
  coordinates?: Array<{ lat: number; lon: number }>;
  distance_meters: number;
  duration_seconds: number;
}

export const getDriverRoute = async (
  originLat: number,
  originLon: number,
  destLat: number,
  destLon: number
): Promise<DriverRouteResponse> => {
  const params = new URLSearchParams({
    origin_lat: String(originLat),
    origin_lon: String(originLon),
    dest_lat: String(destLat),
    dest_lon: String(destLon),
  });
  const res = await api.get<DriverRouteResponse>(`/driver/me/route?${params}`);
  return res.data;
};

// ========== Bookings ==========
export type Booking = {
  id: number;
  /** Chauffeur assigné (si présent dans la réponse API). */
  driver_id?: number;
  pickup_location: string;
  dropoff_location: string;
  scheduled_time: string;
  status: string;
  client_name: string;
  estimated_duration?: string;
  duration_seconds?: number; // Durée estimée du trajet en secondes
  distance_meters?: number; // Distance en mètres
  // ✅ P1-4 Phase 3.3: Déprécié - utiliser client_name à la place
  /** @deprecated Utiliser client_name à la place */
  customer_name?: string;
  client?: {
    id: number;
    first_name: string;
    last_name: string;
    full_name: string;
    birth_date?: string;
    /** Genre client (HOMME/FEMME/AUTRE) pour afficher Madame/Monsieur */
    gender?: string;
    contact_phone?: string; // ✅ P1-4 Phase 3.1: Utiliser client.contact_phone au lieu de client_phone au niveau racine
    phone?: string; // Téléphone principal client
    gp_phone?: string; // Téléphone médecin traitant (optionnel pour bouton Appeler)
    door_code?: string;
    floor?: string;
    access_notes?: string;
  };
  // ✅ P1-4 Phase 3.1: Déprécié - utiliser client.contact_phone à la place
  /** @deprecated Utiliser client.contact_phone à la place */
  client_phone?: string;
  medical_destination?: string;
  wheelchair?: boolean;
  notes?: string;
  is_return: boolean;
  return_time?: string;
  // Nouveaux champs pour les informations médicales
  medical_facility?: string;
  doctor_name?: string;
  hospital_service?: string;
  notes_medical?: string;
  /** Instructions d'accès au point de prise en charge (ex: restaurant, hôtel) */
  pickup_access_notes?: string;
  /** Instructions d'accès à la destination (ex: entrée B, étage 2) */
  dropoff_access_notes?: string;
  // Nouveaux champs pour la chaise roulante
  wheelchair_client_has?: boolean;
  wheelchair_need?: boolean;
  // ✅ P1-4 Phase 3.2: Ajouter champs manquants
  boarded_at?: string; // ISO 8601
  completed_at?: string; // ISO 8601
  // ✅ P1-4 Phase 3.4: Ajouter company_id et company_name
  company_id?: number;
  company_name?: string;
  // ✅ P1-4 Phase 3.5: Ajouter timestamps ISO
  created_at?: string; // ISO 8601
  updated_at?: string; // ISO 8601
  created_at_formatted?: string; // Formaté pour affichage (optionnel)
  updated_at_formatted?: string; // Formaté pour affichage (optionnel)
  // ✅ P1-4 Phase 3.6: Ajouter coordonnées GPS
  pickup_lat?: number;
  pickup_lon?: number;
  dropoff_lat?: number;
  dropoff_lon?: number;
  /** Type de mission : patient_transport | material_delivery */
  mission_type?: "patient_transport" | "material_delivery";
  /** Description de la livraison (requis si mission_type === material_delivery) */
  delivery_description?: string | null;
  [key: string]: any;
};

export const getAssignedTrips = async (options?: { since?: string }): Promise<Booking[]> => {
  try {
    const params = options?.since ? { since: options.since } : {};
    const response = await api.get<Booking[]>("/driver/me/bookings/since", { params });
    return response.data;
  } catch (error: any) {
    // Supprimer les erreurs 401/403/404 car elles sont attendues si l'utilisateur n'est pas un chauffeur
    const status = error?.response?.status;
    if (status === 401 || status === 403 || status === 404) {
      log.debug("getAssignedTrips access denied (not driver)", { status });
      // Retourner un tableau vide au lieu de lancer une erreur
      return [];
    }
    // Pour les autres erreurs, les relancer
    throw error;
  }
};

// ✅ FIX: Ajouter la fonction manquante getCompletedTrips
export const getCompletedTrips = async (
  driverId: number
): Promise<Booking[]> => {
  const response = await api.get<Booking[]>("/driver/me/bookings/all");
  // Filtrer uniquement les courses complétées
  // ✅ P0-1: Utiliser la fonction de normalisation
  return response.data.filter((booking) => isCompletedStatus(booking.status));
};

// Détail d’une course : route conforme à backend driver.py
export const getCompanyTodayTrips = async (): Promise<Booking[]> => {
  try {
    const response = await api.get<Booking[]>("/driver/me/company-bookings/today");
    return response.data;
  } catch (error: any) {
    const status = error?.response?.status;
    if (status === 401 || status === 403 || status === 404) return [];
    throw error;
  }
};

export const getTripDetails = async (bookingId: number): Promise<Booking> => {
  const response = await api.get<Booking>(`/driver/me/bookings/${bookingId}`);
  return response.data;
};

// ✅ P0-1: Import depuis utils pour cohérence
import type { BookingStatus } from "@/utils/bookingStatus";
import { isCompletedStatus } from "@/utils/bookingStatus";
export type { BookingStatus };

/** scope=reservation : annule toute la réservation (aller + retour). Par défaut un seul segment. */
export const updateTripStatus = async (
  bookingId: number,
  status: BookingStatus,
  cancelReason?: "CANCEL" | "RELEASE" | string,
  scope?: "segment" | "reservation"
): Promise<void> => {
  // ✅ P0-1: Normaliser le statut avant envoi (backend accepte CANCELED ou CANCELLED)
  const normalizedStatus = status.toUpperCase() as BookingStatus;
  const payload: {
    status: BookingStatus;
    cancel_reason?: string;
    reason_code?: string;
    scope?: string;
  } = { status: normalizedStatus };
  const isCancel = ["CANCELED", "CANCELLED"].includes(normalizedStatus as string);
  if (cancelReason && isCancel) {
    payload.cancel_reason = cancelReason;
    payload.reason_code = cancelReason; // motif standardisé (NO_SHOW, COMPANY_ISSUE, etc.)
  }
  if (scope === "reservation") {
    payload.scope = "reservation";
  }
  await api.put(`/driver/me/bookings/${bookingId}/status`, payload);
};

export const completeTrip = async (
  bookingId: number,
  isReturn = false
): Promise<void> => {
  const status: BookingStatus = isReturn ? "RETURN_COMPLETED" : "COMPLETED";
  await updateTripStatus(bookingId, status);
};

export type OptimizedRoute = { route: any };
export const getOptimizedRoute = async (
  pickup: string,
  dropoff: string
): Promise<OptimizedRoute> => {
  const response = await api.post<OptimizedRoute>("/ai/optimized-route", {
    pickup,
    dropoff,
  });
  return response.data;
};

export const toggleDriverAvailability = updateDriverAvailability;

// Messages
export type Message = {
  id: number | string;
  company_id?: number;
  sender_id?: number | null;
  receiver_id?: number | null;
  content: string;
  timestamp: string;
  sender_role: "DRIVER" | "COMPANY" | string;
  sender_name?: string | null;
  receiver_name?: string | null;
  _localId?: string | null;
  // Support pour images et PDF
  image?: string | null;
  image_url?: string | null;
  pdf?: string | null;
  pdf_url?: string | null;
  pdf_filename?: string | null;
  pdf_size?: number | null;
};
export const getCompanyMessages = async (
  companyId: number
): Promise<Message[]> => {
  const response = await api.get<Message[]>(`/messages/${companyId}`);
  return response.data;
};

/**
 * ⚡ Phase 4 : Récupère les métriques de performance de l'intercepteur (dev uniquement)
 * Utile pour le debugging et l'analyse des performances
 */
export const getInterceptorPerformanceMetrics = () => {
  if (!__DEV__) {
    return null;
  }

  const totalRequests = interceptorCacheHitCount + interceptorCacheMissCount;

  return {
    cacheHits: interceptorCacheHitCount,
    cacheMisses: interceptorCacheMissCount,
    totalRequests,
    cacheHitRate:
      totalRequests > 0
        ? (interceptorCacheHitCount / totalRequests) * 100
        : 0,
    totalInterceptorRequests: interceptorRequestCount,
  };
};

export const testPushNotification = async (): Promise<{
  ok: boolean;
  results?: Array<{ token_preview: string; platform: string; ok: boolean; error?: string }>;
  tokens_count?: number;
  error?: string;
}> => {
  const res = await api.post("/driver/me/test-push");
  return res.data;
};

export default api;
