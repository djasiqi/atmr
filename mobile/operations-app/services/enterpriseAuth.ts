import AsyncStorage from "@react-native-async-storage/async-storage";
import axios, {
  AxiosError,
  AxiosHeaders,
  InternalAxiosRequestConfig,
} from "axios";
import { Platform } from "react-native";
import Constants from "expo-constants";
import * as Sentry from "@sentry/react-native";
import { waitForAuthReady } from "@/services/authSync";
import { secureStorage, asyncStorage } from "@/services/storage";
import { AuthInvalidError, AuthNotReadyError, isPublicEndpoint } from "@/services/authGuards";
import { invokeForceLogoutEnterprise } from "@/services/authController";
import { logAuthEvent, beginRefreshCycle } from "@/services/authLogging";
import { debugAuthLog, isDebugAuthEnabled } from "@/services/authDebug";
import { getLogger } from "@/utils/logger";
import { sendIngestEvent } from "@/src/config/telemetry";

const log = getLogger("EntAuth");

const debugLog = (data: Record<string, unknown>) => {
  if (__DEV__) {
    try {
      sendIngestEvent(data);
    } catch {
      // ignore
    }
  }
};

const expoExtra = Constants.expoConfig?.extra || {};
const ENV_API_URL = process.env.EXPO_PUBLIC_API_URL;
const PROD_API_URL: string = ENV_API_URL || "";
const DEV_API_URL = (expoExtra.devApiUrl as string) || (expoExtra.publicApiUrl as string);
const APP_VARIANT = String(expoExtra.APP_VARIANT || process.env.APP_VARIANT || "prod");

const getDevHost = (): string => {
  // Sur le web, toujours utiliser localhost (le navigateur tourne sur la même machine)
  if (Platform.OS === 'web') {
    return "localhost";
  }
  
  // Sur mobile (iOS/Android), détecter l'IP locale de la machine de développement
  const legacyHost = (Constants as any)?.manifest?.debuggerHost?.split(":")[0];
  const newHost = (Constants as any)?.expoConfig?.hostUri?.split(":")[0];
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
const API_PREFIX = "/api/v1/company_mobile";

// Détecter le mode développement de manière plus fiable
// En web bundled, __DEV__ peut être false même en développement local
const isDevelopment = () => {
  if (APP_VARIANT === "prod") {
    return false;
  }
  // Vérifier __DEV__ d'abord
  if (__DEV__) {
    return true;
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

const validateProdApiUrl = (value: unknown): string => {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error("EXPO_PUBLIC_API_URL is required in production");
  }
  const normalized = value.trim().replace(/\/$/, "");
  if (!normalized.startsWith("https://")) {
    throw new Error("EXPO_PUBLIC_API_URL must be HTTPS in production");
  }
  if (normalized.includes("localhost") || normalized.includes("127.0.0.1")) {
    throw new Error("EXPO_PUBLIC_API_URL must not point to localhost in production");
  }
  return normalized;
};

// En développement, forcer l'utilisation de getDevHost() pour éviter de pointer vers la production
// Surtout important sur le web où on doit utiliser localhost
const getDevBaseURL = () => {
  if (Platform.OS === 'web') {
    // Sur le web, toujours utiliser localhost en développement
    return `http://localhost:${PORT}${API_PREFIX}`;
  }
  // Sur mobile, utiliser getDevHost() pour détecter l'IP locale
  return `http://${getDevHost()}:${PORT}${API_PREFIX}`;
};

// Déterminer l'URL de base à utiliser
const getBaseURL = () => {
  const isDev = isDevelopment();
  
  if (isDev) {
    // En développement, sur le web, toujours utiliser localhost (ignorer DEV_API_URL si défini à la prod)
    if (Platform.OS === 'web') {
      return getDevBaseURL();
    }
    // Sur mobile, utiliser DEV_API_URL si défini et différent de la prod, sinon getDevBaseURL()
    if (DEV_API_URL && DEV_API_URL !== PROD_API_URL && !DEV_API_URL.includes('api.lirie.ch')) {
      return `${DEV_API_URL.replace(/\/$/, "")}${API_PREFIX}`;
    }
    return getDevBaseURL();
  }
  
  // En production, utiliser PROD_API_URL
  return `${validateProdApiUrl(PROD_API_URL)}${API_PREFIX}`;
};

const baseURL = getBaseURL();

try {
  log.info("baseurl resolved", { baseURL, PROD_API_URL, ENV_API_URL, PORT });
} catch {}

export const ENTERPRISE_SESSION_KEY = "enterprise.session";

type AxiosConfig = InternalAxiosRequestConfig<any> & {
  __isRetryRequest?: boolean;
};

// ✅ Fonction helper pour valider le token
export const hasValidToken = async (): Promise<boolean> => {
  try {
    // ✅ CORRECTION : Utiliser SecureStore au lieu d'AsyncStorage
    let token = await secureStorage.getEnterpriseToken();
    
    // ✅ Si le token manque, vérifier si un refresh token existe (mais ne pas refresh ici)
    // Le refresh sera géré par l'intercepteur si nécessaire
    if (!token) {
      const refreshToken = await secureStorage.getEnterpriseRefreshToken();
      // ✅ Si un refresh token existe, considérer comme "valide" pour permettre l'intercepteur de faire le refresh
      // Cela évite de bloquer les requêtes inutilement
      if (refreshToken) {
        // Refresh token disponible → l'intercepteur gérera le refresh si nécessaire
        // Ne pas retourner false immédiatement, permettre à l'intercepteur de tenter le refresh
        log.info("token missing but refresh available, interceptor will refresh", {});
        return true; // Permettre à la requête de partir, l'intercepteur gérera le refresh
      } else {
        // Pas de refresh token non plus
        if (!__DEV__) {
          Sentry.captureMessage("Guard Pattern: Token manquant", {
            level: "warning",
            tags: {
              type: "token_validation",
              reason: "missing_token",
            },
            contexts: {
              tokenInfo: {
                hasToken: false,
                hasRefreshToken: false,
                timestamp: new Date().toISOString(),
              },
            },
          });
        }
        return false;
      }
    }
    
    // Vérifier que le token a un format JWT valide (3 parties séparées par .)
    if (!token) {
      return false; // Fallback de sécurité
    }
    
    const parts = token.split(".");
    if (parts.length !== 3) {
      log.warn("invalid token format", { partsCount: parts.length });
      
      // ✅ Capturer token invalide pour monitoring
      if (!__DEV__) {
        Sentry.captureMessage("Guard Pattern: Token format invalide", {
          level: "error",
          tags: {
            type: "token_validation",
            reason: "invalid_format",
          },
          contexts: {
            tokenInfo: {
              hasToken: true,
              parts: parts.length,
              timestamp: new Date().toISOString(),
            },
          },
        });
      }
      
      await clearEnterpriseStorage();
      return false;
    }
    
    return true;
  } catch (error) {
    log.warn("token validation error", { error });
    
    // ✅ Capturer erreur validation pour monitoring
    if (!__DEV__) {
      Sentry.captureException(error, {
        tags: {
          type: "token_validation",
          reason: "exception",
        },
      });
    }
    
    return false;
  }
};

// ✅ Fonction pour attendre qu'un token soit disponible (avec timeout)
export const waitForToken = async (
  timeoutMs: number = 5000
): Promise<string | null> => {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeoutMs) {
    // ✅ CORRECTION : Utiliser SecureStore au lieu d'AsyncStorage
    const token = await secureStorage.getEnterpriseToken();
    if (token) {
      return token;
    }
    
    // Attendre 100ms avant de réessayer
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  return null;
};

// ✅ Fonction de retry avec backoff exponentiel
export const retryWithBackoff = async <T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    baseDelay?: number;
    maxDelay?: number;
    shouldRetry?: (error: any) => boolean;
  } = {}
): Promise<T> => {
  const {
    maxRetries = 3,
    baseDelay = 1000,
    maxDelay = 10000,
    shouldRetry = (error) => {
      // Retry sur erreurs réseau ou 5xx
      const status = error?.response?.status;
      return (
        !status || // Erreur réseau
        status >= 500 // Erreur serveur
      );
    },
  } = options;

  let lastError: any;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Ne pas retry si c'est le dernier essai ou si on ne doit pas retry
      if (attempt === maxRetries || !shouldRetry(error)) {
        throw error;
      }

      // Calculer le délai (exponential backoff + jitter)
      const delay = Math.min(
        baseDelay * Math.pow(2, attempt) + Math.random() * 1000,
        maxDelay
      );

      log.warn("retry attempt failed, retrying", {
        attempt: attempt + 1,
        maxRetries,
        delayMs: Math.round(delay),
      });

      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError;
};

export interface EnterpriseUserPayload {
  id: number;
  public_id: string;
  email: string;
  first_name?: string | null;
  last_name?: string | null;
  role: string;
}

export interface EnterpriseCompanyPayload {
  id: number;
  name: string;
  dispatch_mode?: string | null;
}

export interface EnterpriseTokenPayload {
  token: string;
  refresh_token?: string | null;
  user: EnterpriseUserPayload;
  company: EnterpriseCompanyPayload;
  scopes?: string[];
  session_id: string;
  mfa_required?: false;
}

export interface EnterpriseLoginMfaPayload {
  mfa_required: true;
  challenge_id: string;
  methods?: string[];
  ttl?: number;
  message?: string;
}

export type EnterpriseLoginResponse =
  | EnterpriseTokenPayload
  | EnterpriseLoginMfaPayload;

export interface EnterpriseSessionPayload {
  user: EnterpriseUserPayload;
  company: EnterpriseCompanyPayload;
  scopes?: string[];
  session_id: string;
}

export interface EnterpriseLoginParams {
  method?: "password" | "oidc";
  email?: string;
  password?: string;
  id_token?: string;
  provider?: string;
  mfa_code?: string;
  device_id?: string;
}

export interface EnterpriseMfaVerifyParams {
  challenge_id: string;
  code: string;
  device_id?: string;
}

export const enterpriseApi = axios.create({
  baseURL,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// ⚡ CORRECTION : Cache interceptor pour Enterprise API (similaire à Driver API)
// Réduit les lectures SecureStore répétées lors de requêtes simultanées
let enterpriseInterceptorTokenCache: string | null = null;
let enterpriseInterceptorTokenCacheTime = 0;
const ENTERPRISE_INTERCEPTOR_CACHE_TTL = 30000; // 30 secondes

// ⚡ Métriques de performance pour l'intercepteur Enterprise (dev uniquement)
let enterpriseInterceptorCacheHitCount = 0;
let enterpriseInterceptorCacheMissCount = 0;
let enterpriseInterceptorRequestCount = 0;
const ENTERPRISE_INTERCEPTOR_METRICS_LOG_INTERVAL = 100; // Log toutes les 100 requêtes

/**
 * Invalide le cache de l'intercepteur Enterprise (utile lors du logout ou changement de token)
 */
export const invalidateEnterpriseInterceptorCache = () => {
  enterpriseInterceptorTokenCache = null;
  enterpriseInterceptorTokenCacheTime = 0;
  if (__DEV__) {
    enterpriseInterceptorCacheHitCount = 0;
    enterpriseInterceptorCacheMissCount = 0;
    enterpriseInterceptorRequestCount = 0;
  }
};

/** P1.A — Nettoie uniquement les clés auth entreprise (SecureStore + AsyncStorage). */
const clearEnterpriseStorage = async () => {
  await secureStorage.clearEnterpriseAuthOnly();
  invalidateEnterpriseInterceptorCache();
};

const persistEnterpriseSession = async (
  payload: EnterpriseTokenPayload
): Promise<void> => {
  const session = {
    token: payload.token,
    refreshToken: payload.refresh_token ?? null,
    user: payload.user,
    company: {
      id: payload.company.id,
      name: payload.company.name,
      dispatchMode:
        (payload.company as any).dispatchMode ?? payload.company.dispatch_mode,
    },
    scopes: payload.scopes ?? [],
    sessionId: payload.session_id,
  };

  // ✅ CORRECTION : Utiliser SecureStore pour les tokens (sécurisé + cache)
  try {
    await secureStorage.setEnterpriseToken(session.token);
  } catch (error) {
    // ✅ CORRECTION : Gérer les erreurs de stockage (ex: Keychain/Keystore inaccessible)
    log.error("enterprise token save failed", { error });
    throw new Error("Échec de la sauvegarde du token Enterprise");
  }
  
  // Garder AsyncStorage uniquement pour la session complète (données non sensibles)
  try {
    await AsyncStorage.setItem(
      ENTERPRISE_SESSION_KEY,
      JSON.stringify(session)
    );
  } catch (error) {
    // ✅ CORRECTION : Gérer les erreurs de stockage AsyncStorage
    log.error("enterprise session save failed", { error });
    // Ne pas throw ici car le token est déjà sauvegardé dans SecureStore
    // La session AsyncStorage est moins critique
  }

  if (session.refreshToken) {
    try {
      await secureStorage.setEnterpriseRefreshToken(session.refreshToken);
    } catch (error) {
      log.error("enterprise refresh token save failed", { error });
    }
  } else {
    // Ne PAS supprimer l'ancien refresh token si le nouveau est absent.
    // Sur iOS, une réponse partielle ou un crash réseau pourrait perdre le refresh token
    // et rendre la session irrécupérable.
    log.warn("persist: no refresh_token in payload, keeping existing");
  }

  // ✅ Vérifier que le token a bien été sauvegardé
  const savedToken = await secureStorage.getEnterpriseToken();
  
  // ⚡ CORRECTION : Mettre à jour le cache interceptor avec le nouveau token
  if (savedToken) {
    enterpriseInterceptorTokenCache = savedToken;
    enterpriseInterceptorTokenCacheTime = Date.now();
  }
  
  if (!savedToken || savedToken !== session.token) {
    log.error("token not persisted correctly", {
      savedTokenExists: !!savedToken,
      savedTokenMatches: savedToken === session.token,
    });
    
    // ✅ Capturer échec de persistence pour monitoring
    if (!__DEV__) {
      Sentry.captureMessage("Token persistence failed", {
        level: "fatal",
        tags: {
          type: "token_persistence",
          reason: "save_failed",
        },
        contexts: {
          persistenceInfo: {
            tokenLength: session.token.length,
            savedTokenExists: !!savedToken,
            savedTokenMatches: savedToken === session.token,
            timestamp: new Date().toISOString(),
          },
        },
      });
    }
    
    throw new Error("Échec de la persistence du token");
  }

  log.success("session saved and verified", {});
};

// ✅ CORRECTION #2 : Queue de refresh similaire à Driver API
// Évite les refresh multiples simultanés qui causent des "missing token"
let isRefreshing = false;
let tokenRefreshPromise:
  | Promise<string | null | undefined>
  | null = null;
let failedQueue: Array<{
  resolve: (token: string | null | undefined) => void;
  reject: (error: any) => void;
}> = [];

const processQueue = (
  error: any,
  token: string | null | undefined = null
): void => {
  // ✅ CORRECTION : Traiter toutes les requêtes en queue (comme Driver API)
  // Une fois le refresh terminé, toutes les requêtes en queue reçoivent le nouveau token
  while (failedQueue.length > 0) {
    const { resolve, reject } = failedQueue.shift()!;
    if (error) {
      reject(error);
    } else if (token) {
      resolve(token);
    } else {
      // Token manquant après refresh → rejeter avec erreur
      reject(new Error("Token manquant après refresh"));
    }
  }
};

let enterpriseRefreshPayloadPromise: Promise<EnterpriseTokenPayload> | null = null;
let enterpriseRefreshFailedQueue: Array<{
  resolve: (payload: EnterpriseTokenPayload) => void;
  reject: (error: any) => void;
}> = [];

const processEnterpriseRefreshQueue = (
  error: any,
  payload: EnterpriseTokenPayload | null = null
): void => {
  while (enterpriseRefreshFailedQueue.length > 0) {
    const { resolve, reject } = enterpriseRefreshFailedQueue.shift()!;
    if (error) {
      reject(error);
    } else if (payload) {
      resolve(payload);
    } else {
      reject(new Error("Token manquant après refresh"));
    }
  }
};

export const refreshEnterpriseTokenSingleflight = async (
  overrideRefreshToken?: string
): Promise<EnterpriseTokenPayload> => {
  if (enterpriseRefreshPayloadPromise) {
    return enterpriseRefreshPayloadPromise;
  }

  enterpriseRefreshPayloadPromise = (async () => {
    const refreshToken =
      overrideRefreshToken || (await secureStorage.getEnterpriseRefreshToken());
    if (!refreshToken) {
      logAuthEvent("AUTH_REFRESH_FAIL", {
        route: "enterprise",
        reason: "missing_refresh_token",
        refresh_attempted: false,
        outcome: "abort",
      });
      throw new AuthNotReadyError({
        kind: "enterprise",
        reason: "missing_refresh_token",
        url: "/auth/refresh",
      });
    }

    const refreshCycleId = beginRefreshCycle("enterprise");
    logAuthEvent("AUTH_REFRESH_START", { route: "enterprise", refresh_cycle_id: refreshCycleId });
    const response = await axios.post<EnterpriseTokenPayload>(
      `${baseURL}/auth/refresh`,
      { refresh_token: refreshToken },
      {
        headers: { "Content-Type": "application/json" },
        timeout: 10000,
      }
    );

    const payload = response.data;
    await persistEnterpriseSession(payload);

    // Cache interceptor
    enterpriseInterceptorTokenCache = payload.token;
    enterpriseInterceptorTokenCacheTime = Date.now();

    return payload;
  })();

  enterpriseRefreshPayloadPromise
    .then((payload) => processEnterpriseRefreshQueue(null, payload))
    .catch((err) => processEnterpriseRefreshQueue(err, null))
    .finally(() => {
      enterpriseRefreshPayloadPromise = null;
    });

  return enterpriseRefreshPayloadPromise;
};

const refreshAccessToken = async (): Promise<string | null | undefined> => {
  // ✅ CORRECTION : Vérifier d'abord si un refresh est déjà en cours
  // Cette vérification doit être atomique pour éviter les race conditions
  if (isRefreshing && tokenRefreshPromise) {
    // Mettre en queue ✅ (comme Driver API)
    return new Promise<string | null | undefined>((resolve, reject) => {
      failedQueue.push({ resolve, reject });
    });
  }

  // ✅ CORRECTION : Double vérification pour éviter les race conditions
  // Si tokenRefreshPromise existe maintenant (créé entre les deux vérifications), l'utiliser
  if (tokenRefreshPromise) {
    return tokenRefreshPromise;
  }

  // Démarrer un nouveau refresh (lock atomique)
  isRefreshing = true;
  tokenRefreshPromise = (async () => {
      try {
        const payload = await refreshEnterpriseTokenSingleflight();
        const newToken = payload.token;
        log.success("refresh succeeded, new token obtained", {});
        
        processQueue(null, newToken);
        return newToken;
      } catch (error: any) {
        // ✅ CORRECTION : Distinguer les types d'erreurs pour ne pas supprimer les tokens inutilement
        const refreshStatus = error?.response?.status;
        const errorData = error?.response?.data;
        const isNetworkError = !error?.response; // Pas de réponse = erreur réseau
        
        log.error("refresh error", {
          status: refreshStatus,
          error: errorData?.error || error?.message,
          isNetworkError,
          hasResponse: !!error?.response,
        });
        
        // Ne supprimer les tokens que si c'est vraiment un problème d'authentification
        // (401 = refresh token expiré/invalide, 403 = compte désactivé)
        // Ne pas supprimer pour erreurs réseau temporaires ou erreurs serveur
        if (refreshStatus === 401) {
          logAuthEvent("AUTH_REFRESH_FAIL", {
            route: "enterprise",
            status: 401,
            refresh_attempted: true,
            outcome: "logout",
          });
          log.error("refresh token rejected (401)", {});
          const reason = "refresh_rejected_401";
          processQueue(new AuthInvalidError({ route: "enterprise", reason }), null);
          await invokeForceLogoutEnterprise(reason);
          throw new AuthInvalidError({ route: "enterprise", reason });
        } else if (refreshStatus === 403) {
          const serverReason = errorData?.reason;
          if (serverReason === "account_disabled") {
            logAuthEvent("AUTH_REFRESH_FAIL", {
              route: "enterprise",
              status: 403,
              refresh_attempted: true,
              outcome: "logout",
              server_reason: "account_disabled",
            });
            log.error("refresh token rejected (account disabled)", {});
            processQueue(new AuthInvalidError({ route: "enterprise", reason: "account_disabled" }), null);
            await invokeForceLogoutEnterprise("account_disabled");
            throw new AuthInvalidError({ route: "enterprise", reason: "account_disabled" });
          }
          log.warn("refresh 403 without account_disabled, treating as transient", {
            serverError: errorData?.error,
          });
          processQueue(error, null);
          throw error;
        } else if (isNetworkError) {
          log.warn("network error during refresh, tokens kept", {});
          processQueue(error, null);
          throw error;
        } else {
          log.warn("server error during refresh, tokens kept", { status: refreshStatus });
          processQueue(error, null);
          throw error;
        }
      } finally {
        tokenRefreshPromise = null;
        isRefreshing = false;
      }
    })();

  return tokenRefreshPromise;
};

enterpriseApi.interceptors.request.use(
  async (config) => {
    try {
      const rawUrl = String(config.url || "").toLowerCase();
      const isExplicitPublicAuthRoute =
        rawUrl.includes("/auth/login") ||
        rawUrl.includes("/auth/refresh") ||
        rawUrl.includes("/auth/mfa");

      // Verrou: ne jamais tenter de refresh/token sur endpoints publics enterprise
      if (isExplicitPublicAuthRoute) {
        return config;
      }
      // #region agent log
      const isLoginRequest = config.url?.includes("/auth/login");
      const isPublic = isPublicEndpoint(config.url, "enterprise");
      debugLog({location:'enterpriseAuth.ts:268',message:'interceptor request entry',data:{url:config.url,method:config.method,isLoginRequest,baseURL:config.baseURL},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'});
      // #endregion
      
      // ✅ CORRECTION #1 : Guard d'initialisation
      // Attendre que l'auth soit prête avant de permettre les requêtes (sauf login)
      if (!isPublic) {
        try {
          await waitForAuthReady(5000);
        } catch (error) {
          log.warn("auth not ready timeout, request rejected", { url: config.url });
          throw new AuthNotReadyError({
            kind: "enterprise",
            reason: "auth_ready_timeout",
            url: config.url,
          });
        }
      }
      
      const headers =
        config.headers instanceof AxiosHeaders
          ? config.headers
          : new AxiosHeaders(config.headers || {});

      // ✅ Ne pas ajouter de token uniquement pour les requêtes de login
      // Les autres endpoints /auth/ (comme /auth/me/driver-account, /auth/refresh, etc.) nécessitent un token
      if (!isPublic) {
        // Si un token est déjà présent dans les headers (passé explicitement), l'utiliser
        // Sinon, essayer de le récupérer depuis SecureStore avec cache interceptor
        if (!headers.has("Authorization")) {
          // ⚡ CORRECTION : Cache interceptor (30s TTL) pour réduire les lectures SecureStore
          // Similaire à Driver API pour cohérence et performance
          const now = Date.now();
          let token = enterpriseInterceptorTokenCache;
          
          if (!token || now - enterpriseInterceptorTokenCacheTime >= ENTERPRISE_INTERCEPTOR_CACHE_TTL) {
            // Cache invalide ou expiré → lire depuis SecureStore (qui utilise son propre cache 1min)
            token = await secureStorage.getEnterpriseToken();
            enterpriseInterceptorTokenCache = token;
            enterpriseInterceptorTokenCacheTime = now;
            if (__DEV__) {
              enterpriseInterceptorCacheMissCount++;
            }
          } else {
            if (__DEV__) {
              enterpriseInterceptorCacheHitCount++;
            }
          }
          
          // ⚡ Métriques de performance (dev uniquement)
          if (__DEV__) {
            enterpriseInterceptorRequestCount++;
            if (enterpriseInterceptorRequestCount % ENTERPRISE_INTERCEPTOR_METRICS_LOG_INTERVAL === 0) {
              const totalRequests = enterpriseInterceptorCacheHitCount + enterpriseInterceptorCacheMissCount;
              const cacheHitRate = totalRequests > 0
                ? (enterpriseInterceptorCacheHitCount / totalRequests) * 100
                : 0;
              log.info("interceptor cache performance", {
                cacheHitRate: cacheHitRate.toFixed(1),
                hits: enterpriseInterceptorCacheHitCount,
                misses: enterpriseInterceptorCacheMissCount,
                total: enterpriseInterceptorRequestCount,
              });
            }
          }
          // #region agent log
          debugLog({location:'enterpriseAuth.ts:278',message:'token check',data:{url:config.url,hasToken:!!token,isLoginRequest},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'});
          // #endregion
          if (token) {
            headers.set("Authorization", `Bearer ${token}`);
            log.debug("token added to request", { url: config.url });
          } else {
            // ✅ P1 (strict): ne jamais appeler /auth/refresh sans refresh_token
            // Vérifier explicitement avant refresh → fail fast, UX "Connexion…" / "Session expirée"
            const refreshToken = await secureStorage.getEnterpriseRefreshToken();
            if (isDebugAuthEnabled()) {
              debugAuthLog("interceptor_before_refresh", {
                refresh_present: refreshToken ? 1 : 0,
              });
            }
            if (!refreshToken) {
              logAuthEvent("AUTH_ACCESS_ABSENT", {
                kind: "enterprise",
                reason: "missing_refresh_token",
                url: config.url?.slice(0, 100),
              });
              log.warn("auth not ready missing refresh token, request blocked", { url: config.url });
              throw new AuthNotReadyError({
                kind: "enterprise",
                reason: "missing_refresh_token",
                url: "/auth/refresh",
              });
            }
            log.warn("no token, attempting refresh singleflight", { url: config.url });
            try {
              const payload = await refreshEnterpriseTokenSingleflight(refreshToken);
              headers.set("Authorization", `Bearer ${payload.token}`);
            } catch (e) {
              throw new AuthNotReadyError({
                kind: "enterprise",
                reason: "missing_token_and_refresh_failed",
                url: config.url,
              });
            }
          }
        } else {
          log.debug("token already in headers", { url: config.url });
        }
      }

      // ✅ Ne pas ajouter X-Company-ID/X-Session-ID uniquement pour les requêtes de login
      // Les autres endpoints /auth/ (comme /auth/me/driver-account, /auth/refresh, etc.) nécessitent une session
      if (!isLoginRequest) {
        const sessionRaw = await AsyncStorage.getItem(ENTERPRISE_SESSION_KEY);
        // #region agent log
        debugLog({location:'enterpriseAuth.ts:301',message:'session check',data:{url:config.url,hasSession:!!sessionRaw,isLoginRequest},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'});
        // #endregion
        if (sessionRaw) {
          try {
            const session = JSON.parse(sessionRaw);
            if (session?.company?.id && !headers.has("X-Company-ID")) {
              headers.set("X-Company-ID", String(session.company.id));
              log.debug("x-company-id added", { companyId: session.company.id });
            }
            if (session?.sessionId && !headers.has("X-Session-ID")) {
              headers.set("X-Session-ID", session.sessionId);
              log.debug("x-session-id added", { sessionId: session.sessionId });
            }
          } catch (e) {
            log.warn("session parse error for headers", { error: e });
          }
        } else {
          const isPublicEndpoint = config.url?.includes("/auth/") ||
                                    config.url?.includes("/public");
          if (!isPublicEndpoint) {
            log.warn("no session available for request", { url: config.url });
          }
        }
      }

      // ✅ Device/session correlation: toujours envoyer X-Device-ID (stable) sur mobile
      if (!headers.has("X-Device-ID") && Platform.OS !== "web") {
        try {
          const deviceId = await asyncStorage.getOrCreateDeviceId();
          headers.set("X-Device-ID", deviceId);
        } catch (e) {
          log.warn("could not generate x-device-id", { error: e });
        }
      }

      // #region agent log
      const finalHeaders: Record<string, string> = {};
      // Axios v1: config.headers peut être AxiosHeaders, un objet simple, ou undefined.
      // Sur certains chemins (ex: login), `headers` peut ne pas implémenter `forEach`.
      try {
        if (headers && typeof (headers as any).forEach === "function") {
          (headers as any).forEach((value: unknown, key: string) => {
            if (String(key).toLowerCase() === "authorization") {
              finalHeaders[key] = value
                ? `${String(value).substring(0, 20)}...`
                : "";
            } else {
              finalHeaders[key] = String(value);
            }
          });
        } else if (headers && typeof headers === "object") {
          for (const [k, v] of Object.entries(headers as any)) {
            if (String(k).toLowerCase() === "authorization") {
              finalHeaders[k] = v ? `${String(v).substring(0, 20)}...` : "";
            } else if (v !== undefined) {
              finalHeaders[k] = String(v);
            }
          }
        }
      } catch {
        // ne jamais casser la requête juste pour le logging
      }
      debugLog({location:'enterpriseAuth.ts:333',message:'interceptor request exit',data:{url:config.url,headers:finalHeaders,isLoginRequest},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'});
      // #endregion
      config.headers = headers;
      return config;
    } catch (err) {
      // ✅ P1: ne jamais "swallow" une erreur d'auth; rejeter la requête
      return Promise.reject(err);
    }
  },
  (error) => Promise.reject(error)
);

enterpriseApi.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const { response, config } = error;
    const originalConfig = config as AxiosConfig | undefined;
    const rawUrl = String(originalConfig?.url || "").toLowerCase();
    const isExplicitPublicAuthRoute =
      rawUrl.includes("/auth/login") ||
      rawUrl.includes("/auth/refresh") ||
      rawUrl.includes("/auth/mfa");
    if (isExplicitPublicAuthRoute) {
      return Promise.reject(error);
    }

    // ✅ CORRECTION (audit 34-38):
    // - 403 peut être un "forbidden" fonctionnel (rôle/droits) et ne doit pas déclencher un refresh.
    // - On ne tente donc un refresh automatique que sur 401.
    //
    // Exception: si la requête de refresh elle-même échoue (401/403), on nettoie la session enterprise.
    if (originalConfig?.url?.includes("/auth/refresh")) {
      const refreshStatus = response?.status;
      const errorData = response?.data as { error?: string } | undefined;

      if (refreshStatus === 401) {
        logAuthEvent("AUTH_REFRESH_FAIL", {
          route: "enterprise",
          status: 401,
          refresh_attempted: true,
          outcome: "logout",
          source: "refresh_endpoint",
        });
        log.error("interceptor refresh token rejected (401)", { url: originalConfig.url });
        const reason = "refresh_rejected_401";
        processQueue(new AuthInvalidError({ route: "enterprise", reason }), null);
        await invokeForceLogoutEnterprise(reason);
        return Promise.reject(new AuthInvalidError({ route: "enterprise", reason }));
      }
      if (refreshStatus === 403) {
        const serverReason = (errorData as { reason?: string })?.reason;
        if (serverReason === "account_disabled") {
          logAuthEvent("AUTH_REFRESH_FAIL", {
            route: "enterprise",
            status: 403,
            refresh_attempted: true,
            outcome: "logout",
            source: "refresh_endpoint",
            server_reason: "account_disabled",
          });
          log.error("interceptor refresh rejected (account disabled)", { url: originalConfig.url });
          processQueue(new AuthInvalidError({ route: "enterprise", reason: "account_disabled" }), null);
          await invokeForceLogoutEnterprise("account_disabled");
          return Promise.reject(new AuthInvalidError({ route: "enterprise", reason: "account_disabled" }));
        }
        log.warn("interceptor refresh 403 without account_disabled, treating as transient", {
          serverError: errorData?.error,
        });
        return Promise.reject(error);
      }
      // Autres erreurs (réseau, etc.) → ne pas déconnecter
      return Promise.reject(error);
    }

    const isAuthError = response?.status === 401;

    if (isAuthError && originalConfig && !originalConfig.__isRetryRequest) {

      // Si déjà en train de refresh, mettre en queue (via refreshAccessToken)
      // La queue est gérée dans refreshAccessToken() qui utilise failedQueue
      
      try {
        const payload = await refreshEnterpriseTokenSingleflight();
        const newToken = payload.token;
        if (!newToken) {
          // Si refresh échoue, vérifier si c'est une erreur critique
          // Note: refreshAccessToken() gère déjà la déconnexion pour 401/403 (uniquement sur /auth/refresh)
          // On ne doit pas déconnecter à nouveau ici pour éviter les doubles déconnexions
          const refreshError = error;
          const refreshStatus = (refreshError as any)?.response?.status;
          
          log.warn("refresh failed, no new token", { status: refreshStatus });
          
          // refreshAccessToken() a déjà géré la déconnexion si nécessaire
          return Promise.reject(error);
        }

        const headers =
          originalConfig.headers instanceof AxiosHeaders
            ? originalConfig.headers
            : new AxiosHeaders(originalConfig.headers || {});

        headers.set("Authorization", `Bearer ${newToken}`);
        originalConfig.headers = headers;
        originalConfig.__isRetryRequest = true;

        return enterpriseApi(originalConfig);
      } catch (refreshError: any) {
        const refreshStatus = refreshError?.response?.status;
        const isNetworkError = !refreshError?.response;
        
        // ✅ Distinguer les types d'erreurs (comme Driver API)
        if (refreshStatus === 401) {
          logAuthEvent("AUTH_REFRESH_FAIL", {
            route: "enterprise",
            status: 401,
            refresh_attempted: true,
            outcome: "logout",
          });
          log.error("refresh failed, force logout", { status: refreshStatus });
          await invokeForceLogoutEnterprise("refresh_rejected_401");
          return Promise.reject(refreshError);
        } else if (isNetworkError) {
          // Erreur réseau temporaire → ne pas déconnecter
          log.warn("network error during refresh, user stays connected", {});
          return Promise.reject(refreshError);
        } else {
          log.warn("server error during refresh, user stays connected", { status: refreshStatus });
          return Promise.reject(refreshError);
        }
      }
    }
    
    // ✅ 403 = forbidden fonctionnel → ne pas retry automatiquement (audit 34-38)
    if (response?.status === 403 && originalConfig && !originalConfig.__isRetryRequest) {
      log.error("access denied 403", {
        url: originalConfig.url,
        data: response?.data,
        message: error.message,
      });
      // Si c'est un 403, ne pas retry automatiquement.
      return Promise.reject(error);
    }

    return Promise.reject(error);
  }
);

export const loginEnterprise = async (
  params: EnterpriseLoginParams
): Promise<EnterpriseLoginResponse> => {
  try {
    // #region agent log
    const fullURL = `${baseURL}/auth/login`;
    const logData = {
      location:'enterpriseAuth.ts:407',
      message:'loginEnterprise entry',
      data:{
        baseURL,
        fullURL,
        hasEmail:Boolean(params.email),
        hasPassword:Boolean(params.password),
        method:params.method,
        platform:Platform.OS,
        isDev:__DEV__,
        executionEnv:Constants.executionEnvironment,
      },
      timestamp:Date.now(),
      sessionId:'debug-session',
      runId:'run1',
      hypothesisId:'D'
    };
    debugLog(logData);
    // #endregion
    log.info("login request", {
      hasEmail: Boolean(params.email),
      hasPassword: Boolean(params.password),
      baseURL,
      fullURL,
    });

    const response = await enterpriseApi.post<EnterpriseLoginResponse>(
      "/auth/login",
      params
    );
    // #region agent log
    const successLogData = {location:'enterpriseAuth.ts:422',message:'loginEnterprise success',data:{status:response.status,hasToken:!!(response.data as any)?.token,hasMfa:!!(response.data as any)?.mfa_required},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'};
    debugLog(successLogData);
    // #endregion
    log.info("login response received", {
      hasToken: !!(response.data as any)?.token,
      hasMfa: !!(response.data as any)?.mfa_required,
      status: response.status,
    });
    return response.data;
  } catch (err: unknown) {
    // #region agent log
    const errorData: any = {errorType:err instanceof Error ? err.constructor.name : typeof err};
    if (axios.isAxiosError(err)) {
      errorData.status = err.response?.status;
      errorData.statusText = err.response?.statusText;
      errorData.responseData = err.response?.data;
      errorData.url = err.config?.url;
      errorData.baseURL = err.config?.baseURL;
      errorData.fullURL = err.config?.baseURL ? `${err.config.baseURL}${err.config.url}` : err.config?.url;
    }
    const errorLogData = {location:'enterpriseAuth.ts:432',message:'loginEnterprise error',data:errorData,timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'};
    debugLog(errorLogData);
    // #endregion
    const isNetErr =
      (axios.isAxiosError(err) && (err as any)?.code === "ERR_NETWORK") ||
      (axios.isAxiosError(err) && err.message?.toLowerCase().includes("network error"));

    if (!isNetErr) {
      throw err;
    }

    // Fallback via fetch (30s) pour contourner un éventuel souci Axios natif
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30000);
      const res = await fetch(`${baseURL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      clearTimeout(timeout);
      const text = await res.text();
      if (!res.ok) {
        throw new Error(`Enterprise login via fetch a échoué (${res.status}): ${text}`);
      }
      log.success("login via fetch ok", {});
      return JSON.parse(text) as EnterpriseLoginResponse;
    } catch (fallbackError) {
      log.warn("login fallback fetch failed", { error: fallbackError });
      throw err;
    }
  }
};

export const verifyEnterpriseMfa = async (
  params: EnterpriseMfaVerifyParams
): Promise<EnterpriseTokenPayload> => {
  const response = await enterpriseApi.post<EnterpriseTokenPayload>(
    "/auth/mfa/verify",
    params
  );
  return response.data;
};

// ✅ Alias compatible: tous les chemins passent par le singleflight (P2)
export const refreshEnterpriseToken = async (
  refreshToken: string
): Promise<EnterpriseTokenPayload> => {
  return await refreshEnterpriseTokenSingleflight(refreshToken);
};

export const fetchEnterpriseSession = async (
  overrideToken?: string
): Promise<EnterpriseSessionPayload> => {
  const headers = new AxiosHeaders();
  if (overrideToken) {
    headers.set("Authorization", `Bearer ${overrideToken}`);
  }
  const response = await enterpriseApi.get<EnterpriseSessionPayload>(
    "/auth/session",
    { headers }
  );
  return response.data;
};

export function getEnterpriseAuthRecoveryMessage(error: unknown): string | null {
  const reason = (error as any)?.reason;
  const status = (error as any)?.response?.status;
  if (reason === "missing_refresh_token") {
    return "Session entreprise incomplète. Reconnectez-vous pour continuer.";
  }
  if (reason === "auth_ready_timeout") {
    return "La session entreprise n'est pas prête. Reconnectez-vous.";
  }
  if (status === 401) {
    return "Session expirée. Reconnectez-vous pour reprendre.";
  }
  return null;
}
