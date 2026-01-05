import AsyncStorage from "@react-native-async-storage/async-storage";
import axios, {
  AxiosError,
  AxiosHeaders,
  InternalAxiosRequestConfig,
} from "axios";
import { Platform } from "react-native";
import Constants from "expo-constants";

const expoExtra = Constants.expoConfig?.extra || {};
const ENV_API_URL = process.env.EXPO_PUBLIC_API_URL;
const PROD_API_URL: string =
  ENV_API_URL || (expoExtra.publicApiUrl as string) || expoExtra.productionApiUrl || "";
const DEV_API_URL = (expoExtra.devApiUrl as string) || (expoExtra.publicApiUrl as string);

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
  return `${(PROD_API_URL || "").replace(/\/$/, "")}${API_PREFIX}`;
};

const baseURL = getBaseURL();

// Debug: log baseURL résolu (non sensible)
try {
  // eslint-disable-next-line no-console
  console.log("[ENT] baseURL:", baseURL, {
    PROD_API_URL,
    ENV_API_URL,
    PORT,
  });
} catch {}

export const ENTERPRISE_TOKEN_KEY = "enterprise.token";
export const ENTERPRISE_REFRESH_KEY = "enterprise.refresh";
export const ENTERPRISE_SESSION_KEY = "enterprise.session";

type AxiosConfig = InternalAxiosRequestConfig<any> & {
  __isRetryRequest?: boolean;
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

const clearEnterpriseStorage = async () => {
  await AsyncStorage.multiRemove([
    ENTERPRISE_TOKEN_KEY,
    ENTERPRISE_REFRESH_KEY,
    ENTERPRISE_SESSION_KEY,
  ]);
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

  await AsyncStorage.setItem(ENTERPRISE_TOKEN_KEY, session.token);
  await AsyncStorage.setItem(
    ENTERPRISE_SESSION_KEY,
    JSON.stringify(session)
  );

  if (session.refreshToken) {
    await AsyncStorage.setItem(
      ENTERPRISE_REFRESH_KEY,
      session.refreshToken
    );
  } else {
    await AsyncStorage.removeItem(ENTERPRISE_REFRESH_KEY);
  }
};

let tokenRefreshPromise:
  | Promise<string | null | undefined>
  | null = null;

const refreshAccessToken = async (): Promise<string | null | undefined> => {
  if (!tokenRefreshPromise) {
    tokenRefreshPromise = (async () => {
      const refreshToken = await AsyncStorage.getItem(ENTERPRISE_REFRESH_KEY);
      if (!refreshToken) {
        await clearEnterpriseStorage();
        return null;
      }

      try {
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
        return payload.token;
      } catch (error) {
        await clearEnterpriseStorage();
        throw error;
      } finally {
        tokenRefreshPromise = null;
      }
    })();
  }

  return tokenRefreshPromise;
};

enterpriseApi.interceptors.request.use(
  async (config) => {
    try {
      // #region agent log
      const isLoginRequest = config.url?.includes("/auth/login");
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'enterpriseAuth.ts:268',message:'interceptor request entry',data:{url:config.url,method:config.method,isLoginRequest,baseURL:config.baseURL},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
      const headers =
        config.headers instanceof AxiosHeaders
          ? config.headers
          : new AxiosHeaders(config.headers || {});

      // ✅ Ne pas ajouter de token uniquement pour les requêtes de login
      // Les autres endpoints /auth/ (comme /auth/me/driver-account, /auth/refresh, etc.) nécessitent un token
      if (!isLoginRequest) {
        // Si un token est déjà présent dans les headers (passé explicitement), l'utiliser
        // Sinon, essayer de le récupérer depuis AsyncStorage
        if (!headers.has("Authorization")) {
          const token = await AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY);
          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'enterpriseAuth.ts:278',message:'token check',data:{url:config.url,hasToken:!!token,isLoginRequest},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
          // #endregion
          if (token) {
            headers.set("Authorization", `Bearer ${token}`);
            // eslint-disable-next-line no-console
            console.log("[ENT] Token ajouté à la requête depuis AsyncStorage", {
              url: config.url,
              hasToken: !!token,
              tokenLength: token.length,
              tokenPreview: token.substring(0, 20) + "...",
            });
          } else {
            // eslint-disable-next-line no-console
            console.warn("[ENT] Aucun token disponible pour la requête", {
              url: config.url,
            });
          }
        } else {
          // eslint-disable-next-line no-console
          console.log("[ENT] Token déjà présent dans les headers", {
            url: config.url,
          });
        }
      }

      // ✅ Ne pas ajouter X-Company-ID/X-Session-ID uniquement pour les requêtes de login
      // Les autres endpoints /auth/ (comme /auth/me/driver-account, /auth/refresh, etc.) nécessitent une session
      if (!isLoginRequest) {
        const sessionRaw = await AsyncStorage.getItem(ENTERPRISE_SESSION_KEY);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'enterpriseAuth.ts:301',message:'session check',data:{url:config.url,hasSession:!!sessionRaw,isLoginRequest},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
        if (sessionRaw) {
          try {
            const session = JSON.parse(sessionRaw);
            if (session?.company?.id && !headers.has("X-Company-ID")) {
              headers.set("X-Company-ID", String(session.company.id));
              // eslint-disable-next-line no-console
              console.log("[ENT] X-Company-ID ajouté", {
                url: config.url,
                companyId: session.company.id,
              });
            }
            if (session?.sessionId && !headers.has("X-Session-ID")) {
              headers.set("X-Session-ID", session.sessionId);
              // eslint-disable-next-line no-console
              console.log("[ENT] X-Session-ID ajouté", {
                url: config.url,
                sessionId: session.sessionId,
              });
            }
          } catch (e) {
            // eslint-disable-next-line no-console
            console.warn("[ENT] Erreur parsing session pour headers", e);
            // ignore parsing issues, will be refreshed later
          }
        } else {
          // eslint-disable-next-line no-console
          console.warn("[ENT] Aucune session disponible pour ajouter X-Company-ID/X-Session-ID", {
            url: config.url,
          });
        }
      }

      // #region agent log
      const finalHeaders: Record<string, string> = {};
      headers.forEach((value: unknown, key: string) => {
        if (key.toLowerCase() === 'authorization') {
          finalHeaders[key] = value ? `${String(value).substring(0, 20)}...` : '';
        } else {
          finalHeaders[key] = String(value);
        }
      });
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'enterpriseAuth.ts:333',message:'interceptor request exit',data:{url:config.url,headers:finalHeaders,isLoginRequest},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
      config.headers = headers;
    } catch {
      // ignore errors for now
    }
    return config;
  },
  (error) => Promise.reject(error)
);

enterpriseApi.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const { response, config } = error;
    const originalConfig = config as AxiosConfig | undefined;

    if (
      response?.status === 401 &&
      originalConfig &&
      !originalConfig.__isRetryRequest
    ) {
      try {
        const newToken = await refreshAccessToken();
        if (!newToken) {
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
      } catch (refreshError) {
        return Promise.reject(refreshError);
      }
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
    console.log('[DEBUG] loginEnterprise entry:', JSON.stringify(logData, null, 2));
    fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(logData)}).catch((e)=>console.warn('[DEBUG] Log fetch failed:', e));
    // #endregion
    // Log léger pour debug (sans secrets)
    // eslint-disable-next-line no-console
    console.log("[ENT] login request", { hasEmail: Boolean(params.email), hasPassword: Boolean(params.password), baseURL, fullURL });

    const response = await enterpriseApi.post<EnterpriseLoginResponse>(
      "/auth/login",
      params
    );
    // #region agent log
    const successLogData = {location:'enterpriseAuth.ts:422',message:'loginEnterprise success',data:{status:response.status,hasToken:!!(response.data as any)?.token,hasMfa:!!(response.data as any)?.mfa_required},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'};
    console.log('[DEBUG]', JSON.stringify(successLogData));
    fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(successLogData)}).catch((e)=>console.warn('[DEBUG] Log fetch failed:', e));
    // #endregion
    // eslint-disable-next-line no-console
    console.log("[ENT] login response reçue", {
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
    console.error('[DEBUG]', JSON.stringify(errorLogData));
    fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(errorLogData)}).catch((e)=>console.warn('[DEBUG] Log fetch failed:', e));
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
      // eslint-disable-next-line no-console
      console.log("[ENT] login via fetch OK");
      return JSON.parse(text) as EnterpriseLoginResponse;
    } catch (fallbackError) {
      // eslint-disable-next-line no-console
      console.warn("[ENT] login fallback (fetch) échec:", fallbackError);
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

export const refreshEnterpriseToken = async (
  refreshToken: string
): Promise<EnterpriseTokenPayload> => {
  const response = await enterpriseApi.post<EnterpriseTokenPayload>(
    "/auth/refresh",
    { refresh_token: refreshToken }
  );
  return response.data;
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
