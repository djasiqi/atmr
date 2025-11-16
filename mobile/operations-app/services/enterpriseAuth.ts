import AsyncStorage from "@react-native-async-storage/async-storage";
import axios, {
  AxiosError,
  AxiosHeaders,
  InternalAxiosRequestConfig,
} from "axios";
import Constants from "expo-constants";

const expoExtra = Constants.expoConfig?.extra || {};
const ENV_API_URL = process.env.EXPO_PUBLIC_API_URL;
const PROD_API_URL: string =
  ENV_API_URL || (expoExtra.publicApiUrl as string) || expoExtra.productionApiUrl || "";

const getDevHost = (): string => {
  const legacyHost = (Constants as any)?.manifest?.debuggerHost?.split(":")[0];
  const newHost = (Constants as any)?.expoConfig?.hostUri?.split(":")[0];
  const detectedHost = newHost || legacyHost;
  if (
    !detectedHost ||
    detectedHost === "localhost" ||
    detectedHost === "127.0.0.1"
  ) {
    return "172.20.10.2";
  }
  return detectedHost;
};

const ENV_PORT = process.env.EXPO_PUBLIC_BACKEND_PORT;
const PORT = ENV_PORT || expoExtra.backendPort || "5000";
const API_PREFIX = "/api/v1/company_mobile";

// Utiliser l'URL distante si EXPO_PUBLIC_API_URL est défini, même en dev
const USE_REMOTE = Boolean(PROD_API_URL && /^https?:\/\//.test(PROD_API_URL));
const baseURL = USE_REMOTE
  ? `${(PROD_API_URL || "").replace(/\/$/, "")}${API_PREFIX}`
  : `http://${getDevHost()}:${PORT}${API_PREFIX}`;

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
      const headers =
        config.headers instanceof AxiosHeaders
          ? config.headers
          : new AxiosHeaders(config.headers || {});

      const token = await AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY);
      if (token && !headers.has("Authorization")) {
        headers.set("Authorization", `Bearer ${token}`);
      }

      const sessionRaw = await AsyncStorage.getItem(ENTERPRISE_SESSION_KEY);
      if (sessionRaw) {
        try {
          const session = JSON.parse(sessionRaw);
          if (session?.company?.id && !headers.has("X-Company-ID")) {
            headers.set("X-Company-ID", String(session.company.id));
          }
          if (session?.sessionId && !headers.has("X-Session-ID")) {
            headers.set("X-Session-ID", session.sessionId);
          }
        } catch {
          // ignore parsing issues, will be refreshed later
        }
      }

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
    // Log léger pour debug (sans secrets)
    // eslint-disable-next-line no-console
    console.log("[ENT] login request", { hasEmail: Boolean(params.email), hasPassword: Boolean(params.password) });

    const response = await enterpriseApi.post<EnterpriseLoginResponse>(
      "/auth/login",
      params
    );
    return response.data;
  } catch (err: unknown) {
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
