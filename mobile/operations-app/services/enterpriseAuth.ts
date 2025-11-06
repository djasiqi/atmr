import AsyncStorage from "@react-native-async-storage/async-storage";
import axios, { AxiosHeaders } from "axios";
import Constants from "expo-constants";

const expoExtra = Constants.expoConfig?.extra || {};
const PROD_API_URL: string = expoExtra.productionApiUrl || "";

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

const PORT = expoExtra.backendPort || "5000";
const API_PREFIX = "/api/v1/company_mobile";

const baseURL = __DEV__
  ? `http://${getDevHost()}:${PORT}${API_PREFIX}`
  : `${(PROD_API_URL || "").replace(/\/$/, "")}${API_PREFIX}`;

export const ENTERPRISE_TOKEN_KEY = "enterprise.token";
export const ENTERPRISE_REFRESH_KEY = "enterprise.refresh";
export const ENTERPRISE_SESSION_KEY = "enterprise.session";

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
  timeout: 10000,
  headers: { "Content-Type": "application/json" },
});

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

export const loginEnterprise = async (
  params: EnterpriseLoginParams
): Promise<EnterpriseLoginResponse> => {
  const response = await enterpriseApi.post<EnterpriseLoginResponse>(
    "/auth/login",
    params
  );
  return response.data;
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
