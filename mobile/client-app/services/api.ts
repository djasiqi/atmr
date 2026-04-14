import axios, { AxiosError, InternalAxiosRequestConfig, isAxiosError } from 'axios';
import Constants from 'expo-constants';
import { Platform } from 'react-native';

import {
  clearStoredTokens,
  getStoredAccessToken,
  getStoredRefreshToken,
  setStoredTokens,
} from '@/services/tokenStorage';

const expoExtra = Constants.expoConfig?.extra ?? {};
const ENV_API_URL = process.env.EXPO_PUBLIC_API_URL;
const PROD_API_URL = ENV_API_URL;
const DEV_API_URL = (expoExtra as { devApiUrl?: string; publicApiUrl?: string }).devApiUrl
  ?? (expoExtra as { publicApiUrl?: string }).publicApiUrl;
const RAW_APP_VARIANT = String(
  (expoExtra as { APP_VARIANT?: string }).APP_VARIANT ?? process.env.APP_VARIANT ?? 'prod',
);
const APP_VARIANT = RAW_APP_VARIANT;

const getDevHost = (): string => {
  if (Platform.OS === 'web') {
    return 'localhost';
  }
  const legacyHost = (Constants as { manifest?: { debuggerHost?: string } }).manifest?.debuggerHost?.split(':')[0];
  const newHost = (Constants as { expoConfig?: { hostUri?: string } }).expoConfig?.hostUri?.split(':')[0];
  const detectedHost = newHost || legacyHost;
  if (!detectedHost || detectedHost === 'localhost' || detectedHost === '127.0.0.1') {
    return '127.0.0.1';
  }
  return detectedHost;
};

const ENV_PORT = process.env.EXPO_PUBLIC_BACKEND_PORT;
const PORT = ENV_PORT ?? (expoExtra as { backendPort?: string }).backendPort ?? '5000';
const PROD_API_FALLBACK = 'https://api.lirie.ch';

const getDevBaseURL = () => {
  if (Platform.OS === 'web') {
    return `http://127.0.0.1:${PORT}`;
  }
  return `http://${getDevHost()}:${PORT}`;
};

const validateProdApiUrl = (value: unknown): string | null => {
  if (typeof value !== 'string' || !value.trim()) {
    return null;
  }
  const normalized = value.trim().replace(/\/$/, '');
  if (!normalized.startsWith('https://')) {
    return null;
  }
  if (normalized.includes('localhost') || normalized.includes('127.0.0.1')) {
    return null;
  }
  return normalized;
};

const isDevelopment = (): boolean => {
  if (__DEV__) {
    return true;
  }
  if (APP_VARIANT === 'prod') {
    return false;
  }
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    const hostname = window.location?.hostname;
    if (
      hostname === 'localhost'
      || hostname === '127.0.0.1'
      || hostname?.startsWith('192.168.')
      || hostname?.startsWith('10.')
      || hostname?.startsWith('172.16.')
    ) {
      return true;
    }
  }
  if (DEV_API_URL && DEV_API_URL !== PROD_API_URL && !DEV_API_URL.includes('api.lirie.ch')) {
    return true;
  }
  return false;
};

const getBaseURL = (): string => {
  const isDev = isDevelopment();
  if (isDev) {
    if (process.env.EXPO_PUBLIC_USE_PROD_IN_DEV === '1') {
      return validateProdApiUrl(PROD_API_URL) ?? PROD_API_FALLBACK;
    }
    if (Platform.OS === 'web') {
      return getDevBaseURL();
    }
    if (DEV_API_URL && DEV_API_URL !== PROD_API_URL && !DEV_API_URL.includes('api.lirie.ch')) {
      return DEV_API_URL;
    }
    return getDevBaseURL();
  }
  return validateProdApiUrl(PROD_API_URL) ?? PROD_API_FALLBACK;
};

export const apiBaseRoot = getBaseURL().replace(/\/$/, '');
export const apiBaseURL = `${apiBaseRoot}/api/v1`;

let memoryAccessToken: string | null = null;

export function setMemoryAccessToken(token: string | null): void {
  memoryAccessToken = token;
}

export function getMemoryAccessToken(): string | null {
  return memoryAccessToken;
}

export async function hydrateAccessTokenFromStorage(): Promise<void> {
  memoryAccessToken = await getStoredAccessToken();
}

let refreshInFlight: Promise<boolean> | null = null;

async function performTokenRefresh(): Promise<boolean> {
  const refreshToken = await getStoredRefreshToken();
  if (!refreshToken) {
    return false;
  }
  try {
    const res = await axios.post<{
      access_token: string;
      refresh_token?: string;
    }>(
      `${apiBaseURL}/auth/refresh-token`,
      { refresh_token: refreshToken },
      {
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'Expo',
        },
        timeout: 30000,
      },
    );
    const { access_token, refresh_token } = res.data;
    memoryAccessToken = access_token;
    await setStoredTokens(access_token, refresh_token ?? refreshToken);
    return true;
  } catch {
    memoryAccessToken = null;
    await clearStoredTokens();
    return false;
  }
}

function ensureRefresh(): Promise<boolean> {
  if (!refreshInFlight) {
    refreshInFlight = performTokenRefresh().finally(() => {
      refreshInFlight = null;
    });
  }
  return refreshInFlight;
}

export const api = axios.create({
  baseURL: apiBaseURL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'Expo',
  },
});

api.interceptors.request.use((config) => {
  const headers = config.headers ?? {};
  if (memoryAccessToken && !headers.Authorization) {
    headers.Authorization = `Bearer ${memoryAccessToken}`;
  }
  config.headers = headers;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (error: AxiosError) => {
    const original = error.config as InternalAxiosRequestConfig & { _retry?: boolean };
    if (!original || original._retry) {
      return Promise.reject(error);
    }
    if (error.response?.status !== 401) {
      return Promise.reject(error);
    }
    const url = String(original.url ?? '');
    if (url.includes('/auth/refresh-token') || url.includes('/auth/login')) {
      return Promise.reject(error);
    }
    original._retry = true;
    const ok = await ensureRefresh();
    if (!ok) {
      return Promise.reject(error);
    }
    original.headers = original.headers ?? {};
    original.headers.Authorization = `Bearer ${memoryAccessToken}`;
    return api(original);
  },
);

export function getApiErrorMessage(err: unknown): string {
  if (isAxiosError(err)) {
    const data = err.response?.data as { error?: string; message?: string } | undefined;
    return data?.message ?? data?.error ?? err.message ?? 'Erreur réseau';
  }
  if (err instanceof Error) {
    return err.message;
  }
  return 'Erreur inconnue';
}
