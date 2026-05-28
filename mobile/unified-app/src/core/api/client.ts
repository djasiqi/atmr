import axios, { AxiosError } from "axios";
import Constants from "expo-constants";
import * as SecureStore from "../storage/secureStoreCompat";
import { NativeModules, Platform } from "react-native";
import type { InternalAxiosRequestConfig } from "axios";
import {
  BootstrapResponse,
  bootstrapResponseSchema,
  SwitchContextResponse,
  switchContextResponseSchema,
} from "../contracts/auth";
import { buildMockBootstrap, buildMockSwitchContext } from "./mockData";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import { buildSessionDiagHeader } from "../observability/sessionJournal";
import { getRuntimeFlagsVersion, isFeatureEnabled } from "../featureFlags/registry";
import { getNetworkSnapshot } from "../network/networkState";
import { evaluateConnectivityPolicy } from "../network/connectivityPolicy";

const DEFAULT_PROD_API_BASE_URL = "https://api.lirie.ch/api/v1";

function extractHostFromUrl(value: string | null | undefined): string | null {
  if (!value) return null;
  const match = value.match(/https?:\/\/([^/:]+)/i);
  return match?.[1] ?? null;
}

/** Hôte API qu’on peut réaligner sur l’hôte Metro en dev (IP LAN, loopback, émulateur). Pas les domaines de prod. */
function isDevAlignableApiHost(hostname: string): boolean {
  const h = hostname.trim().toLowerCase();
  if (h === "localhost" || h === "127.0.0.1" || h === "[::1]" || h === "::1") return true;
  if (h === "10.0.2.2") return true;
  if (/^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(h)) return true;
  if (/^192\.168\.\d{1,3}\.\d{1,3}$/.test(h)) return true;
  const m = /^172\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/.exec(h);
  if (m) {
    const second = Number(m[1]);
    if (second >= 16 && second <= 31) return true;
  }
  return false;
}

function isLoopbackStyleHost(hostname: string): boolean {
  const h = hostname.trim().toLowerCase();
  return h === "localhost" || h === "127.0.0.1" || h === "[::1]" || h === "::1";
}

/** Cible dev non-loopback (LAN, 10.0.2.2, etc.) : ne pas la réécrire vers localhost Metro. */
function isNonLoopbackDevApiHost(hostname: string): boolean {
  return isDevAlignableApiHost(hostname) && !isLoopbackStyleHost(hostname);
}

function isUnsafeProductionApiUrl(value: string): boolean {
  if (!value.startsWith("https://")) return true;
  const host = extractHostFromUrl(value);
  return Boolean(host && isDevAlignableApiHost(host));
}

function deriveBundleHostForDev(): string | null {
  if (Platform?.OS === "web") {
    const webHost = globalThis?.location?.hostname;
    if (typeof webHost === "string" && webHost.length > 0) {
      return webHost;
    }
  }
  const scriptUrl: string | undefined = NativeModules?.SourceCode?.scriptURL;
  const hostFromScript = extractHostFromUrl(scriptUrl);
  if (hostFromScript) return hostFromScript;
  const hostUri = (Constants.expoConfig as { hostUri?: string } | null)?.hostUri;
  if (!hostUri) return null;
  return hostUri.split(":")[0] ?? null;
}

/** Repli legacy EXPO_PUBLIC_API_URL (operations-app / secrets EAS historiques). */
function resolveApiBaseUrlFromEnv(): string | undefined {
  const direct = process.env.EXPO_PUBLIC_API_BASE_URL?.trim();
  if (direct) return direct;
  const legacy = process.env.EXPO_PUBLIC_API_URL?.trim();
  if (!legacy) return undefined;
  const normalized = legacy.replace(/\/$/, "");
  if (normalized.endsWith("/api/v1")) return normalized;
  return `${normalized}/api/v1`;
}

function resolveBaseUrl(): string {
  const envUrl = resolveApiBaseUrlFromEnv();
  const configUrl = (Constants.expoConfig?.extra as { apiBaseUrl?: string } | undefined)?.apiBaseUrl;
  const chosen = envUrl ?? configUrl ?? DEFAULT_PROD_API_BASE_URL;

  if (!__DEV__ && isUnsafeProductionApiUrl(chosen)) {
    return DEFAULT_PROD_API_BASE_URL;
  }

  // Web dev : suivre l’hôte de la page (localhost/IP LAN) pour éviter qu’une ancienne
  // IP compilée dans le bundle continue d’être utilisée après changement de réseau.
  if (__DEV__ && Platform?.OS === "web") {
    const webHost = globalThis?.location?.hostname;
    const chosenHost = extractHostFromUrl(chosen);
    if (
      typeof webHost === "string" &&
      webHost.length > 0 &&
      chosenHost &&
      webHost !== chosenHost &&
      isDevAlignableApiHost(chosenHost)
    ) {
      return chosen.replace(chosenHost, webHost);
    }
  }

  // Dev : si l’IP LAN du PC change, réaligner une base **locale** sur l’hôte Metro.
  // Ne pas remplacer api.lirie.ch (ou autre prod) par localhost — sinon le mobile appelle https://localhost/...
  if (__DEV__) {
    const bundleHost = deriveBundleHostForDev();
    const chosenHost = extractHostFromUrl(chosen);
    if (bundleHost && chosenHost && bundleHost !== chosenHost && isDevAlignableApiHost(chosenHost)) {
      if (isLoopbackStyleHost(bundleHost) && isNonLoopbackDevApiHost(chosenHost)) {
        return chosen;
      }
      return chosen.replace(chosenHost, bundleHost);
    }
  }
  return chosen;
}

const baseURL = resolveBaseUrl();
const useMockBootstrap = process.env.EXPO_PUBLIC_USE_MOCK_BOOTSTRAP === "1";

// axios default export expose .create ; import nommé non utilisé ici.
// eslint-disable-next-line import/no-named-as-default-member
export const apiClient = axios.create({
  baseURL,
  timeout: 15000,
  withCredentials: true,
});

/** URL de base résolue (correction LAN IP Metro incluse en dev). Utilisée par les bridges Socket.IO. */
export function getResolvedApiBaseUrl(): string {
  return baseURL;
}

let csrfTokenCache: string | null = null;
let csrfFetchInFlight: Promise<string | null> | null = null;
let refreshTokenInFlight: Promise<string | null> | null = null;
let lastRefreshFailureAtMs = 0;
let lastRefreshFailureSignature: string | null = null;
let resumeAttemptId: string | null = null;
/** Toutes les requêtes (driver, company, …) reçoivent le contexte actif pour l’autorisation multi-rôles. */
let activeContextIdForApi: string | null = null;
const REFRESH_TOKEN_STORAGE_KEY = "auth_refresh_token";
const REFRESH_FAILURE_TELEMETRY_COOLDOWN_MS = 10000;
const POST_BOOTSTRAP_REFRESH_SKIP_MS = 5000;
let lastBootstrapAuthSuccessAtMs = 0;

/** Évite un refresh token concurrent juste après login/bootstrap. */
export function markBootstrapAuthFresh(): void {
  lastBootstrapAuthSuccessAtMs = Date.now();
}

function shouldSkipPostBootstrapRefresh(): boolean {
  return Date.now() - lastBootstrapAuthSuccessAtMs < POST_BOOTSTRAP_REFRESH_SKIP_MS;
}

function shouldEmitRefreshFailure(status: number | null, reason: string): boolean {
  const signature = `${status ?? "null"}|${reason}`;
  const now = Date.now();
  if (
    signature === lastRefreshFailureSignature &&
    now - lastRefreshFailureAtMs < REFRESH_FAILURE_TELEMETRY_COOLDOWN_MS
  ) {
    return false;
  }
  lastRefreshFailureSignature = signature;
  lastRefreshFailureAtMs = now;
  return true;
}

export function setActiveContextIdForApi(contextId: string | null) {
  activeContextIdForApi = contextId && contextId.trim().length > 0 ? contextId.trim() : null;
}

function buildTraceId(): string {
  return `mob_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

export function setResumeAttemptCorrelationId(value: string | null) {
  resumeAttemptId = value;
}

async function readRefreshToken(): Promise<string | null> {
  try {
    const value = await SecureStore.getItemAsync(REFRESH_TOKEN_STORAGE_KEY);
    if (typeof value === "string" && value.trim().length > 0) {
      return value;
    }
    return null;
  } catch {
    return null;
  }
}

export async function hasStoredRefreshToken(): Promise<boolean> {
  const token = await readRefreshToken();
  return Boolean(token);
}

async function writeRefreshToken(value: string | null): Promise<void> {
  try {
    if (value && value.trim().length > 0) {
      await SecureStore.setItemAsync(REFRESH_TOKEN_STORAGE_KEY, value);
      return;
    }
    await SecureStore.deleteItemAsync(REFRESH_TOKEN_STORAGE_KEY);
  } catch {
    // Best effort storage: keep runtime flow resilient.
  }
}

function isMutatingMethod(method: string | undefined): boolean {
  const m = String(method || "get").toUpperCase();
  return m === "POST" || m === "PUT" || m === "PATCH" || m === "DELETE";
}

function isDriverEndpoint(url: string): boolean {
  return url.startsWith("/driver/") || url.startsWith("driver/");
}

function resolveAdaptiveTimeoutMs(url: string, defaultTimeoutMs: number): number {
  if (!isFeatureEnabled("driver_http_adaptive_timeout_enabled") || !isDriverEndpoint(url)) {
    return defaultTimeoutMs;
  }
  const policy = evaluateConnectivityPolicy(getNetworkSnapshot());
  const normalMs = Number(process.env.EXPO_PUBLIC_DRIVER_HTTP_TIMEOUT_NORMAL_MS ?? "15000");
  const poorMs = Number(process.env.EXPO_PUBLIC_DRIVER_HTTP_TIMEOUT_POOR_MS ?? "28000");
  const offlineMs = Number(process.env.EXPO_PUBLIC_DRIVER_HTTP_TIMEOUT_OFFLINE_MS ?? "2500");
  const trackingOverrideMs = Number(
    process.env.EXPO_PUBLIC_DRIVER_HTTP_TIMEOUT_TRACKING_MS ?? String(poorMs)
  );
  if (url.includes("/driver/me/location")) {
    return Math.max(1000, trackingOverrideMs);
  }
  if (policy.mode === "offline") {
    return Math.max(1000, offlineMs);
  }
  if (policy.mode === "degraded") {
    return Math.max(1000, poorMs);
  }
  return Math.max(1000, normalMs);
}

function shouldFailFastOffline(url: string): boolean {
  if (!isFeatureEnabled("driver_http_adaptive_timeout_enabled") || !isDriverEndpoint(url)) {
    return false;
  }
  const policy = evaluateConnectivityPolicy(getNetworkSnapshot());
  if (policy.mode !== "offline") {
    return false;
  }
  // Location and status transitions can be queued/retried by runtime queues.
  const failFastCandidates = ["/driver/me/location", "/driver/me/bookings/", "/driver/me/bookings/since"];
  return failFastCandidates.some((candidate) => url.includes(candidate));
}

function hasCsrfHeader(config: InternalAxiosRequestConfig): boolean {
  const h = config.headers;
  if (!h) return false;
  return Boolean(h["X-CSRF-Token"] || h["X-Csrf-Token"]);
}

async function fetchCsrfToken(): Promise<string | null> {
  const { data } = await apiClient.get<{ csrf_token?: string; data?: { csrf_token?: string } }>(
    "/auth/csrf-token"
  );
  return data?.csrf_token ?? data?.data?.csrf_token ?? null;
}

async function ensureCsrfToken(): Promise<string | null> {
  if (csrfTokenCache) return csrfTokenCache;
  if (!csrfFetchInFlight) {
    csrfFetchInFlight = fetchCsrfToken()
      .then((token) => {
        csrfTokenCache = token;
        return token;
      })
      .finally(() => {
        csrfFetchInFlight = null;
      });
  }
  return csrfFetchInFlight;
}

async function refreshAuthToken(): Promise<string | null> {
  const refreshToken = await readRefreshToken();
  if (!refreshToken) {
    return null;
  }
  const payload = { refresh_token: refreshToken };

  let endpointUsed: "refresh-token" | "refresh" = "refresh-token";
  let data: { access_token?: string; token?: string; refresh_token?: string } | null = null;
  try {
    const response = await apiClient.post<{
      access_token?: string;
      token?: string;
      refresh_token?: string;
    }>("/auth/refresh-token", payload);
    data = response.data;
  } catch (error) {
    const err = error as AxiosError;
    const shouldFallback = err.response?.status === 404 || err.response?.status === 405;
    if (!shouldFallback) {
      throw error;
    }
    endpointUsed = "refresh";
    const fallbackResponse = await apiClient.post<{
      access_token?: string;
      token?: string;
      refresh_token?: string;
    }>("/auth/refresh", payload);
    data = fallbackResponse.data;
  }

  emitDriverTelemetry("auth.refresh.endpoint_used", {
    source: "core.api.client",
    endpoint: endpointUsed,
  });
  const token = extractToken(data);
  const nextRefreshToken = extractRefreshToken(data);
  await writeRefreshToken(nextRefreshToken);
  if (token) {
    setAuthToken(token);
  }
  return token;
}

async function ensureRefreshToken(): Promise<string | null> {
  if (!refreshTokenInFlight) {
    refreshTokenInFlight = refreshAuthToken()
      .catch((error) => {
        const err = error as AxiosError;
        const status = err.response?.status ?? null;
        const reason = err.message;
        if (shouldEmitRefreshFailure(status, reason)) {
          emitDriverTelemetry("auth.refresh.failure", {
            source: "core.api.client",
            reason,
            status,
          });
        }
        throw error;
      })
      .finally(() => {
        refreshTokenInFlight = null;
      });
  }
  return refreshTokenInFlight;
}

export async function refreshAuthTokenNow(): Promise<boolean> {
  try {
    const token = await ensureRefreshToken();
    return Boolean(token);
  } catch {
    return false;
  }
}

apiClient.interceptors.request.use(async (config) => {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { recordHttpRequest } = require("../observability/perfInstrumentation") as {
      recordHttpRequest: (url: string) => void;
    };
    recordHttpRequest(String(config.url ?? ""));
  } catch {
    // optional perf instrumentation
  }
  const requestUrl = String(config.url ?? "");
  const effectiveTimeoutMs = resolveAdaptiveTimeoutMs(requestUrl, Number(config.timeout ?? 15000));
  config.timeout = effectiveTimeoutMs;
  config.headers["X-Client-Platform"] = Platform.OS;
  config.headers["X-Session-Diag"] = buildSessionDiagHeader();
  config.headers["X-Trace-Id"] = buildTraceId();
  if (activeContextIdForApi && !config.headers["X-Active-Context-Id"]) {
    config.headers["X-Active-Context-Id"] = activeContextIdForApi;
  }
  if (resumeAttemptId) {
    config.headers["X-Resume-Attempt-Id"] = resumeAttemptId;
  }
  const flagsVersion = getRuntimeFlagsVersion();
  if (flagsVersion) {
    config.headers["X-Feature-Flags-Version"] = flagsVersion;
  }
  if (isFeatureEnabled("driver_http_adaptive_timeout_enabled") && isDriverEndpoint(requestUrl)) {
    const policy = evaluateConnectivityPolicy(getNetworkSnapshot());
    emitDriverTelemetry("driver.network.profile", {
      source: "core.api.client",
      network_profile_active: policy.mode === "degraded" ? "poor" : policy.mode,
      recommended_sync_interval_ms: policy.recommendedSyncIntervalMs,
    });
    emitDriverTelemetry("driver.http.timeout", {
      source: "core.api.client",
      endpoint: requestUrl,
      timeout_ms: effectiveTimeoutMs,
      network_profile_active: policy.mode === "degraded" ? "poor" : policy.mode,
    });
  }
  if (shouldFailFastOffline(requestUrl) && !config.headers["X-Allow-Offline-Attempt"]) {
    throw new AxiosError(
      `Offline fail-fast for ${requestUrl}`,
      "ERR_DRIVER_OFFLINE_FAIL_FAST",
      config
    );
  }
  if (!isMutatingMethod(config.method) || hasCsrfHeader(config)) {
    return config;
  }
  const token = await ensureCsrfToken();
  if (token) {
    config.headers["X-CSRF-Token"] = token;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<{ error?: string; error_message?: string }>) => {
    const originalForAuthRetry = error.config as
      | (InternalAxiosRequestConfig & { _authRetried?: boolean })
      | undefined;
    const requestUrl = String(originalForAuthRetry?.url ?? "");
    const isAuthEndpoint =
      requestUrl.includes("/auth/login") ||
      requestUrl.includes("/auth/refresh") ||
      requestUrl.includes("/auth/refresh-token") ||
      requestUrl.includes("/auth/logout");
    if (
      error.response?.status === 401 &&
      originalForAuthRetry &&
      !originalForAuthRetry._authRetried &&
      !isAuthEndpoint
    ) {
      try {
        const refreshedToken = await ensureRefreshToken();
        if (!refreshedToken) {
          return Promise.reject(error);
        }
        originalForAuthRetry._authRetried = true;
        originalForAuthRetry.headers.Authorization = `Bearer ${refreshedToken}`;
        return await apiClient.request(originalForAuthRetry);
      } catch {
        return Promise.reject(error);
      }
    }

    const responseStatus = error.response?.status ?? null;
    const backendError = (
      error.response?.data?.error_message ??
      error.response?.data?.error ??
      ""
    ).toLowerCase();
    const isCsrfError =
      responseStatus === 403 &&
      (backendError.includes("csrf") ||
        backendError.includes("token csrf") ||
        backendError.includes("token invalide"));
    const original = error.config as (InternalAxiosRequestConfig & { _csrfRetried?: boolean }) | undefined;
    const isCsrfEndpoint = String(original?.url ?? "").includes("/auth/csrf-token");
    if (!isCsrfError || !original || original._csrfRetried || isCsrfEndpoint) {
      return Promise.reject(error);
    }
    try {
      csrfTokenCache = null;
      const freshToken = await ensureCsrfToken();
      if (freshToken) {
        original.headers["X-CSRF-Token"] = freshToken;
      }
      original._csrfRetried = true;
      return await apiClient.request(original);
    } catch {
      return Promise.reject(error);
    }
  }
);

export type ApiCallError = {
  status: number | null;
  code: string;
  message: string;
  reason?: string;
  outcome_class?: "success" | "retryable_error" | "terminal_error";
  retryable?: boolean;
  details?: Record<string, unknown>;
};

/** Aucun objet `response` Axios : la requête n’a pas reçu de réponse HTTP (coupure, TLS, DNS, timeout, etc.). */
function buildNoHttpResponseHint(requestUrl: string, error: AxiosError): string {
  const axiosCode = typeof error.code === "string" ? error.code : "";
  const codePart = axiosCode ? ` code=${axiosCode}` : "";
  const msg = (error.message ?? "").toLowerCase();
  const host = extractHostFromUrl(requestUrl);
  const isLanStyle = host != null && isDevAlignableApiHost(host);
  const timeoutish = axiosCode === "ECONNABORTED" || msg.includes("timeout");

  if (timeoutish) {
    return ` | URL=${requestUrl}${codePart} | Aucune réponse (délai ou coupure avant la fin de la requête).`;
  }
  const tail = isLanStyle
    ? "En dev sur IP locale : même Wi‑Fi que le PC, bon port, HTTP vs HTTPS."
    : "Pas de trame HTTP reçue : vérifie Internet, VPN / pare-feu / proxy, DNS, et la date-heure de l’appareil (souvent lié aux échecs TLS).";
  return ` | URL=${requestUrl}${codePart} | ${tail}`;
}

function toApiError(error: unknown): ApiCallError {
  const e = error as AxiosError<{
    message?: string;
    error_message?: string;
    error?: string;
    error_code?: string;
    reason?: string;
    outcome_class?: "success" | "retryable_error" | "terminal_error";
    retryable?: boolean;
    trace_id?: string;
    activation_session_id?: string;
    masked_email?: string;
    masked_phone?: string;
    details?: Record<string, unknown>;
  }>;
  const endpoint = e.config?.url ?? "";
  const requestUrl = `${e.config?.baseURL ?? ""}${endpoint}`;
  const noHttpResponse = e.response == null && e.code !== "ERR_CANCELED";
  const transportHint = noHttpResponse ? buildNoHttpResponseHint(requestUrl, e) : "";
  return {
    status: e.response?.status ?? null,
    code: e.response?.data?.error_code ?? "UNKNOWN_ERROR",
    message:
      (e.response?.data?.error_message ??
        e.response?.data?.error ??
        e.response?.data?.message ??
        e.message) +
      transportHint,
    reason: e.response?.data?.reason,
    outcome_class: e.response?.data?.outcome_class,
    retryable: e.response?.data?.retryable,
    details: {
      ...(e.response?.data?.details ?? {}),
      trace_id: e.response?.data?.trace_id,
      activation_session_id: e.response?.data?.activation_session_id,
      masked_email: e.response?.data?.masked_email,
      masked_phone: e.response?.data?.masked_phone,
    },
  };
}

function extractToken(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;
  const obj = payload as Record<string, unknown>;
  const tokenCandidate = obj.access_token ?? obj.token;
  if (typeof tokenCandidate === "string" && tokenCandidate.trim().length > 0) {
    return tokenCandidate;
  }
  return null;
}

function extractRefreshToken(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;
  const obj = payload as Record<string, unknown>;
  const tokenCandidate = obj.refresh_token;
  if (typeof tokenCandidate === "string" && tokenCandidate.trim().length > 0) {
    return tokenCandidate;
  }
  return null;
}

export function setAuthToken(token: string | null) {
  if (token) {
    apiClient.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    delete apiClient.defaults.headers.common.Authorization;
    csrfTokenCache = null;
    void writeRefreshToken(null);
  }
}

export function hasAuthToken(): boolean {
  return getAuthAccessToken() != null;
}

/** Jeton d’accès actuel (Bearer), pour handshakes hors Axios (ex. Socket.IO). */
export function getAuthAccessToken(): string | null {
  const raw = apiClient.defaults.headers.common.Authorization;
  const s =
    typeof raw === "string"
      ? raw
      : Array.isArray(raw)
        ? raw.find((x): x is string => typeof x === "string")
        : undefined;
  if (!s?.trim()) return null;
  const m = s.trim().match(/^Bearer\s+(\S.+)$/i);
  const t = m?.[1]?.trim();
  return t && t.length > 0 ? t : null;
}

export async function fetchBootstrap(activeContextId?: string | null): Promise<BootstrapResponse> {
  if (useMockBootstrap) {
    return bootstrapResponseSchema.parse(buildMockBootstrap());
  }
  try {
    const headers = activeContextId ? { "X-Active-Context-Id": activeContextId } : undefined;
    const { data } = await apiClient.get("/auth/bootstrap", { headers });
    const parsed = bootstrapResponseSchema.parse(data);
    markBootstrapAuthFresh();
    return parsed;
  } catch (error) {
    const err = error as AxiosError;
    emitDriverTelemetry("auth.bootstrap.failure", {
      source: "core.api.client",
      context_id: activeContextId ?? null,
      reason: err.message,
      status: err.response?.status ?? null,
      axios_code: err.code ?? null,
    });
    throw toApiError(error);
  }
}

export async function login(email: string, password: string): Promise<void> {
  if (useMockBootstrap) return;
  try {
    const { data } = await apiClient.post("/auth/login", {
      email: email.trim(),
      password,
    });
    const token = extractToken(data);
    const refreshToken = extractRefreshToken(data);
    if (token) {
      setAuthToken(token);
    }
    await writeRefreshToken(refreshToken);
    markBootstrapAuthFresh();
  } catch (error) {
    throw toApiError(error);
  }
}

export async function switchContext(targetContextId: string): Promise<SwitchContextResponse> {
  if (useMockBootstrap) {
    return switchContextResponseSchema.parse(buildMockSwitchContext(targetContextId));
  }
  try {
    const { data } = await apiClient.post<Record<string, unknown>>(
      "/auth/switch-context",
      { target_context_id: targetContextId },
      { timeout: 12_000, headers: { "X-Allow-Offline-Attempt": "1" } }
    );
    // Jetons explicites si le serveur en émet (futur) ; sinon refresh pour réhydrater le header (web/HMR, perte in-memory).
    const inlineAccess = extractToken(data);
    if (inlineAccess) {
      setAuthToken(inlineAccess);
    }
    const inlineRefresh = extractRefreshToken(data);
    if (inlineRefresh) {
      await writeRefreshToken(inlineRefresh);
    }
    const parsed = switchContextResponseSchema.parse(data);
    if (!shouldSkipPostBootstrapRefresh()) {
      void refreshAuthTokenNow().catch(() => false);
    }
    return parsed;
  } catch (error) {
    throw toApiError(error);
  }
}

export async function logoutSession(): Promise<void> {
  if (useMockBootstrap) return;
  try {
    await apiClient.post("/auth/logout");
  } catch (error) {
    // Logout backend best-effort: on purge toujours localement côté session provider.
    throw toApiError(error);
  } finally {
    setAuthToken(null);
  }
}

export type ServiceAreaCheckStatus = "available" | "conditional" | "unavailable";

export type ServiceAreaCheckResponse = {
  status: ServiceAreaCheckStatus;
  reason_code: string;
  message: string;
  next_step: "continue" | "contact_support" | "try_later";
};

export async function checkServiceArea(payload: {
  departure: string;
  destination: string;
  date: string;
  transport_type: string;
}): Promise<ServiceAreaCheckResponse> {
  try {
    const { data } = await apiClient.post<ServiceAreaCheckResponse>(
      "/auth/public/service-area/check",
      payload
    );
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export type PublicPreRequestDraftPayload = {
  draft_id: string;
  departure: string;
  destination: string;
  date: string;
  pickup_time?: string | null;
  trip_type?: "one_way" | "round_trip" | null;
  passengers?: number | null;
  transport_type: string;
  special_requirements?: string | null;
  contact_first_name?: string | null;
  contact_last_name?: string | null;
  contact_email?: string | null;
  contact_phone?: string | null;
  service_area_status?: ServiceAreaCheckStatus | null;
};

export type PublicPreRequestDraftResponse = {
  draft_id: string;
  status: "stored" | "updated";
  server_timestamp: string;
};

export async function upsertPublicPreRequestDraft(
  payload: PublicPreRequestDraftPayload
): Promise<PublicPreRequestDraftResponse> {
  try {
    const { data } = await apiClient.post<PublicPreRequestDraftResponse>(
      "/auth/public/pre-request/draft",
      payload
    );
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export async function fetchPublicPreRequestDraft(
  draftId: string
): Promise<PublicPreRequestDraftPayload | null> {
  try {
    const { data } = await apiClient.get<{
      draft?: PublicPreRequestDraftPayload | null;
    }>(`/auth/public/pre-request/draft/${encodeURIComponent(draftId)}`);
    return data?.draft ?? null;
  } catch (error) {
    const apiError = toApiError(error);
    if (apiError.status === 404) return null;
    throw apiError;
  }
}

export async function consumePublicPreRequestDraft(draftId: string): Promise<void> {
  try {
    await apiClient.post("/auth/public/pre-request/consume", { draft_id: draftId });
  } catch (error) {
    throw toApiError(error);
  }
}

export type PublicBookingStatusResponse = {
  status: "confirmed" | "pending" | "in_progress" | "completed" | "cancelled" | "unknown";
  label: string;
  updated_at: string | null;
  booking_reference: string;
};

export async function fetchPublicBookingStatus(
  token: string
): Promise<PublicBookingStatusResponse> {
  try {
    const { data } = await apiClient.get<PublicBookingStatusResponse>(
      `/auth/public/booking-status?token=${encodeURIComponent(token)}`
    );
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export type GuestBookingPreviewResponse = {
  pricing: {
    amount: number;
    currency: string;
    /** Distance routière (m), comme le preview client authentifié. */
    distance_meters: number;
    duration_seconds: number;
    pricing_profile_id?: number | null;
    pricing_profile_version_id?: number | null;
    pricing_status: string;
    breakdown?: Record<string, unknown> | null;
  };
  workflow: {
    guest_checkout_enabled: boolean;
    payment_required: boolean;
  };
};

export async function previewGuestBooking(payload: {
  departure: string;
  destination: string;
  date: string;
  pickup_time: string;
  trip_type?: "one_way" | "round_trip";
}): Promise<GuestBookingPreviewResponse> {
  try {
    const { data } = await apiClient.post<GuestBookingPreviewResponse>(
      "/auth/public/guest-booking/preview",
      payload
    );
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export type GuestBookingCreateResponse = {
  guest_booking_id: string;
  status: string;
  status_token: string;
  message?: string;
};

export async function createGuestBooking(payload: {
  departure: string;
  destination: string;
  date: string;
  pickup_time: string;
  trip_type?: "one_way" | "round_trip";
  passengers?: number;
  transport_type?: string;
  first_name?: string | null;
  last_name?: string | null;
  email?: string | null;
  phone?: string | null;
  notes?: string | null;
  preview_amount?: number;
}): Promise<GuestBookingCreateResponse> {
  try {
    const { data } = await apiClient.post<GuestBookingCreateResponse>(
      "/auth/public/guest-booking/create",
      payload
    );
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export type GuestSaferpayInitializeResponse = {
  guest_booking_id: string;
  redirect_url: string;
  order_id: string;
  payment_provider: string;
  payment_status: string;
  payment_amount: number;
  currency: string;
};

export async function initializeGuestSaferpay(payload: {
  guest_booking_id: string;
  status_token: string;
  return_url?: string;
}): Promise<GuestSaferpayInitializeResponse> {
  try {
    const { data } = await apiClient.post<GuestSaferpayInitializeResponse>(
      "/auth/public/guest-booking/saferpay/initialize",
      payload
    );
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export type GuestSaferpayAssertResponse = {
  status: string;
  booking_id?: number;
  public_status_token?: string;
  guest_booking_id?: string;
  payment_id?: number;
  payment_provider?: string;
  payment_status?: string;
  pending_verification?: boolean;
  detail?: string;
};

export async function assertGuestSaferpay(payload: {
  guest_booking_id: string;
  status_token: string;
}): Promise<GuestSaferpayAssertResponse> {
  try {
    const { data } = await apiClient.post<GuestSaferpayAssertResponse>(
      "/auth/public/guest-booking/saferpay/assert",
      payload
    );
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export async function fetchGuestBookingStatus(token: string): Promise<{
  guest_booking_id: string;
  status: string;
  booking_id?: number;
  public_status_token?: string;
  departure?: string;
  destination?: string;
  date?: string;
  pickup_time?: string;
  amount?: number;
  currency?: string;
  updated_at?: string;
  linked_to_account?: boolean;
}> {
  try {
    const { data } = await apiClient.get(`/auth/public/guest-booking/status?token=${encodeURIComponent(token)}`);
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export async function linkGuestBookingToAccount(statusToken: string): Promise<{
  status: "linked";
  guest_booking_id: string;
  linked_user_public_id: string;
}> {
  try {
    const { data } = await apiClient.post("/auth/public/guest-booking/link", {
      status_token: statusToken,
    });
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export async function requestPasswordlessOtp(payload: {
  channel: "email" | "phone";
  identifier: string;
}): Promise<{
  otp_session_id: string;
  channel: "email" | "phone";
  masked_identifier: string;
  expires_in_seconds: number;
  debug_code?: string;
}> {
  try {
    const { data } = await apiClient.post("/auth/passwordless/otp/request", payload);
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export async function verifyPasswordlessOtp(payload: {
  otp_session_id: string;
  code: string;
}): Promise<{ access_token: string; refresh_token: string; token_type: string; auth_mode: string }> {
  try {
    const { data } = await apiClient.post("/auth/passwordless/otp/verify", payload);
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}
