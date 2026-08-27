import axios, { AxiosError, isAxiosError } from "axios";
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
import { appendSessionJournalEvent, buildSessionDiagHeader } from "../observability/sessionJournal";
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

/** Dev : réaligne une URL locale sur l'hôte Metro quand l'IP LAN du PC change. */
export function alignDevLocalUrlWithBundleHost(chosen: string): string {
  // Web dev : suivre l'hôte de la page (localhost/IP LAN) pour éviter qu'une ancienne
  // IP compilée dans le bundle continue d'être utilisée après changement de réseau.
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

  return alignDevLocalUrlWithBundleHost(chosen);
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
/** Dernier error_code renvoyé par un refresh/refresh-token échoué (coordinateur de récupération, PR C). */
let lastRefreshErrorCode: string | null = null;

/**
 * P1-C2 : porte terminale refresh. Un refresh token rejeté 401/403 par le
 * backend (révoqué/invalide/replay) ne doit JAMAIS être rejoué sur le réseau.
 * La porte est levée uniquement quand le token stocké change (login/rotation).
 */
type RefreshTerminalState = {
  code: string;
  status: number | null;
  tokenFingerprint: string;
  atMs: number;
};
let refreshTerminalState: RefreshTerminalState | null = null;

/** Empreinte non réversible (djb2 + longueur) — jamais le token en clair. */
function fingerprintRefreshToken(token: string): string {
  let h = 5381;
  for (let i = 0; i < token.length; i += 1) {
    h = ((h << 5) + h + token.charCodeAt(i)) | 0;
  }
  return `${token.length}:${(h >>> 0).toString(36)}`;
}

/**
 * 403 n'est PAS terminal par défaut (ex. incident CSRF ponctuel sur un token
 * valide). Seuls ces error_code explicites (contrat backend futur) terminalisent
 * un 403. Le storm observé passe par le 401 générique de _validate_refresh_token.
 */
const TERMINAL_REFRESH_403_ERROR_CODES = new Set([
  "refresh_token_revoked",
  "refresh_token_invalid",
  "refresh_token_expired",
]);

function markRefreshTerminalIfNeeded(err: AxiosError, refreshToken: string): void {
  const status = err.response?.status ?? null;
  const data = err.response?.data as
    | { error_code?: string; error?: string }
    | undefined;
  const errorCode =
    typeof data?.error_code === "string" && data.error_code ? data.error_code : null;
  const isTerminal =
    status === 401 ||
    (status === 403 &&
      errorCode !== null &&
      TERMINAL_REFRESH_403_ERROR_CODES.has(errorCode));
  if (!isTerminal) {
    return;
  }
  const code =
    errorCode ||
    (typeof data?.error === "string" && data.error) ||
    "refresh_rejected";
  refreshTerminalState = {
    code,
    status,
    tokenFingerprint: fingerprintRefreshToken(refreshToken),
    atMs: Date.now(),
  };
  emitDriverTelemetry("auth.refresh.terminal", {
    source: "core.api.client",
    status,
    error_code: code,
  });
}

export function getLastRefreshErrorCode(): string | null {
  return lastRefreshErrorCode;
}
let resumeAttemptId: string | null = null;
/** Toutes les requêtes (driver, company, …) reçoivent le contexte actif pour l’autorisation multi-rôles. */
let activeContextIdForApi: string | null = null;
const REFRESH_TOKEN_STORAGE_KEY = "auth_refresh_token";
const REFRESH_TOKEN_WRITE_RETRIES = 3;
const REFRESH_TOKEN_WRITE_BACKOFF_MS = [100, 300] as const;
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
    const {
      readRefreshToken: readStrict,
    } = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
    const result = await readStrict();
    if (result.status === "found") return result.value;
    if (result.status === "temporarily_unavailable") {
      emitDriverTelemetry("auth.refresh.read_unavailable", {
        source: "core.api.client",
        cause: result.cause,
      });
      // Jamais convertir en missing / legacy
      return null;
    }
    if (result.status === "permanently_invalidated") {
      emitDriverTelemetry("auth.refresh.permanently_invalidated", {
        source: "core.api.client",
        cause: result.cause,
      });
      return null;
    }
    // missing strict : legacy seulement si pas de migration stricte / pas de marker
    const { readInstallationId } = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
    const installation = await readInstallationId();
    // Si installation stricte existe, migration déjà faite → pas de legacy
    if (installation.status === "found") {
      return null;
    }
  } catch {
    /* fallback legacy encadré ci-dessous */
  }
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

export class RefreshTokenPersistError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RefreshTokenPersistError";
  }
}

/** Erreur contractuelle / locale auth — ne doit jamais être traitée comme panne réseau. */
export class AuthContractError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "AuthContractError";
    this.code = code;
  }
}

function sleepMs(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Headers appareil best-effort (logout / resume) — ne bloque pas si SecureStore échoue. */
async function buildAuthDeviceHeaders(): Promise<Record<string, string>> {
  try {
    return await buildRequiredAuthDeviceHeaders();
  } catch {
    try {
      const { buildDeviceMetadataHeaders } = require("../device/deviceRuntimeMetadata") as {
        buildDeviceMetadataHeaders: () => Record<string, string>;
      };
      return buildDeviceMetadataHeaders();
    } catch {
      return {};
    }
  }
}

/** Headers appareil obligatoires pour login / refresh contrat v1. */
async function buildRequiredAuthDeviceHeaders(): Promise<Record<string, string>> {
   
  const { getStableDeviceId } = require("../notifications/getStableDeviceId") as {
    getStableDeviceId: () => Promise<string>;
  };
  const { buildDeviceMetadataHeaders } = require("../device/deviceRuntimeMetadata") as {
    buildDeviceMetadataHeaders: () => Record<string, string>;
  };
  let deviceId: string;
  try {
    deviceId = await getStableDeviceId();
  } catch {
    throw new AuthContractError(
      "DEVICE_ID_UNAVAILABLE",
      "Impossible de sécuriser la session sur cet appareil. Fermez puis rouvrez l'application et réessayez."
    );
  }
  if (!deviceId.trim()) {
    throw new AuthContractError(
      "DEVICE_ID_UNAVAILABLE",
      "Impossible de sécuriser la session sur cet appareil. Fermez puis rouvrez l'application et réessayez."
    );
  }
  return {
    "X-Device-ID": deviceId,
    ...buildDeviceMetadataHeaders(),
  };
}

async function writeRefreshToken(value: string | null): Promise<void> {
  const isDelete = !value || value.trim().length === 0;
  const attempts = isDelete ? 1 : REFRESH_TOKEN_WRITE_RETRIES;
  let lastError: unknown = null;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const store = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
      if (isDelete) {
        const del = await store.deleteRefreshToken();
        if (del.status !== "ok") {
          throw new RefreshTokenPersistError(del.cause || "delete_failed");
        }
        // Compat legacy key
        try {
          await SecureStore.deleteItemAsync(REFRESH_TOKEN_STORAGE_KEY);
        } catch {
          /* ignore */
        }
      } else {
        const written = await store.writeRefreshToken(value);
        if (written.status !== "ok") {
          throw new RefreshTokenPersistError(written.cause || "write_failed");
        }
        // Compat dual-write legacy pendant rollout
        try {
          await SecureStore.setItemAsync(REFRESH_TOKEN_STORAGE_KEY, value);
        } catch {
          /* ignore */
        }
        // P1-C2 : nouveau token persisté -> la porte terminale est levée.
        refreshTerminalState = null;
      }
      void appendSessionJournalEvent("auth.refresh_token.persist_success", { attempt });
      return;
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await sleepMs(REFRESH_TOKEN_WRITE_BACKOFF_MS[attempt - 1] ?? 300);
      }
    }
  }

  const reason =
    lastError instanceof Error ? lastError.message : "persist_failed";
  void appendSessionJournalEvent("auth.refresh_token.persist_failed", { reason, attempts });
  emitDriverTelemetry("auth.refresh_token.persist_failed", {
    source: "core.api.client",
    reason,
    attempts,
  });
  throw lastError instanceof RefreshTokenPersistError
    ? lastError
    : new RefreshTokenPersistError(reason);
}

function isMutatingMethod(method: string | undefined): boolean {
  const m = String(method || "get").toUpperCase();
  return m === "POST" || m === "PUT" || m === "PATCH" || m === "DELETE";
}

function isDriverEndpoint(url: string): boolean {
  return url.startsWith("/driver/") || url.startsWith("driver/");
}

/** Endpoints personnels chauffeur — exigent un contexte driver actif. */
function isDriverSelfEndpoint(url: string): boolean {
  const normalized = url.startsWith("/") ? url : `/${url}`;
  return (
    normalized.startsWith("/driver/me/") ||
    normalized === "/driver/me" ||
    normalized.startsWith("driver/me/")
  );
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
  const snapshot = getNetworkSnapshot();
  // Android BG : isInternetReachable est souvent false alors que le réseau fonctionne (FGS).
  // Ne fail-fast que si le radio est déconnecté ; la queue tracking gère les retries.
  if (snapshot.connected) {
    return false;
  }
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
  const {
    getAuthEpoch,
    isCurrentAuthEpoch,
    readSessionEnvelope,
  } = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
  // authEpoch capturé AVANT toute I/O réseau
  const epochAtStart = getAuthEpoch();

  const refreshToken = await readRefreshToken();
  if (!refreshToken) {
    return null;
  }
  // P1-C2 : token déjà rejeté comme révoqué/invalide -> pas de rejeu réseau.
  if (refreshTerminalState) {
    if (refreshTerminalState.tokenFingerprint === fingerprintRefreshToken(refreshToken)) {
      if (shouldEmitRefreshFailure(refreshTerminalState.status, "terminal_short_circuit")) {
        emitDriverTelemetry("auth.refresh.terminal_short_circuit", {
          source: "core.api.client",
          error_code: refreshTerminalState.code,
        });
      }
      throw new AuthContractError(
        "AUTH_REFRESH_TERMINAL",
        "Refresh token révoqué : rejeu bloqué, reconnexion requise."
      );
    }
    // Nouveau token stocké (login / rotation) -> porte levée.
    refreshTerminalState = null;
  }
  if (!isCurrentAuthEpoch(epochAtStart)) return null;

  const envelope = await readSessionEnvelope();
  const sessionId =
    envelope.status === "found" ? envelope.value.session_id : "unknown";
  const sourceGen =
    envelope.status === "found" ? envelope.value.refresh_generation : 0;

  const {
    ensurePendingRefreshOperation,
    clearPendingRefreshOperation,
  } = require("../auth/pendingRefreshOperation") as typeof import("../auth/pendingRefreshOperation");
  const pending = await ensurePendingRefreshOperation({
    sessionId,
    sourceRefreshGeneration: sourceGen,
  });

  const payload = { refresh_token: refreshToken };
  const deviceHeaders = await buildRequiredAuthDeviceHeaders();
  const authHeaders = {
    ...deviceHeaders,
    "X-Auth-Contract-Version": "mobile-device-session-v1",
    "Idempotency-Key": pending.operationId,
  };

  let endpointUsed: "refresh-token" | "refresh" = "refresh-token";
  let data: {
    access_token?: string;
    token?: string;
    refresh_token?: string;
    refresh_generation?: number;
  } | null = null;
  try {
    const response = await apiClient.post<{
      access_token?: string;
      token?: string;
      refresh_token?: string;
      refresh_generation?: number;
    }>("/auth/refresh-token", payload, { headers: authHeaders });
    data = response.data;
  } catch (error) {
    const err = error as AxiosError;
    const shouldFallback = err.response?.status === 404 || err.response?.status === 405;
    if (!shouldFallback) {
      markRefreshTerminalIfNeeded(err, refreshToken);
      throw error;
    }
    endpointUsed = "refresh";
    try {
      const fallbackResponse = await apiClient.post<{
        access_token?: string;
        token?: string;
        refresh_token?: string;
      }>("/auth/refresh", payload, { headers: authHeaders });
      data = fallbackResponse.data;
    } catch (fallbackError) {
      markRefreshTerminalIfNeeded(fallbackError as AxiosError, refreshToken);
      throw fallbackError;
    }
  }

  emitDriverTelemetry("auth.refresh.endpoint_used", {
    source: "core.api.client",
    endpoint: endpointUsed,
  });
  if (!isCurrentAuthEpoch(epochAtStart)) return null;

  const token = extractToken(data);
  const nextRefreshToken = extractRefreshToken(data);
  if (!token && !nextRefreshToken) {
    return null;
  }

  const { withSessionCredentialMutation } =
    require("../auth/sessionCredentialMutex") as typeof import("../auth/sessionCredentialMutex");
  const {
    setTrackingAuthTemporarilyUnavailable,
  } = require("../auth/sessionAuthDecision") as typeof import("../auth/sessionAuthDecision");
  const {
    reassertTrackingAuthSessionAfterRefresh,
  } = require("../auth/trackingAuthPresence") as typeof import("../auth/trackingAuthPresence");

  setTrackingAuthTemporarilyUnavailable("refreshing");
  try {
    const applyResult = await withSessionCredentialMutation(epochAtStart, async () => {
      if (nextRefreshToken) {
        await writeRefreshToken(nextRefreshToken);
        if (envelope.status === "found") {
          const nextGen =
            typeof data?.refresh_generation === "number"
              ? data.refresh_generation
              : envelope.value.refresh_generation + 1;
          await (
            require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore")
          ).writeSessionEnvelope({
            ...envelope.value,
            refresh_generation: nextGen,
            last_authenticated_at: new Date().toISOString(),
          });
        }
        await clearPendingRefreshOperation();
      }
      if (token) {
        setAuthToken(token);
        try {
          const { notifyAuthRefreshSuccess } = require("../auth/authRefreshListeners") as {
            notifyAuthRefreshSuccess: () => void;
          };
          notifyAuthRefreshSuccess();
        } catch {
          /* optional */
        }
      }
      return token;
    });

    if (applyResult.status === "stale") {
      return null;
    }
    return applyResult.value;
  } finally {
    setTrackingAuthTemporarilyUnavailable(null);
    // P0-B : ne pas retomber en TRACKING_IDENTITY_UNAVAILABLE après un simple refresh
    await reassertTrackingAuthSessionAfterRefresh().catch(() => undefined);
  }
}

async function ensureRefreshToken(): Promise<string | null> {
  if (!refreshTokenInFlight) {
    refreshTokenInFlight = refreshAuthToken()
      .then((token) => {
        lastRefreshErrorCode = null;
        return token;
      })
      .catch((error) => {
        if (
          error instanceof AuthContractError &&
          error.code === "AUTH_REFRESH_TERMINAL"
        ) {
          // Court-circuit terminal : conserver le code d'origine pour la policy.
          lastRefreshErrorCode =
            refreshTerminalState?.code ?? lastRefreshErrorCode ?? "session_revoked";
          throw error;
        }
        const err = error as AxiosError;
        const status = err.response?.status ?? null;
        const reason = err.message;
        const data = err.response?.data as { error_code?: string; error?: string } | undefined;
        lastRefreshErrorCode =
          (typeof data?.error_code === "string" && data.error_code) ||
          (typeof data?.error === "string" && data.error) ||
          null;
        if (shouldEmitRefreshFailure(status, reason)) {
          emitDriverTelemetry("auth.refresh.failure", {
            source: "core.api.client",
            reason,
            status,
            error_code: lastRefreshErrorCode,
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

type AtmrAxiosRequestConfig = InternalAxiosRequestConfig & {
  _skipBearerAuth?: boolean;
  _p1HarnessStartedAt?: number;
};

apiClient.interceptors.request.use(async (config) => {
  try {
     
    const { recordHttpRequest } = require("../observability/perfInstrumentation") as {
      recordHttpRequest: (url: string) => void;
    };
    recordHttpRequest(String(config.url ?? ""));
  } catch {
    // optional perf instrumentation
  }
  const atmrConfig = config as AtmrAxiosRequestConfig;
  atmrConfig._p1HarnessStartedAt = Date.now();
  try {
    const { emitP1HarnessLog } = require("../observability/p1HarnessLog") as {
      emitP1HarnessLog: (event: string, payload: Record<string, unknown>) => void;
    };
    const url = String(config.url ?? "");
    if (
      url.includes("/auth/") ||
      url.includes("/company_mobile/") ||
      url.includes("/driver/")
    ) {
      emitP1HarnessLog("p1.api.start", {
        source: "api.client",
        method: String(config.method ?? "get").toUpperCase(),
        endpoint: url,
      });
    }
  } catch {
    // optional p1 harness
  }
  if (atmrConfig._skipBearerAuth) {
    // JWT expiré + jwt_required(optional=True) → 401 serveur ; bootstrap doit pouvoir
    // retomber en mode non authentifié sans écran « session expirée ».
    if (atmrConfig.headers) {
      delete atmrConfig.headers.Authorization;
      delete atmrConfig.headers.authorization;
    }
  }
  const requestUrl = String(config.url ?? "");
  if (
    isDriverSelfEndpoint(requestUrl) &&
    !(typeof activeContextIdForApi === "string" && activeContextIdForApi.startsWith("driver:"))
  ) {
    throw new AuthContractError(
      "DRIVER_CONTEXT_INACTIVE",
      "Driver context is not active"
    );
  }
  const effectiveTimeoutMs = resolveAdaptiveTimeoutMs(requestUrl, Number(config.timeout ?? 15000));
  config.timeout = effectiveTimeoutMs;
  try {
     
    const { buildDeviceMetadataHeaders } = require("../device/deviceRuntimeMetadata") as {
      buildDeviceMetadataHeaders: () => Record<string, string>;
    };
    const deviceMetaHeaders = buildDeviceMetadataHeaders();
    for (const [key, value] of Object.entries(deviceMetaHeaders)) {
      if (!config.headers[key]) {
        config.headers[key] = value;
      }
    }
  } catch {
    config.headers["X-Client-Platform"] = Platform.OS;
  }
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
    const offlineSnapshot = getNetworkSnapshot();
    emitDriverTelemetry("driver.http.fail_fast_offline", {
      source: "core.api.client",
      endpoint: requestUrl,
      connected: offlineSnapshot.connected,
      internet_reachable: offlineSnapshot.internetReachable,
      network_type: offlineSnapshot.type,
    });
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
  (response) => {
    try {
      const cfg = response.config as AtmrAxiosRequestConfig;
      const started = cfg._p1HarnessStartedAt;
      const url = String(cfg.url ?? "");
      if (
        started &&
        (url.includes("/auth/") ||
          url.includes("/company_mobile/") ||
          url.includes("/driver/"))
      ) {
        const { emitP1HarnessLog } = require("../observability/p1HarnessLog") as {
          emitP1HarnessLog: (event: string, payload: Record<string, unknown>) => void;
        };
        emitP1HarnessLog("p1.api.end", {
          source: "api.client",
          method: String(cfg.method ?? "get").toUpperCase(),
          endpoint: url,
          status: response.status,
          duration_ms: Date.now() - started,
        });
      }
    } catch {
      // optional p1 harness
    }
    return response;
  },
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
  if (error instanceof AuthContractError) {
    return {
      status: null,
      code: error.code,
      message: error.message,
    };
  }

  if (!isAxiosError(error)) {
    if (error instanceof Error) {
      const code =
        error.message === "storage_locked" ? "STORAGE_UNAVAILABLE" : "CLIENT_RUNTIME_ERROR";
      return {
        status: null,
        code,
        message: error.message || "Erreur locale inattendue.",
      };
    }
    return {
      status: null,
      code: "CLIENT_RUNTIME_ERROR",
      message: "Erreur locale inattendue.",
    };
  }

  const e = error;
  const data = e.response?.data as
    | {
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
        limit?: number;
        active_count?: number;
        sessions?: unknown[];
        resolution_token?: string;
        capabilities?: Record<string, unknown>;
        details?: Record<string, unknown>;
      }
    | undefined;
  const endpoint = e.config?.url ?? "";
  const requestUrl = `${e.config?.baseURL ?? ""}${endpoint}`;
  const isTransportFailure =
    Boolean(e.request) && !e.response && e.code !== "ERR_CANCELED";
  const transportHint = isTransportFailure ? buildNoHttpResponseHint(requestUrl, e) : "";
  const errorCode = data?.error_code ?? "UNKNOWN_ERROR";
  const isDeviceSessionLimit = errorCode === "device_session_limit_reached";
  return {
    status: e.response?.status ?? null,
    code: errorCode,
    message: isDeviceSessionLimit
      ? "Nombre maximal d'appareils atteint. Déconnectez un ancien appareil pour continuer."
      : (data?.error_message ?? data?.error ?? data?.message ?? e.message) + transportHint,
    reason: data?.reason,
    outcome_class: data?.outcome_class,
    retryable: data?.retryable,
    details: {
      ...(data?.details ?? {}),
      trace_id: data?.trace_id,
      activation_session_id: data?.activation_session_id,
      masked_email: data?.masked_email,
      masked_phone: data?.masked_phone,
      ...(Array.isArray(data?.sessions) ? { sessions: data.sessions } : {}),
      ...(typeof data?.limit === "number" ? { limit: data.limit } : {}),
      ...(typeof data?.active_count === "number" ? { active_count: data.active_count } : {}),
      ...(typeof data?.resolution_token === "string"
        ? { resolution_token: data.resolution_token }
        : {}),
      ...(data?.capabilities && typeof data.capabilities === "object"
        ? { capabilities: data.capabilities }
        : {}),
    },
  };
}

/** Remplacement multi-appareils : token + capability strictement true. */
export function canReplaceDeviceSession(details?: Record<string, unknown> | null): boolean {
  if (!details) return false;
  const token = details.resolution_token;
  if (typeof token !== "string" || !token.trim()) return false;
  const caps = details.capabilities;
  if (!caps || typeof caps !== "object") return false;
  return (caps as Record<string, unknown>).device_session_replace === true;
}

export type DeviceSessionInfo = {
  session_id: string;
  device_name?: string | null;
  device_model?: string | null;
  device_code?: string | null;
  last_platform?: string | null;
  last_app_version?: string | null;
  last_seen_at?: string | null;
  is_current?: boolean;
  is_provisional?: boolean;
  status?: string | null;
};

export type DeviceSessionsListResponse = {
  sessions: DeviceSessionInfo[];
  auth_contract_version?: string;
  capabilities?: Record<string, unknown>;
};

async function schedulePendingSessionConfirmation(
  sessionId: string,
  deviceInstallationId: string
): Promise<void> {
  try {
    const {
      writePendingSessionConfirmation,
      flushPendingSessionConfirmation,
    } = require("../auth/pendingSessionConfirmation") as typeof import("../auth/pendingSessionConfirmation");
    await writePendingSessionConfirmation({ sessionId, deviceInstallationId });
    void flushPendingSessionConfirmation().catch(() => undefined);
  } catch {
    /* best-effort */
  }
}

export async function confirmDeviceSession(sessionId: string): Promise<void> {
  if (useMockBootstrap) return;
  try {
    await apiClient.post(
      `/auth/device-sessions/${encodeURIComponent(sessionId)}/confirm`,
      {}
    );
  } catch (error) {
    throw toApiError(error);
  }
}

export async function listDeviceSessions(): Promise<DeviceSessionsListResponse> {
  if (useMockBootstrap) {
    return { sessions: [] };
  }
  try {
    const { data } = await apiClient.get<DeviceSessionsListResponse>("/auth/device-sessions");
    const sessions = Array.isArray(data?.sessions) ? data.sessions : [];
    return {
      sessions,
      auth_contract_version:
        typeof data?.auth_contract_version === "string"
          ? data.auth_contract_version
          : undefined,
      capabilities:
        data?.capabilities && typeof data.capabilities === "object"
          ? data.capabilities
          : undefined,
    };
  } catch (error) {
    throw toApiError(error);
  }
}

export async function revokeDeviceSession(sessionId: string): Promise<void> {
  if (useMockBootstrap) return;
  try {
    await apiClient.delete(`/auth/device-sessions/${encodeURIComponent(sessionId)}`);
  } catch (error) {
    throw toApiError(error);
  }
}

export async function revokeOtherDeviceSessions(): Promise<{
  ok: boolean;
  revoked_sessions: number;
}> {
  if (useMockBootstrap) return { ok: true, revoked_sessions: 0 };
  try {
    const { data } = await apiClient.post<{
      ok?: boolean;
      revoked_sessions?: number;
    }>("/auth/device-sessions/revoke-others", {});
    return {
      ok: data?.ok !== false,
      revoked_sessions:
        typeof data?.revoked_sessions === "number" ? data.revoked_sessions : 0,
    };
  } catch (error) {
    throw toApiError(error);
  }
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
    // Ne plus fire-and-forget wipe SecureStore ici — utiliser clearLocalAuth().
  }
}

/** Purge locale awaitée (access + refresh + recovery + envelope). Incrémente authEpoch. */
export async function clearLocalAuth(): Promise<void> {
  // P1-C2 : plus de token -> plus de porte terminale a maintenir.
  refreshTerminalState = null;
  try {
    const store = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
    const { withCredentialStoreLock } = require("../auth/sessionCredentialMutex") as typeof import("../auth/sessionCredentialMutex");
    await withCredentialStoreLock(async () => {
      store.bumpAuthEpoch();
      await store.clearLocalAuthCredentialsLocked();
    });
  } catch {
    /* ignore */
  }
  try {
    const { clearPendingSessionConfirmation } = require("../auth/pendingSessionConfirmation") as {
      clearPendingSessionConfirmation: () => Promise<void>;
    };
    await clearPendingSessionConfirmation();
  } catch {
    /* ignore */
  }
  setAuthToken(null);
  try {
    await SecureStore.deleteItemAsync(REFRESH_TOKEN_STORAGE_KEY);
  } catch {
    /* ignore */
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

async function fetchBootstrapOnce(
  activeContextId?: string | null,
  skipBearerAuth = false
): Promise<BootstrapResponse> {
  const headers: Record<string, string> = {};
  if (activeContextId) {
    headers["X-Active-Context-Id"] = activeContextId;
  }
  const { data } = await apiClient.get("/auth/bootstrap", {
    headers,
    _skipBearerAuth: skipBearerAuth,
  } as AtmrAxiosRequestConfig);
  return bootstrapResponseSchema.parse(data);
}

export async function fetchBootstrap(activeContextId?: string | null): Promise<BootstrapResponse> {
  if (useMockBootstrap) {
    return bootstrapResponseSchema.parse(buildMockBootstrap());
  }
  const emitBootstrapFailure = (error: unknown, phase: string) => {
    const err = error as AxiosError;
    emitDriverTelemetry("auth.bootstrap.failure", {
      source: "core.api.client",
      context_id: activeContextId ?? null,
      reason: err.message,
      status: err.response?.status ?? null,
      axios_code: err.code ?? null,
      phase,
    });
  };
  try {
    const parsed = await fetchBootstrapOnce(activeContextId, false);
    markBootstrapAuthFresh();
    return parsed;
  } catch (error) {
    const status = isAxiosError(error) ? error.response?.status ?? null : null;
    if (status === 401 || status === 403) {
      setAuthToken(null);
      try {
        const parsed = await fetchBootstrapOnce(activeContextId, true);
        markBootstrapAuthFresh();
        emitDriverTelemetry("auth.bootstrap.recovered_without_bearer", {
          source: "core.api.client",
          context_id: activeContextId ?? null,
          is_authenticated: parsed.is_authenticated,
        });
        return parsed;
      } catch (retryError) {
        emitBootstrapFailure(retryError, "retry_without_bearer");
        throw toApiError(retryError);
      }
    }
    emitBootstrapFailure(error, "initial");
    throw toApiError(error);
  }
}

export async function login(email: string, password: string): Promise<void> {
  if (useMockBootstrap) return;
  try {
    const store = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
    const {
      withCredentialStoreLock,
      withSessionCredentialMutation,
    } = require("../auth/sessionCredentialMutex") as typeof import("../auth/sessionCredentialMutex");
    const {
      enqueueOrphanedLoginRevocation,
      flushOrphanedLoginRevocationInBackground,
    } = require("../auth/authRecoveryCoordinator") as typeof import("../auth/authRecoveryCoordinator");
    const { decodeJwtClaims } = require("../auth/jwtClaims") as typeof import("../auth/jwtClaims");

    // Claim génération au début de l'intention login (avant réseau).
    const loginGeneration = await withCredentialStoreLock(() => store.bumpSessionGeneration());

    // Flush best-effort des pending historiques — ne bloque jamais le login.
    try {
      const recoveryMod = require("../auth/authRecoveryCoordinator") as typeof import("../auth/authRecoveryCoordinator");
      void recoveryMod.flushPendingRevocationTombstone().catch(() => undefined);
    } catch {
      /* ignore */
    }

    const deviceHeaders = await buildRequiredAuthDeviceHeaders();
    const { data } = await apiClient.post(
      "/auth/login",
      {
        email: email.trim(),
        password,
      },
      {
        headers: {
          ...deviceHeaders,
          "X-Auth-Contract-Version": "mobile-device-session-v1",
        },
      }
    );
    const token = extractToken(data);
    const refreshToken = extractRefreshToken(data);
    const responseObj = data && typeof data === "object" ? (data as Record<string, unknown>) : {};
    const recovery = responseObj.recovery_credential;
    const revocationSecret = responseObj.revocation_secret;
    const userObj =
      responseObj.user && typeof responseObj.user === "object"
        ? (responseObj.user as Record<string, unknown>)
        : {};

    // Fail-closed : toutes les écritures doivent réussir avant de publier l'access
    if (!refreshToken || typeof recovery !== "string" || !recovery) {
      emitDriverTelemetry("auth.login.contract_incomplete", {
        source: "core.api.client",
        has_access_token: Boolean(token),
        has_refresh_token: Boolean(refreshToken),
        has_session_id: typeof responseObj.session_id === "string",
        has_recovery_credential: typeof recovery === "string" && Boolean(recovery),
        has_revocation_secret: typeof revocationSecret === "string",
      });
      const sid = responseObj.session_id;
      if (typeof sid === "string" && typeof revocationSecret === "string") {
        try {
          await revokeSessionPending(sid, revocationSecret);
        } catch {
          /* best-effort */
        }
      }
      throw new AuthContractError(
        "AUTH_LOGIN_CONTRACT_INCOMPLETE",
        "Le serveur n'a pas retourné les éléments nécessaires à une session sécurisée."
      );
    }

    const deviceId = deviceHeaders["X-Device-ID"] ?? "unknown";
    const accessClaims = token ? decodeJwtClaims(token) : null;
    const driverIdClaim = accessClaims?.driver_id;
    const sessionId = responseObj.session_id;
    if (typeof sessionId !== "string") {
      throw new AuthContractError(
        "AUTH_LOGIN_CONTRACT_INCOMPLETE",
        "Le serveur n'a pas retourné les éléments nécessaires à une session sécurisée."
      );
    }

    const credentialGenerationRaw = responseObj.credential_generation;
    const credentialGeneration =
      typeof credentialGenerationRaw === "number"
        ? credentialGenerationRaw
        : typeof credentialGenerationRaw === "string" &&
            credentialGenerationRaw.trim() !== "" &&
            Number.isFinite(Number(credentialGenerationRaw))
          ? Number(credentialGenerationRaw)
          : null;

    const envelopePayload = {
      schema_version: 1,
      session_id: sessionId,
      device_installation_id: deviceId,
      user_public_id: String(userObj.public_id ?? ""),
      driver_id: typeof driverIdClaim === "number" ? driverIdClaim : null,
      role: String(userObj.role ?? "driver"),
      active_context_id: null as string | null,
      refresh_generation: Number(
        responseObj.refresh_generation ?? responseObj.session_generation ?? 1
      ),
      // Autorité serveur uniquement — jamais copier refresh_generation.
      credential_generation: credentialGeneration,
      last_authenticated_at: new Date().toISOString(),
      revocation_secret: typeof revocationSecret === "string" ? revocationSecret : null,
    };

    const persistResult = await withSessionCredentialMutation(loginGeneration, async () => {
      // Recovery d'abord (secret successeur le plus critique), puis refresh, puis envelope.
      const recoveryWrite = await store.writeRecoveryCredential(recovery);
      if (recoveryWrite.status !== "ok") {
        throw new AuthContractError(
          "STORAGE_UNAVAILABLE",
          "Stockage sécurisé temporairement indisponible. Fermez puis rouvrez l'application et réessayez."
        );
      }

      const refreshWrite = await store.writeRefreshToken(refreshToken);
      if (refreshWrite.status !== "ok") {
        await store.deleteRecoveryCredential();
        throw new AuthContractError(
          "STORAGE_UNAVAILABLE",
          "Stockage sécurisé temporairement indisponible. Fermez puis rouvrez l'application et réessayez."
        );
      }
      try {
        await SecureStore.setItemAsync(REFRESH_TOKEN_STORAGE_KEY, refreshToken);
      } catch {
        /* ignore */
      }

      const envelopeWrite = await store.writeSessionEnvelope(envelopePayload);
      if (envelopeWrite.status !== "ok") {
        await store.deleteRefreshToken();
        await store.deleteRecoveryCredential();
        if (typeof revocationSecret === "string") {
          await store.appendPendingRevocation({
            operation_id: `login-fail-${Date.now()}`,
            session_id: sessionId,
            device_installation_id: deviceId,
            revocation_secret: revocationSecret,
            created_at: new Date().toISOString(),
            origin: "orphaned_login_cleanup",
          });
        }
        throw new AuthContractError(
          "STORAGE_UNAVAILABLE",
          "Stockage sécurisé temporairement indisponible. Fermez puis rouvrez l'application et réessayez."
        );
      }
      return true;
    });

    if (persistResult.status === "stale") {
      // Login orphelin : enqueue durable sans toucher à la session courante.
      if (typeof revocationSecret === "string") {
        const orphan = await enqueueOrphanedLoginRevocation({
          sessionId,
          deviceInstallationId: deviceId,
          revocationSecret,
        });
        flushOrphanedLoginRevocationInBackground(orphan);
      }
      throw new AuthContractError(
        "AUTH_LOGIN_STALE",
        "Une autre session a pris le relais pendant la connexion. Réessayez."
      );
    }

    if (persistResult.status === "applied") {
      // Accès publié hors mutex (mémoire) — génération encore courante vérifiée à l'instant.
      if (!store.isCurrentSessionGeneration(loginGeneration)) {
        if (typeof revocationSecret === "string") {
          const orphan = await enqueueOrphanedLoginRevocation({
            sessionId,
            deviceInstallationId: deviceId,
            revocationSecret,
          });
          flushOrphanedLoginRevocationInBackground(orphan);
        }
        throw new AuthContractError(
          "AUTH_LOGIN_STALE",
          "Une autre session a pris le relais pendant la connexion. Réessayez."
        );
      }
      if (token) {
        setAuthToken(token);
      }
      markBootstrapAuthFresh();
      void schedulePendingSessionConfirmation(sessionId, deviceId);
    }
  } catch (error) {
    // Best-effort : flush pending créés lors d'un échec d'écriture login.
    try {
      const recoveryMod = require("../auth/authRecoveryCoordinator") as typeof import("../auth/authRecoveryCoordinator");
      void recoveryMod.flushPendingRevocationTombstone().catch(() => undefined);
    } catch {
      /* ignore */
    }
    throw toApiError(error);
  }
}

/**
 * Remplace une session existante après un 409 device_session_limit_reached.
 * Même contrat de persistance locale que login().
 * À n'appeler que si canReplaceDeviceSession(details) (resolution_token présent).
 */
export async function replaceDeviceSessionOnLimit(params: {
  sessionToRevoke: string;
  resolutionToken: string;
}): Promise<void> {
  if (useMockBootstrap) return;
  try {
    const store = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
    const {
      withCredentialStoreLock,
      withSessionCredentialMutation,
    } = require("../auth/sessionCredentialMutex") as typeof import("../auth/sessionCredentialMutex");
    const {
      enqueueOrphanedLoginRevocation,
      flushOrphanedLoginRevocationInBackground,
    } = require("../auth/authRecoveryCoordinator") as typeof import("../auth/authRecoveryCoordinator");
    const { decodeJwtClaims } = require("../auth/jwtClaims") as typeof import("../auth/jwtClaims");

    const loginGeneration = await withCredentialStoreLock(() => store.bumpSessionGeneration());
    const deviceHeaders = await buildRequiredAuthDeviceHeaders();
    const { data } = await apiClient.post(
      "/auth/device-sessions/replace",
      {
        session_to_revoke: params.sessionToRevoke,
        resolution_token: params.resolutionToken,
      },
      {
        headers: {
          ...deviceHeaders,
          "X-Auth-Contract-Version": "mobile-device-session-v1",
        },
      }
    );

    const token = extractToken(data);
    const refreshToken = extractRefreshToken(data);
    const responseObj = data && typeof data === "object" ? (data as Record<string, unknown>) : {};
    const recovery = responseObj.recovery_credential;
    const revocationSecret = responseObj.revocation_secret;
    const userObj =
      responseObj.user && typeof responseObj.user === "object"
        ? (responseObj.user as Record<string, unknown>)
        : {};

    if (!refreshToken || typeof recovery !== "string" || !recovery) {
      const sid = responseObj.session_id;
      if (typeof sid === "string" && typeof revocationSecret === "string") {
        try {
          await revokeSessionPending(sid, revocationSecret);
        } catch {
          /* best-effort */
        }
      }
      throw new AuthContractError(
        "AUTH_LOGIN_CONTRACT_INCOMPLETE",
        "Le serveur n'a pas retourné les éléments nécessaires à une session sécurisée."
      );
    }

    const deviceId = deviceHeaders["X-Device-ID"] ?? "unknown";
    const accessClaims = token ? decodeJwtClaims(token) : null;
    const driverIdClaim = accessClaims?.driver_id;
    const sessionId = responseObj.session_id;
    if (typeof sessionId !== "string") {
      throw new AuthContractError(
        "AUTH_LOGIN_CONTRACT_INCOMPLETE",
        "Le serveur n'a pas retourné les éléments nécessaires à une session sécurisée."
      );
    }

    const credentialGenerationRaw = responseObj.credential_generation;
    const credentialGeneration =
      typeof credentialGenerationRaw === "number"
        ? credentialGenerationRaw
        : typeof credentialGenerationRaw === "string" &&
            credentialGenerationRaw.trim() !== "" &&
            Number.isFinite(Number(credentialGenerationRaw))
          ? Number(credentialGenerationRaw)
          : null;

    const envelopePayload = {
      schema_version: 1,
      session_id: sessionId,
      device_installation_id: deviceId,
      user_public_id: String(userObj.public_id ?? ""),
      driver_id: typeof driverIdClaim === "number" ? driverIdClaim : null,
      role: String(userObj.role ?? "driver"),
      active_context_id: null as string | null,
      refresh_generation: Number(
        responseObj.refresh_generation ?? responseObj.session_generation ?? 1
      ),
      credential_generation: credentialGeneration,
      last_authenticated_at: new Date().toISOString(),
      revocation_secret: typeof revocationSecret === "string" ? revocationSecret : null,
    };

    const persistResult = await withSessionCredentialMutation(loginGeneration, async () => {
      const recoveryWrite = await store.writeRecoveryCredential(recovery);
      if (recoveryWrite.status !== "ok") {
        throw new AuthContractError(
          "STORAGE_UNAVAILABLE",
          "Stockage sécurisé temporairement indisponible. Fermez puis rouvrez l'application et réessayez."
        );
      }
      const refreshWrite = await store.writeRefreshToken(refreshToken);
      if (refreshWrite.status !== "ok") {
        await store.deleteRecoveryCredential();
        throw new AuthContractError(
          "STORAGE_UNAVAILABLE",
          "Stockage sécurisé temporairement indisponible. Fermez puis rouvrez l'application et réessayez."
        );
      }
      try {
        await SecureStore.setItemAsync(REFRESH_TOKEN_STORAGE_KEY, refreshToken);
      } catch {
        /* ignore */
      }
      const envelopeWrite = await store.writeSessionEnvelope(envelopePayload);
      if (envelopeWrite.status !== "ok") {
        await store.deleteRefreshToken();
        await store.deleteRecoveryCredential();
        if (typeof revocationSecret === "string") {
          await store.appendPendingRevocation({
            operation_id: `replace-fail-${Date.now()}`,
            session_id: sessionId,
            device_installation_id: deviceId,
            revocation_secret: revocationSecret,
            created_at: new Date().toISOString(),
            origin: "orphaned_login_cleanup",
          });
        }
        throw new AuthContractError(
          "STORAGE_UNAVAILABLE",
          "Stockage sécurisé temporairement indisponible. Fermez puis rouvrez l'application et réessayez."
        );
      }
      return true;
    });

    if (persistResult.status === "stale") {
      if (typeof revocationSecret === "string") {
        const orphan = await enqueueOrphanedLoginRevocation({
          sessionId,
          deviceInstallationId: deviceId,
          revocationSecret,
        });
        flushOrphanedLoginRevocationInBackground(orphan);
      }
      throw new AuthContractError(
        "AUTH_LOGIN_STALE",
        "Une autre session a pris le relais pendant la connexion. Réessayez."
      );
    }

    if (persistResult.status === "applied") {
      if (!store.isCurrentSessionGeneration(loginGeneration)) {
        if (typeof revocationSecret === "string") {
          const orphan = await enqueueOrphanedLoginRevocation({
            sessionId,
            deviceInstallationId: deviceId,
            revocationSecret,
          });
          flushOrphanedLoginRevocationInBackground(orphan);
        }
        throw new AuthContractError(
          "AUTH_LOGIN_STALE",
          "Une autre session a pris le relais pendant la connexion. Réessayez."
        );
      }
      if (token) {
        setAuthToken(token);
      }
      markBootstrapAuthFresh();
      void schedulePendingSessionConfirmation(sessionId, deviceId);
    }
  } catch (error) {
    try {
      const recoveryMod = require("../auth/authRecoveryCoordinator") as typeof import("../auth/authRecoveryCoordinator");
      void recoveryMod.flushPendingRevocationTombstone().catch(() => undefined);
    } catch {
      /* ignore */
    }
    throw toApiError(error);
  }
}

export async function switchContext(
  targetContextId: string,
  opts?: { sourceContextId?: string | null }
): Promise<SwitchContextResponse> {
  if (useMockBootstrap) {
    return switchContextResponseSchema.parse(buildMockSwitchContext(targetContextId));
  }
  const {
    beginContextSwitchOperation,
    isCurrentContextSwitchOperation,
  } = require("../auth/contextSwitchOperation") as typeof import("../auth/contextSwitchOperation");
  const { isCurrentSessionGeneration } =
    require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
  const { withSessionCredentialMutation } =
    require("../auth/sessionCredentialMutex") as typeof import("../auth/sessionCredentialMutex");

  const operation = beginContextSwitchOperation({
    sourceContextId: opts?.sourceContextId ?? null,
    targetContextId,
  });

  try {
    const { data } = await apiClient.post<Record<string, unknown>>(
      "/auth/switch-context",
      { target_context_id: targetContextId },
      { timeout: 12_000, headers: { "X-Allow-Offline-Attempt": "1" } }
    );

    if (
      !isCurrentContextSwitchOperation(operation.operationId) ||
      !isCurrentSessionGeneration(operation.sessionGenerationId)
    ) {
      throw new AuthContractError(
        "CONTEXT_SWITCH_STALE",
        "Changement de contexte obsolète ignoré."
      );
    }

    const inlineAccess = extractToken(data);
    const inlineRefresh = extractRefreshToken(data);
    if (inlineAccess || inlineRefresh) {
      const applyResult = await withSessionCredentialMutation(
        operation.sessionGenerationId,
        async () => {
          if (
            !isCurrentContextSwitchOperation(operation.operationId) ||
            !isCurrentSessionGeneration(operation.sessionGenerationId)
          ) {
            return "stale" as const;
          }
          if (inlineAccess) {
            setAuthToken(inlineAccess);
          }
          if (inlineRefresh) {
            await writeRefreshToken(inlineRefresh);
          }
          return "applied" as const;
        }
      );
      if (
        applyResult.status === "stale" ||
        (applyResult.status === "applied" && applyResult.value === "stale")
      ) {
        throw new AuthContractError(
          "CONTEXT_SWITCH_STALE",
          "Changement de contexte obsolète ignoré."
        );
      }
    }

    if (
      !isCurrentContextSwitchOperation(operation.operationId) ||
      !isCurrentSessionGeneration(operation.sessionGenerationId)
    ) {
      throw new AuthContractError(
        "CONTEXT_SWITCH_STALE",
        "Changement de contexte obsolète ignoré."
      );
    }

    const parsed = switchContextResponseSchema.parse(data);
    if (!shouldSkipPostBootstrapRefresh()) {
      void refreshAuthTokenNow().catch(() => false);
    }
    return Object.assign(parsed, {
      contextSwitchOperationId: operation.operationId,
      contextSwitchSessionGenerationId: operation.sessionGenerationId,
    });
  } catch (error) {
    throw toApiError(error);
  }
}

export async function logoutSession(opts?: { skipLocalPurge?: boolean }): Promise<void> {
  if (useMockBootstrap) return;
  try {
    const deviceHeaders = await buildAuthDeviceHeaders();
    let body: Record<string, string> | undefined;
    try {
      const store = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
      const envelope = await store.readSessionEnvelope();
      const refresh = await store.readRefreshToken();
      body = {};
      if (envelope.status === "found") {
        body.session_id = envelope.value.session_id;
        body.device_installation_id = envelope.value.device_installation_id;
      }
      if (refresh.status === "found") {
        body.refresh_token = refresh.value;
      }
      if (Object.keys(body).length === 0) body = undefined;
    } catch {
      body = undefined;
    }
    await apiClient.post("/auth/logout", body, { headers: deviceHeaders });
  } catch (error) {
    throw toApiError(error);
  } finally {
    if (!opts?.skipLocalPurge) {
      await clearLocalAuth();
    }
  }
}

/** Reprise session durable après refresh JWT expiré / irrécupérable. */
export async function sessionResumeRequest(): Promise<{
  ok: boolean;
  code: string | null;
  retryable: boolean;
}> {
  try {
    const store = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
    const {
      ensurePendingResumeOperation,
      clearPendingResumeOperation,
    } = require("../auth/pendingResumeOperation") as typeof import("../auth/pendingResumeOperation");
    const { withSessionCredentialMutation } =
      require("../auth/sessionCredentialMutex") as typeof import("../auth/sessionCredentialMutex");

    const envelope = await store.readSessionEnvelope();
    const recovery = await store.readRecoveryCredential();
    if (envelope.status !== "found" || recovery.status !== "found") {
      return { ok: false, code: "missing_recovery", retryable: false };
    }

    const sourceCredGen =
      typeof envelope.value.credential_generation === "number"
        ? envelope.value.credential_generation
        : null;

    const pending = await ensurePendingResumeOperation({
      sessionId: envelope.value.session_id,
      sourceCredentialGeneration: sourceCredGen,
    });

    const deviceHeaders = await buildAuthDeviceHeaders();
    const body: Record<string, unknown> = {
      session_id: envelope.value.session_id,
      device_installation_id: envelope.value.device_installation_id,
      recovery_credential: recovery.value,
      idempotency_key: pending.operationId,
    };
    // Ne pas envoyer client_generation tant que credential_generation n'est pas autoritaire.
    if (sourceCredGen !== null) {
      body.client_generation = sourceCredGen;
    }

    const { data } = await apiClient.post("/auth/session-resume", body, {
      headers: { ...deviceHeaders, "Idempotency-Key": pending.operationId },
    });

    const token = extractToken(data);
    const refreshToken = extractRefreshToken(data);
    const dataObj =
      data && typeof data === "object" ? (data as Record<string, unknown>) : null;
    const nextRecovery = dataObj?.recovery_credential;
    const sessionId = dataObj?.session_id;
    const nextCredGen = dataObj?.credential_generation;
    const nextRefreshGen = dataObj?.refresh_generation;

    if (
      typeof nextRecovery !== "string" ||
      nextRecovery.length === 0 ||
      !refreshToken ||
      typeof sessionId !== "string"
    ) {
      return { ok: false, code: "incomplete_resume_response", retryable: true };
    }

    const epochAtStart = store.getSessionGenerationId();
    const applyResult = await withSessionCredentialMutation(epochAtStart, async () => {
      const recoveryWrite = await store.writeRecoveryCredential(nextRecovery);
      if (recoveryWrite.status !== "ok") {
        throw new Error("recovery_write_failed");
      }
      const refreshWrite = await store.writeRefreshToken(refreshToken);
      if (refreshWrite.status !== "ok") {
        throw new Error("refresh_write_failed");
      }
      try {
        await SecureStore.setItemAsync(REFRESH_TOKEN_STORAGE_KEY, refreshToken);
      } catch {
        /* compat legacy — ignore */
      }
      const envelopeWrite = await store.writeSessionEnvelope({
        ...envelope.value,
        session_id: sessionId,
        credential_generation:
          typeof nextCredGen === "number"
            ? nextCredGen
            : envelope.value.credential_generation ?? null,
        refresh_generation:
          typeof nextRefreshGen === "number"
            ? nextRefreshGen
            : envelope.value.refresh_generation,
        last_authenticated_at: new Date().toISOString(),
      });
      if (envelopeWrite.status !== "ok") {
        throw new Error("envelope_write_failed");
      }
      await clearPendingResumeOperation();
      return true;
    });

    if (applyResult.status !== "applied") {
      // Pending conservé — retry avec le même operationId.
      return { ok: false, code: "storage_stale", retryable: true };
    }

    if (token) {
      setAuthToken(token);
    }
    lastRefreshErrorCode = null;
    markBootstrapAuthFresh();
    return { ok: Boolean(token), code: null, retryable: false };
  } catch (error) {
    const err = error as AxiosError;
    const data = err.response?.data as
      | { error_code?: string; error?: string; retryable?: boolean }
      | undefined;
    const code =
      (typeof data?.error_code === "string" && data.error_code) ||
      (typeof data?.error === "string" && data.error) ||
      (error instanceof Error && error.message.includes("_write_failed")
        ? "storage_unavailable"
        : null);
    lastRefreshErrorCode = code;
    return {
      ok: false,
      code,
      retryable:
        Boolean(data?.retryable) ||
        err.response?.status === 503 ||
        code === "storage_unavailable" ||
        code === "storage_stale",
    };
  }
}

/** Flush d'un tombstone de révocation hors-ligne (idempotent via operation_id). */
export async function revokeSessionPending(
  sessionId: string,
  revocationSecret: string,
  operationId?: string
): Promise<boolean> {
  try {
    const opId = operationId || `revoke-${Date.now()}`;
    await apiClient.post(
      `/auth/sessions/${encodeURIComponent(sessionId)}/revoke-pending`,
      { revocation_secret: revocationSecret, operation_id: opId },
      {
        headers: {
          "Idempotency-Key": opId,
          "X-Auth-Contract-Version": "mobile-device-session-v1",
        },
      }
    );
    return true;
  } catch {
    return false;
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
