/**
 * P0-2 FINAL GATE — réponse HTTP perdue côté client mobile (avant SecureStore).
 *
 * Scénario :
 *   SecureStore=R0, PendingRefreshOperation=OP1, envelope.refresh_generation=N
 *   POST R0+OP1 → backend 200 R0→R1 / gen N+1
 *   réponse perdue AVANT toute écriture SecureStore
 *   retry POST R0+OP1 → même R1, aucune nouvelle rotation
 *   apply client → SecureStore=R1, gen=N+1, pending absente, terminal non déclenché
 */
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import AsyncStorage from "@react-native-async-storage/async-storage";

const mockPost = jest.fn();
const mockGet = jest.fn();
const mockRequest = jest.fn();
const mockRequestUse = jest.fn();
const mockResponseUse = jest.fn();
const mockCommonHeaders: Record<string, unknown> = {};

const secureStore = new Map<string, string>();
const mockGetItemAsync = jest.fn(async (key: string) => secureStore.get(key) ?? null);
const mockSetItemAsync = jest.fn(async (key: string, value: string) => {
  secureStore.set(key, value);
});
const mockDeleteItemAsync = jest.fn(async (key: string) => {
  secureStore.delete(key);
});

class MockAxiosError extends Error {
  isAxiosError = true;
  response?: { status?: number; data?: unknown };
  code?: string;
  config?: unknown;

  constructor(message?: string, code?: string, config?: unknown) {
    super(message);
    this.name = "AxiosError";
    this.code = code;
    this.config = config;
  }
}

jest.mock("axios", () => {
  const isAxiosError = (error: unknown) =>
    Boolean(
      error &&
        typeof error === "object" &&
        (error as { isAxiosError?: boolean }).isAxiosError
    );
  return {
    __esModule: true,
    default: {
      create: jest.fn(() => ({
        post: mockPost,
        get: mockGet,
        request: mockRequest,
        interceptors: {
          request: { use: mockRequestUse },
          response: { use: mockResponseUse },
        },
        defaults: { headers: { common: mockCommonHeaders } },
      })),
      isAxiosError,
    },
    AxiosError: MockAxiosError,
    isAxiosError,
  };
});

jest.mock("expo-constants", () => ({
  __esModule: true,
  default: {
    expoConfig: {
      extra: {
        apiBaseUrl: "https://api.test/api/v1",
      },
    },
  },
}));

jest.mock("expo-secure-store", () => ({
  getItemAsync: (...args: unknown[]) =>
    mockGetItemAsync(...(args as [string])),
  setItemAsync: (...args: unknown[]) =>
    mockSetItemAsync(...(args as [string, string])),
  deleteItemAsync: (...args: unknown[]) =>
    mockDeleteItemAsync(...(args as [string])),
  AFTER_FIRST_UNLOCK: "AFTER_FIRST_UNLOCK",
}));

jest.mock("react-native", () => ({
  NativeModules: {
    SourceCode: {
      scriptURL: undefined,
    },
  },
  Platform: { OS: "ios" },
}));

jest.mock("../observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
}));

jest.mock("../observability/sessionJournal", () => ({
  buildSessionDiagHeader: () => "diag-test",
  appendSessionJournalEvent: jest.fn(),
}));

jest.mock("../notifications/getStableDeviceId", () => ({
  getStableDeviceId: jest.fn().mockResolvedValue("test-device-id"),
}));

jest.mock("expo-application", () => ({
  applicationName: "Lirie Test",
}));

jest.mock("../featureFlags/registry", () => ({
  getRuntimeFlagsVersion: () => null,
  isFeatureEnabled: () => false,
}));

jest.mock("../network/networkState", () => ({
  getNetworkSnapshot: () => ({}),
}));

jest.mock("../network/connectivityPolicy", () => ({
  evaluateConnectivityPolicy: () => ({
    mode: "normal",
    recommendedSyncIntervalMs: 5000,
  }),
}));

jest.mock("../auth/trackingAuthPresence", () => ({
  reassertTrackingAuthSessionAfterRefresh: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("../auth/sessionAuthDecision", () => ({
  setTrackingAuthTemporarilyUnavailable: jest.fn(),
}));

const REFRESH_KEY = "atmr.auth.refresh_token";
const ENVELOPE_KEY = "atmr.auth.session_envelope";
const PENDING_KEY = "@atmr/auth/pending_refresh_operation";

type BackendRotation = {
  access_token: string;
  refresh_token: string;
  refresh_generation: number;
};

describe("P0-2 FINAL GATE — lost HTTP response (mobile client)", () => {
  beforeEach(async () => {
    jest.resetModules();
    secureStore.clear();
    await AsyncStorage.clear();
    mockPost.mockReset();
    mockGet.mockReset();
    mockRequest.mockReset();
    mockRequestUse.mockReset();
    mockResponseUse.mockReset();
    mockGetItemAsync.mockClear();
    mockSetItemAsync.mockClear();
    mockDeleteItemAsync.mockClear();
    for (const key of Object.keys(mockCommonHeaders)) {
      delete mockCommonHeaders[key];
    }
  });

  it("drop avant SecureStore → retry même OP1 → même R1, auth préservée", async () => {
    const R0 = "refresh-token-R0";
    const R1 = "refresh-token-R1";
    const OP1 = "ref-op1-fixed";
    const sessionId = "sess-gate-lost-response";
    const genN = 7;
    const accessA = "access-token-A";
    const accessR1 = "access-token-after-R1";

    // État initial client
    secureStore.set(REFRESH_KEY, R0);
    secureStore.set(
      ENVELOPE_KEY,
      JSON.stringify({
        session_id: sessionId,
        session_epoch: 1,
        refresh_generation: genN,
        last_authenticated_at: new Date().toISOString(),
      })
    );
    await AsyncStorage.setItem(
      PENDING_KEY,
      JSON.stringify({
        operationId: OP1,
        sessionId,
        sourceRefreshGeneration: genN,
        createdAt: new Date().toISOString(),
      })
    );

    // Backend idempotent in-memory : une seule rotation R0→R1 pour OP1
    let rotationCount = 0;
    let dropNextSuccessfulResponse = true;
    const seenIdemKeys: string[] = [];
    const receipts = new Map<string, BackendRotation>();
    mockPost.mockImplementation(async (url: string, body: unknown, config?: { headers?: Record<string, string> }) => {
      expect(String(url)).toContain("/auth/refresh-token");
      const payload = body as { refresh_token?: string };
      const headers = config?.headers ?? {};
      const idem =
        headers["Idempotency-Key"] ||
        headers["idempotency-key"] ||
        (headers as { get?: (k: string) => string }).get?.("Idempotency-Key") ||
        "";
      seenIdemKeys.push(String(idem));
      expect(payload.refresh_token).toBe(R0);
      const receiptKey = `${payload.refresh_token}|${idem}`;
      const existing = receipts.get(receiptKey);
      if (existing) {
        return { data: { ...existing, error_code: "refresh_duplicate" } };
      }
      rotationCount += 1;
      const created: BackendRotation = {
        access_token: accessR1,
        refresh_token: R1,
        refresh_generation: genN + 1,
      };
      receipts.set(receiptKey, created);
      // Réponse HTTP perdue : serveur a commit la rotation, client ne reçoit rien
      // → aucune écriture SecureStore / envelope / clear pending.
      if (dropNextSuccessfulResponse) {
        dropNextSuccessfulResponse = false;
        const lost = new MockAxiosError("Network Error", "ERR_NETWORK");
        throw lost;
      }
      return { data: created };
    });

    const { setAuthToken, hasAuthToken, refreshAuthTokenNow, getLastRefreshErrorCode } =
      require("./client") as typeof import("./client");
    const {
      readPendingRefreshOperation,
      writePendingRefreshOperation,
    } = require("../auth/pendingRefreshOperation") as typeof import("../auth/pendingRefreshOperation");
    const {
      readRefreshToken,
      readSessionEnvelope,
      writeRefreshToken,
      writeSessionEnvelope,
    } = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");

    // Ré-écrire via le store (même chemin que prod) après resetModules
    await writeRefreshToken(R0);
    await writeSessionEnvelope({
      session_id: sessionId,
      session_epoch: 1,
      refresh_generation: genN,
      last_authenticated_at: new Date().toISOString(),
    });
    await writePendingRefreshOperation({
      operationId: OP1,
      sessionId,
      sourceRefreshGeneration: genN,
      createdAt: new Date().toISOString(),
    });

    // Session authentifiée avant le refresh (access en mémoire)
    setAuthToken(accessA);
    expect(hasAuthToken()).toBe(true);

    // 1) Première tentative : rotation serveur faite, réponse perdue avant apply client
    const first = await refreshAuthTokenNow({ force: true });
    expect(first).toBe(false);
    expect(mockPost).toHaveBeenCalled();
    expect(rotationCount).toBe(1);
    expect(seenIdemKeys[0]).toBe(OP1);
    expect(mockPost).toHaveBeenCalledTimes(1);

    // Après drop : SecureStore / pending / gen inchangés
    const storedAfterDrop = await readRefreshToken();
    expect(storedAfterDrop.status).toBe("found");
    if (storedAfterDrop.status === "found") {
      expect(storedAfterDrop.value).toBe(R0);
    }
    const pendingAfterDrop = await readPendingRefreshOperation();
    expect(pendingAfterDrop).not.toBeNull();
    expect(pendingAfterDrop?.operationId).toBe(OP1);
    expect(pendingAfterDrop?.sourceRefreshGeneration).toBe(genN);
    const envAfterDrop = await readSessionEnvelope();
    expect(envAfterDrop.status).toBe("found");
    if (envAfterDrop.status === "found") {
      expect(envAfterDrop.value.refresh_generation).toBe(genN);
    }
    expect(hasAuthToken()).toBe(true);
    expect(getLastRefreshErrorCode()).not.toBe("refresh_replay_detected");
    expect(mockSetItemAsync).not.toHaveBeenCalledWith(REFRESH_KEY, R1);

    // 2) Retry même R0 + OP1 → même R1 (receipt), pas de nouvelle rotation
    const second = await refreshAuthTokenNow({ force: true });
    expect(second).toBe(true);
    expect(rotationCount).toBe(1);
    expect(mockPost).toHaveBeenCalledTimes(2);

    const storedFinal = await readRefreshToken();
    expect(storedFinal.status).toBe("found");
    if (storedFinal.status === "found") {
      expect(storedFinal.value).toBe(R1);
    }
    const envFinal = await readSessionEnvelope();
    expect(envFinal.status).toBe("found");
    if (envFinal.status === "found") {
      expect(envFinal.value.refresh_generation).toBe(genN + 1);
    }
    expect(await readPendingRefreshOperation()).toBeNull();
    expect(hasAuthToken()).toBe(true);
    expect(getLastRefreshErrorCode()).toBeNull();
  });
});
