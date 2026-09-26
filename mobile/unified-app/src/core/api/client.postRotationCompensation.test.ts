/**
 * P0 — post-rotation compensation : NEW jamais écrasé par OLD après rotation serveur.
 * R1–R4
 */
import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockPost = jest.fn();
const mockGet = jest.fn();
const mockRequest = jest.fn();
const mockRequestUse = jest.fn();
const mockResponseUse = jest.fn();
const mockCommonHeaders: Record<string, unknown> = {};

const secureMem = new Map<string, string>();
const mockAsyncMem = new Map<string, string>();

const mockGetItemAsync = jest.fn(async (key: string) => secureMem.get(key) ?? null);
const mockSetItemAsync = jest.fn(async (key: string, value: string) => {
  secureMem.set(key, value);
});
const mockDeleteItemAsync = jest.fn(async (key: string) => {
  secureMem.delete(key);
});

class MockAxiosError extends Error {
  isAxiosError = true;
  response?: { status?: number; data?: unknown };
  constructor(message?: string) {
    super(message);
    this.name = "AxiosError";
  }
}

jest.mock("axios", () => {
  const isAxiosError = (error: unknown) =>
    Boolean(error && typeof error === "object" && (error as { isAxiosError?: boolean }).isAxiosError);
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
  default: { expoConfig: { extra: { apiBaseUrl: "https://api.test/api/v1" } } },
}));

jest.mock("expo-secure-store", () => ({
  AFTER_FIRST_UNLOCK: 0,
  getItemAsync: (...args: unknown[]) => mockGetItemAsync(...(args as [string])),
  setItemAsync: (...args: unknown[]) =>
    mockSetItemAsync(...(args as [string, string])),
  deleteItemAsync: (...args: unknown[]) => mockDeleteItemAsync(...(args as [string])),
}));

jest.mock("react-native", () => ({
  NativeModules: { SourceCode: { scriptURL: undefined } },
  Platform: { OS: "android" },
}));

jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: {
    getItem: jest.fn(async (key: string) => mockAsyncMem.get(key) ?? null),
    setItem: jest.fn(async (key: string, value: string) => {
      mockAsyncMem.set(key, value);
    }),
    removeItem: jest.fn(async (key: string) => {
      mockAsyncMem.delete(key);
    }),
  },
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

const OLD = "refresh-token-OLD";
const NEW = "refresh-token-NEW";

const ENVELOPE = {
  schema_version: 1,
  session_id: "sess-1",
  device_installation_id: "test-device-id",
  user_public_id: "u1",
  driver_id: null,
  role: "driver",
  active_context_id: null,
  refresh_generation: 1,
  last_authenticated_at: new Date().toISOString(),
  revocation_secret: "sec",
};

function seedOldSession() {
  secureMem.set("atmr.auth.refresh_token", OLD);
  secureMem.set("atmr.auth.recovery_credential", "recovery-OLD");
  secureMem.set("atmr.auth.installation_id", "test-device-id");
  secureMem.set("atmr.auth.session_envelope", JSON.stringify(ENVELOPE));
}

describe("P0 post-rotation compensation (R1–R4)", () => {
  beforeEach(() => {
    jest.resetModules();
    secureMem.clear();
    mockAsyncMem.clear();
    mockPost.mockReset();
    mockGet.mockReset();
    mockGetItemAsync.mockReset();
    mockSetItemAsync.mockReset();
    mockDeleteItemAsync.mockReset();
    mockGetItemAsync.mockImplementation(async (key: string) => secureMem.get(key) ?? null);
    mockSetItemAsync.mockImplementation(async (key: string, value: string) => {
      secureMem.set(key, value);
    });
    mockDeleteItemAsync.mockImplementation(async (key: string) => {
      secureMem.delete(key);
    });
    for (const key of Object.keys(mockCommonHeaders)) {
      delete mockCommonHeaders[key];
    }
  });

  it("R1 — envelope fail after NEW write : KEEP NEW, never rewrite OLD", async () => {
    seedOldSession();
    mockPost.mockResolvedValue({
      data: {
        access_token: "access-NEW",
        refresh_token: NEW,
        refresh_generation: 2,
      },
    });
    const strictRefreshWrites: string[] = [];
    mockSetItemAsync.mockImplementation(async (key: string, value: string) => {
      if (key === "atmr.auth.refresh_token") {
        strictRefreshWrites.push(value);
      }
      if (key === "atmr.auth.session_envelope") {
        throw new Error("envelope_persist_fail");
      }
      secureMem.set(key, value);
    });


    const { refreshAuthTokenNow, hasAuthToken } = require("./client") as typeof import("./client");
    const ok = await refreshAuthTokenNow({ force: true });

    expect(ok).toBe(false);
    expect(secureMem.get("atmr.auth.refresh_token")).toBe(NEW);
    expect(secureMem.get("atmr.auth.refresh_token")).not.toBe(OLD);
    expect(strictRefreshWrites.includes(OLD)).toBe(false);
    expect(strictRefreshWrites.includes(NEW)).toBe(true);
    expect(hasAuthToken()).toBe(false);
    expect(secureMem.get("atmr.auth.session_envelope")).toBeUndefined();
  });

  it("R3 spy — writeRefreshToken(OLD) call count = 0 après NEW (via store spy)", async () => {
    seedOldSession();
    mockPost.mockResolvedValue({
      data: { access_token: "a", refresh_token: NEW, refresh_generation: 2 },
    });
    mockSetItemAsync.mockImplementation(async (key: string, value: string) => {
      if (key === "atmr.auth.session_envelope") {
        throw new Error("envelope_fail");
      }
      secureMem.set(key, value);
    });

    const store = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
    const spy = jest.spyOn(store, "writeRefreshToken");

    const { refreshAuthTokenNow } = require("./client") as typeof import("./client");
    await refreshAuthTokenNow({ force: true });

    const oldRewrites = spy.mock.calls.filter((args) => args[0] === OLD);
    expect(oldRewrites.length).toBe(0);
    const newWrites = spy.mock.calls.filter((args) => args[0] === NEW);
    expect(newWrites.length).toBeGreaterThanOrEqual(1);
    spy.mockRestore();
  });

  it("R2 — session-resume envelope fail : KEEP NEW, never restore OLD", async () => {
    seedOldSession();
    mockPost.mockResolvedValue({
      data: {
        access_token: "access-resume",
        refresh_token: NEW,
        recovery_credential: "recovery-NEW",
        session_id: "sess-1",
        credential_generation: 2,
        refresh_generation: 2,
      },
    });
    mockSetItemAsync.mockImplementation(async (key: string, value: string) => {
      if (key === "atmr.auth.session_envelope") {
        throw new Error("envelope_fail");
      }
      secureMem.set(key, value);
    });

    const store = require("../auth/authCredentialStore") as typeof import("../auth/authCredentialStore");
    const spy = jest.spyOn(store, "writeRefreshToken");
    const { sessionResumeRequest, hasAuthToken } = require("./client") as typeof import("./client");

    const result = await sessionResumeRequest();
    expect(result.ok).toBe(false);
    expect(result.code).toBe("storage_unavailable");
    expect(secureMem.get("atmr.auth.refresh_token")).toBe(NEW);
    expect(secureMem.get("atmr.auth.refresh_token")).not.toBe(OLD);
    expect(spy.mock.calls.filter((a) => a[0] === OLD).length).toBe(0);
    expect(hasAuthToken()).toBe(false);
    expect(secureMem.get("atmr.auth.session_envelope")).toBeUndefined();
    spy.mockRestore();
  });

  it("R4 — happy path NEW + envelope OK → authenticated", async () => {
    seedOldSession();
    mockPost.mockResolvedValue({
      data: {
        access_token: "access-NEW",
        refresh_token: NEW,
        refresh_generation: 2,
      },
    });

    const { refreshAuthTokenNow, hasAuthToken } = require("./client") as typeof import("./client");
    const ok = await refreshAuthTokenNow({ force: true });
    expect(ok).toBe(true);
    expect(secureMem.get("atmr.auth.refresh_token")).toBe(NEW);
    expect(hasAuthToken()).toBe(true);
    const envRaw = secureMem.get("atmr.auth.session_envelope");
    expect(envRaw).toBeTruthy();
    const env = JSON.parse(envRaw!);
    expect(env.refresh_generation).toBe(2);
  });
});
