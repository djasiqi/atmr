import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockPost = jest.fn();
const mockGet = jest.fn();
const mockRequest = jest.fn();
const mockRequestUse = jest.fn();
const mockResponseUse = jest.fn();
const mockCommonHeaders: Record<string, unknown> = {};
const mockGetStableDeviceId = jest.fn();
const mockWriteRefreshToken = jest.fn();
const mockWriteRecoveryCredential = jest.fn();
const mockWriteSessionEnvelope = jest.fn();
const mockDeleteRefreshToken = jest.fn();
const mockDeleteRecoveryCredential = jest.fn();
const mockBumpAuthEpoch = jest.fn();
const mockHasPendingRevocationTombstone = jest.fn();

class MockAxiosError extends Error {
  isAxiosError = true;
  response?: { status?: number; data?: unknown };
  request?: unknown;
  config?: { url?: string; baseURL?: string };
  code?: string;

  constructor(message?: string, code?: string) {
    super(message);
    this.name = "AxiosError";
    this.code = code;
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
  default: {
    expoConfig: {
      extra: {
        apiBaseUrl: "https://api.test/api/v1",
      },
    },
  },
}));

jest.mock("expo-secure-store", () => ({
  getItemAsync: jest.fn(),
  setItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
}));

jest.mock("react-native", () => ({
  NativeModules: { SourceCode: { scriptURL: undefined } },
  Platform: { OS: "android" },
}));

jest.mock("../observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
}));

jest.mock("../observability/sessionJournal", () => ({
  buildSessionDiagHeader: () => "diag-test",
  appendSessionJournalEvent: jest.fn(),
}));

jest.mock("../notifications/getStableDeviceId", () => ({
  getStableDeviceId: (...args: unknown[]) => mockGetStableDeviceId(...args),
}));

jest.mock("expo-application", () => ({
  applicationName: "LirieTest",
}));

jest.mock("../auth/authCredentialStore", () => ({
  writeRefreshToken: (...args: unknown[]) => mockWriteRefreshToken(...args),
  writeRecoveryCredential: (...args: unknown[]) => mockWriteRecoveryCredential(...args),
  writeSessionEnvelope: (...args: unknown[]) => mockWriteSessionEnvelope(...args),
  deleteRefreshToken: (...args: unknown[]) => mockDeleteRefreshToken(...args),
  deleteRecoveryCredential: (...args: unknown[]) => mockDeleteRecoveryCredential(...args),
  bumpAuthEpoch: (...args: unknown[]) => mockBumpAuthEpoch(...args),
  bumpSessionGeneration: (...args: unknown[]) => mockBumpAuthEpoch(...args),
  getAuthEpoch: () => 1,
  getSessionGenerationId: () => 1,
  isCurrentAuthEpoch: () => true,
  isCurrentSessionGeneration: () => true,
  appendPendingRevocation: jest.fn(async () => ({ status: "ok" })),
  readSessionEnvelope: jest.fn(),
  readRefreshToken: jest.fn(),
  writeRevocationTombstone: jest.fn(),
  deleteRevocationTombstone: jest.fn(),
  clearLocalAuthCredentialsLocked: jest.fn(async () => undefined),
}));

jest.mock("../auth/sessionCredentialMutex", () => ({
  withCredentialStoreLock: async <T,>(fn: () => Promise<T> | T) => fn(),
  withSessionCredentialMutation: async <T,>(
    _gen: number,
    fn: () => Promise<T> | T
  ) => ({ status: "applied" as const, value: await fn() }),
  claimNextSessionGenerationIfCurrent: async () => ({
    status: "claimed" as const,
    generation: 2,
  }),
}));

jest.mock("../auth/authRecoveryCoordinator", () => ({
  hasPendingRevocationTombstone: (...args: unknown[]) =>
    mockHasPendingRevocationTombstone(...args),
  flushPendingRevocationTombstone: jest.fn(async () => true),
  enqueueOrphanedLoginRevocation: jest.fn(async (args: unknown) => args),
  flushOrphanedLoginRevocationInBackground: jest.fn(),
}));

jest.mock("../auth/jwtClaims", () => ({
  decodeJwtClaims: () => ({ driver_id: 42 }),
}));

jest.mock("../featureFlags/registry", () => ({
  getRuntimeFlagsVersion: () => 1,
  isFeatureEnabled: () => false,
}));

jest.mock("../network/networkState", () => ({
  getNetworkSnapshot: () => ({ isConnected: true, isInternetReachable: true }),
}));

jest.mock("../network/connectivityPolicy", () => ({
  evaluateConnectivityPolicy: () => ({ allowRequest: true }),
}));

describe("login contrat session durable P0", () => {
  beforeEach(() => {
    jest.resetModules();
    mockPost.mockReset();
    mockGetStableDeviceId.mockReset();
    mockWriteRefreshToken.mockReset();
    mockWriteRecoveryCredential.mockReset();
    mockWriteSessionEnvelope.mockReset();
    mockDeleteRefreshToken.mockReset();
    mockDeleteRecoveryCredential.mockReset();
    mockBumpAuthEpoch.mockReset();
    mockHasPendingRevocationTombstone.mockReset();
    mockHasPendingRevocationTombstone.mockResolvedValue(false);
    mockGetStableDeviceId.mockResolvedValue("stable-device-id");
    mockWriteRefreshToken.mockResolvedValue({ status: "ok" });
    mockWriteRecoveryCredential.mockResolvedValue({ status: "ok" });
    mockWriteSessionEnvelope.mockResolvedValue({ status: "ok" });
    mockDeleteRefreshToken.mockResolvedValue({ status: "ok" });
    mockDeleteRecoveryCredential.mockResolvedValue({ status: "ok" });
  });

  it("n'envoie pas POST /auth/login si getStableDeviceId échoue", async () => {
    mockGetStableDeviceId.mockRejectedValue(new Error("device_identity_storage_unavailable"));
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { login } = require("./client") as typeof import("./client");
    await expect(login("a@b.ch", "x")).rejects.toMatchObject({
      code: "DEVICE_ID_UNAVAILABLE",
      status: null,
    });
    expect(mockPost).not.toHaveBeenCalled();
  });

  it("erreur locale sans texte VPN/DNS/TLS", async () => {
    mockGetStableDeviceId.mockRejectedValue(new Error("device_identity_storage_unavailable"));
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { login } = require("./client") as typeof import("./client");
    try {
      await login("a@b.ch", "x");
      throw new Error("expected reject");
    } catch (err) {
      const message = String((err as { message?: string }).message ?? "");
      expect(message).not.toMatch(/VPN|DNS|TLS|trame HTTP/i);
      expect((err as { code?: string }).code).toBe("DEVICE_ID_UNAVAILABLE");
    }
  });

  it("erreur Axios transport affiche le diagnostic réseau", async () => {
    mockPost.mockRejectedValue(
      Object.assign(new MockAxiosError("Network Error", "ERR_NETWORK"), {
        request: {},
        config: { baseURL: "https://api.test/api/v1", url: "/auth/login" },
      })
    );
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { login } = require("./client") as typeof import("./client");
    try {
      await login("a@b.ch", "x");
      throw new Error("expected reject");
    } catch (err) {
      const message = String((err as { message?: string }).message ?? "");
      expect(message).toMatch(/Pas de trame HTTP|VPN|DNS|TLS/i);
      expect(message).toContain("URL=");
    }
  });

  it("réponse 200 incomplète → AUTH_LOGIN_CONTRACT_INCOMPLETE", async () => {
    mockPost.mockResolvedValue({
      data: {
        access_token: "access-only",
        refresh_token: "refresh-token",
        // recovery_credential manquant
      },
    });
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { login } = require("./client") as typeof import("./client");
    try {
      await login("a@b.ch", "x");
      throw new Error("expected reject");
    } catch (err) {
      expect(err).toMatchObject({
        code: "AUTH_LOGIN_CONTRACT_INCOMPLETE",
        status: null,
      });
      const message = String((err as { message?: string }).message ?? "");
      expect(message).not.toMatch(/VPN|DNS|TLS|trame HTTP/i);
    }
  });

  it("échec écriture SecureStore → STORAGE_UNAVAILABLE", async () => {
    mockPost.mockResolvedValue({
      data: {
        access_token: "access",
        refresh_token: "refresh",
        recovery_credential: "recovery",
        revocation_secret: "revocation",
        session_id: "session-1",
        refresh_generation: 1,
        user: { public_id: "u1", role: "driver" },
      },
    });
    mockWriteRefreshToken.mockResolvedValue({ status: "temporarily_unavailable" });
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { login } = require("./client") as typeof import("./client");
    await expect(login("a@b.ch", "x")).rejects.toMatchObject({
      code: "STORAGE_UNAVAILABLE",
      status: null,
    });
  });

  it("login complet persiste avant publication access", async () => {
    mockPost.mockResolvedValue({
      data: {
        access_token: "access",
        refresh_token: "refresh",
        recovery_credential: "recovery",
        revocation_secret: "revocation",
        session_id: "session-1",
        refresh_generation: 1,
        user: { public_id: "u1", role: "driver" },
      },
    });
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { login } = require("./client") as typeof import("./client");
    await login("a@b.ch", "x");
    expect(mockPost).toHaveBeenCalledWith(
      "/auth/login",
      { email: "a@b.ch", password: "x" },
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-Device-ID": "stable-device-id",
          "X-Auth-Contract-Version": "mobile-device-session-v1",
        }),
      })
    );
    expect(mockWriteRefreshToken).toHaveBeenCalled();
    expect(mockWriteRecoveryCredential).toHaveBeenCalled();
    expect(mockWriteSessionEnvelope).toHaveBeenCalled();
    expect(mockBumpAuthEpoch).toHaveBeenCalled();
  });
});
