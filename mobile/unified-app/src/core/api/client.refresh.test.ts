import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockPost = jest.fn();
const mockGet = jest.fn();
const mockRequest = jest.fn();
const mockRequestUse = jest.fn();
const mockResponseUse = jest.fn();
const mockCommonHeaders: Record<string, unknown> = {};
const mockGetItemAsync = jest.fn();
const mockSetItemAsync = jest.fn();
const mockDeleteItemAsync = jest.fn();

class MockAxiosError extends Error {
  isAxiosError = true;
  response?: { status?: number; data?: unknown };
  config?: unknown;

  constructor(message?: string, _code?: string, config?: unknown) {
    super(message);
    this.name = "AxiosError";
    this.config = config;
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
  getItemAsync: (...args: unknown[]) => mockGetItemAsync(...args),
  setItemAsync: (...args: unknown[]) => mockSetItemAsync(...args),
  deleteItemAsync: (...args: unknown[]) => mockDeleteItemAsync(...args),
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

describe("refreshAuthTokenNow", () => {
  beforeEach(() => {
    jest.resetModules();
    mockPost.mockReset();
    mockGet.mockReset();
    mockRequest.mockReset();
    mockRequestUse.mockReset();
    mockResponseUse.mockReset();
    mockGetItemAsync.mockReset();
    mockSetItemAsync.mockReset();
    mockDeleteItemAsync.mockReset();
    for (const key of Object.keys(mockCommonHeaders)) {
      delete mockCommonHeaders[key];
    }
  });

  it("returns false and skips network call when no refresh token exists", async () => {
    mockGetItemAsync.mockResolvedValue(null);
     
    const { refreshAuthTokenNow } = require("./client");

    const refreshed = await refreshAuthTokenNow();

    expect(refreshed).toBe(false);
    expect(mockPost).not.toHaveBeenCalled();
  });

  it("uses singleflight for concurrent refresh calls", async () => {
    mockGetItemAsync.mockResolvedValue("refresh-token-initial");
    mockPost.mockResolvedValue({
      data: {
        access_token: "access-token-next",
        refresh_token: "refresh-token-next",
      },
    });
    mockGetItemAsync.mockImplementation(async () => "refresh-token-next");
     
    const { hasAuthToken, refreshAuthTokenNow } = require("./client");

    const first = refreshAuthTokenNow();
    const second = refreshAuthTokenNow();

    const [firstResult, secondResult] = await Promise.all([first, second]);

    expect(mockPost).toHaveBeenCalledTimes(1);
    expect(firstResult).toBe(true);
    expect(secondResult).toBe(true);
    expect(mockSetItemAsync).toHaveBeenCalledWith("auth_refresh_token", "refresh-token-next");
    expect(hasAuthToken()).toBe(true);
  });

  it("never replays a refresh token rejected 401 (P1-C2 terminal gate)", async () => {
    mockGetItemAsync.mockImplementation(async () => "revoked-token");
    const rejection = new MockAxiosError("Request failed with status code 401");
    rejection.response = {
      status: 401,
      data: { error: "Refresh token invalide" },
    };
    mockPost.mockRejectedValue(rejection);
     
    const { getLastRefreshErrorCode, refreshAuthTokenNow } = require("./client");

    const first = await refreshAuthTokenNow();
    expect(first).toBe(false);
    expect(mockPost).toHaveBeenCalledTimes(1);

    const second = await refreshAuthTokenNow();
    const third = await refreshAuthTokenNow();
    expect(second).toBe(false);
    expect(third).toBe(false);
    // Porte terminale : aucun rejeu reseau du meme token rejete.
    expect(mockPost).toHaveBeenCalledTimes(1);
    expect(getLastRefreshErrorCode()).toBe("Refresh token invalide");
  });

  it("does_not_terminalize_generic_403 (CSRF-style): retry allowed", async () => {
    mockGetItemAsync.mockImplementation(async () => "csrf-blocked-token");
    const rejection = new MockAxiosError("Request failed with status code 403");
    rejection.response = {
      status: 403,
      data: { error: "Token CSRF invalide" },
    };
    mockPost.mockRejectedValue(rejection);
     
    const { refreshAuthTokenNow } = require("./client");

    expect(await refreshAuthTokenNow()).toBe(false);
    expect(await refreshAuthTokenNow()).toBe(false);
    // 403 generique (sans error_code terminal) : PAS de porte, retry autorise.
    expect(mockPost).toHaveBeenCalledTimes(2);
  });

  it("does not gate transient failures (503) - retry allowed", async () => {
    mockGetItemAsync.mockImplementation(async () => "flaky-token");
    const rejection = new MockAxiosError("Request failed with status code 503");
    rejection.response = { status: 503, data: { error: "store_unavailable" } };
    mockPost.mockRejectedValue(rejection);
     
    const { refreshAuthTokenNow } = require("./client");

    expect(await refreshAuthTokenNow()).toBe(false);
    expect(await refreshAuthTokenNow()).toBe(false);
    // Echec transitoire : chaque tentative repart sur le reseau.
    expect(mockPost).toHaveBeenCalledTimes(2);
  });

  it("lifts the terminal gate when the stored refresh token changes", async () => {
    let storedToken = "revoked-token";
    mockGetItemAsync.mockImplementation(async () => storedToken);
    const rejection = new MockAxiosError("Request failed with status code 401");
    rejection.response = {
      status: 401,
      data: { error_code: "session_revoked", error: "session_revoked" },
    };
    mockPost.mockRejectedValue(rejection);
     
    const { refreshAuthTokenNow } = require("./client");

    expect(await refreshAuthTokenNow()).toBe(false);
    expect(await refreshAuthTokenNow()).toBe(false);
    expect(mockPost).toHaveBeenCalledTimes(1);

    // Re-login : nouveau refresh token stocke -> porte levee, reseau reautorise.
    storedToken = "fresh-token-after-login";
    mockPost.mockResolvedValue({
      data: {
        access_token: "access-token-next",
        refresh_token: "fresh-token-after-login",
      },
    });

    expect(await refreshAuthTokenNow()).toBe(true);
    expect(mockPost).toHaveBeenCalledTimes(2);
  });

  it("AUTH-01: 20 concurrent refresh callers produce a single POST", async () => {
    mockGetItemAsync.mockResolvedValue("refresh-token-shared");
    mockPost.mockResolvedValue({
      data: {
        access_token: "access-token-shared",
        refresh_token: "refresh-token-shared-next",
      },
    });
    mockGetItemAsync.mockImplementation(async () => "refresh-token-shared-next");
     
    const { refreshAuthTokenNow } = require("./client");
    const results = await Promise.all(
      Array.from({ length: 20 }, () => refreshAuthTokenNow({ force: true }))
    );
    expect(mockPost).toHaveBeenCalledTimes(1);
    expect(results.every((ok) => ok === true)).toBe(true);
  });

  it("AUTH-02: after refresh 200, no spontaneous second refresh", async () => {
    mockGetItemAsync.mockResolvedValue("refresh-token-ok");
    mockPost.mockResolvedValue({
      data: {
        access_token: "access-token-ok",
        refresh_token: "refresh-token-ok-2",
      },
    });
    mockGetItemAsync.mockImplementation(async () => "refresh-token-ok-2");
     
    const { refreshAuthTokenNow } = require("./client");
    expect(await refreshAuthTokenNow({ force: true })).toBe(true);
    expect(mockPost).toHaveBeenCalledTimes(1);
    // Appels proactifs (socket / recovery) : cooldown post-succès.
    expect(await refreshAuthTokenNow()).toBe(true);
    expect(await refreshAuthTokenNow()).toBe(true);
    expect(mockPost).toHaveBeenCalledTimes(1);
  });

  it("AUTH-02b: post-bootstrap skip blocks proactive refresh", async () => {
    mockGetItemAsync.mockImplementation(async () => "refresh-token-boot");
    mockPost.mockResolvedValue({
      data: {
        access_token: "access-after-login",
        refresh_token: "refresh-token-boot-2",
      },
    });
     
    const { markBootstrapAuthFresh, refreshAuthTokenNow, setAuthToken } = require("./client");
    setAuthToken("access-fresh-from-login");
    markBootstrapAuthFresh();
    expect(await refreshAuthTokenNow()).toBe(true);
    expect(mockPost).not.toHaveBeenCalled();
    // force (401) reste autorisé.
    mockGetItemAsync.mockImplementation(async () => "refresh-token-boot-2");
    expect(await refreshAuthTokenNow({ force: true })).toBe(true);
    expect(mockPost).toHaveBeenCalledTimes(1);
  });

  it("AUTH-03: 429 does not immediately retry refresh", async () => {
    mockGetItemAsync.mockImplementation(async () => "refresh-token-rl");
    const rejection = new MockAxiosError("Request failed with status code 429");
    rejection.response = {
      status: 429,
      data: { error: "rate_limited" },
      headers: { "retry-after": "30" },
    } as { status?: number; data?: unknown; headers?: Record<string, string> };
    mockPost.mockRejectedValue(rejection);
     
    const { refreshAuthTokenNow } = require("./client");
    expect(await refreshAuthTokenNow({ force: true })).toBe(false);
    expect(mockPost).toHaveBeenCalledTimes(1);
    expect(await refreshAuthTokenNow({ force: true })).toBe(false);
    expect(await refreshAuthTokenNow()).toBe(false);
    expect(mockPost).toHaveBeenCalledTimes(1);
  });

  it("AUTH-04: response interceptor does not refresh on refresh-token 401", async () => {
    mockGetItemAsync.mockImplementation(async () => "refresh-token-x");
     
    require("./client");
    expect(mockResponseUse).toHaveBeenCalled();
    const onRejected = mockResponseUse.mock.calls[0]?.[1] as
      | ((error: MockAxiosError) => Promise<unknown>)
      | undefined;
    expect(typeof onRejected).toBe("function");

    const err = new MockAxiosError("Request failed with status code 401");
    err.response = { status: 401, data: { error: "invalid" } };
    err.config = { url: "/auth/refresh-token", headers: {} };

    await expect(onRejected!(err)).rejects.toBe(err);
    expect(mockPost).not.toHaveBeenCalled();

    mockPost.mockResolvedValue({
      data: { access_token: "a", refresh_token: "b" },
    });
    mockGetItemAsync.mockImplementation(async () => "b");
    const { refreshAuthTokenNow } = require("./client");
    expect(await refreshAuthTokenNow({ force: true })).toBe(true);
    expect(mockPost).toHaveBeenCalledTimes(1);
  });
});
