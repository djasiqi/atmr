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
  response?: { status?: number; data?: unknown };
  config?: unknown;

  constructor(message?: string, _code?: string, config?: unknown) {
    super(message);
    this.name = "AxiosError";
    this.config = config;
  }
}

jest.mock("axios", () => ({
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
  },
  AxiosError: MockAxiosError,
}));

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
    // eslint-disable-next-line @typescript-eslint/no-require-imports
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
    // eslint-disable-next-line @typescript-eslint/no-require-imports
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
});
