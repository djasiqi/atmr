import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockHandlers = new Map<string, ((payload?: unknown) => void)[]>();
const mockReconnectHandlers = new Map<string, (() => void)[]>();
const mockDispatch = jest.fn();
const mockSocketEmit = jest.fn();
const mockIsFeatureEnabled = jest.fn();

const mockSocket = {
  on: jest.fn((event: string, callback: (payload?: unknown) => void) => {
    const existing = mockHandlers.get(event) ?? [];
    existing.push(callback);
    mockHandlers.set(event, existing);
    return mockSocket;
  }),
  emit: jest.fn((event: string, payload?: unknown) => {
    mockSocketEmit(event, payload);
  }),
  removeAllListeners: jest.fn(() => {
    mockHandlers.clear();
  }),
  disconnect: jest.fn(),
  io: {
    on: jest.fn((event: string, callback: () => void) => {
      const existing = mockReconnectHandlers.get(event) ?? [];
      existing.push(callback);
      mockReconnectHandlers.set(event, existing);
    }),
  },
};

const mockIo = jest.fn((url?: string, options?: unknown) => {
  void url;
  void options;
  return mockSocket;
});

jest.mock("socket.io-client", () => ({
  io: (url: string, options?: unknown) => mockIo(url, options),
}));

jest.mock("../../../core/realtime/contextRealtimeRouter", () => ({
  contextRealtimeRouter: {
    dispatch: (...args: unknown[]) => mockDispatch(...args),
  },
}));

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (key: string) => mockIsFeatureEnabled(key),
}));

const mockGetAccessToken = jest.fn<() => string | null>(() => "test-jwt");
const mockRefreshAuthTokenNow = jest.fn(() => Promise.resolve(false));
const mockGetResolvedApiBaseUrl = jest.fn(() => "http://192.168.1.103:5000/api/v1");
jest.mock("../../../core/api/client", () => ({
  getAuthAccessToken: () => mockGetAccessToken(),
  refreshAuthTokenNow: () => mockRefreshAuthTokenNow(),
  getResolvedApiBaseUrl: () => mockGetResolvedApiBaseUrl(),
}));

describe("company realtime bridge", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.resetModules();
    mockHandlers.clear();
    mockReconnectHandlers.clear();
    mockDispatch.mockReset();
    mockSocketEmit.mockReset();
    mockIo.mockReset();
    mockIo.mockImplementation(() => mockSocket);
    mockSocket.on.mockClear();
    mockSocket.emit.mockClear();
    mockSocket.disconnect.mockClear();
    mockSocket.removeAllListeners.mockClear();
    mockSocket.io.on.mockClear();
    mockIsFeatureEnabled.mockReset();
    mockGetAccessToken.mockReturnValue("test-jwt");
    mockRefreshAuthTokenNow.mockReset();
    mockRefreshAuthTokenNow.mockImplementation(() => Promise.resolve(false));
    mockGetResolvedApiBaseUrl.mockReset();
    mockGetResolvedApiBaseUrl.mockReturnValue("http://192.168.1.103:5000/api/v1");
    delete process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL;
    delete process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL;
    delete process.env.EXPO_PUBLIC_API_BASE_URL;
  });

  it("marks failed when company realtime feature is disabled", () => {
    mockIsFeatureEnabled.mockImplementation((key: unknown) =>
      key === "company_realtime_enabled" ? false : true
    );
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");

    expect(companyRealtimeBridge.getSnapshot()).toEqual(
      expect.objectContaining({
        status: "failed",
        connected: false,
        contextId: "company:42",
      })
    );
    expect(mockIo).not.toHaveBeenCalled();
  });

  it("fails when access token is missing after wait window", async () => {
    mockGetAccessToken.mockReturnValue(null);
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "wss://company.example.test";
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");
    await jest.runAllTimersAsync();

    expect(companyRealtimeBridge.getSnapshot()).toEqual(
      expect.objectContaining({
        status: "failed",
        connected: false,
        contextId: "company:42",
      })
    );
    expect(mockIo).not.toHaveBeenCalled();
  });

  it("fails when socket URL is missing", () => {
    delete process.env.EXPO_PUBLIC_API_BASE_URL;
    mockGetResolvedApiBaseUrl.mockReturnValue("");
    mockIsFeatureEnabled.mockReturnValue(true);
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");

    expect(companyRealtimeBridge.getSnapshot()).toEqual(
      expect.objectContaining({
        status: "failed",
        connected: false,
      })
    );
    expect(mockIo).not.toHaveBeenCalled();
  });

  it("does not use DRIVER_SOCKET_URL as company socket (configure API origin or COMPANY_SOCKET_URL)", () => {
    delete process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL;
    delete process.env.EXPO_PUBLIC_API_BASE_URL;
    mockGetResolvedApiBaseUrl.mockReturnValue("");
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver-only.example.test";
    mockIsFeatureEnabled.mockReturnValue(true);
    const {
      companyRealtimeBridge,
      getResolvedCompanySocketEnvSource,
      getResolvedCompanySocketUrl,
    } = require("./companyRealtimeBridge");

    expect(getResolvedCompanySocketUrl()).toBe("");
    expect(getResolvedCompanySocketEnvSource()).toBe("none");

    companyRealtimeBridge.connect("company:42");

    expect(mockIo).not.toHaveBeenCalled();
    expect(companyRealtimeBridge.getSnapshot().status).toBe("failed");
  });

  it("uses EXPO_PUBLIC_API_BASE_URL origin when no explicit socket URL", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_API_BASE_URL = "http://10.0.0.1:5000/api/v1";
    mockGetResolvedApiBaseUrl.mockReturnValue("http://10.0.0.1:5000/api/v1");
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:9");

    expect(mockIo).toHaveBeenCalledWith("http://10.0.0.1:5000", expect.anything());
  });

  it("joins room, dispatches events and handles malformed payloads", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "wss://company.example.test";
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");
    expect(mockIo).toHaveBeenCalledTimes(1);

    const connectHandler = mockHandlers.get("connect")?.[0];
    const bookingUpdatedHandler = mockHandlers.get("booking_updated")?.[0];
    connectHandler?.();

    expect(mockSocketEmit).toHaveBeenCalledWith("join_company", undefined);

    bookingUpdatedHandler?.({ mission_id: 101 });
    expect(mockDispatch).toHaveBeenCalledWith(
      "company:42",
      expect.objectContaining({
        event_type: "booking_updated",
        mission_id: 101,
        context_type: "company",
      }),
      { contextType: "company" }
    );

    bookingUpdatedHandler?.("invalid_payload");
    expect(mockDispatch).toHaveBeenCalledTimes(1);
  });

  it("registers a single reconnect_attempt handler", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "wss://company.example.test";
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");
    expect(mockReconnectHandlers.get("reconnect_attempt")?.length).toBe(1);
  });

  it("does not notify listeners on high-frequency lastEventAt-only updates", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "wss://company.example.test";
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    const listener = jest.fn();
    companyRealtimeBridge.subscribe(listener);
    listener.mockClear();

    companyRealtimeBridge.connect("company:42");
    mockHandlers.get("connect")?.[0]?.();
    const callsAfterConnect = listener.mock.calls.length;

    const bookingUpdatedHandler = mockHandlers.get("booking_updated")?.[0];
    bookingUpdatedHandler?.({ mission_id: 1 });
    bookingUpdatedHandler?.({ mission_id: 2 });

    expect(listener.mock.calls.length).toBe(callsAfterConnect);
  });

  it("keeps transport healthy during silence but marks data freshness idle then failed on reconnect exhaustion", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "wss://company.example.test";
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");
    mockHandlers.get("connect")?.[0]?.();
    expect(companyRealtimeBridge.getSnapshot().transportStatus).toBe("healthy");

    jest.advanceTimersByTime(125_000);
    const afterSilence = companyRealtimeBridge.getSnapshot();
    expect(afterSilence.transportStatus).toBe("healthy");
    expect(afterSilence.dataFreshness).toBe("idle");

    mockHandlers.get("connect_error")?.[0]?.({ message: "network down" });
    expect(companyRealtimeBridge.getSnapshot().transportStatus).toBe("reconnecting");
    mockReconnectHandlers.get("reconnect_failed")?.[0]?.();
    expect(companyRealtimeBridge.getSnapshot().transportStatus).toBe("failed");
  });

  it("falls back to polling-only after a handshake timeout", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "http://company.example.test/socket";
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");

    expect(mockIo).toHaveBeenCalledTimes(1);
    expect(mockIo.mock.calls[0]?.[1]).toMatchObject({
      transports: ["websocket"],
    });

    mockHandlers.get("connect_error")?.[0]?.({ message: "timeout" });
    jest.runOnlyPendingTimers();

    expect(mockIo).toHaveBeenCalledTimes(2);
    expect(mockIo.mock.calls[1]?.[1]).toMatchObject({
      transports: ["polling"],
      upgrade: false,
    });
  });

  it("falls back to polling-only after a websocket handshake error", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "http://company.example.test/socket";
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");

    expect(mockIo).toHaveBeenCalledTimes(1);
    expect(mockIo.mock.calls[0]?.[1]).toMatchObject({
      transports: ["websocket"],
    });

    mockHandlers.get("connect_error")?.[0]?.({ message: "websocket error" });
    jest.runOnlyPendingTimers();

    expect(mockIo).toHaveBeenCalledTimes(2);
    expect(mockIo.mock.calls[1]?.[1]).toMatchObject({
      transports: ["polling"],
      upgrade: false,
    });
  });

  it("adds company_id to socket.io query when context is company:N", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "http://example.test";
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:7");
    expect(mockIo.mock.calls[0]?.[1]?.query).toEqual({
      context_id: "company:7",
      company_id: "7",
      surface: "company",
    });
  });

  it("getCompanyNumericIdFromContextId parses company: prefix", () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { getCompanyNumericIdFromContextId } = require("./companyRealtimeBridge");
    expect(getCompanyNumericIdFromContextId("company:1")).toBe("1");
    expect(getCompanyNumericIdFromContextId("  company:42  ")).toBe("42");
    expect(getCompanyNumericIdFromContextId("driver:1")).toBe("");
  });
});
