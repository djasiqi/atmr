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
jest.mock("../../../core/api/client", () => ({
  getAuthAccessToken: () => mockGetAccessToken(),
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
    delete process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL;
    delete process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL;
    delete process.env.EXPO_PUBLIC_API_BASE_URL;
  });

  it("stays idle when company realtime feature is disabled", () => {
    mockIsFeatureEnabled.mockImplementation((key: unknown) =>
      key === "company_realtime_enabled" ? false : true
    );
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");

    expect(companyRealtimeBridge.getSnapshot()).toEqual(
      expect.objectContaining({
        status: "idle",
        connected: false,
        contextId: "company:42",
      })
    );
    expect(mockIo).not.toHaveBeenCalled();
  });

  it("fails when access token is missing after wait window", () => {
    mockGetAccessToken.mockReturnValue(null);
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "wss://company.example.test";
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");
    jest.advanceTimersByTime(25_000);

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

  it("uses EXPO_PUBLIC_API_BASE_URL origin when no explicit socket URL", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_API_BASE_URL = "http://10.0.0.1:5000/api/v1";
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

  it("degrades after realtime silence and transitions on reconnect errors", () => {
    mockIsFeatureEnabled.mockReturnValue(true);
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL = "wss://company.example.test";
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { companyRealtimeBridge } = require("./companyRealtimeBridge");

    companyRealtimeBridge.connect("company:42");
    mockHandlers.get("connect")?.[0]?.();
    expect(companyRealtimeBridge.getSnapshot().status).toBe("healthy");

    jest.advanceTimersByTime(125_000);
    expect(companyRealtimeBridge.getSnapshot().status).toBe("degraded");

    mockHandlers.get("connect_error")?.[0]?.({ message: "network down" });
    expect(companyRealtimeBridge.getSnapshot().status).toBe("reconnecting");
    mockReconnectHandlers.get("reconnect_failed")?.[0]?.();
    expect(companyRealtimeBridge.getSnapshot().status).toBe("failed");
  });
});
