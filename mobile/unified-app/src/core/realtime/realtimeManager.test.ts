import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { realtimeManager } from "./realtimeManager";

const mockHandlers = new Map<string, ((payload?: unknown) => void)[]>();
const mockEmitDriverTelemetry = jest.fn();
const mockSocket = {
  on: jest.fn((event: string, callback: (payload?: unknown) => void) => {
    const existing = mockHandlers.get(event) ?? [];
    existing.push(callback);
    mockHandlers.set(event, existing);
    return mockSocket;
  }),
  removeAllListeners: jest.fn(() => {
    mockHandlers.clear();
  }),
  disconnect: jest.fn(),
};
const mockIo = jest.fn(() => mockSocket) as jest.MockedFunction<(...args: any[]) => typeof mockSocket>;

jest.mock("socket.io-client", () => ({
  // `socket.io-client` typings are tuple-heavy; avoid rest/spread in the mock factory for `tsc`.
  io: (url: string, opts?: unknown) => mockIo(url, opts),
}));

jest.mock("../observability/driverTelemetry", () => ({
  emitDriverTelemetry: (event: string, payload?: unknown) => mockEmitDriverTelemetry(event, payload),
}));

// Helper — renregistre les handlers après chaque reconnexion simulée
function rebindSocketHandlers() {
  mockSocket.on.mockImplementation((event: string, cb: (p?: unknown) => void) => {
    const existing = mockHandlers.get(event) ?? [];
    existing.push(cb);
    mockHandlers.set(event, existing);
    return mockSocket;
  });
}

function fireConnectError(message: string, code?: string) {
  const handler = mockHandlers.get("connect_error")?.[0];
  handler?.({ message, data: code ? { code } : undefined } as unknown as Error);
}

const IDLE_SNAPSHOT = {
  activeContextId: null,
  connected: false,
  mode: "idle",
  desiredTransport: "polling",
  actualTransport: "idle",
  lastEventAt: null,
  lastError: null,
  reconnectAttempts: 0,
  reconnectBackoffMs: 0,
  authAttempts: 0,
  authExhausted: false,
  authErrorCode: null,
  transportAuthority: "polling",
  degradedMode: false,
  degradedModeSince: null,
  reconnectWindowStartedAtMs: null,
  reconnectWindowAttempts: 0,
};

describe("realtime manager", () => {
  beforeEach(() => {
    mockHandlers.clear();
    mockIo.mockClear();
    mockSocket.on.mockClear();
    mockSocket.disconnect.mockClear();
    mockSocket.removeAllListeners.mockClear();
    mockEmitDriverTelemetry.mockClear();
    realtimeManager.disconnect();
    delete process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL;
    delete process.env.EXPO_PUBLIC_REALTIME_DEGRADED_HYSTERESIS_MS;
  });

  it("connects and switches context safely", () => {
    realtimeManager.connect("client:self");
    expect(realtimeManager.getSnapshot()).toEqual({
      activeContextId: "client:self",
      connected: true,
      mode: "polling",
      desiredTransport: "polling",
      actualTransport: "polling",
      lastEventAt: null,
      lastError: null,
      reconnectAttempts: 0,
      reconnectBackoffMs: 0,
      authAttempts: 0,
      authExhausted: false,
      authErrorCode: null,
      transportAuthority: "polling",
      degradedMode: false,
      degradedModeSince: null,
      reconnectWindowStartedAtMs: null,
      reconnectWindowAttempts: 0,
    });

    realtimeManager.onContextSwitch("driver:42");
    expect(realtimeManager.getSnapshot()).toEqual({
      activeContextId: "driver:42",
      connected: true,
      mode: "polling",
      desiredTransport: "polling",
      actualTransport: "polling",
      lastEventAt: null,
      lastError: null,
      reconnectAttempts: 0,
      reconnectBackoffMs: 0,
      authAttempts: 0,
      authExhausted: false,
      authErrorCode: null,
      transportAuthority: "polling",
      degradedMode: false,
      degradedModeSince: null,
      reconnectWindowStartedAtMs: null,
      reconnectWindowAttempts: 0,
    });
  });

  it("falls back to polling when socket url is not configured", () => {
    realtimeManager.connect("driver:42", { enableSocket: true });
    expect(mockIo).not.toHaveBeenCalled();
    expect(realtimeManager.getSnapshot()).toEqual({
      activeContextId: "driver:42",
      connected: true,
      mode: "polling",
      desiredTransport: "socket",
      actualTransport: "polling",
      lastEventAt: null,
      lastError: "EXPO_PUBLIC_DRIVER_SOCKET_URL not configured",
      reconnectAttempts: 0,
      reconnectBackoffMs: 0,
      authAttempts: 0,
      authExhausted: false,
      authErrorCode: null,
      transportAuthority: "polling",
      degradedMode: false,
      degradedModeSince: expect.any(String),
      reconnectWindowStartedAtMs: null,
      reconnectWindowAttempts: 0,
    });
  });

  it("sets degraded after hysteresis when socket url is missing and socket is never established", () => {
    jest.useFakeTimers();
    // DEGRADED_HYSTERESIS_MS est figé à l’import (défaut 8000) — avancer 8000 ms + marge
    realtimeManager.connect("driver:42", { enableSocket: true });
    expect(mockIo).not.toHaveBeenCalled();
    expect(realtimeManager.getSnapshot().degradedMode).toBe(false);
    expect(realtimeManager.getSnapshot().transportAuthority).toBe("polling");
    jest.advanceTimersByTime(8000);
    const snap = realtimeManager.getSnapshot();
    expect(snap.degradedMode).toBe(true);
    expect(snap.transportAuthority).toBe("degraded");
    jest.useRealTimers();
  });

  it("emits driver events and schedules reconnect after non-auth socket errors", () => {
    jest.useFakeTimers();
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    const receivedEvents: unknown[] = [];
    const unsubscribe = realtimeManager.subscribeDriverEvents((event) => {
      receivedEvents.push(event);
    });

    realtimeManager.connect("driver:42", { enableSocket: true });
    expect(mockIo).toHaveBeenCalledTimes(1);

    const connectError = mockHandlers.get("connect_error")?.[0];
    const missionEvent = mockHandlers.get("driver_mission_event")?.[0];
    expect(connectError).toBeDefined();
    expect(missionEvent).toBeDefined();

    connectError?.({ message: "socket down" } as unknown as Error);
    expect(realtimeManager.getSnapshot().mode).toBe("polling");
    expect(realtimeManager.getSnapshot().lastError).toBe("socket down");
    expect(realtimeManager.getSnapshot().authAttempts).toBe(0);
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "realtime.socket.disconnect",
      expect.objectContaining({ source: "core.realtime.manager", reason: "socket down" })
    );

    missionEvent?.({ mission_id: 101, event_type: "mission_updated" });
    expect(receivedEvents).toEqual([{ mission_id: 101, event_type: "mission_updated" }]);
    expect(realtimeManager.getSnapshot().lastEventAt).not.toBeNull();

    rebindSocketHandlers();
    jest.advanceTimersByTime(5000);
    expect(mockIo).toHaveBeenCalledTimes(2);
    const reconnect = mockHandlers.get("connect")?.[0];
    reconnect?.();
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "realtime.socket.reconnect",
      expect.objectContaining({ source: "core.realtime.manager" })
    );

    unsubscribe();
    jest.useRealTimers();
  });

  it("disconnects on logout", () => {
    realtimeManager.connect("driver:42");
    realtimeManager.disconnect();
    expect(realtimeManager.getSnapshot()).toEqual(IDLE_SNAPSHOT);
  });

  // ─── Auth recovery ────────────────────────────────────────────────────────

  it("increments authAttempts on 401 without triggering exhaustion early", () => {
    jest.useFakeTimers();
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });

    fireConnectError("401 Unauthorized");

    expect(realtimeManager.getSnapshot().authAttempts).toBe(1);
    expect(realtimeManager.getSnapshot().authExhausted).toBe(false);
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "realtime.auth.retry",
      expect.objectContaining({ retry_count: 1, terminal: false })
    );

    jest.useRealTimers();
  });

  it("sets authExhausted after 5 auth failures and fires onAuthExhausted", () => {
    jest.useFakeTimers();
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });

    const exhaustedCb = jest.fn();
    const unsub = realtimeManager.onAuthExhausted(exhaustedCb);

    for (let i = 0; i < 5; i++) {
      fireConnectError("401 Unauthorized");
      // Avance le timer pour permettre la reconnexion avant le prochain cycle
      mockHandlers.delete("connect_error");
      rebindSocketHandlers();
      jest.advanceTimersByTime(60_000);
    }

    expect(realtimeManager.getSnapshot().authExhausted).toBe(true);
    expect(exhaustedCb).toHaveBeenCalledWith("exhausted", undefined);

    unsub();
    jest.useRealTimers();
  });

  it("exhausts immediately on terminal code without scheduling reconnect", () => {
    jest.useFakeTimers();
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });

    const exhaustedCb = jest.fn();
    const unsub = realtimeManager.onAuthExhausted(exhaustedCb);

    fireConnectError("Forbidden", "session_revoked");

    expect(realtimeManager.getSnapshot().authExhausted).toBe(true);
    expect(exhaustedCb).toHaveBeenCalledWith("terminal", "session_revoked");
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "realtime.auth.retry",
      expect.objectContaining({ terminal: true, error_code: "session_revoked" })
    );

    // Aucun reconnect ne doit être planifié après un code terminal
    jest.advanceTimersByTime(60_000);
    expect(mockIo).toHaveBeenCalledTimes(1);

    unsub();
    jest.useRealTimers();
  });

  it("resets authAttempts and authExhausted on successful connect", () => {
    jest.useFakeTimers();
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });

    fireConnectError("401 Unauthorized");
    expect(realtimeManager.getSnapshot().authAttempts).toBe(1);

    rebindSocketHandlers();
    jest.advanceTimersByTime(5000);
    const connectHandler = mockHandlers.get("connect")?.[0];
    connectHandler?.();

    expect(realtimeManager.getSnapshot().authAttempts).toBe(0);
    expect(realtimeManager.getSnapshot().authExhausted).toBe(false);

    jest.useRealTimers();
  });

  it("onAuthExhausted can be unsubscribed before exhaustion", () => {
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });

    const cb = jest.fn();
    const unsub = realtimeManager.onAuthExhausted(cb);
    unsub();

    fireConnectError("Forbidden", "account_disabled");
    expect(cb).not.toHaveBeenCalled();
  });

  it("caps reconnect storms within a fixed window", () => {
    jest.useFakeTimers();
    process.env.EXPO_PUBLIC_REALTIME_RECONNECT_WINDOW_CAP = "0";
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });

    fireConnectError("socket down");
    rebindSocketHandlers();
    jest.advanceTimersByTime(1000);
    fireConnectError("socket down");

    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "realtime.reconnect.cap_reached",
      expect.objectContaining({ reconnect_attempt_window_cap: 0 })
    );
    expect(realtimeManager.getSnapshot().transportAuthority).toBe("degraded");
    delete process.env.EXPO_PUBLIC_REALTIME_RECONNECT_WINDOW_CAP;
    jest.useRealTimers();
  });
});
