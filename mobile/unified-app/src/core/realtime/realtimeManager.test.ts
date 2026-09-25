import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  resetDriverSessionNetworkGateForTests,
  setDriverSessionNetworkReady,
} from "../network/driverSessionNetworkGate";
import { realtimeManager } from "./realtimeManager";

const mockHandlers = new Map<string, ((payload?: unknown) => void)[]>();
const mockEmitDriverTelemetry = jest.fn();
const mockRefreshAuthTokenNow = jest.fn<() => Promise<boolean>>();
const mockRequestWarmAuthRecovery = jest.fn<() => Promise<string>>();
const mockGetAuthAccessToken = jest.fn<() => string | null>();
const mockSocket = {
  connected: true,
  on: jest.fn((event: string, callback: (payload?: unknown) => void) => {
    const existing = mockHandlers.get(event) ?? [];
    existing.push(callback);
    mockHandlers.set(event, existing);
    return mockSocket;
  }),
  emit: jest.fn(),
  removeAllListeners: jest.fn(() => {
    mockHandlers.clear();
  }),
  disconnect: jest.fn(),
  io: {
    on: jest.fn(),
  },
};
const mockIo = jest.fn(() => mockSocket) as jest.MockedFunction<(...args: any[]) => typeof mockSocket>;

jest.mock("socket.io-client", () => ({
  // `socket.io-client` typings are tuple-heavy; avoid rest/spread in the mock factory for `tsc`.
  io: (url: string, opts?: unknown) => mockIo(url, opts),
}));

const mockResolveDriverSocketUrl = jest.fn<() => string | null>();

jest.mock("./resolveDriverSocketUrl", () => ({
  resolveDriverSocketUrl: () => mockResolveDriverSocketUrl(),
}));

jest.mock("../observability/driverTelemetry", () => ({
  emitDriverTelemetry: (event: string, payload?: unknown) => mockEmitDriverTelemetry(event, payload),
}));
jest.mock("../api/client", () => ({
  refreshAuthTokenNow: () => mockRefreshAuthTokenNow(),
  getAuthAccessToken: () => mockGetAuthAccessToken(),
}));

jest.mock("../auth/authWarmRecoveryBridge", () => ({
  requestWarmAuthRecovery: (reason: string) => mockRequestWarmAuthRecovery(reason),
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
    resetDriverSessionNetworkGateForTests();
    setDriverSessionNetworkReady(true);
    mockHandlers.clear();
    mockIo.mockClear();
    mockSocket.on.mockClear();
    mockSocket.emit.mockClear();
    mockSocket.io.on.mockClear();
    mockSocket.disconnect.mockClear();
    mockSocket.removeAllListeners.mockClear();
    mockEmitDriverTelemetry.mockClear();
    mockRefreshAuthTokenNow.mockReset();
    mockRequestWarmAuthRecovery.mockReset();
    mockGetAuthAccessToken.mockReset();
    mockRefreshAuthTokenNow.mockResolvedValue(true);
    mockRequestWarmAuthRecovery.mockResolvedValue("recovered");
    mockGetAuthAccessToken.mockReturnValue("token-test");
    mockResolveDriverSocketUrl.mockImplementation(
      () => process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL ?? null
    );
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

  it("bloque le socket chauffeur avant SESSION_READY", () => {
    resetDriverSessionNetworkGateForTests();
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });
    expect(mockIo).not.toHaveBeenCalled();
    expect(realtimeManager.getSnapshot().mode).toBe("idle");
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "realtime.connect.blocked_before_session_ready",
      expect.objectContaining({ context_id: "driver:42" })
    );
  });

  it("falls back to polling when socket url is not configured", () => {
    mockResolveDriverSocketUrl.mockReturnValue(null);
    realtimeManager.connect("driver:42", { enableSocket: true });
    expect(mockIo).not.toHaveBeenCalled();
    expect(realtimeManager.getSnapshot()).toEqual({
      activeContextId: "driver:42",
      connected: true,
      mode: "polling",
      desiredTransport: "socket",
      actualTransport: "polling",
      lastEventAt: null,
      lastError: "Driver socket URL not configured",
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
    mockResolveDriverSocketUrl.mockReturnValue(null);
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
    expect(mockIo).toHaveBeenCalledTimes(1);
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

    expect(realtimeManager.getSnapshot().authAttempts).toBeGreaterThanOrEqual(1);
    expect(exhaustedCb).not.toHaveBeenCalledWith("terminal", expect.anything());

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

  it("P0-5 final: generic disconnect/backoff reconnect → 0 warm recovery", async () => {
    jest.useFakeTimers();
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });
    mockRequestWarmAuthRecovery.mockClear();

    // disconnect → scheduleReconnect → backoff → connectSocket
    const disconnect = mockHandlers.get("disconnect")?.[0];
    disconnect?.();
    rebindSocketHandlers();
    jest.advanceTimersByTime(5000);
    await Promise.resolve();
    disconnect?.();
    rebindSocketHandlers();
    jest.advanceTimersByTime(10000);
    await Promise.resolve();

    expect(mockRequestWarmAuthRecovery).not.toHaveBeenCalled();
    jest.useRealTimers();
  });

  it("P0-5 final: auth handshake failure → exactly one warm recovery", async () => {
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });
    mockRequestWarmAuthRecovery.mockClear();

    fireConnectError("Unauthorized", "invalid_token");
    await Promise.resolve();
    await Promise.resolve();

    expect(mockRequestWarmAuthRecovery).toHaveBeenCalledTimes(1);
    expect(mockRequestWarmAuthRecovery).toHaveBeenCalledWith("socket_auth_failure");
  });

  it("P0-5 final: rapid auth errors → single-flight (pas de storm séquentiel concurrent)", async () => {
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    let resolveRecovery!: (v: string) => void;
    const pending = new Promise<string>((r) => {
      resolveRecovery = r;
    });
    mockRequestWarmAuthRecovery.mockReturnValue(pending);

    realtimeManager.connect("driver:42", { enableSocket: true });
    mockRequestWarmAuthRecovery.mockClear();
    mockRequestWarmAuthRecovery.mockReturnValue(pending);

    fireConnectError("Unauthorized");
    fireConnectError("Forbidden");
    fireConnectError("401");
    await Promise.resolve();

    expect(mockRequestWarmAuthRecovery).toHaveBeenCalledTimes(1);
    expect(mockRequestWarmAuthRecovery).toHaveBeenCalledWith("socket_auth_failure");
    resolveRecovery("recovered");
    await Promise.resolve();
  });

  it("does not apply token after logout disconnect (generation/epoch)", async () => {
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });
    expect(mockIo).toHaveBeenCalledTimes(1);

    realtimeManager.disconnect();
    expect(realtimeManager.getSnapshot().mode).toBe("idle");
    mockGetAuthAccessToken.mockClear();

    // reconnect_attempt sur ancien socket ne doit plus toucher l'auth
    const reconnectAttempt = mockSocket.io.on.mock.calls.find(
      (c) => c[0] === "reconnect_attempt"
    )?.[1] as (() => void) | undefined;
    reconnectAttempt?.();
    await Promise.resolve();

    expect(realtimeManager.getSnapshot().mode).toBe("idle");
  });

  it("ignores stale socket connect callbacks after context switch", () => {
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:1", { enableSocket: true });
    const staleConnect = mockHandlers.get("connect")?.[0];
    expect(staleConnect).toBeDefined();

    realtimeManager.onContextSwitch("driver:2", { enableSocket: true });
    expect(realtimeManager.getSnapshot().activeContextId).toBe("driver:2");

    staleConnect?.();
    expect(realtimeManager.getSnapshot().activeContextId).toBe("driver:2");
  });

  it("sends wrapped batch payload with tracking session metadata", () => {
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = "wss://driver.example.test";
    realtimeManager.connect("driver:42", { enableSocket: true });
    const connectHandler = mockHandlers.get("connect")?.[0];
    connectHandler?.();
    const ok = realtimeManager.sendDriverLocationBatch([
      {
        tracking_event_id: "evt-1",
        tracking_session_id: "sess-1",
        batch_id: "batch-1",
        position_id: "pos-1",
        sequence_id: 1,
        mission_id: 12,
        latitude: 1,
        longitude: 2,
      },
    ]);
    expect(ok).toBe(true);
    expect(mockSocket.emit).toHaveBeenCalledWith(
      "driver_location_batch",
      expect.objectContaining({
        tracking_session_id: "sess-1",
        batch_id: "batch-1",
      })
    );
  });
});
