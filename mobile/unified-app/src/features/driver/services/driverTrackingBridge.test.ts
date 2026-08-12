import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  getDriverTrackingBridgeSnapshot,
  hardStopDriverContextRuntime,
  setDriverTrackingPresenceContext,
  startDriverTrackingBridge,
  stopDriverTrackingBridge,
  updateDriverTrackingBridgeStatus,
} from "./driverTrackingBridge";
import { driverTrackingQueue } from "./driverTrackingQueue";
import {
  markPresenceDisclosureAccepted,
  __resetLiveTrackingDisclosureSessionForTests,
} from "./liveTrackingDisclosureSession";
import type { DriverLocationPayload } from "../types";
import type { DriverTelemetryEventName } from "../../../core/observability/driverTelemetry";

const mockRequestForegroundPermissionsAsync = jest.fn() as jest.MockedFunction<
  typeof import("expo-location").requestForegroundPermissionsAsync
>;
const mockRequestBackgroundPermissionsAsync = jest.fn() as jest.MockedFunction<
  typeof import("expo-location").requestBackgroundPermissionsAsync
>;
const mockGetCurrentPositionAsync = jest.fn() as jest.MockedFunction<
  typeof import("expo-location").getCurrentPositionAsync
>;
const mockWatchPositionAsync = jest.fn() as jest.MockedFunction<
  typeof import("expo-location").watchPositionAsync
>;
const mockSendDriverLocation = jest.fn() as jest.MockedFunction<
  typeof import("../api/driverHttp").sendDriverLocation
>;
const mockEmitDriverTelemetry = jest.fn() as jest.MockedFunction<
  typeof import("../../../core/observability/driverTelemetry").emitDriverTelemetry
>;
const mockWarn = jest.spyOn(console, "warn").mockImplementation(() => undefined);
const mockAsyncStorageGetItem = jest.fn<(key: string) => Promise<string | null>>();
const mockAsyncStorageSetItem = jest.fn<
  (key: string, value: string) => Promise<void>
>();
const mockAsyncStorage = {
  getItem: mockAsyncStorageGetItem,
  setItem: mockAsyncStorageSetItem,
};

// `var` : accessible dans la factory jest.mock (hoisting).
// eslint-disable-next-line no-var
var __appStateTest: {
  handlers: Array<(state: "active" | "inactive" | "background") => void>;
  currentState: "active" | "inactive" | "background";
} = { handlers: [], currentState: "active" };

function emitAppState(next: "active" | "inactive" | "background") {
  __appStateTest.currentState = next;
  for (const handler of [...__appStateTest.handlers]) {
    handler(next);
  }
}

jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: mockAsyncStorage,
}));

jest.mock("expo-battery", () => ({
  __esModule: true,
  getBatteryLevelAsync: jest.fn().mockResolvedValue(0.85),
}));

jest.mock("../../../core/observability/gpsFidelityTrace", () => ({
  emitBatteryBaselineIfTracing: jest.fn(),
}));

jest.mock("expo-location", () => ({
  requestForegroundPermissionsAsync: () => mockRequestForegroundPermissionsAsync(),
  requestBackgroundPermissionsAsync: () => mockRequestBackgroundPermissionsAsync(),
  getCurrentPositionAsync: (options?: unknown) => mockGetCurrentPositionAsync(options as any),
  watchPositionAsync: (options: unknown, cb: unknown) =>
    mockWatchPositionAsync(options as any, cb as any),
  Accuracy: { Balanced: "balanced", High: "high" },
}));

jest.mock("react-native", () => ({
  AppState: {
    get currentState() {
      return __appStateTest.currentState;
    },
    addEventListener: (
      _event: string,
      callback: (state: "active" | "inactive" | "background") => void
    ) => {
      __appStateTest.handlers.push(callback);
      return { remove: jest.fn() };
    },
  },
  Platform: {
    OS: "android",
    select: (config: Record<string, unknown>) => config.android ?? config.default,
  },
}));

jest.mock("../api/driverHttp", () => ({
  sendDriverLocation: (payload: DriverLocationPayload) => mockSendDriverLocation(payload),
}));

jest.mock("./trackingSessionsApi", () => ({
  registerTrackingSession: jest.fn(async () => ({
    tracking_session_id: "test-session",
    session_generation: 1,
  })),
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (event: DriverTelemetryEventName, payload: Record<string, unknown>) =>
    mockEmitDriverTelemetry(event, payload as any),
}));

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (flag: string) =>
    flag === "tracking_persistent_runtime_enabled" ||
    flag === "tracking_http_fallback_enabled" ||
    flag === "tracking_background_enabled",
}));

jest.mock("@sentry/react-native", () => ({
  addBreadcrumb: jest.fn(),
  captureMessage: jest.fn(),
}));

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    isDriverSocketReady: jest.fn(() => false),
  },
}));

jest.mock("../missionBarAndroid", () => ({
  showMissionBarAndroid: jest.fn(),
  hideMissionBarAndroid: jest.fn(),
}));

jest.mock("../missionBarIOS", () => ({
  startMissionLiveActivity: jest.fn(),
  stopMissionLiveActivity: jest.fn(),
  updateMissionLiveActivity: jest.fn(),
}));

jest.mock("./backgroundLocationTask", () => ({
  ensureNativeTrackingWhileForeground: jest.fn().mockResolvedValue(undefined),
  initializeBackgroundLocationTask: jest.fn(),
  resumePendingNativeTrackingIfNeeded: jest.fn().mockResolvedValue(undefined),
  setBackgroundTrackingMissionContext: jest.fn().mockResolvedValue(undefined),
  stopBackgroundLocationTask: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("./trackingContextLease", () => ({
  setTrackingContextLeaseDriverActive: jest.fn().mockResolvedValue(undefined),
  setTrackingContextLeaseInactive: jest.fn().mockResolvedValue(undefined),
  setTrackingContextLeaseSwitching: jest.fn().mockResolvedValue(undefined),
  readTrackingContextLease: jest.fn().mockResolvedValue({
    state: "driver_active",
    contextId: "driver:1",
    driverId: 1,
    sessionGenerationId: 1,
    trackingGenerationId: "trk-test",
    trackingIdentityId: "driver:1:company:1",
    missionId: null,
    missionContextVersion: 1,
    updatedAt: Date.now(),
  }),
  leaseAllowsTransport: () => true,
  leaseAllowsCapture: () => true,
}));

jest.mock("../tracking/TrackingRecoveryOrchestrator", () => ({
  tickTrackingRecovery: jest.fn().mockResolvedValue({
    recoveryStage: "HEALTHY",
    recoveryGeneration: 0,
    startedAt: 0,
    nextCheckAt: 0,
    attemptCount: 0,
    lastEvidence: null,
  }),
  runTrackingRecoveryCascade: jest.fn().mockResolvedValue(undefined),
}));

describe("driver tracking bridge", () => {
  beforeEach(async () => {
    jest.useFakeTimers();
    mockWarn.mockClear();
    // Ne pas vider les handlers : TrackingManager + bridge s’abonnent une seule fois.
    __appStateTest.currentState = "active";
    __resetLiveTrackingDisclosureSessionForTests();
    mockRequestForegroundPermissionsAsync.mockReset();
    mockRequestBackgroundPermissionsAsync.mockReset();
    mockGetCurrentPositionAsync.mockReset();
    mockWatchPositionAsync.mockReset();
    mockSendDriverLocation.mockReset();
    mockEmitDriverTelemetry.mockReset();
    mockAsyncStorageGetItem.mockReset();
    mockAsyncStorageSetItem.mockReset();
    mockAsyncStorageGetItem.mockResolvedValue(null);
    mockAsyncStorageSetItem.mockResolvedValue(undefined);
    mockRequestForegroundPermissionsAsync.mockResolvedValue({
      // Ne pas importer `expo-modules-core` ici: ça exécute du code natif incompatible avec Jest node.
      status: "granted" as any,
      expires: "never",
      granted: true,
      canAskAgain: true,
    });
    mockRequestBackgroundPermissionsAsync.mockResolvedValue({
      status: "granted" as any,
      expires: "never",
      granted: true,
      canAskAgain: true,
    });
    mockGetCurrentPositionAsync.mockResolvedValue({
      coords: {
        latitude: 46.5,
        longitude: 6.6,
        altitude: 400,
        accuracy: 7,
        altitudeAccuracy: 1,
        heading: 90,
        speed: 5,
      },
      timestamp: Date.now(),
    });
    mockWatchPositionAsync.mockResolvedValue({ remove: jest.fn() } as any);
    mockSendDriverLocation.mockResolvedValue({ ack_status: "accepted" });
    await stopDriverTrackingBridge();
    setDriverTrackingPresenceContext({ available: false, windowOpen: false });
  });

  afterEach(async () => {
    await stopDriverTrackingBridge();
    setDriverTrackingPresenceContext({ available: false, windowOpen: false });
    __resetLiveTrackingDisclosureSessionForTests();
    jest.useRealTimers();
  });

  it("handles permission denied without sending points", async () => {
    mockRequestForegroundPermissionsAsync.mockResolvedValueOnce({
      status: "denied" as any,
      expires: "never",
      granted: false,
      canAskAgain: true,
    });
    startDriverTrackingBridge(7, "ASSIGNED");
    await Promise.resolve();

    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    expect(getDriverTrackingBridgeSnapshot().permission).toBe("denied");
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "tracking.permission.denied",
      expect.objectContaining({ source: "driver.tracking.bridge", mission_id: 7 })
    );
  });

  it("applies backoff after send failure to avoid flooding", async () => {
    mockSendDriverLocation.mockRejectedValueOnce(new Error("offline"));
    startDriverTrackingBridge(8, "IN_PROGRESS");
    await jest.advanceTimersByTimeAsync(0);

    // File persistante : l’échec HTTP est géré dans la queue (retry), pas via TrackingManager.
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "tracking.queue.http_send_failure",
      expect.objectContaining({ source: "driver.tracking.queue", mission_id: 8 })
    );

    const callsAfterFirstFailure = mockSendDriverLocation.mock.calls.length;
    await jest.advanceTimersByTimeAsync(1000);
    expect(mockSendDriverLocation.mock.calls.length).toBe(callsAfterFirstFailure);
  });

  it("stops tracking when mission status becomes ineligible via updateDriverTrackingBridgeStatus", async () => {
    startDriverTrackingBridge(9, "ASSIGNED");
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(true);
    updateDriverTrackingBridgeStatus("COMPLETED");
    await stopDriverTrackingBridge();
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);
    expect(getDriverTrackingBridgeSnapshot().missionId).toBeNull();
  });

  it("restarts loop correctly on stop/start resume cycle", async () => {
    startDriverTrackingBridge(10, "EN_ROUTE");
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(true);
    await stopDriverTrackingBridge();
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);
    startDriverTrackingBridge(10, "EN_ROUTE");
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(true);
  });

  it("STOP-GATE TRACKING-01: resync queueDepth after mission completion", async () => {
    const getSnapshotSpy = jest.spyOn(driverTrackingQueue, "getSnapshot").mockResolvedValue({
      queueDepth: 0,
      oldestQueuedAt: null,
      newestQueuedAt: null,
      oldestItemAgeMs: null,
    });

    startDriverTrackingBridge(11, "IN_PROGRESS");
    await stopDriverTrackingBridge();

    const after = getDriverTrackingBridgeSnapshot();
    expect(after.isRunning).toBe(false);
    expect(after.queueDepth).toBe(0);
    expect(getSnapshotSpy).toHaveBeenCalled();
    getSnapshotSpy.mockRestore();
  });

  it("flushPoint définit nowIso avant enqueue (régression ReferenceError)", async () => {
    const enqueueSpy = jest.spyOn(driverTrackingQueue, "enqueue").mockResolvedValue(undefined);
    jest.spyOn(driverTrackingQueue, "flush").mockResolvedValue({
      sent: 1,
      backendAcked: 1,
      socketEmitted: 0,
      dropped: 0,
      retried: 0,
      queueDepth: 0,
      flushPathUsed: "socket_batch",
      lastBackendAckAt: Date.now(),
      lastBackendAckStatus: "accepted",
      oldestItemAgeMs: null,
      networkProfile: "normal",
    });

    startDriverTrackingBridge(13, "IN_PROGRESS");
    await jest.advanceTimersByTimeAsync(0);

    expect(enqueueSpy).toHaveBeenCalled();
    const payload = enqueueSpy.mock.calls[0]?.[0]?.payload;
    expect(payload?.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(() => new Date(payload!.timestamp!).toISOString()).not.toThrow();

    enqueueSpy.mockRestore();
  });

  it("deduplicates concurrent stopDriverTrackingBridge calls", async () => {
    const flushSpy = jest.spyOn(driverTrackingQueue, "flush").mockResolvedValue({
      sent: 0,
      backendAcked: 0,
      socketEmitted: 0,
      dropped: 0,
      retried: 0,
      queueDepth: 0,
      flushPathUsed: "http_fallback",
      lastBackendAckAt: null,
      oldestItemAgeMs: null,
      networkProfile: "normal",
    });

    startDriverTrackingBridge(12, "EN_ROUTE");
    await jest.advanceTimersByTimeAsync(0);
    flushSpy.mockClear();
    await Promise.all([stopDriverTrackingBridge(), stopDriverTrackingBridge()]);
    expect(flushSpy.mock.calls.length).toBe(1);
    flushSpy.mockRestore();
  });

  describe("présence bornée par fenêtre (P0-F TIME) + transitions AppState", () => {
    it("available + FG hors fenêtre + disclosure => tracking OFF", async () => {
      markPresenceDisclosureAccepted();
      setDriverTrackingPresenceContext({ available: true, windowOpen: false });
      await jest.advanceTimersByTimeAsync(0);
      await Promise.resolve();

      expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);
      expect(mockWatchPositionAsync).not.toHaveBeenCalled();
    });

    it("FG → BG hors fenêtre => reste OFF", async () => {
      markPresenceDisclosureAccepted();
      setDriverTrackingPresenceContext({ available: true, windowOpen: false });
      await jest.advanceTimersByTimeAsync(0);
      await Promise.resolve();
      expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);

      emitAppState("background");
      await Promise.resolve();
      expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);
    });

    it("BG → FG hors fenêtre => ne redémarre pas", async () => {
      markPresenceDisclosureAccepted();
      setDriverTrackingPresenceContext({ available: true, windowOpen: false });
      await jest.advanceTimersByTimeAsync(0);
      await Promise.resolve();

      emitAppState("background");
      await Promise.resolve();
      expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);

      mockWatchPositionAsync.mockClear();
      emitAppState("active");
      await jest.advanceTimersByTimeAsync(0);
      await Promise.resolve();
      expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);
      expect(mockWatchPositionAsync).not.toHaveBeenCalled();
    });

    it("dans la fenêtre FG → BG => reste ON et passe High → Balanced", async () => {
      markPresenceDisclosureAccepted();
      setDriverTrackingPresenceContext({ available: true, windowOpen: true });
      await jest.advanceTimersByTimeAsync(0);
      await Promise.resolve();
      expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(true);
      const fgOpts = mockWatchPositionAsync.mock.calls[0]?.[0] as { accuracy?: string };
      expect(fgOpts?.accuracy).toBe("high");

      mockWatchPositionAsync.mockClear();
      emitAppState("background");
      await jest.advanceTimersByTimeAsync(0);
      await Promise.resolve();
      expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(true);
      const bgOpts = mockWatchPositionAsync.mock.calls.at(-1)?.[0] as { accuracy?: string };
      expect(bgOpts?.accuracy).toBe("balanced");
    });

    it("windowOpen=false stoppe la présence même en FG + available", async () => {
      emitAppState("active");
      markPresenceDisclosureAccepted();
      setDriverTrackingPresenceContext({ available: true, windowOpen: true });
      await jest.advanceTimersByTimeAsync(0);
      await Promise.resolve();
      expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(true);

      setDriverTrackingPresenceContext({ available: true, windowOpen: false });
      await Promise.resolve();
      expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);
      expect(getDriverTrackingBridgeSnapshot().appState).toBe("active");
    });
  });

  it("hardStopDriverContextRuntime n'appelle pas flush et await clear taskContext", async () => {
    const flushSpy = jest.spyOn(driverTrackingQueue, "flush").mockResolvedValue({
      sent: 0,
      backendAcked: 0,
      socketEmitted: 0,
      dropped: 0,
      retried: 0,
      queueDepth: 0,
      flushPathUsed: "http_fallback",
      lastBackendAckAt: null,
      lastBackendAckStatus: null,
      lastBackendAckRequestEventId: null,
      lastBackendAckServerEventId: null,
      oldestItemAgeMs: null,
      networkProfile: "normal",
      socketEmittedEventIds: [],
      ingestedEventIds: [],
      persistedEventIds: [],
      retryEventIds: [],
    });
    const bg = require("./backgroundLocationTask") as {
      setBackgroundTrackingMissionContext: jest.Mock;
      stopBackgroundLocationTask: jest.Mock;
    };
    let clearResolved = false;
    bg.setBackgroundTrackingMissionContext.mockImplementation(async () => {
      clearResolved = true;
    });
    bg.stopBackgroundLocationTask.mockClear();

    startDriverTrackingBridge(99, "EN_ROUTE" as never);
    await jest.advanceTimersByTimeAsync(0);
    await Promise.resolve();
    flushSpy.mockClear();

    await hardStopDriverContextRuntime("context_left_driver");

    expect(flushSpy).not.toHaveBeenCalled();
    expect(clearResolved).toBe(true);
    expect(bg.setBackgroundTrackingMissionContext).toHaveBeenCalledWith(null, null);
    expect(bg.stopBackgroundLocationTask).toHaveBeenCalledWith("context_left_driver");
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "tracking.context.hard_stop",
      expect.objectContaining({ reason: "context_left_driver" })
    );
    flushSpy.mockRestore();
  });
});
