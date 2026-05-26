import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockHasStarted = jest.fn<() => Promise<boolean>>();
const mockStart = jest.fn<() => Promise<void>>();
const mockGetFg = jest.fn<() => Promise<{ status: string; granted: boolean }>>();
const mockGetBg = jest.fn<() => Promise<{ status: string; granted: boolean }>>();
const mockRequestFg = jest.fn<() => Promise<{ granted: boolean }>>();
const mockRequestBg = jest.fn<() => Promise<{ granted: boolean }>>();
const mockEmit = jest.fn();

jest.mock("react-native", () => ({
  AppState: { currentState: "active" },
  Platform: { OS: "android" },
}));

jest.mock("expo-battery", () => ({
  getBatteryLevelAsync: jest.fn().mockResolvedValue(0.9),
}));

jest.mock("expo-location", () => ({
  hasStartedLocationUpdatesAsync: () => mockHasStarted(),
  startLocationUpdatesAsync: (...args: unknown[]) => mockStart(...args),
  stopLocationUpdatesAsync: jest.fn().mockResolvedValue(undefined),
  getForegroundPermissionsAsync: () => mockGetFg(),
  getBackgroundPermissionsAsync: () => mockGetBg(),
  requestForegroundPermissionsAsync: () => mockRequestFg(),
  requestBackgroundPermissionsAsync: () => mockRequestBg(),
  Accuracy: { Balanced: "balanced", Low: "low" },
}));

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn().mockResolvedValue(null),
  setItem: jest.fn().mockResolvedValue(undefined),
  removeItem: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (key: string) =>
    key === "tracking_background_enabled" || key === "tracking_presence_mode_enabled",
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) => mockEmit(...args),
}));

jest.mock("./backgroundRuntimeCompat", () => ({
  canUseBackgroundLocation: () => true,
  describeBackgroundRuntime: () => "dev_client_or_standalone",
}));

jest.mock("./driverTrackingQueue", () => ({
  driverTrackingQueue: {
    enqueue: jest.fn(),
    getSnapshot: jest.fn().mockResolvedValue({ queueDepth: 0 }),
    flush: jest.fn().mockResolvedValue({
      queueDepth: 0,
      sent: 0,
      backendAcked: 0,
      socketEmitted: 0,
      dropped: 0,
    }),
  },
}));

jest.mock("expo-task-manager", () => ({
  defineTask: jest.fn(),
}));

// eslint-disable-next-line @typescript-eslint/no-require-imports
const bgTask = require("./backgroundLocationTask") as typeof import("./backgroundLocationTask");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const trackingRuntime = require("./trackingRuntime") as typeof import("./trackingRuntime");

describe("backgroundLocationTask", () => {
  beforeEach(() => {
    trackingRuntime.__resetTrackingRuntimeForTests();
    mockHasStarted.mockReset();
    mockStart.mockReset();
    mockEmit.mockReset();
    mockGetFg.mockResolvedValue({ status: "granted", granted: true });
    mockGetBg.mockResolvedValue({ status: "granted", granted: true });
    mockRequestFg.mockResolvedValue({ granted: true });
    mockRequestBg.mockResolvedValue({ granted: true });
    mockHasStarted.mockResolvedValue(false);
    mockStart.mockResolvedValue(undefined);
    bgTask.__resetBackgroundLocationTaskStateForTests();
  });

  afterEach(() => {
    jest.useRealTimers();
    bgTask.__resetBackgroundLocationTaskStateForTests();
  });

  it("getNativeTaskLifecycleStatus exposes taskDefined and taskStarted", async () => {
    bgTask.initializeBackgroundLocationTask();
    mockHasStarted.mockResolvedValue(true);
    const status = await bgTask.getNativeTaskLifecycleStatus();
    expect(status.taskDefined).toBe(true);
    expect(status.taskStarted).toBe(true);
  });

  it("emits start_failed when startLocationUpdatesAsync throws", async () => {
    bgTask.initializeBackgroundLocationTask();
    mockStart.mockRejectedValueOnce(new Error("Foreground service cannot be started"));

    await bgTask.ensureNativeTrackingWhileForeground(42, "EN_ROUTE", {}, "test_start");

    expect(mockEmit).toHaveBeenCalledWith(
      "tracking.background.start_failed",
      expect.objectContaining({
        failure_reason: "start_exception",
      })
    );
    expect(trackingRuntime.getTrackingRuntimeSnapshot().lastNativeStartError).toContain("test_start");
  });

  it("records startup_timeout when watchdog exhausts without task started", async () => {
    jest.useFakeTimers();
    bgTask.initializeBackgroundLocationTask();
    mockHasStarted.mockResolvedValue(false);
    mockStart.mockImplementation(() => Promise.resolve());

    void bgTask.ensureNativeTrackingWhileForeground(7, "IN_PROGRESS", {}, "watchdog_test");
    await Promise.resolve();
    await jest.advanceTimersByTimeAsync(31_000);
    await Promise.resolve();

    const snap = trackingRuntime.getTrackingRuntimeSnapshot();
    expect(snap.lastNativeStartError).toContain("startup_timeout");
    expect(mockEmit).toHaveBeenCalledWith(
      "tracking.background.start_failed",
      expect.objectContaining({ failure_reason: "startup_timeout" })
    );
  });
});
