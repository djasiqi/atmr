import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockEmit = jest.fn();
const mockApiPost = jest.fn() as jest.MockedFunction<
  (url: string, body?: unknown) => Promise<{ data: unknown }>
>;

const mockGetForegroundPermissionsAsync = jest.fn() as jest.MockedFunction<
  () => Promise<{ status: string }>
>;
const mockGetBackgroundPermissionsAsync = jest.fn() as jest.MockedFunction<
  () => Promise<{ status: string }>
>;
const mockHasServicesEnabledAsync = jest.fn() as jest.MockedFunction<
  () => Promise<boolean>
>;

const mockGetBatteryLevelAsync = jest.fn() as jest.MockedFunction<() => Promise<number>>;
const mockGetBatteryStateAsync = jest.fn() as jest.MockedFunction<() => Promise<number>>;

const mockGetNativeTaskLifecycleStatus = jest.fn() as jest.MockedFunction<
  () => Promise<{ taskDefined: boolean; taskStarted: boolean }>
>;
const mockCheckBatteryOptimizationStatus = jest.fn() as jest.MockedFunction<
  () => Promise<{ isIgnoring: boolean | null; checked: boolean }>
>;
const mockGetDriverTrackingBridgeSnapshot = jest.fn();
const mockGetDriverTrackingPresenceWindowActive = jest.fn(() => false);

// --- Diagnostic Lot 1 mocks ---
const mockIsLowPowerModeEnabledAsync = jest.fn() as jest.MockedFunction<
  () => Promise<boolean>
>;
const mockGetLastKnownPositionAsync = jest.fn() as jest.MockedFunction<
  () => Promise<{ coords: { accuracy: number | null } } | null>
>;
const mockBackgroundFetchGetStatusAsync = jest.fn() as jest.MockedFunction<
  () => Promise<number | null>
>;
const mockGetTrackingRuntimeSnapshot = jest.fn(() => ({
  lastTaskInvokedAt: null as number | null,
  lastNativeStartError: null,
  lastNativeStartErrorAt: null,
  pendingFgsStart: { active: false },
  missionId: null,
  mode: "off",
  nativeStartDiagnostics: {
    native_start_phase: null,
    native_start_error: null,
    native_task_defined: null,
    native_started_before: null,
    native_started_after: null,
  },
}));

let mockPlatformOS: "android" | "ios" | "web" = "android";
let appStateChangeHandler: ((next: string) => void) | null = null;
const mockAppStateRemove = jest.fn();
const mockAppStateAddEventListener = jest.fn(
  (event: string, cb: (next: string) => void) => {
    if (event === "change") {
      appStateChangeHandler = cb;
    }
    return { remove: mockAppStateRemove };
  }
);

jest.mock("react-native", () => ({
  AppState: {
    addEventListener: (event: string, cb: (next: string) => void) =>
      mockAppStateAddEventListener(event, cb),
    currentState: "active",
  },
  get Platform() {
    return { OS: mockPlatformOS };
  },
}));

jest.mock("expo-location", () => ({
  getForegroundPermissionsAsync: () => mockGetForegroundPermissionsAsync(),
  getBackgroundPermissionsAsync: () => mockGetBackgroundPermissionsAsync(),
  hasServicesEnabledAsync: () => mockHasServicesEnabledAsync(),
  getLastKnownPositionAsync: () => mockGetLastKnownPositionAsync(),
}));

jest.mock("expo-battery", () => ({
  getBatteryLevelAsync: () => mockGetBatteryLevelAsync(),
  getBatteryStateAsync: () => mockGetBatteryStateAsync(),
  isLowPowerModeEnabledAsync: () => mockIsLowPowerModeEnabledAsync(),
}));

jest.mock("expo-application", () => ({
  nativeApplicationVersion: "1.42.3",
}));

jest.mock("expo-device", () => ({
  manufacturer: "Apple",
  modelName: "iPhone 12",
  osVersion: "17.4",
}));

jest.mock("expo-background-fetch", () => ({
  getStatusAsync: () => mockBackgroundFetchGetStatusAsync(),
  BackgroundFetchStatus: { Restricted: 1, Denied: 2, Available: 3 },
}));

jest.mock("./trackingRuntime", () => ({
  getTrackingRuntimeSnapshot: () => mockGetTrackingRuntimeSnapshot(),
}));

jest.mock("../../../core/api/client", () => ({
  apiClient: {
    post: (url: string, body?: unknown) => mockApiPost(url, body),
  },
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) => mockEmit(...args),
}));

jest.mock("./backgroundLocationTask", () => ({
  getNativeTaskLifecycleStatus: () => mockGetNativeTaskLifecycleStatus(),
}));

jest.mock("./batteryOptimization", () => ({
  checkBatteryOptimizationStatus: () => mockCheckBatteryOptimizationStatus(),
}));

jest.mock("./driverTrackingBridge", () => ({
  getDriverTrackingBridgeSnapshot: () => mockGetDriverTrackingBridgeSnapshot(),
  getDriverTrackingPresenceWindowActive: () => mockGetDriverTrackingPresenceWindowActive(),
}));

 
const heartbeat = require("./deviceHealthHeartbeat") as typeof import("./deviceHealthHeartbeat");

function setHappyPathDefaults() {
  mockGetForegroundPermissionsAsync.mockResolvedValue({ status: "granted" });
  mockGetBackgroundPermissionsAsync.mockResolvedValue({ status: "granted" });
  mockHasServicesEnabledAsync.mockResolvedValue(true);
  mockGetBatteryLevelAsync.mockResolvedValue(0.82);
  mockGetBatteryStateAsync.mockResolvedValue(2 /* CHARGING */);
  mockGetNativeTaskLifecycleStatus.mockResolvedValue({
    taskDefined: true,
    taskStarted: true,
  });
  mockCheckBatteryOptimizationStatus.mockResolvedValue({ isIgnoring: true, checked: true });
  mockGetDriverTrackingBridgeSnapshot.mockReturnValue({
    missionId: 42,
    lastWatchAt: new Date(Date.now() - 5_000).toISOString(),
  });
  mockApiPost.mockResolvedValue({ data: { ok: true } });
  mockIsLowPowerModeEnabledAsync.mockResolvedValue(false);
  mockGetLastKnownPositionAsync.mockResolvedValue({ coords: { accuracy: 12 } });
  mockBackgroundFetchGetStatusAsync.mockResolvedValue(3 /* Available */);
  mockGetTrackingRuntimeSnapshot.mockReturnValue({
    lastTaskInvokedAt: Date.now() - 8_000,
    lastNativeStartError: null,
    lastNativeStartErrorAt: null,
    pendingFgsStart: { active: false },
    missionId: null,
    mode: "off",
    nativeStartDiagnostics: {
      native_start_phase: null,
      native_start_error: null,
      native_task_defined: null,
      native_started_before: null,
      native_started_after: null,
    },
  });
}

describe("deviceHealthHeartbeat", () => {
  beforeEach(() => {
    mockPlatformOS = "android";
    appStateChangeHandler = null;
    mockEmit.mockReset();
    mockApiPost.mockReset();
    mockGetForegroundPermissionsAsync.mockReset();
    mockGetBackgroundPermissionsAsync.mockReset();
    mockHasServicesEnabledAsync.mockReset();
    mockGetBatteryLevelAsync.mockReset();
    mockGetBatteryStateAsync.mockReset();
    mockGetNativeTaskLifecycleStatus.mockReset();
    mockCheckBatteryOptimizationStatus.mockReset();
    mockGetDriverTrackingBridgeSnapshot.mockReset();
    mockGetDriverTrackingPresenceWindowActive.mockReset();
    mockGetDriverTrackingPresenceWindowActive.mockReturnValue(false);
    mockIsLowPowerModeEnabledAsync.mockReset();
    mockGetLastKnownPositionAsync.mockReset();
    mockBackgroundFetchGetStatusAsync.mockReset();
    mockGetTrackingRuntimeSnapshot.mockReset();
    mockAppStateAddEventListener.mockClear();
    mockAppStateRemove.mockClear();
    setHappyPathDefaults();
    heartbeat.__resetDeviceHealthHeartbeatForTests();
  });

  afterEach(() => {
    heartbeat.__resetDeviceHealthHeartbeatForTests();
    jest.useRealTimers();
  });

  describe("collectDeviceHealth", () => {
    it("returns null constraint_reason on a healthy device", async () => {
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload).toEqual(
        expect.objectContaining({
          kind: "tracking_health",
          fgs_running: true,
          fg_permission: "granted",
          bg_permission: "granted",
          gps_provider_enabled: true,
          battery_optimized: false,
          is_charging: true,
          constraint_reason: null,
        })
      );
      expect(payload.battery_level).toBeCloseTo(0.82, 5);
      expect(payload.last_fix_age_seconds).toBeGreaterThanOrEqual(0);
      expect(payload.last_fix_age_seconds).toBeLessThan(60);
      expect(payload.fix_success_rate_last_5min).toBeNull();
    });

    it("flags permission_fg_denied when foreground permission denied", async () => {
      mockGetForegroundPermissionsAsync.mockResolvedValue({ status: "denied" });
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.fg_permission).toBe("denied");
      expect(payload.constraint_reason).toBe("permission_fg_denied");
    });

    it("flags permission_bg_denied when only background permission denied", async () => {
      mockGetBackgroundPermissionsAsync.mockResolvedValue({ status: "denied" });
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.bg_permission).toBe("denied");
      expect(payload.constraint_reason).toBe("permission_bg_denied");
    });

    it("flags gps_provider_disabled when location services off", async () => {
      mockHasServicesEnabledAsync.mockResolvedValue(false);
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.gps_provider_enabled).toBe(false);
      expect(payload.constraint_reason).toBe("gps_provider_disabled");
    });

    it("flags battery_optimized on Android when not exempted from Doze", async () => {
      mockCheckBatteryOptimizationStatus.mockResolvedValue({ isIgnoring: false, checked: true });
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.battery_optimized).toBe(true);
      expect(payload.constraint_reason).toBe("battery_optimized");
    });

    it("does not flag battery_optimized on iOS even if helper returns null", async () => {
      mockPlatformOS = "ios";
      mockCheckBatteryOptimizationStatus.mockResolvedValue({ isIgnoring: null, checked: false });
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.battery_optimized).toBe(false);
      expect(payload.constraint_reason).toBeNull();
    });

    it("flags fgs_not_running when a mission is active and FGS is down", async () => {
      mockGetNativeTaskLifecycleStatus.mockResolvedValue({
        taskDefined: true,
        taskStarted: false,
      });
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.fgs_running).toBe(false);
      expect(payload.constraint_reason).toBe("fgs_not_running");
    });

    it("does not flag fgs_not_running when no mission and no presence window", async () => {
      mockGetNativeTaskLifecycleStatus.mockResolvedValue({
        taskDefined: true,
        taskStarted: false,
      });
      mockGetDriverTrackingBridgeSnapshot.mockReturnValue({
        missionId: null,
        lastWatchAt: null,
      });
      mockGetDriverTrackingPresenceWindowActive.mockReturnValue(false);
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.fgs_running).toBe(false);
      expect(payload.constraint_reason).toBeNull();
    });

    it("flags fix_stale when last watch older than 5 minutes", async () => {
      mockGetDriverTrackingBridgeSnapshot.mockReturnValue({
        missionId: 99,
        lastWatchAt: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
      });
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.last_fix_age_seconds).toBeGreaterThanOrEqual(600 - 1);
      expect(payload.constraint_reason).toBe("fix_stale");
    });

    it("returns null is_charging when battery state unknown", async () => {
      mockGetBatteryStateAsync.mockResolvedValue(0 /* UNKNOWN */);
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.is_charging).toBeNull();
    });

    it("[Lot 1] includes app/os version, native fix age, and null iOS signals on Android", async () => {
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.app_version).toBe("1.42.3");
      expect(payload.os_version).toBe("17.4");
      expect(payload.native_task_running).toBe(true);
      // lastTaskInvokedAt = now - 8s -> ~8s
      expect(payload.native_last_fix_age_seconds).toBeGreaterThanOrEqual(7);
      expect(payload.native_last_fix_age_seconds).toBeLessThan(60);
      // iOS-only signals are null on Android
      expect(payload.ios_low_power_mode).toBeNull();
      expect(payload.ios_background_refresh_status).toBeNull();
      expect(payload.ios_accuracy_authorization).toBeNull();
    });

    it("[Lot 1] populates iOS background signals on iOS", async () => {
      mockPlatformOS = "ios";
      mockIsLowPowerModeEnabledAsync.mockResolvedValue(true);
      mockBackgroundFetchGetStatusAsync.mockResolvedValue(2 /* Denied */);
      mockGetLastKnownPositionAsync.mockResolvedValue({
        coords: { accuracy: 3000 },
      });
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.ios_low_power_mode).toBe(true);
      expect(payload.ios_background_refresh_status).toBe("denied");
      expect(payload.ios_accuracy_authorization).toBe("reduced");
    });

    it("[Lot 1] infers full accuracy when last fix is precise on iOS", async () => {
      mockPlatformOS = "ios";
      mockGetLastKnownPositionAsync.mockResolvedValue({
        coords: { accuracy: 15 },
      });
      const payload = await heartbeat.collectDeviceHealth();
      expect(payload.ios_accuracy_authorization).toBe("full");
    });
  });

  describe("sendDeviceHealth", () => {
    it("POSTs to /driver/me/device-health and emits success telemetry", async () => {
      const payload = await heartbeat.collectDeviceHealth();
      await heartbeat.sendDeviceHealth(payload);

      expect(mockApiPost).toHaveBeenCalledTimes(1);
      expect(mockApiPost.mock.calls[0]?.[0]).toBe("/driver/me/device-health");
      expect(mockApiPost.mock.calls[0]?.[1]).toEqual(payload);

      const sentEvent = mockEmit.mock.calls.find(
        (call) => call[0] === "tracking.device_health.sent"
      );
      expect(sentEvent).toBeDefined();
      expect(sentEvent?.[1]).toEqual(
        expect.objectContaining({
          source: "driver.device_health",
          kind: "tracking_health",
          fgs_running: true,
        })
      );
    });

    it("emits send_failed telemetry with http_status when POST fails", async () => {
      const error = Object.assign(new Error("server boom"), {
        response: { status: 503 },
      });
      mockApiPost.mockRejectedValueOnce(error);
      const payload = await heartbeat.collectDeviceHealth();
      await heartbeat.sendDeviceHealth(payload);

      const failed = mockEmit.mock.calls.find(
        (call) => call[0] === "tracking.device_health.send_failed"
      );
      expect(failed).toBeDefined();
      expect(failed?.[1]).toEqual(
        expect.objectContaining({
          source: "driver.device_health",
          error: "server boom",
          http_status: 503,
        })
      );
    });
  });

  describe("startDeviceHealthHeartbeat", () => {
    it("ticks every 120s and throttles premature AppState -> active sends", async () => {
      jest.useFakeTimers();
      const stop = heartbeat.startDeviceHealthHeartbeat();
      expect(typeof stop).toBe("function");

      // Initial tick fired synchronously (async resolution).
      await jest.advanceTimersByTimeAsync(0);
      const initialCalls = mockApiPost.mock.calls.length;
      expect(initialCalls).toBe(1);

      await jest.advanceTimersByTimeAsync(60_000);
      expect(mockApiPost.mock.calls.length).toBe(1);

      await jest.advanceTimersByTimeAsync(60_000);
      expect(mockApiPost.mock.calls.length).toBe(2);

      // Simulate AppState -> active: collected immediately but HTTP send is throttled.
      expect(appStateChangeHandler).toBeTruthy();
      appStateChangeHandler?.("active");
      await jest.advanceTimersByTimeAsync(0);
      expect(mockApiPost.mock.calls.length).toBe(2);

      stop();
    });

    it("is a no-op on web", () => {
      mockPlatformOS = "web";
      const stop = heartbeat.startDeviceHealthHeartbeat();
      expect(typeof stop).toBe("function");
      expect(mockAppStateAddEventListener).not.toHaveBeenCalled();
    });

    it("stop() removes interval and AppState listener", async () => {
      jest.useFakeTimers();
      const stop = heartbeat.startDeviceHealthHeartbeat();
      await jest.advanceTimersByTimeAsync(0);
      const initial = mockApiPost.mock.calls.length;

      stop();
      await jest.advanceTimersByTimeAsync(60_000 * 5);
      expect(mockApiPost.mock.calls.length).toBe(initial);
      expect(mockAppStateRemove).toHaveBeenCalled();
    });
  });

  describe("triggerDeviceHealthNow", () => {
    it("sends one heartbeat with trigger_reason", async () => {
      await heartbeat.triggerDeviceHealthNow("fix_stale");
      expect(mockApiPost).toHaveBeenCalledTimes(1);
      const body = mockApiPost.mock.calls[0]?.[1] as Record<string, unknown>;
      expect(body.trigger_reason).toBe("fix_stale");
      expect(body.kind).toBe("tracking_health");

      const sent = mockEmit.mock.calls.find(
        (call) => call[0] === "tracking.device_health.sent"
      );
      expect(sent?.[1]).toEqual(
        expect.objectContaining({ trigger_reason: "fix_stale" })
      );
    });

    it("is a no-op on web", async () => {
      mockPlatformOS = "web";
      await heartbeat.triggerDeviceHealthNow("any_reason");
      expect(mockApiPost).not.toHaveBeenCalled();
    });
  });
});
