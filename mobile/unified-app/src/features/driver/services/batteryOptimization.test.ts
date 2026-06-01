import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

import {
  __resetBatteryOptimizationCacheForTests,
  checkBatteryOptimizationStatus,
  requestIgnoreBatteryOptimizations,
} from "./batteryOptimization";

const mockEmit = jest.fn();
const mockIsBatteryOptimizationEnabledAsync = jest.fn<() => Promise<boolean>>();
const mockStartActivityAsync = jest.fn<(action: string, params?: Record<string, unknown>) => Promise<unknown>>();

let mockPlatformOS: "android" | "ios" = "android";

jest.mock("react-native", () => ({
  get Platform() {
    return { OS: mockPlatformOS };
  },
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) => mockEmit(...args),
}));

jest.mock("expo-battery", () => ({
  isBatteryOptimizationEnabledAsync: () => mockIsBatteryOptimizationEnabledAsync(),
}));

jest.mock("expo-intent-launcher", () => ({
  startActivityAsync: (action: string, params?: Record<string, unknown>) =>
    mockStartActivityAsync(action, params),
  ActivityAction: {
    REQUEST_IGNORE_BATTERY_OPTIMIZATIONS: "android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS",
    IGNORE_BATTERY_OPTIMIZATION_SETTINGS: "android.settings.IGNORE_BATTERY_OPTIMIZATION_SETTINGS",
  },
}));

jest.mock("expo-application", () => ({
  applicationId: "ch.liri.operations",
}));

describe("batteryOptimization service", () => {
  beforeEach(() => {
    mockPlatformOS = "android";
    mockEmit.mockReset();
    mockIsBatteryOptimizationEnabledAsync.mockReset();
    mockStartActivityAsync.mockReset();
    __resetBatteryOptimizationCacheForTests();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe("checkBatteryOptimizationStatus", () => {
    it("returns null on iOS without calling native", async () => {
      mockPlatformOS = "ios";
      const result = await checkBatteryOptimizationStatus();
      expect(result).toEqual({ isIgnoring: null, checked: false });
      expect(mockIsBatteryOptimizationEnabledAsync).not.toHaveBeenCalled();
      expect(mockEmit).not.toHaveBeenCalled();
    });

    it("returns isIgnoring=true when optimisation disabled", async () => {
      mockIsBatteryOptimizationEnabledAsync.mockResolvedValue(false);
      const result = await checkBatteryOptimizationStatus();
      expect(result).toEqual({ isIgnoring: true, checked: true });
    });

    it("returns isIgnoring=false when optimisation active and emits detected once per state", async () => {
      mockIsBatteryOptimizationEnabledAsync.mockResolvedValue(true);
      const first = await checkBatteryOptimizationStatus();
      const second = await checkBatteryOptimizationStatus();
      expect(first).toEqual({ isIgnoring: false, checked: true });
      expect(second).toEqual({ isIgnoring: false, checked: true });
      const detected = mockEmit.mock.calls.filter(
        (call) => call[0] === "tracking.battery_optimization.detected"
      );
      expect(detected).toHaveLength(1);
    });

    it("emits exempted when status flips from false to true", async () => {
      mockIsBatteryOptimizationEnabledAsync
        .mockResolvedValueOnce(true)
        .mockResolvedValueOnce(false);
      await checkBatteryOptimizationStatus();
      await checkBatteryOptimizationStatus();
      const exempted = mockEmit.mock.calls.filter(
        (call) => call[0] === "tracking.battery_optimization.exempted"
      );
      expect(exempted).toHaveLength(1);
    });

    it("emits check_failed when expo-battery throws", async () => {
      mockIsBatteryOptimizationEnabledAsync.mockRejectedValue(new Error("native boom"));
      const result = await checkBatteryOptimizationStatus();
      expect(result).toEqual({ isIgnoring: null, checked: false });
      expect(mockEmit).toHaveBeenCalledWith(
        "tracking.battery_optimization.check_failed",
        expect.objectContaining({ reason: "native boom" })
      );
    });
  });

  describe("requestIgnoreBatteryOptimizations", () => {
    it("is a no-op on iOS", async () => {
      mockPlatformOS = "ios";
      const result = await requestIgnoreBatteryOptimizations();
      expect(result).toEqual({ intent: null, opened: false });
      expect(mockStartActivityAsync).not.toHaveBeenCalled();
    });

    it("fires REQUEST_IGNORE intent with package data and emits user_action", async () => {
      mockStartActivityAsync.mockResolvedValue(undefined);
      const result = await requestIgnoreBatteryOptimizations();
      expect(result).toEqual({ intent: "request_ignore", opened: true });
      expect(mockStartActivityAsync).toHaveBeenCalledWith(
        "android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS",
        { data: "package:ch.liri.operations" }
      );
      expect(mockEmit).toHaveBeenCalledWith(
        "tracking.battery_optimization.user_action",
        expect.objectContaining({ action: "open_request_dialog" })
      );
    });

    it("falls back to IGNORE_BATTERY_OPTIMIZATION_SETTINGS when request intent fails", async () => {
      mockStartActivityAsync
        .mockRejectedValueOnce(new Error("ActivityNotFoundException"))
        .mockResolvedValueOnce(undefined);
      const result = await requestIgnoreBatteryOptimizations();
      expect(result).toEqual({ intent: "settings", opened: true });
      expect(mockStartActivityAsync.mock.calls[0]?.[0]).toBe(
        "android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS"
      );
      expect(mockStartActivityAsync.mock.calls[1]?.[0]).toBe(
        "android.settings.IGNORE_BATTERY_OPTIMIZATION_SETTINGS"
      );
    });

    it("reports unavailable when both intents fail", async () => {
      mockStartActivityAsync.mockRejectedValue(new Error("no activity"));
      const result = await requestIgnoreBatteryOptimizations();
      expect(result).toEqual({ intent: null, opened: false });
      const unavailable = mockEmit.mock.calls.filter(
        (call) => call[0] === "driver.battery_optimization.unavailable"
      );
      expect(unavailable.length).toBeGreaterThanOrEqual(2);
    });
  });
});
