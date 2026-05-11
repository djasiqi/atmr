import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  getDriverTrackingBridgeSnapshot,
  startDriverTrackingBridge,
  stopDriverTrackingBridge,
  updateDriverTrackingBridgeStatus,
} from "./driverTrackingBridge";
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

jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: mockAsyncStorage,
}));

jest.mock("expo-battery", () => ({
  __esModule: true,
  getBatteryLevelAsync: jest.fn().mockResolvedValue(0.85),
}));

jest.mock("expo-location", () => ({
  requestForegroundPermissionsAsync: () => mockRequestForegroundPermissionsAsync(),
  requestBackgroundPermissionsAsync: () => mockRequestBackgroundPermissionsAsync(),
  getCurrentPositionAsync: (options?: unknown) => mockGetCurrentPositionAsync(options as any),
  Accuracy: { Balanced: "balanced" },
}));

jest.mock("react-native", () => ({
  AppState: {
    currentState: "active",
    addEventListener: () => ({ remove: jest.fn() }),
  },
  Platform: {
    OS: "android",
    select: (config: Record<string, unknown>) => config.android ?? config.default,
  },
}));

jest.mock("../api/driverHttp", () => ({
  sendDriverLocation: (payload: DriverLocationPayload) => mockSendDriverLocation(payload),
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (event: DriverTelemetryEventName, payload: Record<string, unknown>) =>
    mockEmitDriverTelemetry(event, payload as any),
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

describe("driver tracking bridge", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockWarn.mockClear();
    mockRequestForegroundPermissionsAsync.mockReset();
    mockRequestBackgroundPermissionsAsync.mockReset();
    mockGetCurrentPositionAsync.mockReset();
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
    mockSendDriverLocation.mockResolvedValue({ ack_status: "accepted" });
    stopDriverTrackingBridge();
  });

  afterEach(() => {
    stopDriverTrackingBridge();
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

    const snapshot = getDriverTrackingBridgeSnapshot();
    expect(snapshot.consecutiveFailures).toBe(1);
    expect(snapshot.backoffUntilMs).toBeGreaterThan(Date.now());
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "tracking.send.backoff",
      expect.objectContaining({ source: "driver.tracking.bridge", mission_id: 8, retry_count: 1 })
    );

    const callsAfterFirstFailure = mockSendDriverLocation.mock.calls.length;
    await jest.advanceTimersByTimeAsync(1000);
    expect(mockSendDriverLocation.mock.calls.length).toBe(callsAfterFirstFailure);
  });

  it("stops tracking when mission status becomes ineligible", () => {
    startDriverTrackingBridge(9, "ASSIGNED");
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(true);
    updateDriverTrackingBridgeStatus("COMPLETED");
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);
    expect(getDriverTrackingBridgeSnapshot().missionId).toBeNull();
  });

  it("restarts loop correctly on stop/start resume cycle", () => {
    startDriverTrackingBridge(10, "EN_ROUTE");
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(true);
    stopDriverTrackingBridge();
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(false);
    startDriverTrackingBridge(10, "EN_ROUTE");
    expect(getDriverTrackingBridgeSnapshot().isRunning).toBe(true);
  });
});
