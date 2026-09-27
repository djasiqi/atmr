import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockReadLease = jest.fn();
const mockFlush = jest.fn(async () => undefined);
const mockSnapshot = jest.fn(async () => ({ queueDepth: 0 }));
const mockRestart = jest.fn(async () => undefined);
const mockHealth = jest.fn(async () => undefined);
const mockEmit = jest.fn();

jest.mock("react-native", () => ({
  Platform: { OS: "android" },
}));

jest.mock("expo-task-manager", () => ({
  defineTask: jest.fn(),
}));

jest.mock("../src/core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) => mockEmit(...args),
}));

jest.mock("../src/core/featureFlags/registry", () => ({
  isFeatureEnabled: () => true,
}));

jest.mock("../src/features/driver/services/trackingContextLease", () => ({
  readTrackingContextLease: () => mockReadLease(),
  leaseAllowsTransport: (lease: { state?: string } | null) => lease?.state === "driver_active",
}));

jest.mock("../src/features/driver/services/driverTrackingBridge", () => ({
  flushDriverTrackingQueueNow: () => mockFlush(),
  getDriverTrackingQueueSnapshot: () => mockSnapshot(),
}));

jest.mock("../src/features/driver/services/backgroundLocationTask", () => ({
  resumePendingNativeTrackingIfNeeded: jest.fn(async () => undefined),
  restartNativeTrackingFromWake: (...args: unknown[]) => mockRestart(...args),
  initializeBackgroundLocationTask: jest.fn(),
}));

jest.mock("../src/features/driver/services/deviceHealthHeartbeat", () => ({
  triggerDeviceHealthNow: (...args: unknown[]) => mockHealth(...args),
}));

import { runDriverLocationSelfHealTick } from "./locationTask";

describe("runDriverLocationSelfHealTick", () => {
  beforeEach(() => {
    mockReadLease.mockReset();
    mockFlush.mockClear();
    mockRestart.mockClear();
    mockHealth.mockClear();
    mockEmit.mockClear();
  });

  it("revient avant GPS, file et santé si le bail n’est pas chauffeur", async () => {
    mockReadLease.mockResolvedValue({ state: "inactive" });
    await expect(runDriverLocationSelfHealTick()).resolves.toBe("NoData");
    expect(mockFlush).not.toHaveBeenCalled();
    expect(mockRestart).not.toHaveBeenCalled();
    expect(mockHealth).not.toHaveBeenCalled();
    expect(mockEmit).toHaveBeenCalledWith(
      "tracking.background.task.skipped",
      expect.objectContaining({ reason: "lease_not_driver_active" })
    );
  });

  it("exécute le self-heal quand le bail chauffeur est actif", async () => {
    mockReadLease.mockResolvedValue({ state: "driver_active" });
    await expect(runDriverLocationSelfHealTick()).resolves.toBe("NewData");
    expect(mockFlush).toHaveBeenCalledTimes(1);
    expect(mockRestart).toHaveBeenCalledWith("background_task_tick");
    expect(mockHealth).toHaveBeenCalledWith("background_task_tick");
  });
});
