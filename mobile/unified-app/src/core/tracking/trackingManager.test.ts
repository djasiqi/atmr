import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { TrackingManager, type TrackingTickResult } from "./trackingManager";

jest.mock("react-native", () => ({
  AppState: {
    currentState: "active",
    addEventListener: () => ({ remove: jest.fn() }),
  },
}));

describe("tracking manager", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("runs immediate tick and periodic loop when started", async () => {
    const onTick = jest.fn(async (): Promise<TrackingTickResult> => "success");
    const manager = new TrackingManager({
      foregroundIntervalMs: 100,
      backgroundIntervalMs: 200,
      maxBackoffMs: 1000,
      onTick,
    });

    manager.start("mission_live");
    await jest.advanceTimersByTimeAsync(0);
    expect(onTick).toHaveBeenCalledTimes(1);
    expect(manager.getSnapshot().isRunning).toBe(true);

    await jest.advanceTimersByTimeAsync(220);
    expect(onTick).toHaveBeenCalledTimes(3);
    manager.dispose();
  });

  it("applies backoff after failures and recovers after success", async () => {
    const onTick = jest
      .fn(async (): Promise<TrackingTickResult> => "success")
      .mockResolvedValueOnce("failed")
      .mockResolvedValueOnce("success");
    const onFailure = jest.fn();
    const onRecovered = jest.fn();
    const manager = new TrackingManager({
      foregroundIntervalMs: 100,
      backgroundIntervalMs: 200,
      maxBackoffMs: 1000,
      onTick,
      onFailure,
      onRecovered,
    });

    manager.start("mission_live");
    await jest.advanceTimersByTimeAsync(0);
    const failureSnapshot = manager.getSnapshot();
    expect(failureSnapshot.consecutiveFailures).toBe(1);
    expect(failureSnapshot.backoffUntilMs).toBeGreaterThan(Date.now());
    expect(onFailure).toHaveBeenCalled();

    await jest.advanceTimersByTimeAsync(2500);
    expect(onRecovered).toHaveBeenCalled();
    expect(manager.getSnapshot().consecutiveFailures).toBe(0);
    manager.dispose();
  });

  it("updates cadence intervals at runtime", async () => {
    const onTick = jest.fn(async (): Promise<TrackingTickResult> => "success");
    const manager = new TrackingManager({
      foregroundIntervalMs: 2_000,
      backgroundIntervalMs: 2_000,
      maxBackoffMs: 10_000,
      onTick,
    });
    manager.start("mission_live");
    await jest.advanceTimersByTimeAsync(0);
    expect(onTick).toHaveBeenCalledTimes(1);

    manager.setIntervals({ foregroundIntervalMs: 1_000, backgroundIntervalMs: 1_200 });
    await jest.advanceTimersByTimeAsync(2_200);
    expect(onTick.mock.calls.length).toBeGreaterThanOrEqual(3);
    manager.dispose();
  });
});
