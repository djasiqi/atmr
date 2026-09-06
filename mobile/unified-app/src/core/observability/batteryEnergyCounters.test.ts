import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

jest.mock("react-native", () => ({
  Platform: { OS: "ios" },
  AppState: { currentState: "active" },
}));

import {
  flushBatteryEnergyMinuteNowForTests,
  recordBatteryCallback,
  recordBatteryEnqueue,
  recordBatteryPutSuccess,
  resetBatteryEnergyCountersForTests,
  setBatteryEnergyInstrEnabledForTests,
  setBatteryNativeTaskActive,
  setBatteryWatchActive,
} from "./batteryEnergyCounters";
import { setDriverTelemetrySinkForTests } from "./driverTelemetry";

jest.mock("../device/deviceRuntimeMetadata", () => ({
  resolveDeviceRuntimeMetadata: () => ({
    model: "iPhone XR",
    appVersion: "1.0.13",
  }),
}));

describe("batteryEnergyCounters", () => {
  const events: { name: string; payload: Record<string, unknown> }[] = [];

  beforeEach(() => {
    events.length = 0;
    resetBatteryEnergyCountersForTests();
    setBatteryEnergyInstrEnabledForTests(true);
    setDriverTelemetrySinkForTests((name, payload) => {
      events.push({ name, payload: payload as Record<string, unknown> });
    });
  });

  afterEach(() => {
    resetBatteryEnergyCountersForTests();
    setBatteryEnergyInstrEnabledForTests(null);
    setDriverTelemetrySinkForTests(null);
  });

  it("ne fait rien si le flag est OFF", () => {
    setBatteryEnergyInstrEnabledForTests(false);
    recordBatteryCallback({ source: "native_task", recordedAt: "2026-09-05T10:00:00.000Z" });
    recordBatteryEnqueue({
      source: "native_task",
      recordedAt: "2026-09-05T10:00:00.000Z",
      eventId: "e1",
    });
    flushBatteryEnergyMinuteNowForTests();
    expect(events).toHaveLength(0);
  });

  it("compte unique vs duplicate recorded_at et les sources", () => {
    const ts = "2026-09-05T10:00:01.000Z";
    recordBatteryCallback({
      source: "native_task",
      recordedAt: ts,
      trackingMode: "mission_live",
      appState: "active",
    });
    recordBatteryCallback({ source: "js_watch", recordedAt: ts, appState: "active" });
    recordBatteryEnqueue({
      source: "native_task",
      recordedAt: ts,
      eventId: "e1",
      queueDepth: 2,
      trackingMode: "mission_live",
    });
    recordBatteryEnqueue({
      source: "bridge_tick",
      recordedAt: ts,
      eventId: "e2",
      queueDepth: 3,
    });
    recordBatteryPutSuccess({
      eventId: "e1",
      recordedAt: ts,
      queuedAtMs: Date.now() - 400,
      queueDepth: 1,
    });
    setBatteryNativeTaskActive(true);
    setBatteryWatchActive(true);
    flushBatteryEnergyMinuteNowForTests();

    expect(events).toHaveLength(1);
    expect(events[0]?.name).toBe("tracking.battery.minute");
    const p = events[0]?.payload ?? {};
    expect(p.native_callbacks).toBe(1);
    expect(p.js_callbacks).toBe(1);
    expect(p.unique_fixes).toBe(1);
    expect(p.duplicate_fixes).toBe(1);
    expect(p.enqueues).toBe(2);
    expect(p.enqueue_native).toBe(1);
    expect(p.enqueue_bridge_tick).toBe(1);
    expect(p.put_success).toBe(1);
    expect(p.native_task_active).toBe(true);
    expect(p.js_watch_active).toBe(true);
    expect(p.tracking_mode).toBe("mission_live");
    expect(p.native_callbacks).not.toBe(p.enqueues);
    expect(p.unique_fixes).not.toBe(p.enqueues);
    expect(p.same_recorded_at_reused).toBe(true);
    expect(p.layers_not_collapsed).toBe(true);
    expect(p.queue_depth_enqueue_min).toBe(2);
    expect(p.queue_depth_enqueue_max).toBe(3);
    expect(p.queue_depth_enqueue_last).toBe(3);
    expect(p.queue_depth_drain_last).toBe(1);
    expect(p.queue_depth_drain_min).toBe(1);
  });
});
