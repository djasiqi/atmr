/**
 * Monitoring continu des prérequis tracking pendant une mission active.
 * Tick 60s: permissions, batterie, service actif → remontée device-health.
 */
import { AppState, Platform } from "react-native";

import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { triggerDeviceHealthNow } from "./deviceHealthHeartbeat";
import { checkBatteryOptimizationStatus } from "./batteryOptimization";

const MONITOR_INTERVAL_MS = 60_000;

let activeStop: (() => void) | null = null;
let lastConstraintReason: string | null = null;

type MonitorCheckResult = {
  ok: boolean;
  constraintReason: string | null;
};

async function runMonitorChecks(): Promise<MonitorCheckResult> {
  if (Platform.OS === "web") {
    return { ok: true, constraintReason: null };
  }

  const Location = await import("expo-location").catch(() => null);
  if (!Location) {
    return { ok: false, constraintReason: "location_module_unavailable" };
  }

  const [fg, bg, gpsEnabled, battery] = await Promise.all([
    Location.getForegroundPermissionsAsync().catch(() => ({ status: "undetermined" })),
    Location.getBackgroundPermissionsAsync().catch(() => ({ status: "undetermined" })),
    Location.hasServicesEnabledAsync().catch(() => false),
    checkBatteryOptimizationStatus(),
  ]);

  if (fg.status !== "granted") {
    return { ok: false, constraintReason: "permission_fg_denied" };
  }
  if (bg.status !== "granted") {
    return { ok: false, constraintReason: "permission_bg_denied" };
  }
  if (!gpsEnabled) {
    return { ok: false, constraintReason: "gps_provider_disabled" };
  }
  if (Platform.OS === "android" && battery.checked && battery.isIgnoring === false) {
    return { ok: false, constraintReason: "battery_optimized" };
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const bgTask = require("./backgroundLocationTask") as typeof import("./backgroundLocationTask");
    const lifecycle = await bgTask.getNativeTaskLifecycleStatus();
    if (!lifecycle.taskStarted) {
      return { ok: false, constraintReason: "fgs_not_running" };
    }
  } catch {
    return { ok: false, constraintReason: "tracking_runtime_unavailable" };
  }

  return { ok: true, constraintReason: null };
}

async function tickMonitor(): Promise<void> {
  const result = await runMonitorChecks();
  if (result.constraintReason !== lastConstraintReason) {
    lastConstraintReason = result.constraintReason;
    emitDriverTelemetry("tracking.health_monitor.constraint_changed", {
      source: "driver.background_tracking_health_monitor",
      constraint_reason: result.constraintReason,
      ok: result.ok,
    });
  }
  await triggerDeviceHealthNow(
    result.ok ? "health_monitor_ok" : `health_monitor:${result.constraintReason}`
  );
}

export function startBackgroundTrackingHealthMonitor(): () => void {
  if (Platform.OS === "web") {
    return () => undefined;
  }
  if (activeStop) {
    return activeStop;
  }

  let stopped = false;
  void tickMonitor();
  const timer = setInterval(() => {
    if (!stopped) void tickMonitor();
  }, MONITOR_INTERVAL_MS);

  const appSub = AppState.addEventListener("change", (next) => {
    if (next === "active" && !stopped) {
      void tickMonitor();
    }
  });

  const stop = () => {
    if (stopped) return;
    stopped = true;
    clearInterval(timer);
    appSub.remove();
    lastConstraintReason = null;
    if (activeStop === stop) {
      activeStop = null;
    }
  };

  activeStop = stop;
  return stop;
}

export function stopBackgroundTrackingHealthMonitor(): void {
  activeStop?.();
}

/** Test-only */
export function __resetBackgroundTrackingHealthMonitorForTests(): void {
  stopBackgroundTrackingHealthMonitor();
}
