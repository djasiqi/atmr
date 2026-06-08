/**
 * Monitoring continu des prérequis tracking pendant une mission active.
 * Tick 60s: permissions, batterie, service actif → remontée device-health.
 */
import { AppState, Platform } from "react-native";

import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { getTrackingSnapshot } from "../tracking";
import { isTrackingActiveStatus } from "../domain/status";
import { triggerDeviceHealthNow } from "./deviceHealthHeartbeat";
import { evaluateMissionTrackingCapability } from "./missionLiveTrackingEligibility";
import {
  readTrackingOnboarded,
  setTrackingNeedsAttention,
} from "./trackingReadinessPersistence";

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

  const capability = await evaluateMissionTrackingCapability({ forLiveTransition: false });
  return {
    ok: capability.capable,
    constraintReason: capability.constraintReason,
  };
}

async function emitProductTelemetryOnConstraintChange(
  previous: string | null,
  next: string | null
): Promise<void> {
  if (previous === next || next === null) return;

  const tracking = getTrackingSnapshot();
  const missionId = tracking.missionId;
  const missionLive =
    missionId != null &&
    tracking.missionStatus != null &&
    isTrackingActiveStatus(tracking.missionStatus);
  if (!missionLive) return;

  const wasOnboarded = await readTrackingOnboarded().catch(() => false);

  if (
    next === "permission_bg_denied" ||
    next === "permission_fg_denied"
  ) {
    emitDriverTelemetry("tracking.permission_revoked_during_mission", {
      source: "driver.background_tracking_health_monitor",
      mission_id: missionId,
      was_onboarded: wasOnboarded,
      constraint_reason: next,
    });
  }

  if (next === "fgs_not_running") {
    emitDriverTelemetry("tracking.fgs_stopped_during_mission", {
      source: "driver.background_tracking_health_monitor",
      mission_id: missionId,
      constraint_reason: next,
    });
  }
}

async function tickMonitor(): Promise<void> {
  const result = await runMonitorChecks();
  if (result.constraintReason !== lastConstraintReason) {
    await emitProductTelemetryOnConstraintChange(
      lastConstraintReason,
      result.constraintReason
    );
    lastConstraintReason = result.constraintReason;
    emitDriverTelemetry("tracking.health_monitor.constraint_changed", {
      source: "driver.background_tracking_health_monitor",
      constraint_reason: result.constraintReason,
      ok: result.ok,
    });
  }

  if (!result.ok) {
    void setTrackingNeedsAttention(true).catch(() => undefined);
  } else {
    void setTrackingNeedsAttention(false).catch(() => undefined);
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
