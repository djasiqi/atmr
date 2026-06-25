import * as Sentry from "@sentry/react-native";
import * as Location from "expo-location";
import { AppStateStatus } from "react-native";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";

export const STALE_RESTART_THRESHOLD = 2;
export const STALE_RESTART_COOLDOWN_MS = 300_000;
export const WATCH_RESTART_MAX_PER_HOUR = 3;
export const ANTI_ZOMBIE_FIX_AGE_SEC = 60;

export type SelfHealBridgeSlice = {
  watchSubscription: Location.LocationSubscription | null;
  staleFallbackTimeouts: number;
  staleFallbackBlockedUntilMs: number;
  lastWatchAtMs: number | null;
  lastWatchedPosition: Location.LocationObject | null;
  lastWatchRestartAtMs: number;
  watchRestartTimestampsMs: number[];
  missionId: number | null;
};

export type SelfHealActions = {
  stopWatch: () => void;
  stopBackground: (reason: string) => Promise<void>;
  ensureNativeForeground: () => Promise<void>;
  ensureLocationWatch: () => Promise<void>;
  triggerDeviceHealth: (reason: string) => void;
};

export function createSelfHealSlice(): Pick<
  SelfHealBridgeSlice,
  "lastWatchRestartAtMs" | "watchRestartTimestampsMs"
> {
  return {
    lastWatchRestartAtMs: 0,
    watchRestartTimestampsMs: [],
  };
}

function pruneRestartTimestamps(timestamps: number[], nowMs: number): number[] {
  const hourAgo = nowMs - 3_600_000;
  return timestamps.filter((t) => t >= hourAgo);
}

export function shouldForceRestartWatch(
  slice: SelfHealBridgeSlice,
  nowMs: number = Date.now()
): boolean {
  if (!isFeatureEnabled("tracking_self_heal_watch_restart_enabled")) {
    return false;
  }
  if (slice.staleFallbackTimeouts < STALE_RESTART_THRESHOLD) {
    return false;
  }
  if (nowMs - slice.lastWatchRestartAtMs < STALE_RESTART_COOLDOWN_MS) {
    return false;
  }
  const recent = pruneRestartTimestamps(slice.watchRestartTimestampsMs, nowMs);
  return recent.length < WATCH_RESTART_MAX_PER_HOUR;
}

export async function forceRestartTrackingWatch(
  reason: string,
  slice: SelfHealBridgeSlice,
  actions: SelfHealActions,
  appState: AppStateStatus
): Promise<boolean> {
  if (!isFeatureEnabled("tracking_self_heal_watch_restart_enabled")) {
    return false;
  }
  const nowMs = Date.now();
  const recent = pruneRestartTimestamps(slice.watchRestartTimestampsMs, nowMs);
  if (recent.length >= WATCH_RESTART_MAX_PER_HOUR) {
    Sentry.captureMessage("tracking.watch.restart.exhausted", { level: "warning" });
    emitDriverTelemetry("tracking.watch.restart.exhausted", {
      reason,
      mission_id: slice.missionId,
    });
    return false;
  }

  actions.stopWatch();
  await actions.stopBackground("self_heal_restart").catch(() => undefined);
  await Location.requestForegroundPermissionsAsync().catch(() => undefined);

  slice.staleFallbackTimeouts = 0;
  slice.staleFallbackBlockedUntilMs = 0;
  slice.lastWatchAtMs = null;
  slice.lastWatchedPosition = null;
  slice.lastWatchRestartAtMs = nowMs;
  slice.watchRestartTimestampsMs = [...recent, nowMs];

  await actions.ensureNativeForeground();
  await actions.ensureLocationWatch();

  actions.triggerDeviceHealth("stale_fallback_restart");
  emitDriverTelemetry("tracking.watch.restarted", {
    reason,
    mission_id: slice.missionId,
    app_state: appState,
  });
  return true;
}

let lastAntiZombieTriggeredAtMs = 0;
const ANTI_ZOMBIE_COOLDOWN_MS = 60_000;

export function getFixAgeSeconds(
  lastFixProducedAtMs: number | null,
  nowMs: number = Date.now()
): number | null {
  if (lastFixProducedAtMs == null) {
    return null;
  }
  return Math.max(0, (nowMs - lastFixProducedAtMs) / 1000);
}

export function shouldTriggerAntiZombie(input: {
  isTrackingRunning: boolean;
  lastFixProducedAtMs: number | null;
  lastSentAt: string | null;
  nowMs?: number;
}): boolean {
  if (!isFeatureEnabled("tracking_self_heal_watch_restart_enabled")) {
    return false;
  }
  if (!input.isTrackingRunning) {
    return false;
  }
  const nowMs = input.nowMs ?? Date.now();
  if (nowMs - lastAntiZombieTriggeredAtMs < ANTI_ZOMBIE_COOLDOWN_MS) {
    return false;
  }
  const fixAge = getFixAgeSeconds(input.lastFixProducedAtMs, nowMs);
  if (fixAge === null || fixAge <= ANTI_ZOMBIE_FIX_AGE_SEC) {
    return false;
  }
  if (input.lastSentAt) {
    const sentAgeSec = (nowMs - Date.parse(input.lastSentAt)) / 1000;
    if (sentAgeSec <= ANTI_ZOMBIE_FIX_AGE_SEC) {
      return false;
    }
  }
  return true;
}

export function markAntiZombieTriggered(nowMs: number = Date.now()): void {
  lastAntiZombieTriggeredAtMs = nowMs;
}

export function resetAntiZombieForTests(): void {
  lastAntiZombieTriggeredAtMs = 0;
}
