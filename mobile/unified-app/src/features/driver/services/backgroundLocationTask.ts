import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Location from "expo-location";
import * as Battery from "expo-battery";
import { AppStateStatus } from "react-native";
import { DriverMissionStatus } from "../types";
import { isTrackingActiveStatus } from "../domain/status";
import { driverTrackingQueue } from "./driverTrackingQueue";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { resolveTrackingCadence } from "../../../core/tracking/cadenceResolver";
import {
  canUseBackgroundLocation,
  describeBackgroundRuntime,
} from "./backgroundRuntimeCompat";

type BackgroundTrackingTaskMode = "mission" | "presence_window";

type BackgroundTaskRuntimeContext = {
  missionId: number | null;
  missionStatus: DriverMissionStatus | null;
  taskMode: BackgroundTrackingTaskMode;
  updatedAt: string;
};

const BACKGROUND_LOCATION_TASK_NAME = "background-location-task";
const TASK_CONTEXT_STORAGE_KEY = "@driver:bg_tracking_context_v1";
const KILL_SWITCH_KEY = "driver_background_tracking_enabled";
const BACKGROUND_INTERVAL_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_GPS_BACKGROUND_INTERVAL_MS ?? "20000"
);
const BACKGROUND_DISTANCE_METERS = Number(
  process.env.EXPO_PUBLIC_DRIVER_GPS_BACKGROUND_DISTANCE_METERS ?? "10"
);
const FOREGROUND_SERVICE_TITLE =
  process.env.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_TITLE ?? "Lirie Unified est active";
const FOREGROUND_SERVICE_BODY =
  process.env.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_BODY ??
  "Suivi de la localisation en cours pour votre mission active.";
const FOREGROUND_SERVICE_COLOR =
  process.env.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_COLOR ?? "#0A7F59";
const LOW_BATTERY_THRESHOLD = Number(process.env.EXPO_PUBLIC_DRIVER_LOW_BATTERY_THRESHOLD ?? "0.2");
const BACKGROUND_INTERVAL_LOW_BATTERY_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_GPS_BACKGROUND_INTERVAL_LOW_BATTERY_MS ?? "60000"
);

let taskDefined = false;
const inMemoryStorage = new Map<string, string>();

async function readStorage(key: string): Promise<string | null> {
  const storage = AsyncStorage as unknown as {
    getItem?: (input: string) => Promise<string | null>;
  };
  if (typeof storage?.getItem === "function") {
    return storage.getItem(key);
  }
  return inMemoryStorage.get(key) ?? null;
}

async function writeStorage(key: string, value: string): Promise<void> {
  const storage = AsyncStorage as unknown as {
    setItem?: (input: string, output: string) => Promise<void>;
  };
  if (typeof storage?.setItem === "function") {
    await storage.setItem(key, value);
    return;
  }
  inMemoryStorage.set(key, value);
}

async function removeStorage(key: string): Promise<void> {
  const storage = AsyncStorage as unknown as {
    removeItem?: (input: string) => Promise<void>;
  };
  if (typeof storage?.removeItem === "function") {
    await storage.removeItem(key);
    return;
  }
  inMemoryStorage.delete(key);
}

function parseBooleanLike(value: string | null): boolean | null {
  if (!value) return null;
  const normalized = value.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  return null;
}

async function isKillSwitchEnabled(): Promise<boolean> {
  const value = await readStorage(KILL_SWITCH_KEY);
  const parsed = parseBooleanLike(value);
  // Legacy behavior from ops: key means "enabled". kill-switch is active when explicitly false.
  if (parsed === false) return true;
  return false;
}

async function readTaskContext(): Promise<BackgroundTaskRuntimeContext | null> {
  const raw = await readStorage(TASK_CONTEXT_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<BackgroundTaskRuntimeContext>;
    if (!parsed || typeof parsed !== "object") return null;
    return {
      missionId:
        typeof parsed.missionId === "number" && Number.isFinite(parsed.missionId)
          ? parsed.missionId
          : null,
      missionStatus:
        typeof parsed.missionStatus === "string" ? (parsed.missionStatus as DriverMissionStatus) : null,
      taskMode:
        parsed.taskMode === "presence_window" ? "presence_window" : "mission",
      updatedAt:
        typeof parsed.updatedAt === "string" ? parsed.updatedAt : new Date().toISOString(),
    };
  } catch {
    return null;
  }
}

async function writeTaskContext(context: BackgroundTaskRuntimeContext | null): Promise<void> {
  if (!context) {
    await removeStorage(TASK_CONTEXT_STORAGE_KEY);
    return;
  }
  await writeStorage(TASK_CONTEXT_STORAGE_KEY, JSON.stringify(context));
}

function resolveBackgroundTrackingMode(
  missionStatus: DriverMissionStatus | null,
  taskMode: BackgroundTrackingTaskMode = "mission"
): "mission_live" | "availability_presence" {
  if (taskMode === "presence_window") {
    return "availability_presence";
  }
  if (missionStatus === "ASSIGNED" && isFeatureEnabled("tracking_presence_mode_enabled")) {
    return "availability_presence";
  }
  return "mission_live";
}

function defineTaskIfNeeded() {
  if (taskDefined) return;
  if (!canUseBackgroundLocation()) return;
  let TaskManager: { defineTask?: unknown };
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    TaskManager = require("expo-task-manager");
  } catch {
    return;
  }
  if (typeof TaskManager?.defineTask !== "function") return;
  TaskManager.defineTask(
    BACKGROUND_LOCATION_TASK_NAME,
    async ({
      data,
      error,
    }: {
      data?: { locations?: Location.LocationObject[] };
      error?: Error;
    }) => {
      if (error) {
        emitDriverTelemetry("tracking.background.task.error", {
          source: "driver.services.backgroundLocationTask",
          reason: error.message,
          task_name: BACKGROUND_LOCATION_TASK_NAME,
        });
        return;
      }
      if (!isFeatureEnabled("tracking_background_enabled")) return;
      if (await isKillSwitchEnabled()) {
        emitDriverTelemetry("tracking.background.task.skipped", {
          source: "driver.services.backgroundLocationTask",
          reason: "kill_switch_enabled",
          task_name: BACKGROUND_LOCATION_TASK_NAME,
        });
        return;
      }
      const context = await readTaskContext();
      if (!context) {
        emitDriverTelemetry("tracking.background.task.skipped", {
          source: "driver.services.backgroundLocationTask",
          reason: "no_active_context",
          task_name: BACKGROUND_LOCATION_TASK_NAME,
        });
        return;
      }
      const isMissionContext =
        context.taskMode === "mission" &&
        context.missionId != null &&
        isTrackingActiveStatus(context.missionStatus);
      const isPresenceContext = context.taskMode === "presence_window";
      if (!isMissionContext && !isPresenceContext) {
        emitDriverTelemetry("tracking.background.task.skipped", {
          source: "driver.services.backgroundLocationTask",
          reason: "context_ineligible",
          task_mode: context.taskMode,
          task_name: BACKGROUND_LOCATION_TASK_NAME,
        });
        return;
      }

      const locations = Array.isArray(data?.locations) ? data?.locations : [];
      const mode = resolveBackgroundTrackingMode(context.missionStatus, context.taskMode);
      for (const location of locations) {
        const timestamp = new Date(location.timestamp ?? Date.now()).toISOString();
        await driverTrackingQueue.enqueue({
          missionId: context.missionId,
          appState: "background" as AppStateStatus,
          locationMode: mode,
          payload: {
            latitude: location.coords.latitude,
            longitude: location.coords.longitude,
            accuracy: location.coords.accuracy ?? undefined,
            heading: location.coords.heading ?? undefined,
            speed: location.coords.speed ?? undefined,
            missionId: context.missionId ?? null,
            isBackground: true,
            timestamp,
            locationMode: mode,
          },
        });
      }
      const queueSnapshot = await driverTrackingQueue.getSnapshot();
      const cadence = isFeatureEnabled("tracking_adaptive_cadence_enabled")
        ? resolveTrackingCadence({
            mode,
            appState: "background",
            queueDepth: queueSnapshot.queueDepth,
            socketReady: false,
            consecutiveFailures: 0,
            previousProfile: null,
            profileSinceMs: Date.now(),
            nowMs: Date.now(),
          })
        : {
            networkProfile: "normal" as const,
            foregroundIntervalMs: 8_000,
            backgroundIntervalMs: BACKGROUND_INTERVAL_MS,
            ackStaleMs: 75_000,
          };
      const flushResult = await driverTrackingQueue.flush({
        ackStaleMs: cadence.ackStaleMs,
        networkProfile: cadence.networkProfile,
        forceHttpFallback: true,
      });
      emitDriverTelemetry("tracking.background.task.flush", {
        source: "driver.services.backgroundLocationTask",
        task_name: BACKGROUND_LOCATION_TASK_NAME,
        mission_id: context.missionId,
        task_mode: context.taskMode,
        queue_depth: flushResult.queueDepth,
        sent: flushResult.sent,
        backend_acked: flushResult.backendAcked,
        socket_emitted: flushResult.socketEmitted,
        network_profile_active: cadence.networkProfile,
        dropped: flushResult.dropped,
      });
    }
  );
  taskDefined = true;
}

async function ensurePermissions(): Promise<boolean> {
  if (
    typeof Location.requestForegroundPermissionsAsync !== "function" ||
    typeof Location.requestBackgroundPermissionsAsync !== "function"
  ) {
    return false;
  }
  const fg = await Location.requestForegroundPermissionsAsync();
  if (!fg.granted) return false;
  const bg = await Location.requestBackgroundPermissionsAsync().catch(() => ({ granted: false }));
  return !!bg.granted;
}

export async function setBackgroundTrackingMissionContext(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null,
  taskMode: BackgroundTrackingTaskMode = "mission"
) {
  if (missionId == null && taskMode !== "presence_window") {
    await writeTaskContext(null);
    return;
  }
  await writeTaskContext({
    missionId,
    missionStatus,
    taskMode,
    updatedAt: new Date().toISOString(),
  });
}

type StartBackgroundOptions = {
  /** Mode présence (07h–19h) sans mission active. */
  presenceWindow?: boolean;
};

export async function startBackgroundLocationTaskIfEligible(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null,
  options: StartBackgroundOptions = {}
) {
  if (
    typeof Location.hasStartedLocationUpdatesAsync !== "function" ||
    typeof Location.startLocationUpdatesAsync !== "function"
  ) {
    return;
  }
  if (!isFeatureEnabled("tracking_background_enabled")) return;
  const isPresenceWindow = options.presenceWindow === true;
  const taskMode: BackgroundTrackingTaskMode = isPresenceWindow ? "presence_window" : "mission";
  if (!isPresenceWindow && (missionId == null || !isTrackingActiveStatus(missionStatus))) return;
  if (!canUseBackgroundLocation()) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.services.backgroundLocationTask",
      reason: "runtime_unsupported",
      runtime: describeBackgroundRuntime(),
      mission_id: missionId,
      task_mode: taskMode,
      task_name: BACKGROUND_LOCATION_TASK_NAME,
    });
    return;
  }
  if (await isKillSwitchEnabled()) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.services.backgroundLocationTask",
      reason: "kill_switch_enabled",
      mission_id: missionId,
      task_mode: taskMode,
      task_name: BACKGROUND_LOCATION_TASK_NAME,
    });
    return;
  }
  defineTaskIfNeeded();
  const granted = await ensurePermissions();
  if (!granted) {
    emitDriverTelemetry("tracking.permission.denied", {
      source: "driver.services.backgroundLocationTask",
      mission_id: missionId,
      task_mode: taskMode,
      app_state: "background",
    });
    return;
  }
  await setBackgroundTrackingMissionContext(missionId, missionStatus, taskMode);
  const hasStarted = await Location.hasStartedLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME);
  if (hasStarted) return;
  const batteryLevel = await Battery.getBatteryLevelAsync().catch(() => null);
  const isLowBattery = typeof batteryLevel === "number" && batteryLevel <= LOW_BATTERY_THRESHOLD;
  const intervalBase =
    resolveBackgroundTrackingMode(missionStatus, taskMode) === "availability_presence"
      ? Math.max(BACKGROUND_INTERVAL_MS, 90_000)
      : BACKGROUND_INTERVAL_MS;
  const effectiveIntervalMs = isLowBattery
    ? Math.max(intervalBase, BACKGROUND_INTERVAL_LOW_BATTERY_MS)
    : intervalBase;

  await Location.startLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME, {
    accuracy: isLowBattery ? Location.Accuracy.Low : Location.Accuracy.Balanced,
    timeInterval: effectiveIntervalMs,
    distanceInterval: BACKGROUND_DISTANCE_METERS,
    pausesUpdatesAutomatically: false,
    showsBackgroundLocationIndicator: true,
    foregroundService: {
      notificationTitle: FOREGROUND_SERVICE_TITLE,
      notificationBody: FOREGROUND_SERVICE_BODY,
      notificationColor: FOREGROUND_SERVICE_COLOR,
    },
  });

  emitDriverTelemetry("tracking.background.task.started", {
    source: "driver.services.backgroundLocationTask",
    task_name: BACKGROUND_LOCATION_TASK_NAME,
    mission_id: missionId,
    task_mode: taskMode,
    interval_ms: effectiveIntervalMs,
    distance_m: BACKGROUND_DISTANCE_METERS,
    low_battery_mode: isLowBattery,
    battery_level: batteryLevel,
  });
}

export async function stopBackgroundLocationTask(reason: string) {
  await setBackgroundTrackingMissionContext(null, null);
  if (
    typeof Location.hasStartedLocationUpdatesAsync !== "function" ||
    typeof Location.stopLocationUpdatesAsync !== "function"
  ) {
    return;
  }
  const hasStarted = await Location.hasStartedLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME).catch(
    () => false
  );
  if (!hasStarted) return;
  await Location.stopLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME).catch(() => undefined);
  emitDriverTelemetry("tracking.background.task.stopped", {
    source: "driver.services.backgroundLocationTask",
    task_name: BACKGROUND_LOCATION_TASK_NAME,
    reason,
  });
}

export function initializeBackgroundLocationTask() {
  if (!isFeatureEnabled("tracking_background_enabled")) return;
  if (!canUseBackgroundLocation()) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.services.backgroundLocationTask",
      reason: "runtime_unsupported_init",
      runtime: describeBackgroundRuntime(),
      task_name: BACKGROUND_LOCATION_TASK_NAME,
    });
    return;
  }
  defineTaskIfNeeded();
}

