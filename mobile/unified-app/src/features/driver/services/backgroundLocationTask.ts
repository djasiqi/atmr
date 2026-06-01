import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Location from "expo-location";
import * as Battery from "expo-battery";
import { AppState, AppStateStatus, Platform } from "react-native";
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
import {
  clearNativeStartFailure,
  clearPendingFgsStart,
  getPendingFgsStart,
  recordNativeStartFailure,
  setLastTaskInvokedAt,
  setPendingFgsStart,
} from "./trackingRuntime";

/**
 * Notifie le heartbeat de santé tracking en cas d'échec de démarrage natif —
 * lazy-required pour éviter le cycle deviceHealthHeartbeat <-> backgroundLocationTask.
 * Fire-and-forget : toute erreur réseau est swallow par le module heartbeat.
 */
function notifyDeviceHealthOnNativeStartFailure(reason: string): void {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./deviceHealthHeartbeat") as typeof import("./deviceHealthHeartbeat");
    if (typeof mod.triggerDeviceHealthNow === "function") {
      void mod.triggerDeviceHealthNow(`native_start_failure:${reason}`).catch(() => undefined);
    }
  } catch {
    /* noop */
  }
}

type BackgroundTrackingTaskMode = "mission" | "presence_window";

type BackgroundTaskRuntimeContext = {
  missionId: number | null;
  missionStatus: DriverMissionStatus | null;
  taskMode: BackgroundTrackingTaskMode;
  updatedAt: string;
};

export const BACKGROUND_LOCATION_TASK_NAME = "background-location-task";

const FGS_START_TIMEOUT_MS = 30_000;
const FGS_WATCHDOG_RETRY_MS = 2_500;

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
let bgStartInProgress = false;
let watchdogActive = false;
let watchdogTimer: ReturnType<typeof setInterval> | null = null;
let watchdogStartedAt = 0;
let watchdogMissionId: number | null = null;
let watchdogMissionStatus: DriverMissionStatus | null = null;
let watchdogPresenceWindow = false;
let watchdogReason = "watchdog";

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
        typeof parsed.missionStatus === "string"
          ? (parsed.missionStatus as DriverMissionStatus)
          : null,
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

export function isBackgroundLocationTaskDefined(): boolean {
  return taskDefined;
}

export async function getNativeTaskLifecycleStatus(): Promise<{
  taskDefined: boolean;
  taskStarted: boolean;
}> {
  const defined = taskDefined;
  let started = false;
  if (typeof Location.hasStartedLocationUpdatesAsync === "function") {
    started = await Location.hasStartedLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME).catch(
      () => false
    );
  }
  return { taskDefined: defined, taskStarted: started };
}

export async function isNativeBackgroundTrackingRunning(): Promise<boolean> {
  const { taskStarted } = await getNativeTaskLifecycleStatus();
  return taskStarted;
}

async function readPermissionSnapshot(): Promise<{
  fg: "granted" | "denied" | "undetermined";
  bg: "granted" | "denied" | "undetermined";
}> {
  try {
    const [fg, bg] = await Promise.all([
      Location.getForegroundPermissionsAsync(),
      Location.getBackgroundPermissionsAsync(),
    ]);
    const toStatus = (s: { status?: string }): "granted" | "denied" | "undetermined" =>
      s?.status === "granted" ? "granted" : s?.status === "denied" ? "denied" : "undetermined";
    return { fg: toStatus(fg), bg: toStatus(bg) };
  } catch {
    return { fg: "undetermined", bg: "undetermined" };
  }
}

function stopNativeTrackingWatchdog(): void {
  watchdogActive = false;
  if (watchdogTimer) {
    clearInterval(watchdogTimer);
    watchdogTimer = null;
  }
}

async function emitRegistrationStatus(): Promise<void> {
  const lifecycle = await getNativeTaskLifecycleStatus();
  emitDriverTelemetry("tracking.background.task.registration_status", {
    source: "driver.services.backgroundLocationTask",
    task_name: BACKGROUND_LOCATION_TASK_NAME,
    task_defined: lifecycle.taskDefined,
    task_started: lifecycle.taskStarted,
    runtime: describeBackgroundRuntime(),
    bg_flag_enabled: isFeatureEnabled("tracking_background_enabled"),
  });
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

      const locations = Array.isArray(data?.locations) ? data.locations : [];
      setLastTaskInvokedAt(Date.now());
      emitDriverTelemetry("tracking.background.task_invoked", {
        source: "driver.services.backgroundLocationTask",
        task_name: BACKGROUND_LOCATION_TASK_NAME,
        locations_count: locations.length,
        timestamp: Date.now(),
        mission_id: context.missionId,
        task_mode: context.taskMode,
      });

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
  presenceWindow?: boolean;
};

async function startBackgroundLocationTaskIfEligibleInternal(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null,
  options: StartBackgroundOptions = {},
  startReason: string
): Promise<boolean> {
  if (
    typeof Location.hasStartedLocationUpdatesAsync !== "function" ||
    typeof Location.startLocationUpdatesAsync !== "function"
  ) {
    return false;
  }
  if (!isFeatureEnabled("tracking_background_enabled")) return false;
  const isPresenceWindow = options.presenceWindow === true;
  const taskMode: BackgroundTrackingTaskMode = isPresenceWindow ? "presence_window" : "mission";
  if (!isPresenceWindow && (missionId == null || !isTrackingActiveStatus(missionStatus))) {
    return false;
  }
  if (!canUseBackgroundLocation()) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.services.backgroundLocationTask",
      reason: "runtime_unsupported",
      runtime: describeBackgroundRuntime(),
      mission_id: missionId,
      task_mode: taskMode,
      task_name: BACKGROUND_LOCATION_TASK_NAME,
    });
    return false;
  }
  if (await isKillSwitchEnabled()) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.services.backgroundLocationTask",
      reason: "kill_switch_enabled",
      mission_id: missionId,
      task_mode: taskMode,
      task_name: BACKGROUND_LOCATION_TASK_NAME,
    });
    return false;
  }

  defineTaskIfNeeded();
  const lifecycleBefore = await getNativeTaskLifecycleStatus();
  const perms = await readPermissionSnapshot();

  const granted = await ensurePermissions();
  if (!granted) {
    emitDriverTelemetry("tracking.permission.denied", {
      source: "driver.services.backgroundLocationTask",
      mission_id: missionId,
      task_mode: taskMode,
      app_state: AppState.currentState,
    });
    recordNativeStartFailure({
      reason: "permission_denied",
      error: "background or foreground location not granted",
    });
    notifyDeviceHealthOnNativeStartFailure("permission_denied");
    emitDriverTelemetry("tracking.background.start_failed", {
      source: "driver.services.backgroundLocationTask",
      reason: startReason,
      failure_reason: "permission_denied",
      task_defined: lifecycleBefore.taskDefined,
      task_started: false,
      app_state: AppState.currentState,
      fg_permission: perms.fg,
      bg_permission: perms.bg,
      mission_id: missionId,
    });
    return false;
  }

  await setBackgroundTrackingMissionContext(missionId, missionStatus, taskMode);
  const hasStarted = await Location.hasStartedLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME);
  if (hasStarted) {
    clearNativeStartFailure();
    clearPendingFgsStart();
    return true;
  }

  const batteryLevel = await Battery.getBatteryLevelAsync().catch(() => null);
  const isLowBattery = typeof batteryLevel === "number" && batteryLevel <= LOW_BATTERY_THRESHOLD;
  const intervalBase =
    resolveBackgroundTrackingMode(missionStatus, taskMode) === "availability_presence"
      ? Math.max(BACKGROUND_INTERVAL_MS, 90_000)
      : BACKGROUND_INTERVAL_MS;
  const effectiveIntervalMs = isLowBattery
    ? Math.max(intervalBase, BACKGROUND_INTERVAL_LOW_BATTERY_MS)
    : intervalBase;

  emitDriverTelemetry("tracking.background.start_requested", {
    source: "driver.services.backgroundLocationTask",
    reason: startReason,
    task_defined: lifecycleBefore.taskDefined,
    task_started: lifecycleBefore.taskStarted,
    app_state: AppState.currentState,
    fg_permission: perms.fg,
    bg_permission: perms.bg,
    mission_id: missionId,
    task_mode: taskMode,
  });

  const locationOptions: Location.LocationTaskOptions = {
    accuracy: isLowBattery ? Location.Accuracy.Low : Location.Accuracy.Balanced,
    timeInterval: effectiveIntervalMs,
    distanceInterval: BACKGROUND_DISTANCE_METERS,
    pausesUpdatesAutomatically: false,
    showsBackgroundLocationIndicator: true,
    foregroundService: {
      notificationTitle: FOREGROUND_SERVICE_TITLE,
      notificationBody: FOREGROUND_SERVICE_BODY,
      notificationColor: FOREGROUND_SERVICE_COLOR,
      killServiceOnDestroy: false,
    },
  };
  if (Platform.OS === "ios") {
    locationOptions.activityType = Location.ActivityType.AutomotiveNavigation;
  }

  try {
    await Location.startLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME, locationOptions);

    const lifecycleAfter = await getNativeTaskLifecycleStatus();

    emitDriverTelemetry("tracking.background.start_success", {
      source: "driver.services.backgroundLocationTask",
      reason: startReason,
      task_defined: lifecycleAfter.taskDefined,
      task_started: lifecycleAfter.taskStarted,
      app_state: AppState.currentState,
      mission_id: missionId,
      task_mode: taskMode,
    });

    if (!lifecycleAfter.taskStarted) {
      recordNativeStartFailure({
        reason: startReason,
        error: "startLocationUpdatesAsync returned without active native task",
      });
      notifyDeviceHealthOnNativeStartFailure("native_task_inactive");
      return false;
    }

    clearNativeStartFailure();
    clearPendingFgsStart();

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
    return true;
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    recordNativeStartFailure({ reason: startReason, error: message });
    notifyDeviceHealthOnNativeStartFailure("start_exception");
    emitDriverTelemetry("tracking.background.start_failed", {
      source: "driver.services.backgroundLocationTask",
      reason: startReason,
      failure_reason: "start_exception",
      error: message,
      task_defined: lifecycleBefore.taskDefined,
      task_started: false,
      app_state: AppState.currentState,
      fg_permission: perms.fg,
      bg_permission: perms.bg,
      mission_id: missionId,
    });
    return false;
  }
}

async function runWatchdogTick(): Promise<void> {
  if (!watchdogActive) return;
  if (bgStartInProgress) return;

  const lifecycle = await getNativeTaskLifecycleStatus();
  if (lifecycle.taskStarted) {
    stopNativeTrackingWatchdog();
    clearPendingFgsStart();
    clearNativeStartFailure();
    return;
  }

  if (Date.now() - watchdogStartedAt >= FGS_START_TIMEOUT_MS) {
    recordNativeStartFailure({
      reason: "startup_timeout",
      error: `FGS not running after ${FGS_START_TIMEOUT_MS}ms`,
    });
    notifyDeviceHealthOnNativeStartFailure("startup_timeout");
    emitDriverTelemetry("tracking.background.start_failed", {
      source: "driver.services.backgroundLocationTask",
      reason: watchdogReason,
      failure_reason: "startup_timeout",
      task_defined: lifecycle.taskDefined,
      task_started: false,
      app_state: AppState.currentState,
      mission_id: watchdogMissionId,
    });
    stopNativeTrackingWatchdog();
    return;
  }

  if (AppState.currentState !== "active") {
    return;
  }

  if (bgStartInProgress) return;
  bgStartInProgress = true;
  try {
    await startBackgroundLocationTaskIfEligibleInternal(
      watchdogMissionId,
      watchdogMissionStatus,
      { presenceWindow: watchdogPresenceWindow },
      watchdogReason
    );
  } finally {
    bgStartInProgress = false;
  }
}

function startNativeTrackingWatchdog(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null,
  options: StartBackgroundOptions,
  reason: string
): void {
  watchdogMissionId = missionId;
  watchdogMissionStatus = missionStatus;
  watchdogPresenceWindow = options.presenceWindow === true;
  watchdogReason = reason;
  watchdogStartedAt = Date.now();
  if (watchdogActive && watchdogTimer) {
    return;
  }
  watchdogActive = true;
  void runWatchdogTick();
  watchdogTimer = setInterval(() => {
    void runWatchdogTick();
  }, FGS_WATCHDOG_RETRY_MS);
}

/**
 * Démarre le tracking natif (FGS) tant que l'app est au premier plan, avec watchdog et defer BG.
 */
export async function ensureNativeTrackingWhileForeground(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null,
  options: StartBackgroundOptions = {},
  reason = "ensure_native_tracking"
): Promise<void> {
  if (Platform.OS === "web") return;
  if (!isFeatureEnabled("tracking_background_enabled")) return;
  if (!canUseBackgroundLocation()) return;

  const isPresenceWindow = options.presenceWindow === true;
  if (!isPresenceWindow && (missionId == null || !isTrackingActiveStatus(missionStatus))) {
    return;
  }

  const lifecycle = await getNativeTaskLifecycleStatus();
  if (lifecycle.taskStarted) {
    stopNativeTrackingWatchdog();
    clearPendingFgsStart();
    return;
  }

  if (AppState.currentState !== "active") {
    setPendingFgsStart({
      active: true,
      reason: "android_fgs_requires_foreground",
      missionId,
      deferredAt: Date.now(),
    });
    emitDriverTelemetry("tracking.background.start_deferred", {
      source: "driver.services.backgroundLocationTask",
      reason,
      app_state: AppState.currentState,
      mission_id: missionId,
      task_defined: lifecycle.taskDefined,
      task_started: lifecycle.taskStarted,
    });
    return;
  }

  clearPendingFgsStart();

  if (bgStartInProgress) {
    startNativeTrackingWatchdog(missionId, missionStatus, options, reason);
    return;
  }

  bgStartInProgress = true;
  try {
    await startBackgroundLocationTaskIfEligibleInternal(missionId, missionStatus, options, reason);
    const lifecycle = await getNativeTaskLifecycleStatus();
    if (lifecycle.taskStarted) {
      stopNativeTrackingWatchdog();
      return;
    }
    startNativeTrackingWatchdog(missionId, missionStatus, options, reason);
  } finally {
    bgStartInProgress = false;
  }
}

export async function restartNativeTrackingFromWake(reason = "silent_push_wake"): Promise<void> {
  if (Platform.OS === "web") return;
  try {
    await resumePendingNativeTrackingIfNeeded();
    const ctx = await readTaskContext();
    if (ctx?.taskMode === "presence_window") {
      await ensureNativeTrackingWhileForeground(null, null, { presenceWindow: true }, reason);
      return;
    }
    if (ctx?.missionId != null) {
      await ensureNativeTrackingWhileForeground(
        ctx.missionId,
        ctx.missionStatus,
        {},
        reason
      );
    }
    emitDriverTelemetry("tracking.background.wake_restart", {
      source: "driver.services.backgroundLocationTask",
      reason,
      mission_id: ctx?.missionId ?? null,
    });
  } catch (error) {
    emitDriverTelemetry("tracking.background.wake_restart_failed", {
      source: "driver.services.backgroundLocationTask",
      reason,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

/**
 * Fallback iOS : relance les updates avec cadence réduite si le tracking principal est inactif.
 */
export async function ensureIosSignificantLocationFallback(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null
): Promise<void> {
  if (Platform.OS !== "ios") return;
  const lifecycle = await getNativeTaskLifecycleStatus();
  if (lifecycle.taskStarted) return;
  await ensureNativeTrackingWhileForeground(missionId, missionStatus, {}, "ios_significant_fallback");
}

export async function resumePendingNativeTrackingIfNeeded(): Promise<void> {
  const pending = getPendingFgsStart();
  if (!pending.active) return;
  const ctx = await readTaskContext();
  if (ctx?.taskMode === "presence_window") {
    await ensureNativeTrackingWhileForeground(null, null, { presenceWindow: true }, "app_resume");
    return;
  }
  if (ctx?.missionId != null) {
    await ensureNativeTrackingWhileForeground(
      ctx.missionId,
      ctx.missionStatus,
      {},
      "app_resume_pending"
    );
  }
}

export async function startBackgroundLocationTaskIfEligible(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null,
  options: StartBackgroundOptions = {}
) {
  await ensureNativeTrackingWhileForeground(
    missionId,
    missionStatus,
    options,
    "start_background_location_task"
  );
}

export async function stopBackgroundLocationTask(reason: string) {
  stopNativeTrackingWatchdog();
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
  void emitRegistrationStatus();
}

/** Test-only reset */
export function __resetBackgroundLocationTaskStateForTests(): void {
  taskDefined = false;
  bgStartInProgress = false;
  stopNativeTrackingWatchdog();
}
