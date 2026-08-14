import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Location from "expo-location";
import * as Battery from "expo-battery";
import { AppState, AppStateStatus, Platform } from "react-native";
import { DriverMissionStatus, type DriverMission } from "../types";
import { isTrackingActiveStatus } from "../domain/status";
import { resolveMissionTrackingMode } from "../domain/resolveMissionTrackingMode";
import { driverTrackingQueue } from "./driverTrackingQueue";
import { createCaptureId } from "./captureId";
import { trackingQueueStore } from "./trackingQueueStore";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { PRODUCTION_LOCALE } from "../../../i18n/productionLocale";
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
  clearNativeStartDiagnostics,
  recordNativeStartDiagnostics,
  recordNativeStartFailure,
  setLastTaskInvokedAt,
  setPendingFgsStart,
} from "./trackingRuntime";
import {
  leaseAllowsCapture,
  leaseAllowsTransport,
  readTrackingContextLease,
} from "./trackingContextLease";
import { validateNativeOwnerForHeadless } from "./trackingRuntimeRegistry";
import { ensureTrackingAuthAvailabilityForHeadless } from "../../../core/auth/trackingAuthPresence";
import { isWithinTrackingWindow } from "./trackingWindow";
import {
  canAttemptNativeStartNow,
  ensureNativeLifecycleAppStateBridge,
  getNativeLifecycleInFlight,
  requestNativeRecover,
  requestNativeStart,
  requestNativeStop,
  __resetNativeTrackingLifecycleForTests,
  type NativeStartRunResult,
  type NativeStopRunResult,
} from "./nativeTrackingLifecycle";

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

/** P0-A instrumentation — corrélation START/STOP (pas de changement de décision métier). */
type NativeLifecycleErrorFields = {
  error_name: string | null;
  error_code: string | null;
  error_message: string;
  error_stack: string | null;
};

function createNativeLifecycleOpId(kind: "start" | "stop"): string {
  return `nlo_${kind}_${createCaptureId()}`;
}

function extractNativeLifecycleError(error: unknown): NativeLifecycleErrorFields {
  if (error instanceof Error) {
    const withCode = error as Error & { code?: unknown };
    const rawCode = withCode.code;
    const error_code =
      typeof rawCode === "string" || typeof rawCode === "number" ? String(rawCode) : null;
    return {
      error_name: error.name || null,
      error_code,
      error_message: error.message,
      error_stack: typeof error.stack === "string" ? error.stack.slice(0, 1500) : null,
    };
  }
  return {
    error_name: null,
    error_code: null,
    error_message: String(error),
    error_stack: null,
  };
}

function formatNativeStartErrorForHealth(
  opId: string,
  reason: string,
  fields: Pick<NativeLifecycleErrorFields, "error_name" | "error_code" | "error_message">
): string {
  const name = fields.error_name ?? "Error";
  const code = fields.error_code ? `/${fields.error_code}` : "";
  return `[${opId}] ${reason}: ${name}${code}: ${fields.error_message}`;
}

type BackgroundTrackingTaskMode = "mission" | "presence_window";

type MissionSchedulingSnapshot = Pick<DriverMission, "scheduled_time" | "time_confirmed" | "scheduling">;

type NativeTrackingOwnerPersist = {
  trackingGenerationId: string;
  sessionGenerationId: number;
  trackingIdentityId: string;
  missionContextVersion: number;
  /** Mission portée par le propriétaire natif (null = présence). */
  missionId: number | null;
  driverId: number;
};

type BackgroundTaskRuntimeContext = {
  missionId: number | null;
  missionStatus: DriverMissionStatus | null;
  missionScheduling: MissionSchedulingSnapshot | null;
  taskMode: BackgroundTrackingTaskMode;
  updatedAt: string;
  nativeOwner?: NativeTrackingOwnerPersist | null;
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
/** Mission active : intervalle temporel seul (0 = pas de filtre distance, fix immobile). */
const BACKGROUND_MISSION_DISTANCE_METERS = Number(
  process.env.EXPO_PUBLIC_DRIVER_GPS_BACKGROUND_MISSION_DISTANCE_METERS ?? "0"
);
const FOREGROUND_SERVICE_TITLE =
  process.env.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_TITLE ?? PRODUCTION_LOCALE.fgsNotificationTitle;
const FOREGROUND_SERVICE_BODY_MISSION =
  process.env.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_BODY ??
  PRODUCTION_LOCALE.fgsNotificationBodyMission;
const FOREGROUND_SERVICE_BODY_PRESENCE =
  process.env.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_BODY_PRESENCE ??
  PRODUCTION_LOCALE.fgsNotificationBodyPresence;

function resolveBackgroundDistanceMeters(taskMode: BackgroundTrackingTaskMode): number {
  if (taskMode === "presence_window") {
    return BACKGROUND_DISTANCE_METERS;
  }
  return BACKGROUND_MISSION_DISTANCE_METERS;
}

function resolveForegroundServiceNotification(
  taskMode: BackgroundTrackingTaskMode
): { title: string; body: string } {
  if (taskMode === "presence_window") {
    return {
      title: FOREGROUND_SERVICE_TITLE,
      body: FOREGROUND_SERVICE_BODY_PRESENCE,
    };
  }
  return {
    title: FOREGROUND_SERVICE_TITLE,
    body: FOREGROUND_SERVICE_BODY_MISSION,
  };
}
const FOREGROUND_SERVICE_COLOR =
  process.env.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_COLOR ?? "#0A7F59";
const LOW_BATTERY_THRESHOLD = Number(process.env.EXPO_PUBLIC_DRIVER_LOW_BATTERY_THRESHOLD ?? "0.2");
const BACKGROUND_INTERVAL_LOW_BATTERY_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_GPS_BACKGROUND_INTERVAL_LOW_BATTERY_MS ?? "60000"
);

/**
 * P0-F — qualité GPS BG : mission_live ignore la dégradation basse batterie
 * (Accuracy.High + cadence mission). Seule availability_presence peut assouplir.
 */
export function resolveBackgroundGpsQuality(input: {
  trackingMode: string;
  isLowBattery: boolean;
  missionIntervalMs?: number;
  presenceMinIntervalMs?: number;
  lowBatteryIntervalMs?: number;
}): { accuracy: Location.Accuracy; timeIntervalMs: number; batteryDegradesGps: boolean } {
  const missionIntervalMs = input.missionIntervalMs ?? BACKGROUND_INTERVAL_MS;
  const presenceMinIntervalMs = input.presenceMinIntervalMs ?? 90_000;
  const lowBatteryIntervalMs =
    input.lowBatteryIntervalMs ?? BACKGROUND_INTERVAL_LOW_BATTERY_MS;
  const isMissionLive = input.trackingMode === "mission_live";
  const isPresence = input.trackingMode === "availability_presence";

  if (isMissionLive) {
    return {
      accuracy: Location.Accuracy.High,
      timeIntervalMs: missionIntervalMs,
      batteryDegradesGps: false,
    };
  }

  const intervalBase = isPresence
    ? Math.max(missionIntervalMs, presenceMinIntervalMs)
    : missionIntervalMs;
  const batteryDegradesGps = input.isLowBattery;
  const timeIntervalMs = batteryDegradesGps
    ? Math.max(intervalBase, lowBatteryIntervalMs)
    : intervalBase;
  const accuracy = batteryDegradesGps
    ? Location.Accuracy.Low
    : Location.Accuracy.Balanced;

  return { accuracy, timeIntervalMs, batteryDegradesGps };
}

let taskDefined = false;
let bgStartInProgress = false;
let watchdogActive = false;
let watchdogTimer: ReturnType<typeof setInterval> | null = null;
let watchdogStartedAt = 0;
let watchdogMissionId: number | null = null;
let watchdogMissionStatus: DriverMissionStatus | null = null;
let watchdogPresenceWindow = false;
let watchdogReason = "watchdog";

/** Sérialise start/stop/mutations de contexte (anti-TOCTOU présence ↔ mission). */
let lifecycleLockTail: Promise<void> = Promise.resolve();

async function withBackgroundTrackingLifecycleLock<T>(
  fn: () => Promise<T>
): Promise<T> {
  const previous = lifecycleLockTail;
  let release!: () => void;
  lifecycleLockTail = new Promise<void>((resolve) => {
    release = resolve;
  });
  await previous;
  try {
    return await fn();
  } finally {
    release();
  }
}

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
    const owner = parsed.nativeOwner as NativeTrackingOwnerPersist | null | undefined;
    return {
      missionId:
        typeof parsed.missionId === "number" && Number.isFinite(parsed.missionId)
          ? parsed.missionId
          : null,
      missionStatus:
        typeof parsed.missionStatus === "string"
          ? (parsed.missionStatus as DriverMissionStatus)
          : null,
      missionScheduling:
        parsed.missionScheduling && typeof parsed.missionScheduling === "object"
          ? (parsed.missionScheduling as MissionSchedulingSnapshot)
          : null,
      taskMode:
        parsed.taskMode === "presence_window" ? "presence_window" : "mission",
      updatedAt:
        typeof parsed.updatedAt === "string" ? parsed.updatedAt : new Date().toISOString(),
      nativeOwner:
        owner &&
        typeof owner.trackingGenerationId === "string" &&
        typeof owner.trackingIdentityId === "string" &&
        typeof owner.driverId === "number" &&
        Number.isFinite(owner.driverId)
          ? {
              trackingGenerationId: owner.trackingGenerationId,
              sessionGenerationId:
                typeof owner.sessionGenerationId === "number"
                  ? owner.sessionGenerationId
                  : 0,
              trackingIdentityId: owner.trackingIdentityId,
              missionContextVersion:
                typeof owner.missionContextVersion === "number"
                  ? owner.missionContextVersion
                  : 0,
              missionId:
                owner.missionId === null || owner.missionId === undefined
                  ? null
                  : typeof owner.missionId === "number" && Number.isFinite(owner.missionId)
                    ? owner.missionId
                    : null,
              driverId: owner.driverId,
            }
          : null,
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
  taskMode: BackgroundTrackingTaskMode = "mission",
  scheduling: MissionSchedulingSnapshot | null = null
): "mission_live" | "availability_presence" {
  if (taskMode === "presence_window") {
    return "availability_presence";
  }
  if (missionStatus) {
    const mission: DriverMission = {
      id: 0,
      status: missionStatus,
      scheduled_time: scheduling?.scheduled_time ?? null,
      time_confirmed: scheduling?.time_confirmed ?? null,
      scheduling: scheduling?.scheduling ?? null,
    };
    const mode = resolveMissionTrackingMode(mission);
    if (mode) return mode;
  }
  return "mission_live";
}

export function isBackgroundLocationTaskDefined(): boolean {
  return taskDefined;
}

function isTaskManagerTaskDefined(): boolean {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const TaskManager = require("expo-task-manager") as {
      isTaskDefined?: (taskName: string) => boolean;
    };
    if (typeof TaskManager?.isTaskDefined === "function") {
      return TaskManager.isTaskDefined(BACKGROUND_LOCATION_TASK_NAME);
    }
  } catch {
    /* noop */
  }
  return false;
}

type TaskManagerModule = {
  isTaskDefined?: (taskName: string) => boolean;
  isTaskRegisteredAsync?: (taskName: string) => Promise<boolean>;
  unregisterTaskAsync?: (taskName: string) => Promise<void>;
};

function readTaskManagerModule(): TaskManagerModule | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    return require("expo-task-manager") as TaskManagerModule;
  } catch {
    return null;
  }
}

function isBenignTaskLifecycleError(error: unknown): boolean {
  const message = (error instanceof Error ? error.message : String(error)).toLowerCase();
  return (
    message.includes("tasknotfound") ||
    message.includes("not found for app id") ||
    message.includes("unregistertaskasync")
  );
}

async function isBackgroundLocationTaskRegistered(): Promise<boolean> {
  const TaskManager = readTaskManagerModule();
  if (typeof TaskManager?.isTaskRegisteredAsync !== "function") {
    return false;
  }
  return TaskManager.isTaskRegisteredAsync(BACKGROUND_LOCATION_TASK_NAME).catch(() => false);
}

/**
 * STOP Expo Location — corps exécuté uniquement via requestNativeStop (P0-A).
 * Ne jamais appeler directement depuis un run() imbriqué d'un autre requestNative*.
 */
async function stopNativeBackgroundLocationUpdatesUnlocked(
  stopReason = "stop_native_background"
): Promise<NativeStopRunResult> {
  if (
    typeof Location.hasStartedLocationUpdatesAsync !== "function" ||
    typeof Location.stopLocationUpdatesAsync !== "function"
  ) {
    return { ok: true, nativeStopped: false };
  }

  defineTaskIfNeeded();

  const stopAttemptId = createNativeLifecycleOpId("stop");
  const stopRequestedAt = Date.now();
  const appStateAtRequest = AppState.currentState;
  const priorContext = await readTaskContext();
  const inFlight = getNativeLifecycleInFlight();

  const [hasStartedBefore, isRegisteredBefore] = await Promise.all([
    readNativeLocationUpdatesStarted(),
    isBackgroundLocationTaskRegistered(),
  ]);

  if (!isRegisteredBefore && !hasStartedBefore) {
    return { ok: true, nativeStopped: false };
  }

  if (!isRegisteredBefore) {
    emitDriverTelemetry("tracking.background.task.stop_skipped", {
      source: "driver.services.backgroundLocationTask",
      task_name: BACKGROUND_LOCATION_TASK_NAME,
      reason: "task_not_registered",
      stop_attempt_id: stopAttemptId,
      stop_requested_at: stopRequestedAt,
      stop_reason: stopReason,
      app_state_at_request: appStateAtRequest,
      has_started_flag: hasStartedBefore,
      isTaskRegisteredAsync_before: false,
      hasStartedLocationUpdatesAsync_before: hasStartedBefore,
      stop_in_flight: inFlight.stop_in_flight,
      start_in_flight: inFlight.start_in_flight,
      native_owner: priorContext?.nativeOwner ?? null,
      mission_id: priorContext?.missionId ?? null,
    });
    return { ok: true, nativeStopped: false };
  }

  emitDriverTelemetry("tracking.background.stop_requested", {
    source: "driver.services.backgroundLocationTask",
    task_name: BACKGROUND_LOCATION_TASK_NAME,
    stop_attempt_id: stopAttemptId,
    stop_requested_at: stopRequestedAt,
    stop_reason: stopReason,
    app_state_at_request: appStateAtRequest,
    isTaskRegisteredAsync_before: isRegisteredBefore,
    hasStartedLocationUpdatesAsync_before: hasStartedBefore,
    stop_in_flight: inFlight.stop_in_flight,
    start_in_flight: inFlight.start_in_flight,
    native_owner: priorContext?.nativeOwner ?? null,
    mission_id: priorContext?.missionId ?? null,
  });

  try {
    await Location.stopLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME);
    const [hasStartedAfter, isRegisteredAfter] = await Promise.all([
      readNativeLocationUpdatesStarted(),
      isBackgroundLocationTaskRegistered(),
    ]);
    const inFlightAfter = getNativeLifecycleInFlight();
    emitDriverTelemetry("tracking.background.stop_success", {
      source: "driver.services.backgroundLocationTask",
      task_name: BACKGROUND_LOCATION_TASK_NAME,
      stop_attempt_id: stopAttemptId,
      stop_requested_at: stopRequestedAt,
      stop_reason: stopReason,
      app_state_at_request: appStateAtRequest,
      app_state_at_resolve: AppState.currentState,
      isTaskRegisteredAsync_before: isRegisteredBefore,
      hasStartedLocationUpdatesAsync_before: hasStartedBefore,
      isTaskRegisteredAsync_after: isRegisteredAfter,
      hasStartedLocationUpdatesAsync_after: hasStartedAfter,
      stop_in_flight: inFlightAfter.stop_in_flight,
      start_in_flight: inFlightAfter.start_in_flight,
      native_owner: priorContext?.nativeOwner ?? null,
      mission_id: priorContext?.missionId ?? null,
    });
    return { ok: true, nativeStopped: true };
  } catch (error) {
    const errFields = extractNativeLifecycleError(error);
    if (!isBenignTaskLifecycleError(error)) {
      const inFlightFail = getNativeLifecycleInFlight();
      emitDriverTelemetry("tracking.background.stop_failed", {
        source: "driver.services.backgroundLocationTask",
        task_name: BACKGROUND_LOCATION_TASK_NAME,
        stop_attempt_id: stopAttemptId,
        stop_requested_at: stopRequestedAt,
        stop_reason: stopReason,
        app_state_at_request: appStateAtRequest,
        app_state_at_resolve: AppState.currentState,
        isTaskRegisteredAsync_before: isRegisteredBefore,
        hasStartedLocationUpdatesAsync_before: hasStartedBefore,
        stop_in_flight: inFlightFail.stop_in_flight,
        start_in_flight: inFlightFail.start_in_flight,
        native_owner: priorContext?.nativeOwner ?? null,
        mission_id: priorContext?.missionId ?? null,
        error: errFields.error_message,
        error_name: errFields.error_name,
        error_code: errFields.error_code,
        error_stack: errFields.error_stack,
      });
      // Alias historique (listeners / dashboards existants)
      emitDriverTelemetry("tracking.background.task.stop_failed", {
        source: "driver.services.backgroundLocationTask",
        task_name: BACKGROUND_LOCATION_TASK_NAME,
        stop_attempt_id: stopAttemptId,
        error: errFields.error_message,
        error_name: errFields.error_name,
        error_code: errFields.error_code,
      });
      return {
        ok: false,
        nativeStopped: false,
        errorCode: errFields.error_code,
        errorName: errFields.error_name,
        errorMessage: errFields.error_message,
      };
    }
    return { ok: true, nativeStopped: true };
  }
}

async function stopNativeBackgroundLocationUpdatesSafely(
  stopReason = "stop_native_background"
): Promise<void> {
  await requestNativeStop({
    reason: stopReason,
    run: () => stopNativeBackgroundLocationUpdatesUnlocked(stopReason),
  });
}

async function readNativeLocationUpdatesStarted(): Promise<boolean> {
  if (typeof Location.hasStartedLocationUpdatesAsync !== "function") return false;
  return Location.hasStartedLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME).catch(() => false);
}

export async function getNativeTaskLifecycleStatus(): Promise<{
  taskDefined: boolean;
  taskStarted: boolean;
}> {
  const defined = taskDefined || isTaskManagerTaskDefined();
  const started = await readNativeLocationUpdatesStarted();
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

      // P0 : lease AVANT taskContext / SQLite / enqueue (fail-closed réseau).
      const lease = await readTrackingContextLease();
      if (!leaseAllowsCapture(lease)) {
        emitDriverTelemetry("tracking.background.task.skipped", {
          source: "driver.services.backgroundLocationTask",
          reason:
            lease?.state === "switching"
              ? "lease_switching_capture_blocked"
              : "context_not_driver",
          lease_state: lease?.state ?? "absent",
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

      const auth = await ensureTrackingAuthAvailabilityForHeadless();
      const authUsable =
        auth.kind === "SESSION_AVAILABLE" ||
        auth.kind === "AUTH_TEMPORARILY_UNAVAILABLE";
      const ownerCheck = validateNativeOwnerForHeadless({
        owner: context.nativeOwner ?? null,
        lease,
        authUsable,
      });
      // Pendant switching : capture locale OK si owner présent et lease fromDriver ;
      // owner/lease génération ne matchent que sur driver_active.
      if (lease?.state === "driver_active" && !ownerCheck.ok) {
        emitDriverTelemetry("tracking.background.task.skipped", {
          source: "driver.services.backgroundLocationTask",
          reason: ownerCheck.reason,
          task_name: BACKGROUND_LOCATION_TASK_NAME,
        });
        return;
      }
      // Gate capture : contexte task vs owner (mission / version).
      if (
        lease?.state === "driver_active" &&
        context.nativeOwner &&
        (context.nativeOwner.missionId !== context.missionId ||
          (lease.missionId !== undefined &&
            context.nativeOwner.missionId !== lease.missionId) ||
          (typeof lease.missionContextVersion === "number" &&
            context.nativeOwner.missionContextVersion !== lease.missionContextVersion))
      ) {
        emitDriverTelemetry("tracking.background.task.skipped", {
          source: "driver.services.backgroundLocationTask",
          reason: "mission_or_version_mismatch",
          task_name: BACKGROUND_LOCATION_TASK_NAME,
          context_mission_id: context.missionId,
          owner_mission_id: context.nativeOwner.missionId,
          owner_mission_context_version: context.nativeOwner.missionContextVersion,
          lease_mission_id: lease.state === "driver_active" ? lease.missionId : null,
          lease_mission_context_version:
            lease.state === "driver_active" ? lease.missionContextVersion : null,
        });
        return;
      }
      if (lease?.state === "switching") {
        if (!context.nativeOwner) {
          emitDriverTelemetry("tracking.background.task.skipped", {
            source: "driver.services.backgroundLocationTask",
            reason: "missing_native_owner",
            task_name: BACKGROUND_LOCATION_TASK_NAME,
          });
          return;
        }
        if (!authUsable) {
          emitDriverTelemetry("tracking.background.task.skipped", {
            source: "driver.services.backgroundLocationTask",
            reason: "auth_not_usable",
            task_name: BACKGROUND_LOCATION_TASK_NAME,
          });
          return;
        }
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

      // TIME-3D : présence hors fenêtre Zurich → aucune capture + arrêt conditionnel (génération)
      if (isPresenceContext && !isWithinTrackingWindow(new Date())) {
        const expectedGenerationId =
          context.nativeOwner?.trackingGenerationId ?? null;
        const expectedMissionContextVersion =
          context.nativeOwner?.missionContextVersion ?? null;
        emitDriverTelemetry("tracking.background.task.skipped", {
          source: "driver.services.backgroundLocationTask",
          reason: "presence_window_closed",
          task_name: BACKGROUND_LOCATION_TASK_NAME,
        });
        await stopPresenceWindowIfStillCurrent({
          expectedGenerationId,
          expectedMissionContextVersion,
          reason: "presence_window_closed",
        });
        return;
      }

      // Sondage headless AVANT tout enfilage/flush : un handle SQLite non durable ou un schéma
      // pas prêt doit bloquer net (jamais de fallback HTTP silencieux depuis le background task).
      const health = await trackingQueueStore.initAndHealthcheckHeadless();
      if (!health.durable || !health.schemaReady) {
        emitDriverTelemetry("sqlite_headless_init_failed", {
          source: "driver.services.backgroundLocationTask",
          task_name: BACKGROUND_LOCATION_TASK_NAME,
          durable: health.durable,
          schema_ready: health.schemaReady,
          recovered: health.recovered,
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

      const mode = resolveBackgroundTrackingMode(
        context.missionStatus,
        context.taskMode,
        context.missionScheduling
      );
      for (const location of locations) {
        const timestamp = new Date(location.timestamp ?? Date.now()).toISOString();
        const lat = location.coords.latitude;
        const lon = location.coords.longitude;
        const osIdRaw = (location as unknown as { id?: unknown }).id;
        const osId = typeof osIdRaw === "string" ? osIdRaw : null;
        const captureId = osId ? `os:${osId}` : createCaptureId();
        const trackingGenerationId =
          context.nativeOwner?.trackingGenerationId ?? null;
        const missionContextVersion =
          context.nativeOwner?.missionContextVersion ?? null;
        await driverTrackingQueue.enqueue({
          missionId: context.missionId,
          appState: "background" as AppStateStatus,
          locationMode: mode,
          captureId,
          trackingGenerationId,
          missionContextVersion,
          payload: {
            latitude: lat,
            longitude: lon,
            accuracy: location.coords.accuracy ?? undefined,
            heading: location.coords.heading ?? undefined,
            speed: location.coords.speed ?? undefined,
            missionId: context.missionId ?? null,
            isBackground: true,
            timestamp,
            locationMode: mode,
            trackingGenerationId,
            missionContextVersion,
          },
        });
      }

      // Transport : uniquement si lease driver_active
      if (!leaseAllowsTransport(lease)) {
        emitDriverTelemetry("tracking.background.task.skipped", {
          source: "driver.services.backgroundLocationTask",
          reason: "transport_blocked_lease",
          lease_state: lease?.state ?? "absent",
          task_name: BACKGROUND_LOCATION_TASK_NAME,
          locations_count: locations.length,
        });
        return;
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
      let flushResult = await driverTrackingQueue.flush({
        ackStaleMs: cadence.ackStaleMs,
        networkProfile: cadence.networkProfile,
        forceHttpFallback: true,
      });
      if (flushResult.sent === 0 && locations.length > 0 && flushResult.queueDepth > 0) {
        flushResult = await driverTrackingQueue.flush({
          ackStaleMs: cadence.ackStaleMs,
          networkProfile: cadence.networkProfile,
          forceHttpFallback: true,
        });
      }
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
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const bridge = require("./driverTrackingBridge") as typeof import("./driverTrackingBridge");
        if (typeof bridge.syncBridgeQueueDepthFromPersistence === "function") {
          void bridge.syncBridgeQueueDepthFromPersistence();
        }
      } catch {
        /* noop — resync best-effort */
      }
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
  taskMode: BackgroundTrackingTaskMode = "mission",
  scheduling?: MissionSchedulingSnapshot | null,
  nativeOwner?: NativeTrackingOwnerPersist | null
) {
  if (missionId == null && taskMode !== "presence_window") {
    await writeTaskContext(null);
    return;
  }
  const prior = await readTaskContext();
  // undefined = conserver ; null = clear explicite ; object = remplacer
  const resolvedOwner =
    nativeOwner === undefined ? (prior?.nativeOwner ?? null) : nativeOwner;
  await writeTaskContext({
    missionId,
    missionStatus,
    missionScheduling: scheduling ?? null,
    taskMode,
    updatedAt: new Date().toISOString(),
    nativeOwner: resolvedOwner,
  });
}

type StartBackgroundOptions = {
  presenceWindow?: boolean;
  scheduling?: MissionSchedulingSnapshot | null;
  /** Owner natif à persister avec le contexte (évite write sans owner pendant switch). */
  nativeOwner?: NativeTrackingOwnerPersist | null;
};

async function startBackgroundLocationTaskIfEligibleInternal(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null,
  options: StartBackgroundOptions = {},
  startReason: string
): Promise<NativeStartRunResult> {
  if (
    typeof Location.hasStartedLocationUpdatesAsync !== "function" ||
    typeof Location.startLocationUpdatesAsync !== "function"
  ) {
    return { ok: false, nativeStarted: false };
  }
  if (!isFeatureEnabled("tracking_background_enabled")) {
    return { ok: false, nativeStarted: false };
  }
  const isPresenceWindow = options.presenceWindow === true;
  const taskMode: BackgroundTrackingTaskMode = isPresenceWindow ? "presence_window" : "mission";
  if (!isPresenceWindow && (missionId == null || !isTrackingActiveStatus(missionStatus))) {
    return { ok: false, nativeStarted: false };
  }
  // TIME-3C : refuser tout start/reprise présence hors fenêtre Europe/Zurich
  if (isPresenceWindow && !isWithinTrackingWindow(new Date())) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.services.backgroundLocationTask",
      reason: "presence_window_closed",
      runtime: describeBackgroundRuntime(),
      mission_id: missionId,
      task_mode: taskMode,
      task_name: BACKGROUND_LOCATION_TASK_NAME,
      start_reason: startReason,
    });
    return { ok: false, nativeStarted: false };
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
    return { ok: false, nativeStarted: false };
  }
  if (await isKillSwitchEnabled()) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.services.backgroundLocationTask",
      reason: "kill_switch_enabled",
      mission_id: missionId,
      task_mode: taskMode,
      task_name: BACKGROUND_LOCATION_TASK_NAME,
    });
    return { ok: false, nativeStarted: false };
  }

  defineTaskIfNeeded();
  const lifecycleBefore = await getNativeTaskLifecycleStatus();
  const perms = await readPermissionSnapshot();

  const granted = await ensurePermissions();
  if (!granted) {
    recordNativeStartDiagnostics({
      native_start_phase: "ensurePermissions",
      native_start_error: "background or foreground location not granted",
      native_task_defined: lifecycleBefore.taskDefined || isTaskManagerTaskDefined(),
      native_started_before: lifecycleBefore.taskStarted,
      native_started_after: false,
    });
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
    return { ok: false, nativeStarted: false, errorMessage: "permission_denied" };
  }

  await setBackgroundTrackingMissionContext(
    missionId,
    missionStatus,
    taskMode,
    options.scheduling ?? null
  );

  const startAttemptId = createNativeLifecycleOpId("start");
  const startRequestedAt = Date.now();
  const appStateAtRequest = AppState.currentState;
  const priorContext = await readTaskContext();
  const [hasStartedBefore, isRegisteredBefore] = await Promise.all([
    readNativeLocationUpdatesStarted(),
    isBackgroundLocationTaskRegistered(),
  ]);

  const inFlightAtRequest = getNativeLifecycleInFlight();
  const lifecycleCorrelationBase = {
    start_attempt_id: startAttemptId,
    start_requested_at: startRequestedAt,
    start_reason: startReason,
    app_state_at_request: appStateAtRequest,
    isTaskRegisteredAsync_before: isRegisteredBefore,
    hasStartedLocationUpdatesAsync_before: hasStartedBefore,
    stop_in_flight: inFlightAtRequest.stop_in_flight,
    start_in_flight: inFlightAtRequest.start_in_flight,
    native_owner: priorContext?.nativeOwner ?? null,
    mission_id: missionId,
    task_mode: taskMode,
  };

  if (hasStartedBefore) {
    clearNativeStartFailure();
    clearNativeStartDiagnostics();
    clearPendingFgsStart();
    emitDriverTelemetry("tracking.background.start_requested", {
      source: "driver.services.backgroundLocationTask",
      reason: startReason,
      ...lifecycleCorrelationBase,
      outcome: "already_started",
      task_defined: lifecycleBefore.taskDefined,
      task_started: true,
      app_state: appStateAtRequest,
      fg_permission: perms.fg,
      bg_permission: perms.bg,
    });
    emitDriverTelemetry("tracking.background.start_success", {
      source: "driver.services.backgroundLocationTask",
      reason: startReason,
      ...lifecycleCorrelationBase,
      outcome: "already_started",
      app_state_at_resolve: AppState.currentState,
      isTaskRegisteredAsync_after: isRegisteredBefore,
      hasStartedLocationUpdatesAsync_after: true,
      task_defined: lifecycleBefore.taskDefined,
      task_started: true,
      app_state: AppState.currentState,
    });
    return { ok: true, nativeStarted: true, invokedNativeStart: false };
  }

  const batteryLevel = await Battery.getBatteryLevelAsync().catch(() => null);
  const isLowBattery = typeof batteryLevel === "number" && batteryLevel <= LOW_BATTERY_THRESHOLD;
  const resolvedTrackingMode = resolveBackgroundTrackingMode(
    missionStatus,
    taskMode,
    options.scheduling ?? null
  );
  const gpsQuality = resolveBackgroundGpsQuality({
    trackingMode: resolvedTrackingMode,
    isLowBattery,
  });
  const effectiveIntervalMs = gpsQuality.timeIntervalMs;

  emitDriverTelemetry("tracking.background.start_requested", {
    source: "driver.services.backgroundLocationTask",
    reason: startReason,
    ...lifecycleCorrelationBase,
    task_defined: lifecycleBefore.taskDefined,
    task_started: lifecycleBefore.taskStarted,
    app_state: appStateAtRequest,
    fg_permission: perms.fg,
    bg_permission: perms.bg,
  });

  const effectiveDistanceMeters = resolveBackgroundDistanceMeters(taskMode);

  const locationOptions: Location.LocationTaskOptions = {
    // P0-F : mission_live = High + cadence mission même batterie faible.
    // Présence flotte : Low/Balanced + cadence allongée si batterie ≤20 %.
    accuracy: gpsQuality.accuracy,
    timeInterval: effectiveIntervalMs,
    distanceInterval: effectiveDistanceMeters,
    pausesUpdatesAutomatically: false,
    showsBackgroundLocationIndicator: true,
    foregroundService: (() => {
      const fgsNotification = resolveForegroundServiceNotification(taskMode);
      return {
        notificationTitle: fgsNotification.title,
        notificationBody: fgsNotification.body,
        notificationColor: FOREGROUND_SERVICE_COLOR,
        killServiceOnDestroy: false,
      };
    })(),
  };
  if (Platform.OS === "ios") {
    locationOptions.activityType = Location.ActivityType.AutomotiveNavigation;
  }

  recordNativeStartDiagnostics({
    native_start_phase: "before_startLocationUpdatesAsync",
    native_task_defined: lifecycleBefore.taskDefined || isTaskManagerTaskDefined(),
    native_started_before: hasStartedBefore,
    native_started_after: null,
    native_start_error: null,
  });

  try {
    recordNativeStartDiagnostics({ native_start_phase: "startLocationUpdatesAsync" });
    await Location.startLocationUpdatesAsync(BACKGROUND_LOCATION_TASK_NAME, locationOptions);

    const [hasStartedAfter, isRegisteredAfter] = await Promise.all([
      readNativeLocationUpdatesStarted(),
      isBackgroundLocationTaskRegistered(),
    ]);
    const lifecycleAfter = await getNativeTaskLifecycleStatus();
    const inFlightAfter = getNativeLifecycleInFlight();
    recordNativeStartDiagnostics({
      native_start_phase: "after_startLocationUpdatesAsync",
      native_task_defined: lifecycleAfter.taskDefined,
      native_started_after: lifecycleAfter.taskStarted,
    });

    emitDriverTelemetry("tracking.background.start_success", {
      source: "driver.services.backgroundLocationTask",
      reason: startReason,
      ...lifecycleCorrelationBase,
      start_in_flight: inFlightAfter.start_in_flight,
      stop_in_flight: inFlightAfter.stop_in_flight,
      outcome: "started",
      app_state_at_resolve: AppState.currentState,
      isTaskRegisteredAsync_after: isRegisteredAfter,
      hasStartedLocationUpdatesAsync_after: hasStartedAfter,
      task_defined: lifecycleAfter.taskDefined,
      task_started: lifecycleAfter.taskStarted,
      app_state: AppState.currentState,
    });

    if (!lifecycleAfter.taskStarted) {
      const inactiveError = "startLocationUpdatesAsync returned without active native task";
      const healthError = formatNativeStartErrorForHealth(startAttemptId, startReason, {
        error_name: "NativeTaskInactive",
        error_code: null,
        error_message: inactiveError,
      });
      recordNativeStartDiagnostics({
        native_start_phase: "after_startLocationUpdatesAsync",
        native_start_error: healthError,
        native_started_after: false,
      });
      recordNativeStartFailure({
        reason: startReason,
        error: `[${startAttemptId}] ${inactiveError}`,
      });
      notifyDeviceHealthOnNativeStartFailure("native_task_inactive");
      emitDriverTelemetry("tracking.background.start_failed", {
        source: "driver.services.backgroundLocationTask",
        reason: startReason,
        failure_reason: "native_task_inactive",
        ...lifecycleCorrelationBase,
        start_in_flight: inFlightAfter.start_in_flight,
        stop_in_flight: inFlightAfter.stop_in_flight,
        app_state_at_resolve: AppState.currentState,
        isTaskRegisteredAsync_after: isRegisteredAfter,
        hasStartedLocationUpdatesAsync_after: hasStartedAfter,
        error: inactiveError,
        error_name: "NativeTaskInactive",
        error_code: null,
        error_stack: null,
        task_defined: lifecycleAfter.taskDefined,
        task_started: false,
        app_state: AppState.currentState,
        fg_permission: perms.fg,
        bg_permission: perms.bg,
      });
      return {
        ok: false,
        nativeStarted: false,
        invokedNativeStart: true,
        errorName: "NativeTaskInactive",
        errorMessage: inactiveError,
      };
    }

    clearNativeStartFailure();
    clearNativeStartDiagnostics();
    clearPendingFgsStart();

    emitDriverTelemetry("tracking.background.task.started", {
      source: "driver.services.backgroundLocationTask",
      task_name: BACKGROUND_LOCATION_TASK_NAME,
      start_attempt_id: startAttemptId,
      mission_id: missionId,
      task_mode: taskMode,
      interval_ms: effectiveIntervalMs,
      distance_m: effectiveDistanceMeters,
      low_battery_mode: isLowBattery,
      battery_degrades_gps: gpsQuality.batteryDegradesGps,
      battery_level: batteryLevel,
    });
    return { ok: true, nativeStarted: true, invokedNativeStart: true };
  } catch (error: unknown) {
    const errFields = extractNativeLifecycleError(error);
    const healthError = formatNativeStartErrorForHealth(startAttemptId, startReason, errFields);
    const inFlightFail = getNativeLifecycleInFlight();
    recordNativeStartDiagnostics({
      native_start_phase: "startLocationUpdatesAsync",
      native_start_error: healthError,
      native_started_after: false,
    });
    recordNativeStartFailure({
      reason: startReason,
      error: healthError,
    });
    notifyDeviceHealthOnNativeStartFailure("start_exception");
    emitDriverTelemetry("tracking.background.start_failed", {
      source: "driver.services.backgroundLocationTask",
      reason: startReason,
      failure_reason: "start_exception",
      ...lifecycleCorrelationBase,
      start_in_flight: inFlightFail.start_in_flight,
      stop_in_flight: inFlightFail.stop_in_flight,
      app_state_at_resolve: AppState.currentState,
      error: errFields.error_message,
      error_name: errFields.error_name,
      error_code: errFields.error_code,
      error_stack: errFields.error_stack,
      task_defined: lifecycleBefore.taskDefined,
      task_started: false,
      app_state: AppState.currentState,
      fg_permission: perms.fg,
      bg_permission: perms.bg,
    });
    return {
      ok: false,
      nativeStarted: false,
      invokedNativeStart: true,
      errorCode: errFields.error_code,
      errorName: errFields.error_name,
      errorMessage: errFields.error_message,
    };
  }
}

function isRecoverStartReason(reason: string): boolean {
  return (
    reason.includes("fgs_recover") ||
    reason.includes("anti_zombie") ||
    reason.includes("recover") ||
    reason.includes("watchdog")
  );
}

/**
 * Point d'entrée sérialisé P0-A pour tout START/RECOVER natif.
 * Ne jamais appeler startBackgroundLocationTaskIfEligibleInternal hors de ce helper
 * (sauf depuis le `run` injecté ici).
 */
async function requestEligibleNativeStart(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null,
  options: StartBackgroundOptions,
  startReason: string
): Promise<NativeStartRunResult> {
  const run = () =>
    startBackgroundLocationTaskIfEligibleInternal(
      missionId,
      missionStatus,
      options,
      startReason
    );
  const ctrl = isRecoverStartReason(startReason)
    ? await requestNativeRecover({ reason: startReason, run })
    : await requestNativeStart({ reason: startReason, run });
  if (ctrl.result) return ctrl.result;
  return {
    ok: false,
    nativeStarted: false,
    errorMessage: ctrl.outcome,
  };
}

async function runWatchdogTick(): Promise<void> {
  if (!watchdogActive) return;
  if (bgStartInProgress) return;

  // TIME-3C : ne pas relancer une présence hors fenêtre
  if (watchdogPresenceWindow && !isWithinTrackingWindow(new Date())) {
    stopNativeTrackingWatchdog();
    clearPendingFgsStart();
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.services.backgroundLocationTask",
      reason: "presence_window_closed_watchdog",
      mission_id: watchdogMissionId,
    });
    return;
  }

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

  // P0-A : pas de spam START sous BLOCKED / op en cours
  if (!canAttemptNativeStartNow()) {
    return;
  }

  if (bgStartInProgress) return;
  bgStartInProgress = true;
  try {
    await requestEligibleNativeStart(
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
  if (!isFeatureEnabled("tracking_background_enabled")) {
    recordNativeStartDiagnostics({
      native_start_phase: "ensureNativeTrackingWhileForeground",
      native_start_error: "tracking_background_enabled=false",
    });
    return;
  }
  if (!canUseBackgroundLocation()) {
    recordNativeStartDiagnostics({
      native_start_phase: "ensureNativeTrackingWhileForeground",
      native_start_error: "runtime_unsupported",
    });
    return;
  }

  const isPresenceWindow = options.presenceWindow === true;
  if (!isPresenceWindow && (missionId == null || !isTrackingActiveStatus(missionStatus))) {
    return;
  }
  // TIME-3C : ne pas écrire/reprendre un contexte présence hors fenêtre
  if (isPresenceWindow && !isWithinTrackingWindow(new Date())) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.services.backgroundLocationTask",
      reason: "presence_window_closed",
      mission_id: missionId,
      task_mode: "presence_window",
      start_reason: reason,
    });
    return;
  }

  const lifecycle = await getNativeTaskLifecycleStatus();
  const taskMode: BackgroundTrackingTaskMode = isPresenceWindow ? "presence_window" : "mission";
  const priorContext = await readTaskContext();
  const desiredOwner =
    options.nativeOwner === undefined ? undefined : options.nativeOwner;

  // Android : FGS déjà démarré avec owner mission/version obsolète → hard restart.
  const ownerVersionMismatch =
    Platform.OS === "android" &&
    lifecycle.taskStarted &&
    desiredOwner != null &&
    priorContext?.nativeOwner != null &&
    (priorContext.nativeOwner.missionId !== desiredOwner.missionId ||
      priorContext.nativeOwner.missionContextVersion !==
        desiredOwner.missionContextVersion);

  await withBackgroundTrackingLifecycleLock(async () => {
    await setBackgroundTrackingMissionContext(
      missionId,
      missionStatus,
      taskMode,
      options.scheduling ?? null,
      desiredOwner
    );
  });

  if (ownerVersionMismatch) {
    emitDriverTelemetry("tracking.background.fgs_hard_restart", {
      source: "driver.services.backgroundLocationTask",
      reason: `${reason}:owner_version_mismatch`,
      mission_id: missionId,
      prior_mission_id: priorContext?.nativeOwner?.missionId ?? null,
      prior_mission_context_version:
        priorContext?.nativeOwner?.missionContextVersion ?? null,
      desired_mission_id: desiredOwner?.missionId ?? null,
      desired_mission_context_version: desiredOwner?.missionContextVersion ?? null,
    });
    // P0-A : STOP puis START sérialisés (jamais imbriqués dans un même run)
    await stopNativeBackgroundLocationUpdatesSafely(`${reason}:owner_version_mismatch`);
    clearPendingFgsStart();
    stopNativeTrackingWatchdog();
    if (AppState.currentState === "active") {
      await requestEligibleNativeStart(
        missionId,
        missionStatus,
        options,
        `${reason}:owner_version_mismatch`
      );
    } else {
      setPendingFgsStart({
        active: true,
        reason: "android_fgs_owner_version_mismatch",
        missionId,
        deferredAt: Date.now(),
      });
    }
    return;
  }

  const contextUpgradedToMission =
    lifecycle.taskStarted &&
    priorContext?.taskMode === "presence_window" &&
    taskMode === "mission" &&
    missionId != null;

  if (lifecycle.taskStarted && !contextUpgradedToMission) {
    stopNativeTrackingWatchdog();
    clearPendingFgsStart();
    const stillRunning = await Location.hasStartedLocationUpdatesAsync(
      BACKGROUND_LOCATION_TASK_NAME
    ).catch(() => false);
    if (!stillRunning) {
      emitDriverTelemetry("tracking.background.fgs_recover", {
        source: "driver.services.backgroundLocationTask",
        reason,
        mission_id: missionId,
        task_mode: taskMode,
      });
      await requestEligibleNativeStart(
        missionId,
        missionStatus,
        options,
        `${reason}:fgs_recover`
      );
    }
    return;
  }

  if (contextUpgradedToMission && AppState.currentState === "active") {
    await stopNativeBackgroundLocationUpdatesSafely(`${reason}:context_upgrade_to_mission`);
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
    await requestEligibleNativeStart(missionId, missionStatus, options, reason);
    const lifecycleAfter = await getNativeTaskLifecycleStatus();
    if (lifecycleAfter.taskStarted) {
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
      if (!isWithinTrackingWindow(new Date())) {
        await stopPresenceWindowIfStillCurrent({
          expectedGenerationId: ctx.nativeOwner?.trackingGenerationId ?? null,
          expectedMissionContextVersion: ctx.nativeOwner?.missionContextVersion ?? null,
          reason: "presence_window_closed_wake",
        });
        return;
      }
      await ensureNativeTrackingWhileForeground(null, null, { presenceWindow: true }, reason);
      return;
    }
    if (ctx?.missionId != null) {
      await ensureNativeTrackingWhileForeground(
        ctx.missionId,
        ctx.missionStatus,
        { nativeOwner: ctx.nativeOwner ?? undefined },
        reason
      );
    }
    // P6 : tick recovery au wake (event-driven).
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const recovery = require("../tracking/TrackingRecoveryOrchestrator") as typeof import("../tracking/TrackingRecoveryOrchestrator");
      if (typeof recovery.tickTrackingRecovery === "function") {
        void recovery
          .tickTrackingRecovery(
            Date.now(),
            {
              // Tick neutre : reason seule ne démarre pas de cascade (P6).
              reason: `wake:${reason}`,
            },
            {
              restartWatch: async () => undefined,
              restartFgs: async (r) => {
                if (ctx?.missionId != null) {
                  await ensureNativeTrackingWhileForeground(
                    ctx.missionId,
                    ctx.missionStatus,
                    { nativeOwner: ctx.nativeOwner ?? undefined },
                    r
                  );
                }
              },
              restartEngine: async () => undefined,
            }
          )
          .catch(() => undefined);
      }
    } catch {
      /* noop */
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
  const lease = await readTrackingContextLease();
  if (!leaseAllowsTransport(lease)) {
    emitDriverTelemetry("tracking.background.resume.rejected_lease", {
      source: "driver.services.backgroundLocationTask",
      lease_state: lease?.state ?? "absent",
    });
    await stopBackgroundLocationTask("lease_not_driver_active");
    return;
  }
  const ctx = await readTaskContext();
  if (!ctx?.nativeOwner) {
    emitDriverTelemetry("tracking.background.resume.rejected_missing_owner", {
      source: "driver.services.backgroundLocationTask",
    });
    await stopBackgroundLocationTask("missing_native_owner");
    return;
  }
  const auth = await ensureTrackingAuthAvailabilityForHeadless();
  const authUsable =
    auth.kind === "SESSION_AVAILABLE" || auth.kind === "AUTH_TEMPORARILY_UNAVAILABLE";
  const ownerCheck = validateNativeOwnerForHeadless({
    owner: ctx.nativeOwner,
    lease,
    authUsable,
  });
  if (!ownerCheck.ok) {
    emitDriverTelemetry("tracking.background.resume.rejected_stale_owner", {
      source: "driver.services.backgroundLocationTask",
      reason: ownerCheck.reason,
      tracking_generation_id: ctx.nativeOwner.trackingGenerationId,
    });
    await stopBackgroundLocationTask("stale_native_owner");
    return;
  }
  if (ctx?.taskMode === "presence_window") {
    if (!isWithinTrackingWindow(new Date())) {
      await stopPresenceWindowIfStillCurrent({
        expectedGenerationId: ctx.nativeOwner?.trackingGenerationId ?? null,
        expectedMissionContextVersion: ctx.nativeOwner?.missionContextVersion ?? null,
        reason: "presence_window_closed_resume",
      });
      return;
    }
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

/**
 * Arrêt présence race-safe : no-op si le contexte n'est plus presence_window
 * ou si la génération / missionContextVersion a changé (mission devenue active).
 */
export async function stopPresenceWindowIfStillCurrent(opts: {
  expectedGenerationId: string | null;
  expectedMissionContextVersion: number | null;
  reason: string;
}): Promise<boolean> {
  return withBackgroundTrackingLifecycleLock(async () => {
    const ctx = await readTaskContext();
    if (!ctx || ctx.taskMode !== "presence_window") {
      emitDriverTelemetry("tracking.background.presence_stop_noop", {
        source: "driver.services.backgroundLocationTask",
        reason: opts.reason,
        noop_reason: "not_presence_window",
        current_task_mode: ctx?.taskMode ?? null,
      });
      return false;
    }
    const gen = ctx.nativeOwner?.trackingGenerationId ?? null;
    const ver = ctx.nativeOwner?.missionContextVersion ?? null;
    if (
      opts.expectedGenerationId != null &&
      gen !== opts.expectedGenerationId
    ) {
      emitDriverTelemetry("tracking.background.presence_stop_noop", {
        source: "driver.services.backgroundLocationTask",
        reason: opts.reason,
        noop_reason: "generation_mismatch",
        expected_generation: opts.expectedGenerationId,
        current_generation: gen,
      });
      return false;
    }
    if (
      opts.expectedMissionContextVersion != null &&
      ver !== opts.expectedMissionContextVersion
    ) {
      emitDriverTelemetry("tracking.background.presence_stop_noop", {
        source: "driver.services.backgroundLocationTask",
        reason: opts.reason,
        noop_reason: "mission_context_version_mismatch",
        expected_version: opts.expectedMissionContextVersion,
        current_version: ver,
      });
      return false;
    }
    stopNativeTrackingWatchdog();
    await writeTaskContext(null);
    await stopNativeBackgroundLocationUpdatesSafely(opts.reason);
    emitDriverTelemetry("tracking.background.task.stopped", {
      source: "driver.services.backgroundLocationTask",
      task_name: BACKGROUND_LOCATION_TASK_NAME,
      reason: opts.reason,
      presence_stop: true,
    });
    return true;
  });
}

export async function stopBackgroundLocationTask(reason: string) {
  await withBackgroundTrackingLifecycleLock(async () => {
    stopNativeTrackingWatchdog();
    await setBackgroundTrackingMissionContext(null, null);
    await stopNativeBackgroundLocationUpdatesSafely(reason);
    emitDriverTelemetry("tracking.background.task.stopped", {
      source: "driver.services.backgroundLocationTask",
      task_name: BACKGROUND_LOCATION_TASK_NAME,
      reason,
    });
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
  ensureNativeLifecycleAppStateBridge();
  defineTaskIfNeeded();
  void emitRegistrationStatus();
}

/** Test-only reset */
export function __resetBackgroundLocationTaskStateForTests(): void {
  taskDefined = false;
  bgStartInProgress = false;
  lifecycleLockTail = Promise.resolve();
  stopNativeTrackingWatchdog();
  __resetNativeTrackingLifecycleForTests();
}
