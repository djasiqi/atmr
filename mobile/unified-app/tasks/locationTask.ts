/**
 * Tâche périodique de self-heal GPS (F3 / PR D2).
 *
 * Chemin unique `expo-background-task` (SDK 54+, minimumInterval en minutes).
 * Rôle : maintenance best-effort (flush / santé / self-heal) — PAS watchdog GPS temps réel.
 * Ne purge JAMAIS les tokens, ne fait JAMAIS de logout.
 *
 * TaskManager.defineTask DOIT être au scope module (bundle global).
 */
import { Platform } from "react-native";
import * as TaskManager from "expo-task-manager";
import { emitDriverTelemetry } from "../src/core/observability/driverTelemetry";
import { isFeatureEnabled } from "../src/core/featureFlags/registry";

const DRIVER_LOCATION_TASK = "driver-location-background-task";
/** BackgroundTask : intervalle en minutes (minimum Android = 15). */
const BACKGROUND_INTERVAL_MINUTES = 15;

type TickResult = "NewData" | "NoData" | "Failed";

/**
 * Lecture du bail uniquement. Tant que le contexte n’est pas chauffeur actif,
 * aucun module GPS / file / santé n’est chargé.
 */
async function isDriverTransportLeaseActive(): Promise<boolean> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const leaseMod = require("../src/features/driver/services/trackingContextLease") as typeof import("../src/features/driver/services/trackingContextLease");
    const lease = await leaseMod.readTrackingContextLease();
    return leaseMod.leaseAllowsTransport(lease);
  } catch {
    return false;
  }
}

export async function runDriverLocationSelfHealTick(): Promise<TickResult> {
  if (!(await isDriverTransportLeaseActive())) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.tasks.locationTask",
      reason: "lease_not_driver_active",
      task_name: DRIVER_LOCATION_TASK,
    });
    return "NoData";
  }

  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const bridge = require("../src/features/driver/services/driverTrackingBridge") as typeof import("../src/features/driver/services/driverTrackingBridge");
  await bridge.flushDriverTrackingQueueNow();
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const bgTask = require("../src/features/driver/services/backgroundLocationTask") as typeof import("../src/features/driver/services/backgroundLocationTask");
    await bgTask.resumePendingNativeTrackingIfNeeded();
    await bgTask.restartNativeTrackingFromWake("background_task_tick");
  } catch {
    /* noop — self-heal best-effort */
  }
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const health = require("../src/features/driver/services/deviceHealthHeartbeat") as {
      triggerDeviceHealthNow?: (reason: string) => Promise<void>;
    };
    if (typeof health.triggerDeviceHealthNow === "function") {
      await health.triggerDeviceHealthNow("background_task_tick");
    }
  } catch {
    /* noop */
  }
  const snapshot = await bridge.getDriverTrackingQueueSnapshot();
  emitDriverTelemetry("tracking.background.task.tick", {
    source: "driver.tasks.locationTask",
    queue_depth: snapshot.queueDepth,
  });
  return "NewData";
}

// Scope module obligatoire (Expo BackgroundTask / TaskManager)
if (Platform.OS !== "web" && typeof TaskManager.defineTask === "function") {
  TaskManager.defineTask(DRIVER_LOCATION_TASK, async () => {
    try {
      await runDriverLocationSelfHealTick();
      try {
        const BackgroundTask = await import("expo-background-task");
        const Result = (BackgroundTask as {
          BackgroundTaskResult?: { Success?: number; Failed?: number };
        }).BackgroundTaskResult;
        return Result?.Success ?? 1;
      } catch {
        return 1;
      }
    } catch {
      try {
        const BackgroundTask = await import("expo-background-task");
        const Result = (BackgroundTask as {
          BackgroundTaskResult?: { Success?: number; Failed?: number };
        }).BackgroundTaskResult;
        return Result?.Failed ?? 2;
      } catch {
        return 2;
      }
    }
  });
}

export async function registerDriverBackgroundTasks(): Promise<void> {
  if (Platform.OS === "web") return;
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const runtime = require("../src/features/driver/services/backgroundRuntimeCompat") as typeof import("../src/features/driver/services/backgroundRuntimeCompat");
  if (!runtime.canUseBackgroundLocation()) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.tasks.locationTask",
      reason: "runtime_unsupported",
      runtime: runtime.describeBackgroundRuntime(),
      task_name: DRIVER_LOCATION_TASK,
    });
    return;
  }
  if (!isFeatureEnabled("tracking_background_enabled")) return;

  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const bgTask = require("../src/features/driver/services/backgroundLocationTask") as typeof import("../src/features/driver/services/backgroundLocationTask");
  bgTask.initializeBackgroundLocationTask();

  try {
    if (typeof TaskManager.isTaskRegisteredAsync !== "function") return;
    const already = await TaskManager.isTaskRegisteredAsync(DRIVER_LOCATION_TASK);
    if (already) return;

    const BackgroundTask = await import("expo-background-task");
    if (typeof (BackgroundTask as { registerTaskAsync?: unknown }).registerTaskAsync !== "function") {
      emitDriverTelemetry("tracking.background.task.unavailable", {
        source: "driver.tasks.locationTask",
        task_name: DRIVER_LOCATION_TASK,
        reason: "registerTaskAsync_missing",
      });
      return;
    }
    await (BackgroundTask as {
      registerTaskAsync: (
        name: string,
        opts: { minimumInterval?: number }
      ) => Promise<void>;
    }).registerTaskAsync(DRIVER_LOCATION_TASK, {
      minimumInterval: BACKGROUND_INTERVAL_MINUTES,
    });
    emitDriverTelemetry("tracking.background.task.registered", {
      source: "driver.tasks.locationTask",
      task_name: DRIVER_LOCATION_TASK,
      min_interval_min: BACKGROUND_INTERVAL_MINUTES,
      api: "expo-background-task",
    });
  } catch (err) {
    emitDriverTelemetry("tracking.background.task.unavailable", {
      source: "driver.tasks.locationTask",
      task_name: DRIVER_LOCATION_TASK,
      reason: err instanceof Error ? err.message : String(err),
    });
  }
}

// Enregistrement OS uniquement à l’entrée du runtime chauffeur
// (`app/(app)/(driver)/_layout.tsx`). Le defineTask ci-dessus reste global.
