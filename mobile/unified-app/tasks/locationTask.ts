/**
 * Tâche périodique de self-heal GPS (PR D2).
 *
 * Préfère `expo-background-task` (SDK 54+) ; bascule sur `expo-background-fetch`
 * si le module n'est pas installé. Ne décide JAMAIS de l'absence de session,
 * ne purge JAMAIS les tokens, ne fait JAMAIS de logout.
 */
import { Platform } from "react-native";
import { emitDriverTelemetry } from "../src/core/observability/driverTelemetry";
import { isFeatureEnabled } from "../src/core/featureFlags/registry";
import {
  flushDriverTrackingQueueNow,
  getDriverTrackingQueueSnapshot,
} from "../src/features/driver/services/driverTrackingBridge";
import {
  canUseBackgroundLocation,
  describeBackgroundRuntime,
} from "../src/features/driver/services/backgroundRuntimeCompat";
import { initializeBackgroundLocationTask } from "../src/features/driver/services/backgroundLocationTask";

const DRIVER_LOCATION_TASK = "driver-location-background-task";
const BACKGROUND_INTERVAL_SECONDS = 60 * 15;

type TickResult = "NewData" | "NoData" | "Failed";

async function runSelfHealTick(): Promise<TickResult> {
  await flushDriverTrackingQueueNow();
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
  const snapshot = await getDriverTrackingQueueSnapshot();
  emitDriverTelemetry("tracking.background.task.tick", {
    source: "driver.tasks.locationTask",
    queue_depth: snapshot.queueDepth,
  });
  return "NewData";
}

async function registerWithBackgroundTask(
  TaskManager: {
    defineTask: (name: string, fn: () => Promise<unknown>) => void;
    isTaskRegisteredAsync: (name: string) => Promise<boolean>;
  }
): Promise<boolean> {
  try {
    const BackgroundTask = await import("expo-background-task");
    if (typeof (BackgroundTask as { registerTaskAsync?: unknown }).registerTaskAsync !== "function") {
      return false;
    }
    TaskManager.defineTask(DRIVER_LOCATION_TASK, async () => {
      await runSelfHealTick();
      const Result = (BackgroundTask as {
        BackgroundTaskResult?: { Success?: number; Failed?: number };
      }).BackgroundTaskResult;
      return Result?.Success ?? 1;
    });
    const already = await TaskManager.isTaskRegisteredAsync(DRIVER_LOCATION_TASK);
    if (already) return true;
    await (BackgroundTask as {
      registerTaskAsync: (
        name: string,
        opts: { minimumInterval?: number }
      ) => Promise<void>;
    }).registerTaskAsync(DRIVER_LOCATION_TASK, {
      minimumInterval: BACKGROUND_INTERVAL_SECONDS,
    });
    emitDriverTelemetry("tracking.background.task.registered", {
      source: "driver.tasks.locationTask",
      task_name: DRIVER_LOCATION_TASK,
      min_interval_s: BACKGROUND_INTERVAL_SECONDS,
      api: "expo-background-task",
    });
    return true;
  } catch {
    return false;
  }
}

async function registerWithBackgroundFetch(
  TaskManager: {
    defineTask: (name: string, fn: () => Promise<unknown>) => void;
    isTaskRegisteredAsync: (name: string) => Promise<boolean>;
  }
): Promise<boolean> {
  try {
    const BackgroundFetch = await import("expo-background-fetch");
    TaskManager.defineTask(DRIVER_LOCATION_TASK, async () => {
      await runSelfHealTick();
      return BackgroundFetch.BackgroundFetchResult.NewData;
    });
    const already = await TaskManager.isTaskRegisteredAsync(DRIVER_LOCATION_TASK);
    if (already) return true;
    if (typeof BackgroundFetch.registerTaskAsync !== "function") return false;
    await BackgroundFetch.registerTaskAsync(DRIVER_LOCATION_TASK, {
      minimumInterval: BACKGROUND_INTERVAL_SECONDS,
      stopOnTerminate: false,
      startOnBoot: true,
    });
    emitDriverTelemetry("tracking.background.task.registered", {
      source: "driver.tasks.locationTask",
      task_name: DRIVER_LOCATION_TASK,
      min_interval_s: BACKGROUND_INTERVAL_SECONDS,
      api: "expo-background-fetch",
    });
    return true;
  } catch {
    return false;
  }
}

export async function registerDriverBackgroundTasks(): Promise<void> {
  if (Platform.OS === "web") return;
  if (!canUseBackgroundLocation()) {
    emitDriverTelemetry("tracking.background.task.skipped", {
      source: "driver.tasks.locationTask",
      reason: "runtime_unsupported",
      runtime: describeBackgroundRuntime(),
      task_name: DRIVER_LOCATION_TASK,
    });
    return;
  }
  if (!isFeatureEnabled("tracking_background_enabled")) return;

  initializeBackgroundLocationTask();

  try {
    const taskManagerModule = await import("expo-task-manager");
    const TaskManager = taskManagerModule;
    if (typeof TaskManager.defineTask !== "function") return;
    if (typeof TaskManager.isTaskRegisteredAsync !== "function") return;

    const ok =
      (await registerWithBackgroundTask(TaskManager))
      || (await registerWithBackgroundFetch(TaskManager));
    if (!ok) {
      emitDriverTelemetry("tracking.background.task.unavailable", {
        source: "driver.tasks.locationTask",
        task_name: DRIVER_LOCATION_TASK,
      });
    }
  } catch {
    emitDriverTelemetry("tracking.background.task.unavailable", {
      source: "driver.tasks.locationTask",
      task_name: DRIVER_LOCATION_TASK,
    });
  }
}

if (Platform.OS !== "web") {
  void registerDriverBackgroundTasks();
}
