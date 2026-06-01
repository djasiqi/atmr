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
const BACKGROUND_FETCH_INTERVAL_SECONDS = 60 * 15;

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
    const [taskManagerModule, backgroundFetchModule] = await Promise.all([
      import("expo-task-manager"),
      import("expo-background-fetch"),
    ]);
    const TaskManager = taskManagerModule;
    const BackgroundFetch = backgroundFetchModule;

    if (typeof TaskManager.defineTask !== "function") return;

    TaskManager.defineTask(DRIVER_LOCATION_TASK, async () => {
      await flushDriverTrackingQueueNow();
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const bgTask = require("../src/features/driver/services/backgroundLocationTask") as typeof import("../src/features/driver/services/backgroundLocationTask");
        await bgTask.resumePendingNativeTrackingIfNeeded();
        await bgTask.restartNativeTrackingFromWake("background_fetch_tick");
      } catch {
        /* noop */
      }
      const snapshot = await getDriverTrackingQueueSnapshot();
      emitDriverTelemetry("tracking.background.task.tick", {
        source: "driver.tasks.locationTask",
        queue_depth: snapshot.queueDepth,
      });
      return BackgroundFetch.BackgroundFetchResult.NewData;
    });

    if (typeof TaskManager.isTaskRegisteredAsync !== "function") return;

    const alreadyRegistered = await TaskManager.isTaskRegisteredAsync(DRIVER_LOCATION_TASK);
    if (alreadyRegistered) return;

    if (typeof BackgroundFetch.registerTaskAsync !== "function") return;

    await BackgroundFetch.registerTaskAsync(DRIVER_LOCATION_TASK, {
      minimumInterval: BACKGROUND_FETCH_INTERVAL_SECONDS,
      stopOnTerminate: false,
      startOnBoot: true,
    });
    emitDriverTelemetry("tracking.background.task.registered", {
      source: "driver.tasks.locationTask",
      task_name: DRIVER_LOCATION_TASK,
      min_interval_s: BACKGROUND_FETCH_INTERVAL_SECONDS,
    });
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
