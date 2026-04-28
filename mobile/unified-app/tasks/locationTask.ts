import { emitDriverTelemetry } from "../src/core/observability/driverTelemetry";
import { isFeatureEnabled } from "../src/core/featureFlags/registry";
import {
  flushDriverTrackingQueueNow,
  getDriverTrackingQueueSnapshot,
} from "../src/features/driver/services/driverTrackingBridge";
import { initializeBackgroundLocationTask } from "../src/features/driver/services/backgroundLocationTask";

const DRIVER_LOCATION_TASK = "driver-location-background-task";
const BACKGROUND_FETCH_INTERVAL_SECONDS = 60 * 15;

async function tryRegisterTask() {
  if (!isFeatureEnabled("tracking_background_enabled")) return;
  // Register Expo Location background task definition used by startLocationUpdatesAsync.
  initializeBackgroundLocationTask();
  try {
    // Optional native module path: task is registered only when Expo modules are available.
    const [taskManagerModule, backgroundFetchModule] = await Promise.all([
      import("expo-task-manager"),
      import("expo-background-fetch"),
    ]);
    const TaskManager = taskManagerModule;
    const BackgroundFetch = backgroundFetchModule;

    TaskManager.defineTask(DRIVER_LOCATION_TASK, async () => {
      await flushDriverTrackingQueueNow();
      const snapshot = await getDriverTrackingQueueSnapshot();
      emitDriverTelemetry("tracking.background.task.tick", {
        source: "driver.tasks.locationTask",
        queue_depth: snapshot.queueDepth,
      });
      return BackgroundFetch.BackgroundFetchResult.NewData;
    });

    void (async () => {
      const alreadyRegistered = await TaskManager.isTaskRegisteredAsync(DRIVER_LOCATION_TASK);
      if (alreadyRegistered) return;
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
    })();
  } catch {
    emitDriverTelemetry("tracking.background.task.unavailable", {
      source: "driver.tasks.locationTask",
      task_name: DRIVER_LOCATION_TASK,
    });
  }
}

void tryRegisterTask();
