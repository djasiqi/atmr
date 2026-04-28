import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";

export async function openBatteryOptimizationSettings(): Promise<void> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const intentLauncher = require("expo-intent-launcher");
    await intentLauncher.startActivityAsync(
      intentLauncher.ActivityAction.IGNORE_BATTERY_OPTIMIZATION_SETTINGS
    );
  } catch (error) {
    emitDriverTelemetry("driver.battery_optimization.unavailable", {
      source: "driver.battery_optimization",
      reason: error instanceof Error ? error.message : "intent_unavailable",
    });
  }
}
