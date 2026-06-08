import * as Battery from "expo-battery";
import { emitDriverTelemetry } from "./driverTelemetry";

const TRACE_ENABLED = process.env.EXPO_PUBLIC_GPS_FIDELITY_TRACE === "1";
const BATTERY_SNAPSHOT_MS = 5 * 60 * 1000;

let lastBatterySnapshotAt = 0;

/** Snapshot batterie périodique pour campagne terrain E1 / PR5. */
export async function emitBatteryBaselineIfTracing(source: string): Promise<void> {
  if (!TRACE_ENABLED) return;
  const now = Date.now();
  if (now - lastBatterySnapshotAt < BATTERY_SNAPSHOT_MS) return;
  lastBatterySnapshotAt = now;
  try {
    const [level, state] = await Promise.all([
      Battery.getBatteryLevelAsync(),
      Battery.getBatteryStateAsync(),
    ]);
    emitDriverTelemetry("tracking.bridge.health", {
      source,
      gps_fidelity_trace: true,
      battery_level_pct: Math.round(level * 1000) / 10,
      battery_state: state,
      snapshot_ts: new Date(now).toISOString(),
    });
  } catch {
    // Non bloquant
  }
}

export function isGpsFidelityTraceEnabled(): boolean {
  return TRACE_ENABLED;
}
