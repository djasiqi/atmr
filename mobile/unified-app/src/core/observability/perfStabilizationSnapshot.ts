/**
 * Artefacts comparatifs Sprint 0 / after_A* pour stabilisation Unified App.
 * Export JSON via __exportPerfStabilizationSnapshot__ en __DEV__.
 */

import { exportPerfInstrumentationSnapshot } from "./perfInstrumentation";

export type PerfStabilizationSnapshotLabel =
  | "baseline"
  | "after_A1"
  | "after_A2"
  | "after_A3"
  | "after_B"
  | "after_C";

export type PerfStabilizationKpiRow = {
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
  count: number;
};

export type PerfStabilizationSnapshot = {
  label: PerfStabilizationSnapshotLabel | string;
  generated_at: string;
  latencies: {
    mission_open?: PerfStabilizationKpiRow;
    fleet_map_open?: PerfStabilizationKpiRow;
    context_switch_total?: PerfStabilizationKpiRow;
  };
  memory: {
    heap_used_mb?: PerfStabilizationKpiRow;
    heap_peak_mb?: PerfStabilizationKpiRow;
    gc_events?: number;
  };
  network: {
    http_requests_per_min?: number;
    invalidate_booking_updated_per_min?: number;
    invalidate_driver_location_per_min?: number;
  };
  fleet: {
    enrich_fleet_drivers_duration_ms?: PerfStabilizationKpiRow;
  };
  instrumentation_row_count: number;
  top_by_sum_ms: ReturnType<typeof exportPerfInstrumentationSnapshot>["top_by_sum_ms"];
};

function rowFromInstrumentation(
  category: string,
  subKey: string
): PerfStabilizationKpiRow | undefined {
  const report = exportPerfInstrumentationSnapshot();
  const match = report.rows.find((r) => r.category === category && r.sub_key === subKey);
  if (!match || match.count === 0) return undefined;
  return {
    p50_ms: match.p50_ms,
    p95_ms: match.p95_ms,
    max_ms: match.max_ms,
    count: match.count,
  };
}

function sumInvalidateCounts(subKeyPrefix: string): number {
  const report = exportPerfInstrumentationSnapshot();
  return report.rows
    .filter((r) => r.category === "invalidate" && r.sub_key.startsWith(subKeyPrefix))
    .reduce((acc, r) => acc + r.count, 0);
}

export function buildPerfStabilizationSnapshot(
  label: PerfStabilizationSnapshotLabel | string
): PerfStabilizationSnapshot {
  const report = exportPerfInstrumentationSnapshot();

  const heapUsed = rowFromInstrumentation("heap", "used_mb");
  const heapPeak = rowFromInstrumentation("heap", "peak_mb");

  return {
    label,
    generated_at: new Date().toISOString(),
    latencies: {
      mission_open: rowFromInstrumentation("page_load", "mission"),
      fleet_map_open: rowFromInstrumentation("page_load", "fleet-map"),
      context_switch_total: rowFromInstrumentation("context_switch", "total"),
    },
    memory: {
      heap_used_mb: heapUsed,
      heap_peak_mb: heapPeak,
      gc_events: report.rows
        .filter((r) => r.category === "js_long_task")
        .reduce((acc, r) => acc + r.count, 0),
    },
    network: {
      http_requests_per_min: report.rows
        .filter((r) => r.category === "http")
        .reduce((acc, r) => acc + r.count, 0),
      invalidate_booking_updated_per_min: sumInvalidateCounts("booking"),
      invalidate_driver_location_per_min: sumInvalidateCounts("drivers.locations"),
    },
    fleet: {
      enrich_fleet_drivers_duration_ms: rowFromInstrumentation(
        "fleet_map",
        "enrich_fleet_drivers_ms"
      ),
    },
    instrumentation_row_count: report.rows.length,
    top_by_sum_ms: report.top_by_sum_ms,
  };
}

export function serializePerfStabilizationSnapshot(
  label: PerfStabilizationSnapshotLabel | string
): string {
  return JSON.stringify(buildPerfStabilizationSnapshot(label), null, 2);
}

declare global {
  var __exportPerfStabilizationSnapshot__:
    | ((label?: PerfStabilizationSnapshotLabel | string) => string)
    | undefined;
}

if (typeof __DEV__ !== "undefined" && __DEV__) {
  globalThis.__exportPerfStabilizationSnapshot__ = (label = "baseline") =>
    serializePerfStabilizationSnapshot(label);
}
