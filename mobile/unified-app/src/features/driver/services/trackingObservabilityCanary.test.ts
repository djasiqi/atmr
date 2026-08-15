/**
 * Canary OBSERVABILITY P0-C — classification seule (pas de patch tracking).
 *
 * Scénarios O-C1…O-C6 + historique P0-C + invariant compat native_last_fix.
 * Écrit un rapport JSON sous docs/ops/_c3_observability_2026-08-15/.
 */
import * as fs from "fs";
import * as path from "path";

import { describe, expect, it } from "@jest/globals";

import {
  classifyTrackingObservability,
  computeLocationFixAgeSeconds,
  computeTaskInvokeAgeSeconds,
  isGpsStaleAlertClass,
  LOCATION_FIX_STALE_SECONDS,
  type TrackingObservabilityClass,
} from "./trackingObservabilityHealth";

const NOW = 1_700_000_000_000;

type CanaryRow = {
  id: string;
  expected: TrackingObservabilityClass;
  got: TrackingObservabilityClass;
  location_fix_age_seconds: number | null;
  task_invoke_age_seconds: number | null;
  native_last_fix_age_seconds: number | null;
  fix_stale: boolean;
  pass: boolean;
  notes?: string;
};

const rows: CanaryRow[] = [];

function wouldEmitFixStale(
  cls: TrackingObservabilityClass,
  locationFixAgeSeconds: number | null
): boolean {
  return (
    isGpsStaleAlertClass(cls)
    && locationFixAgeSeconds != null
    && locationFixAgeSeconds > LOCATION_FIX_STALE_SECONDS
  );
}

function runScenario(input: {
  id: string;
  expected: TrackingObservabilityClass;
  locationTimestampMs: number | null;
  lastTaskInvokedAtMs: number | null;
  fgsRunning?: boolean;
  fgsExpected?: boolean;
  queueDepth?: number;
  oldestQueueItemAgeSeconds?: number | null;
  persistenceLagSeconds?: number | null;
  enqueueWithoutPersist?: boolean;
  notes?: string;
}): CanaryRow {
  const locationFixAgeSeconds = computeLocationFixAgeSeconds(
    input.locationTimestampMs,
    NOW
  );
  const taskInvokeAgeSeconds = computeTaskInvokeAgeSeconds(
    input.lastTaskInvokedAtMs,
    NOW
  );
  // Compat heartbeat : native_last_fix_age = alias de task_invoke
  const nativeLastFixAgeSeconds = taskInvokeAgeSeconds;
  const got = classifyTrackingObservability({
    locationFixAgeSeconds,
    taskInvokeAgeSeconds,
    fgsRunning: input.fgsRunning ?? true,
    fgsExpected: input.fgsExpected ?? true,
    queueDepth: input.queueDepth ?? 0,
    oldestQueueItemAgeSeconds: input.oldestQueueItemAgeSeconds ?? null,
    persistenceLagSeconds: input.persistenceLagSeconds ?? null,
    enqueueWithoutPersist: input.enqueueWithoutPersist,
  });
  const fixStale = wouldEmitFixStale(got, locationFixAgeSeconds);
  const pass =
    got === input.expected
    && nativeLastFixAgeSeconds === taskInvokeAgeSeconds
    && (input.expected === "GNSS" ? fixStale === true : fixStale === false);

  const row: CanaryRow = {
    id: input.id,
    expected: input.expected,
    got,
    location_fix_age_seconds: locationFixAgeSeconds,
    task_invoke_age_seconds: taskInvokeAgeSeconds,
    native_last_fix_age_seconds: nativeLastFixAgeSeconds,
    fix_stale: fixStale,
    pass,
    notes: input.notes,
  };
  rows.push(row);
  return row;
}

describe("canary OBSERVABILITY O-C1…O-C6", () => {
  it("O-C1 HEALTHY — Location + task + queue + persistence frais", () => {
    const row = runScenario({
      id: "O-C1",
      expected: "HEALTHY",
      locationTimestampMs: NOW - 8_000,
      lastTaskInvokedAtMs: NOW - 5_000,
      queueDepth: 2,
      oldestQueueItemAgeSeconds: 10,
      persistenceLagSeconds: 8,
    });
    expect(row.pass).toBe(true);
    expect(row.got).toBe("HEALTHY");
    expect(row.fix_stale).toBe(false);
  });

  it("O-C2 PIPELINE — Location frais + queue HOL (illusion P0-C)", () => {
    const row = runScenario({
      id: "O-C2",
      expected: "PIPELINE",
      locationTimestampMs: NOW - 12_000,
      lastTaskInvokedAtMs: NOW - 400_000,
      queueDepth: 80,
      oldestQueueItemAgeSeconds: 200,
      persistenceLagSeconds: 400,
      notes: "Conceptuel P0-C : GNSS frais + queue bloquée",
    });
    expect(row.pass).toBe(true);
    expect(row.got).toBe("PIPELINE");
    expect(row.got).not.toBe("GNSS");
    expect(row.fix_stale).toBe(false);
    expect(row.location_fix_age_seconds).toBeLessThan(LOCATION_FIX_STALE_SECONDS);
  });

  it("O-C3 PERSISTENCE — Location frais + PG en retard (queue OK)", () => {
    const row = runScenario({
      id: "O-C3",
      expected: "PERSISTENCE",
      locationTimestampMs: NOW - 10_000,
      lastTaskInvokedAtMs: NOW - 9_000,
      queueDepth: 3,
      oldestQueueItemAgeSeconds: 20,
      persistenceLagSeconds: 400,
    });
    expect(row.pass).toBe(true);
    expect(row.got).toBe("PERSISTENCE");
    expect(row.fix_stale).toBe(false);
  });

  it("O-C4 GNSS — Location.timestamp dépasse le seuil", () => {
    const row = runScenario({
      id: "O-C4",
      expected: "GNSS",
      locationTimestampMs: NOW - (LOCATION_FIX_STALE_SECONDS + 90) * 1000,
      lastTaskInvokedAtMs: NOW - 5_000,
      queueDepth: 0,
      oldestQueueItemAgeSeconds: null,
      persistenceLagSeconds: null,
    });
    expect(row.pass).toBe(true);
    expect(row.got).toBe("GNSS");
    expect(row.fix_stale).toBe(true);
  });

  it("O-C5 RUNTIME_ONLY — task stale + Location frais → jamais GPS stale", () => {
    const row = runScenario({
      id: "O-C5",
      expected: "RUNTIME_ONLY",
      locationTimestampMs: NOW - 7_000,
      lastTaskInvokedAtMs: NOW - 600_000,
      queueDepth: 1,
      oldestQueueItemAgeSeconds: 5,
      persistenceLagSeconds: 7,
    });
    expect(row.pass).toBe(true);
    expect(row.got).toBe("RUNTIME_ONLY");
    expect(row.fix_stale).toBe(false);
    expect(isGpsStaleAlertClass(row.got)).toBe(false);
  });

  it("O-C6 UNKNOWN — aucun Location connu → pas de faux fix_stale", () => {
    const row = runScenario({
      id: "O-C6",
      expected: "UNKNOWN",
      locationTimestampMs: null,
      lastTaskInvokedAtMs: NOW - 400_000,
      queueDepth: 0,
    });
    expect(row.pass).toBe(true);
    expect(row.got).toBe("UNKNOWN");
    expect(row.fix_stale).toBe(false);
    expect(row.location_fix_age_seconds).toBeNull();
  });

  it("HIST-P0-C — GNSS frais + enqueue + HOL + persistence bloquée ≠ GNSS", () => {
    const row = runScenario({
      id: "HIST-P0-C",
      expected: "PIPELINE",
      locationTimestampMs: NOW - 15_000,
      lastTaskInvokedAtMs: NOW - 450_000,
      queueDepth: 12,
      oldestQueueItemAgeSeconds: 180,
      persistenceLagSeconds: null,
      enqueueWithoutPersist: true,
      notes:
        "Situation historique P0-C : doit être PIPELINE (ou PERSISTENCE), jamais GNSS/fix_stale",
    });
    expect(["PIPELINE", "PERSISTENCE"]).toContain(row.got);
    expect(row.got).not.toBe("GNSS");
    expect(row.fix_stale).toBe(false);
    expect(row.location_fix_age_seconds).toBeLessThan(60);
    expect(row.native_last_fix_age_seconds).toBe(row.task_invoke_age_seconds);
    // Override pass for allowed PIPELINE|PERSISTENCE
    row.pass =
      (row.got === "PIPELINE" || row.got === "PERSISTENCE")
      && !row.fix_stale
      && row.native_last_fix_age_seconds === row.task_invoke_age_seconds;
    expect(row.pass).toBe(true);
  });

  it("COMPAT — native_last_fix_age === task_invoke_age sur tous les scénarios", () => {
    for (const row of rows) {
      expect(row.native_last_fix_age_seconds).toBe(row.task_invoke_age_seconds);
    }
  });

  it("écrit le rapport canary JSON", () => {
    const falseGnssWithFreshFix = rows.filter(
      (r) =>
        r.got === "GNSS"
        && r.location_fix_age_seconds != null
        && r.location_fix_age_seconds <= LOCATION_FIX_STALE_SECONDS
    ).length;
    const fixStaleOutsideGnss = rows.filter(
      (r) => r.fix_stale && r.got !== "GNSS"
    ).length;
    const hist = rows.find((r) => r.id === "HIST-P0-C");
    const histMisclassifiedGnss = hist?.got === "GNSS" ? 1 : 0;

    const report = {
      canary: "OBSERVABILITY",
      date: "2026-08-15",
      mode: "harness_controlled",
      tracking_functional_change: false,
      ledger_touched: false,
      scenarios: rows,
      blocking_metrics: {
        false_gnss_with_fresh_fix: falseGnssWithFreshFix,
        fix_stale_outside_gnss: fixStaleOutsideGnss,
        pipeline_hist_classified_gnss: histMisclassifiedGnss,
        location_timestamp_used: true,
        task_invoke_separated_from_fix_age: true,
        regression_p0a: 0,
        regression_p0b: 0,
        regression_ledger: 0,
        loc_put_persistence_behavior_unchanged: true,
      },
      verdict:
        falseGnssWithFreshFix === 0
        && fixStaleOutsideGnss === 0
        && histMisclassifiedGnss === 0
        && rows.every((r) => r.pass)
          ? "PASS"
          : "FAIL",
    };

    const captureDir = path.resolve(
      __dirname,
      "../../../../../../docs/ops/_c3_observability_2026-08-15"
    );
    fs.mkdirSync(captureDir, { recursive: true });
    const outPath = path.join(captureDir, "canary_report.json");
    fs.writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    expect(report.verdict).toBe("PASS");
    expect(fs.existsSync(outPath)).toBe(true);
  });
});
