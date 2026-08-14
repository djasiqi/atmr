/**
 * Tests O1–O7 — classification observabilité P0-C (mesure seule).
 */
import { describe, expect, it } from "@jest/globals";

import {
  classifyTrackingObservability,
  computeLocationFixAgeSeconds,
  computeTaskInvokeAgeSeconds,
  isGpsStaleAlertClass,
  LOCATION_FIX_STALE_SECONDS,
} from "./trackingObservabilityHealth";

const NOW = 1_700_000_000_000;

describe("trackingObservabilityHealth O1–O7", () => {
  it("O1 vrai Location récent → fix_age faible + HEALTHY", () => {
    const age = computeLocationFixAgeSeconds(NOW - 5_000, NOW);
    expect(age).toBe(5);
    const cls = classifyTrackingObservability({
      locationFixAgeSeconds: age,
      taskInvokeAgeSeconds: 5,
      fgsRunning: true,
      fgsExpected: true,
      queueDepth: 0,
      oldestQueueItemAgeSeconds: null,
      persistenceLagSeconds: 5,
    });
    expect(cls).toBe("HEALTHY");
    expect(isGpsStaleAlertClass(cls)).toBe(false);
  });

  it("O2 vieux Location → GNSS stale", () => {
    const age = computeLocationFixAgeSeconds(
      NOW - (LOCATION_FIX_STALE_SECONDS + 60) * 1000,
      NOW
    );
    expect(age).toBeGreaterThan(LOCATION_FIX_STALE_SECONDS);
    const cls = classifyTrackingObservability({
      locationFixAgeSeconds: age,
      taskInvokeAgeSeconds: 5,
      fgsRunning: true,
      fgsExpected: true,
      queueDepth: 0,
      oldestQueueItemAgeSeconds: null,
      persistenceLagSeconds: null,
    });
    expect(cls).toBe("GNSS");
    expect(isGpsStaleAlertClass(cls)).toBe(true);
  });

  it("O3 task ancien + fix récent → RUNTIME_ONLY (jamais GPS stale)", () => {
    const fixAge = computeLocationFixAgeSeconds(NOW - 8_000, NOW);
    const taskAge = computeTaskInvokeAgeSeconds(NOW - 600_000, NOW);
    expect(fixAge).toBe(8);
    expect(taskAge).toBe(600);
    const cls = classifyTrackingObservability({
      locationFixAgeSeconds: fixAge,
      taskInvokeAgeSeconds: taskAge,
      fgsRunning: true,
      fgsExpected: true,
      queueDepth: 0,
      oldestQueueItemAgeSeconds: null,
      persistenceLagSeconds: 8,
    });
    expect(cls).toBe("RUNTIME_ONLY");
    expect(isGpsStaleAlertClass(cls)).toBe(false);
  });

  it("O4 fix récent + queue bloquée → PIPELINE", () => {
    const cls = classifyTrackingObservability({
      locationFixAgeSeconds: 10,
      taskInvokeAgeSeconds: 10,
      fgsRunning: true,
      fgsExpected: true,
      queueDepth: 80,
      oldestQueueItemAgeSeconds: 200,
      persistenceLagSeconds: 200,
    });
    expect(cls).toBe("PIPELINE");
    expect(isGpsStaleAlertClass(cls)).toBe(false);
  });

  it("O5 fix récent + PG en retard → PERSISTENCE", () => {
    const cls = classifyTrackingObservability({
      locationFixAgeSeconds: 12,
      taskInvokeAgeSeconds: 12,
      fgsRunning: true,
      fgsExpected: true,
      queueDepth: 2,
      oldestQueueItemAgeSeconds: 10,
      persistenceLagSeconds: 400,
    });
    expect(cls).toBe("PERSISTENCE");
    expect(isGpsStaleAlertClass(cls)).toBe(false);
  });

  it("O5b enqueue sans persist (illusion P0-C) → PERSISTENCE ou PIPELINE, pas GNSS", () => {
    const cls = classifyTrackingObservability({
      locationFixAgeSeconds: 15,
      taskInvokeAgeSeconds: 400,
      fgsRunning: true,
      fgsExpected: true,
      queueDepth: 12,
      oldestQueueItemAgeSeconds: 180,
      persistenceLagSeconds: null,
      enqueueWithoutPersist: true,
    });
    // Queue oldest bloquée → PIPELINE prioritaire (situation type HOL)
    expect(cls).toBe("PIPELINE");
    expect(isGpsStaleAlertClass(cls)).toBe(false);
  });

  it("O6 aucun Location connu → UNKNOWN, pas faux GPS stale", () => {
    expect(computeLocationFixAgeSeconds(null, NOW)).toBeNull();
    const cls = classifyTrackingObservability({
      locationFixAgeSeconds: null,
      taskInvokeAgeSeconds: 400,
      fgsRunning: true,
      fgsExpected: true,
      queueDepth: 0,
      oldestQueueItemAgeSeconds: null,
      persistenceLagSeconds: null,
    });
    expect(cls).toBe("UNKNOWN");
    expect(isGpsStaleAlertClass(cls)).toBe(false);
  });

  it("O7 timestamp futur/invalide → métrique protégée (null)", () => {
    expect(computeLocationFixAgeSeconds(NOW + 600_000, NOW)).toBeNull();
    expect(computeLocationFixAgeSeconds(Number.NaN, NOW)).toBeNull();
    expect(computeLocationFixAgeSeconds(42, NOW)).toBeNull();
    // secondes Unix plausibles → converties
    const sec = Math.floor(NOW / 1000) - 3;
    expect(computeLocationFixAgeSeconds(sec, NOW)).toBe(3);
  });
});
