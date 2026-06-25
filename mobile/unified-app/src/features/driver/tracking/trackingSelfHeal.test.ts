import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

jest.mock(
  "expo-location",
  () => ({
    requestForegroundPermissionsAsync: async () => ({ granted: true }),
    LocationSubscription: class {},
  }),
  { virtual: true }
);

jest.mock(
  "@sentry/react-native",
  () => ({ captureMessage: () => undefined, addBreadcrumb: () => undefined }),
  { virtual: true }
);

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: () => undefined,
}));

const mockIsFeatureEnabled = jest.fn<(key: string) => boolean>();
jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (key: string) => mockIsFeatureEnabled(key),
}));

import {
  ANTI_ZOMBIE_FIX_AGE_SEC,
  COLD_START_THRESHOLD_SEC,
  resetAntiZombieForTests,
  resetColdStartForTests,
  shouldTriggerAntiZombie,
  shouldTriggerColdStart,
} from "./trackingSelfHeal";

describe("shouldTriggerColdStart", () => {
  const NOW = 1_000_000_000_000;

  beforeEach(() => {
    mockIsFeatureEnabled.mockReset();
    mockIsFeatureEnabled.mockReturnValue(true);
    resetColdStartForTests();
  });

  afterEach(() => {
    resetColdStartForTests();
  });

  it("déclenche quand mission active, tracking arrêté et jamais envoyé", () => {
    expect(
      shouldTriggerColdStart({
        hasActiveMission: true,
        isTrackingRunning: false,
        lastSentAt: null,
        nowMs: NOW,
      })
    ).toBe(true);
  });

  it("ne déclenche pas si le flag est OFF", () => {
    mockIsFeatureEnabled.mockReturnValue(false);
    expect(
      shouldTriggerColdStart({
        hasActiveMission: true,
        isTrackingRunning: false,
        lastSentAt: null,
        nowMs: NOW,
      })
    ).toBe(false);
  });

  it("ne déclenche pas sans mission active", () => {
    expect(
      shouldTriggerColdStart({
        hasActiveMission: false,
        isTrackingRunning: false,
        lastSentAt: null,
        nowMs: NOW,
      })
    ).toBe(false);
  });

  it("ne déclenche pas quand le tracking tourne (couvert par anti-zombie)", () => {
    expect(
      shouldTriggerColdStart({
        hasActiveMission: true,
        isTrackingRunning: true,
        lastSentAt: null,
        nowMs: NOW,
      })
    ).toBe(false);
  });

  it("déclenche si la dernière position est plus vieille que le seuil", () => {
    const lastSentAt = new Date(NOW - (COLD_START_THRESHOLD_SEC + 10) * 1000).toISOString();
    expect(
      shouldTriggerColdStart({
        hasActiveMission: true,
        isTrackingRunning: false,
        lastSentAt,
        nowMs: NOW,
      })
    ).toBe(true);
  });

  it("ne déclenche pas si une position récente a été envoyée", () => {
    const lastSentAt = new Date(NOW - 10 * 1000).toISOString();
    expect(
      shouldTriggerColdStart({
        hasActiveMission: true,
        isTrackingRunning: false,
        lastSentAt,
        nowMs: NOW,
      })
    ).toBe(false);
  });
});

describe("shouldTriggerAntiZombie — aucun fix jamais produit", () => {
  const NOW = 1_000_000_000_000;

  beforeEach(() => {
    mockIsFeatureEnabled.mockReset();
    mockIsFeatureEnabled.mockReturnValue(true);
    resetAntiZombieForTests();
  });

  afterEach(() => {
    resetAntiZombieForTests();
  });

  it("déclenche si le runtime tourne depuis > seuil sans aucun fix", () => {
    expect(
      shouldTriggerAntiZombie({
        isTrackingRunning: true,
        lastFixProducedAtMs: null,
        lastSentAt: null,
        trackingStartedAtMs: NOW - (ANTI_ZOMBIE_FIX_AGE_SEC + 10) * 1000,
        nowMs: NOW,
      })
    ).toBe(true);
  });

  it("ne déclenche pas si le runtime vient de démarrer (grâce au seuil)", () => {
    expect(
      shouldTriggerAntiZombie({
        isTrackingRunning: true,
        lastFixProducedAtMs: null,
        lastSentAt: null,
        trackingStartedAtMs: NOW - 5 * 1000,
        nowMs: NOW,
      })
    ).toBe(false);
  });

  it("rétro-compat : sans trackingStartedAtMs et sans fix → false", () => {
    expect(
      shouldTriggerAntiZombie({
        isTrackingRunning: true,
        lastFixProducedAtMs: null,
        lastSentAt: null,
        nowMs: NOW,
      })
    ).toBe(false);
  });

  it("ne déclenche pas si le tracking n'est pas en cours", () => {
    expect(
      shouldTriggerAntiZombie({
        isTrackingRunning: false,
        lastFixProducedAtMs: null,
        lastSentAt: null,
        trackingStartedAtMs: NOW - 10 * 60 * 1000,
        nowMs: NOW,
      })
    ).toBe(false);
  });

  it("déclenche si le dernier ENVOI est périmé même avec un fix local frais (zombie)", () => {
    expect(
      shouldTriggerAntiZombie({
        isTrackingRunning: true,
        lastFixProducedAtMs: NOW, // fix natif/local « frais »
        lastSentAt: new Date(NOW - (ANTI_ZOMBIE_FIX_AGE_SEC + 30) * 1000).toISOString(),
        trackingStartedAtMs: NOW - 10 * 60 * 1000,
        nowMs: NOW,
      })
    ).toBe(true);
  });

  it("ne déclenche pas si le dernier envoi est récent", () => {
    expect(
      shouldTriggerAntiZombie({
        isTrackingRunning: true,
        lastFixProducedAtMs: NOW - 10 * 60 * 1000, // fix ancien
        lastSentAt: new Date(NOW - 10 * 1000).toISOString(), // envoi récent
        trackingStartedAtMs: NOW - 10 * 60 * 1000,
        nowMs: NOW,
      })
    ).toBe(false);
  });
});
