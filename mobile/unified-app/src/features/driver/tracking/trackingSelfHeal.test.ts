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
  forceRestartTrackingWatch,
  resetAntiZombieForTests,
  resetColdStartForTests,
  shouldTriggerAntiZombie,
  shouldTriggerColdStart,
  type SelfHealActions,
  type SelfHealBridgeSlice,
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

describe("shouldTriggerAntiZombie — aucun fix jamais produit (D5 UNKNOWN)", () => {
  const NOW = 1_000_000_000_000;

  beforeEach(() => {
    mockIsFeatureEnabled.mockReset();
    mockIsFeatureEnabled.mockReturnValue(true);
    resetAntiZombieForTests();
  });

  afterEach(() => {
    resetAntiZombieForTests();
  });

  it("D5 T4 : null/null + startedAge>60s → PAS d'anti-zombie (UNKNOWN, pas Unregister)", () => {
    expect(
      shouldTriggerAntiZombie({
        isTrackingRunning: true,
        lastFixProducedAtMs: null,
        lastSentAt: null,
        trackingStartedAtMs: NOW - (ANTI_ZOMBIE_FIX_AGE_SEC + 10) * 1000,
        nowMs: NOW,
      })
    ).toBe(false);
  });

  it("ne déclenche pas si le runtime vient de démarrer sans preuve de fraîcheur", () => {
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

describe("forceRestartTrackingWatch — D5 L1 non destructif", () => {
  beforeEach(() => {
    mockIsFeatureEnabled.mockReset();
    mockIsFeatureEnabled.mockReturnValue(true);
  });

  function makeSlice(): SelfHealBridgeSlice {
    return {
      watchSubscription: null,
      staleFallbackTimeouts: 2,
      staleFallbackBlockedUntilMs: 0,
      lastWatchAtMs: null,
      lastWatchedPosition: null,
      lastWatchRestartAtMs: 0,
      watchRestartTimestampsMs: [],
      missionId: 38224,
    };
  }

  function makeActions(stopBackground: jest.Mock): SelfHealActions {
    return {
      stopWatch: jest.fn(),
      stopBackground: stopBackground as unknown as SelfHealActions["stopBackground"],
      ensureNativeForeground: jest.fn(async () => undefined),
      ensureLocationWatch: jest.fn(async () => undefined),
      triggerDeviceHealth: jest.fn(),
    };
  }

  it("T4 : L1 par défaut n'appelle pas stopBackground (pas d'Unregister)", async () => {
    const stopBackground = jest.fn(async () => undefined);
    const ok = await forceRestartTrackingWatch(
      "anti_zombie_fix_stale",
      makeSlice(),
      makeActions(stopBackground),
      "active"
    );
    expect(ok).toBe(true);
    expect(stopBackground).not.toHaveBeenCalled();
  });

  it("L2 optionnel appelle stopBackground uniquement si allowDestructiveRestart", async () => {
    const stopBackground = jest.fn(async () => undefined);
    const ok = await forceRestartTrackingWatch(
      "native_proof_failed",
      makeSlice(),
      makeActions(stopBackground),
      "active",
      { allowDestructiveRestart: true }
    );
    expect(ok).toBe(true);
    expect(stopBackground).toHaveBeenCalledWith("self_heal_restart");
  });
});
