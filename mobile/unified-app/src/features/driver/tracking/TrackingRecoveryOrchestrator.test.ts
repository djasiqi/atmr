/**
 * Tests P6 — FSM recovery event-driven (sans sleep).
 */
import { beforeEach, describe, expect, it, jest } from "@jest/globals";

jest.mock("@react-native-async-storage/async-storage", () => {
  const store = new Map<string, string>();
  return {
    getItem: jest.fn(async (k: string) => store.get(k) ?? null),
    setItem: jest.fn(async (k: string, v: string) => {
      store.set(k, v);
    }),
    removeItem: jest.fn(async (k: string) => {
      store.delete(k);
    }),
    __store: store,
  };
});

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
}));

const mockIsFeatureEnabled = jest.fn(() => true);
jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (key: string) => mockIsFeatureEnabled(key),
}));

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    getSnapshot: () => ({ activeContextId: null }),
    connect: jest.fn(),
  },
}));

import {
  __resetTrackingRecoveryForTests,
  EXHAUSTED_BACKOFF_MS,
  tickTrackingRecovery,
  type RecoveryHandlers,
} from "./TrackingRecoveryOrchestrator";

const unhealthy = {
  reason: "anti_zombie",
  fixRecent: false as const,
  watchAlive: false as const,
};

describe("TrackingRecoveryOrchestrator P6", () => {
  const handlers: RecoveryHandlers = {
    restartWatch: jest.fn(async () => undefined),
    restartFgs: jest.fn(async () => undefined),
    restartEngine: jest.fn(async () => undefined),
    reconnectTransport: jest.fn(async () => undefined),
  };

  beforeEach(() => {
    __resetTrackingRecoveryForTests();
    mockIsFeatureEnabled.mockReturnValue(true);
    (handlers.restartWatch as jest.Mock).mockClear();
    (handlers.restartFgs as jest.Mock).mockClear();
    (handlers.restartEngine as jest.Mock).mockClear();
    (handlers.reconnectTransport as jest.Mock).mockClear();
    const AsyncStorage = require("@react-native-async-storage/async-storage") as {
      __store: Map<string, string>;
    };
    AsyncStorage.__store.clear();
  });

  it("flag off → restartWatch seul et reste HEALTHY", async () => {
    mockIsFeatureEnabled.mockReturnValue(false);
    const state = await tickTrackingRecovery(
      Date.now(),
      { reason: "anti_zombie", fixRecent: false },
      handlers
    );
    expect(handlers.restartWatch).toHaveBeenCalledTimes(1);
    expect(handlers.restartFgs).not.toHaveBeenCalled();
    expect(state.recoveryStage).toBe("HEALTHY");
  });

  it("avance HEALTHY → VERIFY_WATCH sans sleep", async () => {
    const now = 1_000_000;
    const state = await tickTrackingRecovery(now, unhealthy, handlers);
    expect(handlers.restartWatch).toHaveBeenCalledTimes(1);
    expect(state.recoveryStage).toBe("VERIFY_WATCH");
    expect(state.nextCheckAt).toBe(now + 5_000);
    expect(state.recoveryGeneration).toBe(1);
  });

  it("100 ticks neutres (reason seule) → 0 restartWatch", async () => {
    const now = 1_000_000;
    for (let i = 0; i < 100; i += 1) {
      await tickTrackingRecovery(
        now + i,
        { reason: `neutral_tick_${i}` },
        handlers
      );
    }
    expect(handlers.restartWatch).not.toHaveBeenCalled();
    expect(handlers.restartFgs).not.toHaveBeenCalled();
    expect(handlers.restartEngine).not.toHaveBeenCalled();
  });

  it("respecte nextCheckAt (pas d'avance prématurée)", async () => {
    const now = 1_000_000;
    await tickTrackingRecovery(now, unhealthy, handlers);
    (handlers.restartWatch as jest.Mock).mockClear();
    const mid = await tickTrackingRecovery(now + 1_000, unhealthy, handlers);
    expect(mid.recoveryStage).toBe("VERIFY_WATCH");
    expect(handlers.restartFgs).not.toHaveBeenCalled();
  });

  it("après fenêtre : VERIFY_WATCH → VERIFY_FGS via restartFgs", async () => {
    const now = 1_000_000;
    await tickTrackingRecovery(now, unhealthy, handlers);
    const next = await tickTrackingRecovery(now + 5_001, unhealthy, handlers);
    expect(handlers.restartFgs).toHaveBeenCalledTimes(1);
    expect(next.recoveryStage).toBe("VERIFY_FGS");
  });

  it("preuve de santé → retour HEALTHY en conservant recoveryGeneration", async () => {
    const now = 1_000_000;
    const started = await tickTrackingRecovery(now, unhealthy, handlers);
    expect(started.recoveryGeneration).toBe(1);
    const recovered = await tickTrackingRecovery(
      now + 100,
      { reason: "fix_ok", fixRecent: true },
      handlers
    );
    expect(recovered.recoveryStage).toBe("HEALTHY");
    expect(recovered.recoveryGeneration).toBe(1);
  });

  it("cycle non résolu → DEGRADED_EXHAUSTED sans nouvelle cascade immédiate", async () => {
    let now = 1_000_000;
    // HEALTHY → VERIFY_WATCH
    let state = await tickTrackingRecovery(now, unhealthy, handlers);
    expect(state.recoveryStage).toBe("VERIFY_WATCH");
    const generation = state.recoveryGeneration;
    (handlers.restartWatch as jest.Mock).mockClear();

    // VERIFY_WATCH → VERIFY_FGS
    now += 5_001;
    state = await tickTrackingRecovery(now, unhealthy, handlers);
    expect(state.recoveryStage).toBe("VERIFY_FGS");

    // VERIFY_FGS → VERIFY_ACK
    now += 5_001;
    state = await tickTrackingRecovery(now, {
      ...unhealthy,
      fgsAlive: false,
    }, handlers);
    expect(state.recoveryStage).toBe("VERIFY_ACK");
    expect(handlers.reconnectTransport).toHaveBeenCalled();

    // VERIFY_ACK → REBUILD_RUNTIME
    now += 5_001;
    state = await tickTrackingRecovery(now, {
      ...unhealthy,
      transportOk: false,
      ackRecent: false,
    }, handlers);
    expect(state.recoveryStage).toBe("REBUILD_RUNTIME");
    expect(handlers.restartEngine).toHaveBeenCalled();

    // REBUILD_RUNTIME non résolu → DEGRADED_EXHAUSTED
    now += 5_001;
    state = await tickTrackingRecovery(now, unhealthy, handlers);
    expect(state.recoveryStage).toBe("DEGRADED_EXHAUSTED");
    expect(state.nextCheckAt).toBe(now + EXHAUSTED_BACKOFF_MS);
    expect(state.recoveryGeneration).toBe(generation);

    // Tick immédiat unhealthy → pas de nouvelle cascade
    (handlers.restartWatch as jest.Mock).mockClear();
    const blocked = await tickTrackingRecovery(now + 1, unhealthy, handlers);
    expect(blocked.recoveryStage).toBe("DEGRADED_EXHAUSTED");
    expect(handlers.restartWatch).not.toHaveBeenCalled();
  });

  it("après cooldown : génération N → N+1 sur nouvelle cascade", async () => {
    let now = 1_000_000;
    let state = await tickTrackingRecovery(now, unhealthy, handlers);
    // Avancer jusqu'à DEGRADED_EXHAUSTED
    for (let i = 0; i < 4; i += 1) {
      now = state.nextCheckAt + 1;
      state = await tickTrackingRecovery(now, unhealthy, handlers);
    }
    expect(state.recoveryStage).toBe("DEGRADED_EXHAUSTED");
    expect(state.recoveryGeneration).toBe(1);

    (handlers.restartWatch as jest.Mock).mockClear();
    const afterCooldown = await tickTrackingRecovery(
      state.nextCheckAt + 1,
      unhealthy,
      handlers
    );
    expect(afterCooldown.recoveryStage).toBe("VERIFY_WATCH");
    expect(afterCooldown.recoveryGeneration).toBe(2);
    expect(handlers.restartWatch).toHaveBeenCalledTimes(1);
  });

  it("preuve healthy arrête la cascade en cours", async () => {
    const now = 1_000_000;
    await tickTrackingRecovery(now, unhealthy, handlers);
    (handlers.restartFgs as jest.Mock).mockClear();
    const recovered = await tickTrackingRecovery(
      now + 5_001,
      { reason: "recovered", fixRecent: true, watchAlive: true },
      handlers
    );
    expect(recovered.recoveryStage).toBe("HEALTHY");
    expect(recovered.recoveryGeneration).toBe(1);
    expect(handlers.restartFgs).not.toHaveBeenCalled();
    expect(handlers.restartEngine).not.toHaveBeenCalled();
  });
});
