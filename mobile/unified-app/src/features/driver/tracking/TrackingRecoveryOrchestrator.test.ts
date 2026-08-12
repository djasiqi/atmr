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
  tickTrackingRecovery,
  type RecoveryHandlers,
} from "./TrackingRecoveryOrchestrator";

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
    const state = await tickTrackingRecovery(
      now,
      { reason: "anti_zombie", fixRecent: false, watchAlive: false },
      handlers
    );
    expect(handlers.restartWatch).toHaveBeenCalledTimes(1);
    expect(state.recoveryStage).toBe("VERIFY_WATCH");
    expect(state.nextCheckAt).toBe(now + 5_000);
  });

  it("respecte nextCheckAt (pas d'avance prématurée)", async () => {
    const now = 1_000_000;
    await tickTrackingRecovery(
      now,
      { reason: "anti_zombie", fixRecent: false },
      handlers
    );
    (handlers.restartWatch as jest.Mock).mockClear();
    const mid = await tickTrackingRecovery(
      now + 1_000,
      { reason: "anti_zombie", fixRecent: false },
      handlers
    );
    expect(mid.recoveryStage).toBe("VERIFY_WATCH");
    expect(handlers.restartFgs).not.toHaveBeenCalled();
  });

  it("après fenêtre : VERIFY_WATCH → VERIFY_FGS via restartFgs", async () => {
    const now = 1_000_000;
    await tickTrackingRecovery(
      now,
      { reason: "anti_zombie", fixRecent: false },
      handlers
    );
    const next = await tickTrackingRecovery(
      now + 5_001,
      { reason: "anti_zombie", fixRecent: false, watchAlive: false },
      handlers
    );
    expect(handlers.restartFgs).toHaveBeenCalledTimes(1);
    expect(next.recoveryStage).toBe("VERIFY_FGS");
  });

  it("preuve de santé → retour HEALTHY", async () => {
    const now = 1_000_000;
    await tickTrackingRecovery(
      now,
      { reason: "anti_zombie", fixRecent: false },
      handlers
    );
    const recovered = await tickTrackingRecovery(
      now + 100,
      { reason: "fix_ok", fixRecent: true },
      handlers
    );
    expect(recovered.recoveryStage).toBe("HEALTHY");
  });
});
