/**
 * Tests P0 — trackingContextLease (capture vs transport, crash-safe).
 */
import { beforeEach, describe, expect, it } from "@jest/globals";

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

import {
  __resetTrackingContextLeaseForTests,
  leaseAllowsCapture,
  leaseAllowsTransport,
  readTrackingContextLease,
  reconcileTrackingContextLeaseFromBootstrap,
  setTrackingContextLeaseDriverActive,
  setTrackingContextLeaseInactive,
  setTrackingContextLeaseSwitching,
  restoreTrackingContextLeaseDriverActiveFromSwitching,
} from "./trackingContextLease";

describe("trackingContextLease", () => {
  beforeEach(() => {
    __resetTrackingContextLeaseForTests();
    const AsyncStorage = require("@react-native-async-storage/async-storage") as {
      __store: Map<string, string>;
    };
    AsyncStorage.__store.clear();
  });

  it("driver_active autorise capture et transport", async () => {
    const lease = await setTrackingContextLeaseDriverActive({
      contextId: "driver:42",
      driverId: 42,
      sessionGenerationId: 3,
      trackingGenerationId: "trk-a",
      trackingIdentityId: "driver:42:company:1",
      missionId: 10,
      missionContextVersion: 2,
    });
    expect(leaseAllowsCapture(lease)).toBe(true);
    expect(leaseAllowsTransport(lease)).toBe(true);
    expect(lease.missionId).toBe(10);
    expect(lease.missionContextVersion).toBe(2);
  });

  it("refuse missionContextVersion non fini", async () => {
    await expect(
      setTrackingContextLeaseDriverActive({
        contextId: "driver:42",
        driverId: 42,
        sessionGenerationId: 3,
        trackingGenerationId: "trk-a",
        trackingIdentityId: "driver:42:company:1",
        missionId: null,
        missionContextVersion: Number.NaN,
      })
    ).rejects.toThrow(/missionContextVersion must be a finite number/);
  });

  it("switching depuis driver : capture ON, transport OFF", async () => {
    const previous = await setTrackingContextLeaseDriverActive({
      contextId: "driver:42",
      driverId: 42,
      sessionGenerationId: 3,
      trackingGenerationId: "trk-a",
      trackingIdentityId: "driver:42:company:1",
      missionId: 7,
      missionContextVersion: 1,
    });
    const switching = await setTrackingContextLeaseSwitching({
      fromDriver: true,
      previousDriverActive: previous,
    });
    expect(leaseAllowsCapture(switching)).toBe(true);
    expect(leaseAllowsTransport(switching)).toBe(false);
  });

  it("inactive : 0 capture, 0 transport", async () => {
    const lease = await setTrackingContextLeaseInactive();
    expect(leaseAllowsCapture(lease)).toBe(false);
    expect(leaseAllowsTransport(lease)).toBe(false);
    expect(leaseAllowsCapture(null)).toBe(false);
  });

  it("échec switch restaure driver_active depuis previous", async () => {
    const previous = await setTrackingContextLeaseDriverActive({
      contextId: "driver:7",
      driverId: 7,
      sessionGenerationId: 1,
      trackingGenerationId: "trk-x",
      trackingIdentityId: "driver:7:company:1",
      missionId: 3,
      missionContextVersion: 4,
    });
    await setTrackingContextLeaseSwitching({
      fromDriver: true,
      previousDriverActive: previous,
    });
    const restored = await restoreTrackingContextLeaseDriverActiveFromSwitching();
    expect(restored?.state).toBe("driver_active");
    expect(restored?.driverId).toBe(7);
    expect(restored?.missionId).toBe(3);
    expect(leaseAllowsTransport(restored)).toBe(true);
  });

  it("crash switching + bootstrap company → inactive (pas de promote)", async () => {
    await setTrackingContextLeaseSwitching({ fromDriver: true });
    const lease = await reconcileTrackingContextLeaseFromBootstrap({
      activeContextId: "company:1",
      activeContextType: "company",
      isAuthenticated: true,
    });
    expect(lease.state).toBe("inactive");
    expect(leaseAllowsTransport(lease)).toBe(false);
  });

  it("crash switching + bootstrap driver + snapshot → restore driver_active", async () => {
    const previous = await setTrackingContextLeaseDriverActive({
      contextId: "driver:9",
      driverId: 9,
      sessionGenerationId: 2,
      trackingGenerationId: "trk-y",
      trackingIdentityId: "driver:9:company:1",
      missionId: null,
      missionContextVersion: 1,
    });
    await setTrackingContextLeaseSwitching({
      fromDriver: true,
      previousDriverActive: previous,
    });
    const lease = await reconcileTrackingContextLeaseFromBootstrap({
      activeContextId: "driver:9",
      activeContextType: "driver",
      isAuthenticated: true,
    });
    expect(lease.state).toBe("driver_active");
    if (lease.state === "driver_active") {
      expect(lease.trackingGenerationId).toBe("trk-y");
    }
  });

  it("persiste et relit après reset mémoire (simule process death)", async () => {
    await setTrackingContextLeaseDriverActive({
      contextId: "driver:5",
      driverId: 5,
      sessionGenerationId: 8,
      trackingGenerationId: "trk-persist",
      trackingIdentityId: "driver:5:company:1",
      missionId: 55,
      missionContextVersion: 9,
    });
    __resetTrackingContextLeaseForTests();
    const lease = await readTrackingContextLease();
    expect(lease?.state).toBe("driver_active");
    if (lease?.state === "driver_active") {
      expect(lease.trackingGenerationId).toBe("trk-persist");
      expect(lease.missionId).toBe(55);
      expect(lease.missionContextVersion).toBe(9);
    }
  });

  it("legacy v1 : fail-closed (jamais réutilisé comme autorité)", async () => {
    const AsyncStorage = require("@react-native-async-storage/async-storage") as {
      __store: Map<string, string>;
      getItem: jest.Mock;
      setItem: jest.Mock;
    };
    AsyncStorage.__store.set(
      "@driver:tracking_context_lease_v1",
      JSON.stringify({
        state: "driver_active",
        contextId: "driver:1",
        driverId: 1,
        sessionGenerationId: 1,
        trackingGenerationId: "legacy",
        trackingIdentityId: "driver:1:company:1",
        updatedAt: Date.now(),
      })
    );
    const lease = await readTrackingContextLease();
    expect(lease?.state).toBe("inactive");
    expect(leaseAllowsTransport(lease)).toBe(false);
    expect(AsyncStorage.__store.has("@driver:tracking_context_lease_v1")).toBe(false);
  });
});
