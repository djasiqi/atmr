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
    });
    expect(leaseAllowsCapture(lease)).toBe(true);
    expect(leaseAllowsTransport(lease)).toBe(true);
  });

  it("switching depuis driver : capture ON, transport OFF", async () => {
    const previous = await setTrackingContextLeaseDriverActive({
      contextId: "driver:42",
      driverId: 42,
      sessionGenerationId: 3,
      trackingGenerationId: "trk-a",
      trackingIdentityId: "driver:42:company:1",
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
    });
    await setTrackingContextLeaseSwitching({
      fromDriver: true,
      previousDriverActive: previous,
    });
    const restored = await restoreTrackingContextLeaseDriverActiveFromSwitching();
    expect(restored?.state).toBe("driver_active");
    expect(restored?.driverId).toBe(7);
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
    });
    __resetTrackingContextLeaseForTests();
    const lease = await readTrackingContextLease();
    expect(lease?.state).toBe("driver_active");
    if (lease?.state === "driver_active") {
      expect(lease.trackingGenerationId).toBe("trk-persist");
    }
  });
});
