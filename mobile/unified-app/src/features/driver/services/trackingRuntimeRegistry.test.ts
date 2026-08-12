/**
 * Tests Phase 1C — génération runtime GPS.
 */
import { beforeEach, describe, expect, it } from "@jest/globals";

jest.mock("../../../core/auth/authCredentialStore", () => ({
  getSessionGenerationId: jest.fn(() => 10),
}));

const mockTerminalListeners = new Set<(event: unknown) => void>();

jest.mock("../../../core/auth/sessionAuthDecision", () => ({
  TRACKING_AUTH_EFFECT_POLICY: {
    explicit_logout: { stop: true, quarantine: true, restartAllowed: false },
    temporary_refresh_failure: { stop: false, quarantine: false, restartAllowed: true },
    account_revoked: { stop: true, quarantine: true, restartAllowed: false },
    auth_exhausted_socket: { stop: false, quarantine: false, restartAllowed: true },
  },
  getTrackingAuthAvailability: jest.fn(() => ({
    kind: "SESSION_AVAILABLE",
    sessionGenerationId: 10,
    trackingIdentityId: "driver:1:company:9",
    driverId: 1,
  })),
  subscribeToTrackingAuthTerminalEvents: (listener: (event: unknown) => void) => {
    mockTerminalListeners.add(listener);
    return () => mockTerminalListeners.delete(listener);
  },
}));

import {
  __resetTrackingRuntimeRegistryForTests,
  canFlushDurableEvent,
  captureActiveRuntime,
  clearActiveRuntimeIfGeneration,
  isNativeOwnerCurrent,
  isRuntimeActive,
  runIfRuntimeActive,
  startOrJoinTrackingRuntime,
  stopTrackingRuntime,
  toNativeTrackingOwner,
  updateMissionContext,
} from "./trackingRuntimeRegistry";

describe("trackingRuntimeRegistry Phase 1C", () => {
  beforeEach(() => {
    __resetTrackingRuntimeRegistryForTests();
  });

  it("mission change keeps generation and bumps missionContextVersion", async () => {
    const r1 = await startOrJoinTrackingRuntime({
      driverId: 1,
      companyId: 9,
      missionId: 100,
      missionStatus: "EN_ROUTE" as never,
    });
    const gen = r1.identity.trackingGenerationId;
    const versionBefore = r1.missionContext.missionContextVersion;
    const r2 = await startOrJoinTrackingRuntime({
      driverId: 1,
      companyId: 9,
      missionId: 200,
      missionStatus: "EN_ROUTE" as never,
    });
    expect(r2.identity.trackingGenerationId).toBe(gen);
    expect(r2.missionContext.missionId).toBe(200);
    expect(r2.missionContext.missionContextVersion).toBeGreaterThan(versionBefore);
  });

  it("toNativeTrackingOwner inclut missionId et isNativeOwnerCurrent compare mission+version", async () => {
    const r1 = await startOrJoinTrackingRuntime({
      driverId: 1,
      companyId: 9,
      missionId: 10,
      missionStatus: "EN_ROUTE" as never,
    });
    const owner = toNativeTrackingOwner(r1);
    expect(owner.missionId).toBe(10);
    expect(owner.missionContextVersion).toBe(r1.missionContext.missionContextVersion);
    expect(isNativeOwnerCurrent(owner)).toBe(true);

    updateMissionContext(20, "EN_ROUTE" as never);
    expect(isNativeOwnerCurrent(owner)).toBe(false);

    const ownerAfter = toNativeTrackingOwner(captureActiveRuntime()!);
    expect(ownerAfter.missionId).toBe(20);
    expect(isNativeOwnerCurrent(ownerAfter)).toBe(true);
  });

  it("stale stop is ignored after new generation", async () => {
    const r1 = await startOrJoinTrackingRuntime({
      driverId: 1,
      companyId: 9,
      missionId: 1,
      missionStatus: "EN_ROUTE" as never,
    });
    const g1 = r1.identity.trackingGenerationId;
    await stopTrackingRuntime(
      {
        expectedTrackingGenerationId: g1,
        reason: "manual_stop",
        quarantinePolicy: "none",
      },
      { invokePhysicalStop: false }
    );
    const r2 = await startOrJoinTrackingRuntime({
      driverId: 1,
      companyId: 9,
      missionId: 2,
      missionStatus: "EN_ROUTE" as never,
      forceNewGeneration: true,
    });
    const stale = await stopTrackingRuntime(
      {
        expectedTrackingGenerationId: g1,
        reason: "manual_stop",
        quarantinePolicy: "none",
      },
      { invokePhysicalStop: false }
    );
    expect(stale.status).toBe("ignored_stale_stop");
    expect(captureActiveRuntime()?.identity.trackingGenerationId).toBe(
      r2.identity.trackingGenerationId
    );
  });

  it("runIfRuntimeActive ignores after generation replaced", async () => {
    const r1 = await startOrJoinTrackingRuntime({
      driverId: 1,
      companyId: 9,
      missionId: 1,
      missionStatus: "EN_ROUTE" as never,
    });
    const identity = r1.identity;
    await stopTrackingRuntime(
      {
        expectedTrackingGenerationId: identity.trackingGenerationId,
        reason: "runtime_replaced",
        quarantinePolicy: "none",
      },
      { invokePhysicalStop: false }
    );
    await startOrJoinTrackingRuntime({
      driverId: 1,
      companyId: 9,
      missionId: 2,
      missionStatus: "EN_ROUTE" as never,
      forceNewGeneration: true,
    });
    const result = await runIfRuntimeActive(identity, async () => "should-not");
    expect(result.status).toBe("ignored_stale_runtime");
    expect(isRuntimeActive(identity)).toBe(false);
  });

  it("mission context snapshot is not rewritten by later update", async () => {
    const r1 = await startOrJoinTrackingRuntime({
      driverId: 1,
      companyId: 9,
      missionId: 10,
      missionStatus: "EN_ROUTE" as never,
    });
    const snapshot = { ...r1.missionContext };
    updateMissionContext(99, "COMPLETED" as never);
    expect(snapshot.missionId).toBe(10);
    expect(captureActiveRuntime()?.missionContext.missionId).toBe(99);
  });

  it("flush durable allowed for same identity even if generation inactive", async () => {
    const r1 = await startOrJoinTrackingRuntime({
      driverId: 1,
      companyId: 9,
      missionId: 1,
      missionStatus: "EN_ROUTE" as never,
    });
    const identityId = r1.identity.trackingIdentityId;
    clearActiveRuntimeIfGeneration(r1.identity.trackingGenerationId);
    expect(
      canFlushDurableEvent({
        trackingIdentityId: identityId,
        partitionQuarantined: false,
      })
    ).toBe(true);
    expect(
      canFlushDurableEvent({
        trackingIdentityId: identityId,
        partitionQuarantined: true,
      })
    ).toBe(false);
    expect(
      canFlushDurableEvent({
        trackingIdentityId: "driver:2:company:9",
        partitionQuarantined: false,
      })
    ).toBe(false);
  });
});
