/**
 * Tests P0 — validateNativeOwnerForHeadless (sans activeRuntime).
 */
import { beforeEach, describe, expect, it } from "@jest/globals";

jest.mock("../../../core/auth/authCredentialStore", () => ({
  getSessionGenerationId: jest.fn(() => 10),
}));

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
    trackingIdentityId: "driver:42:company:1",
    driverId: 42,
  })),
  subscribeToTrackingAuthTerminalEvents: () => () => undefined,
}));

import {
  __resetTrackingRuntimeRegistryForTests,
  toNativeTrackingOwner,
  validateNativeOwnerForHeadless,
  startOrJoinTrackingRuntime,
  isNativeOwnerCurrent,
  captureActiveRuntime,
} from "./trackingRuntimeRegistry";

describe("validateNativeOwnerForHeadless", () => {
  beforeEach(() => {
    __resetTrackingRuntimeRegistryForTests();
  });

  it("autorise owner+lease valides même si activeRuntime est null", async () => {
    const runtime = await startOrJoinTrackingRuntime({
      driverId: 42,
      companyId: 1,
      missionId: 1,
      missionStatus: "EN_ROUTE" as never,
    });
    const owner = toNativeTrackingOwner(runtime);
    expect(owner.driverId).toBe(42);
    expect(owner.missionId).toBe(1);
    // Simule process death : clear runtime mémoire
    __resetTrackingRuntimeRegistryForTests();
    expect(captureActiveRuntime()).toBeNull();
    expect(isNativeOwnerCurrent(owner)).toBe(false);

    const check = validateNativeOwnerForHeadless({
      owner,
      lease: {
        state: "driver_active",
        contextId: "driver:42",
        driverId: 42,
        sessionGenerationId: owner.sessionGenerationId,
        trackingGenerationId: owner.trackingGenerationId,
        trackingIdentityId: owner.trackingIdentityId,
        missionId: owner.missionId,
        missionContextVersion: owner.missionContextVersion,
      },
      authUsable: true,
    });
    expect(check).toEqual({ ok: true });
  });

  it("refuse génération obsolète", () => {
    const check = validateNativeOwnerForHeadless({
      owner: {
        driverId: 42,
        sessionGenerationId: 1,
        trackingGenerationId: "old",
        trackingIdentityId: "driver:42:company:1",
        missionContextVersion: 1,
        missionId: 1,
      },
      lease: {
        state: "driver_active",
        contextId: "driver:42",
        driverId: 42,
        sessionGenerationId: 1,
        trackingGenerationId: "new",
        trackingIdentityId: "driver:42:company:1",
        missionId: 1,
        missionContextVersion: 1,
      },
      authUsable: true,
    });
    expect(check.ok).toBe(false);
    if (!check.ok) {
      expect(check.reason).toBe("tracking_generation_mismatch");
    }
  });

  it("refuse missionId / missionContextVersion mismatch vs lease", () => {
    const baseOwner = {
      driverId: 42,
      sessionGenerationId: 1,
      trackingGenerationId: "trk",
      trackingIdentityId: "driver:42:company:1",
      missionContextVersion: 2,
      missionId: 10,
    };
    const baseLease = {
      state: "driver_active" as const,
      contextId: "driver:42",
      driverId: 42,
      sessionGenerationId: 1,
      trackingGenerationId: "trk",
      trackingIdentityId: "driver:42:company:1",
      missionId: 10,
      missionContextVersion: 2,
    };
    const missionMismatch = validateNativeOwnerForHeadless({
      owner: { ...baseOwner, missionId: 99 },
      lease: baseLease,
      authUsable: true,
    });
    expect(missionMismatch.ok).toBe(false);
    if (!missionMismatch.ok) {
      expect(missionMismatch.reason).toBe("mission_id_mismatch");
    }
    const versionMismatch = validateNativeOwnerForHeadless({
      owner: baseOwner,
      lease: { ...baseLease, missionContextVersion: 1 },
      authUsable: true,
    });
    expect(versionMismatch.ok).toBe(false);
    if (!versionMismatch.ok) {
      expect(versionMismatch.reason).toBe("mission_context_version_mismatch");
    }
  });

  it("refuse lease inactive", () => {
    const check = validateNativeOwnerForHeadless({
      owner: {
        driverId: 42,
        sessionGenerationId: 1,
        trackingGenerationId: "trk",
        trackingIdentityId: "driver:42:company:1",
        missionContextVersion: 1,
        missionId: 1,
      },
      lease: { state: "inactive" },
      authUsable: true,
    });
    expect(check.ok).toBe(false);
  });

  it("refuse owner absent", () => {
    const check = validateNativeOwnerForHeadless({
      owner: null,
      lease: {
        state: "driver_active",
        contextId: "driver:42",
        driverId: 42,
        sessionGenerationId: 1,
        trackingGenerationId: "trk",
        trackingIdentityId: "driver:42:company:1",
        missionId: 1,
        missionContextVersion: 1,
      },
      authUsable: true,
    });
    expect(check.ok).toBe(false);
    if (!check.ok) {
      expect(check.reason).toBe("missing_native_owner");
    }
  });
});
