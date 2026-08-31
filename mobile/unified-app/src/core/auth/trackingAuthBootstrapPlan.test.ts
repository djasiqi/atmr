/**
 * JZ-R1-AUTH-HYDRATION-FIX-17 — plan bootstrap (pure).
 */
import { describe, expect, it } from "@jest/globals";
import { planTrackingAuthPublishOnBootstrap } from "./trackingAuthBootstrapPlan";
import type { TrackingContextLease } from "../../features/driver/services/trackingContextLease";

function driverActiveLease(driverId = 42): TrackingContextLease {
  return {
    state: "driver_active",
    contextId: `driver:${driverId}`,
    driverId,
    sessionGenerationId: 7,
    trackingGenerationId: "trk-cold",
    trackingIdentityId: `driver:${driverId}:company:1`,
    missionId: null,
    missionContextVersion: 0,
    updatedAt: Date.now(),
  };
}

describe("planTrackingAuthPublishOnBootstrap (FIX-17)", () => {
  it("A. cold start + lease driver_active → republish_from_lease", () => {
    const plan = planTrackingAuthPublishOnBootstrap({
      isAuthenticated: true,
      contextType: "driver",
      contextDriverId: 42,
      lease: driverActiveLease(42),
    });
    expect(plan).toEqual({
      action: "republish_from_lease",
      driverId: 42,
      trackingIdentityId: "driver:42:company:1",
      sessionGenerationId: 7,
    });
  });

  it("B. idempotent : même plan si déjà hydraté (décision pure)", () => {
    const input = {
      isAuthenticated: true,
      contextType: "driver" as const,
      contextDriverId: 42,
      lease: driverActiveLease(42),
    };
    expect(planTrackingAuthPublishOnBootstrap(input)).toEqual(
      planTrackingAuthPublishOnBootstrap(input)
    );
    expect(planTrackingAuthPublishOnBootstrap(input).action).toBe(
      "republish_from_lease"
    );
  });

  it("C. DRIVER fresh sans lease actif → acquire_and_publish", () => {
    const plan = planTrackingAuthPublishOnBootstrap({
      isAuthenticated: true,
      contextType: "driver",
      contextDriverId: 42,
      lease: { state: "inactive", updatedAt: 0 },
    });
    expect(plan).toEqual({ action: "acquire_and_publish", driverId: 42 });
  });

  it("D. non authentifié → none (logout / session absente)", () => {
    const plan = planTrackingAuthPublishOnBootstrap({
      isAuthenticated: false,
      contextType: "driver",
      contextDriverId: 42,
      lease: driverActiveLease(42),
    });
    expect(plan).toEqual({ action: "none", reason: "not_authenticated" });
  });

  it("E. contexte COMPANY → none", () => {
    const plan = planTrackingAuthPublishOnBootstrap({
      isAuthenticated: true,
      contextType: "company",
      contextDriverId: null,
      lease: { state: "inactive", updatedAt: 0 },
    });
    expect(plan).toEqual({ action: "none", reason: "not_driver_context" });
  });

  it("F. auth unusable / driver id absent → none", () => {
    expect(
      planTrackingAuthPublishOnBootstrap({
        isAuthenticated: true,
        contextType: "driver",
        contextDriverId: null,
        lease: driverActiveLease(42),
      })
    ).toEqual({ action: "none", reason: "driver_id_unresolvable" });
  });

  it("F2. lease driver_active mismatch → none (pas de publish)", () => {
    const plan = planTrackingAuthPublishOnBootstrap({
      isAuthenticated: true,
      contextType: "driver",
      contextDriverId: 42,
      lease: driverActiveLease(999),
    });
    expect(plan).toEqual({ action: "none", reason: "lease_driver_mismatch" });
  });

  it("G. identité lease consommable après republish plan", () => {
    const lease = driverActiveLease(42);
    const plan = planTrackingAuthPublishOnBootstrap({
      isAuthenticated: true,
      contextType: "driver",
      contextDriverId: 42,
      lease,
    });
    expect(plan.action).toBe("republish_from_lease");
    if (plan.action === "republish_from_lease") {
      expect(plan.trackingIdentityId).toBe(lease.trackingIdentityId);
      expect(plan.sessionGenerationId).toBe(lease.sessionGenerationId);
    }
  });

  it("H. switching lease → acquire_and_publish (pas de republish lease partiel)", () => {
    const plan = planTrackingAuthPublishOnBootstrap({
      isAuthenticated: true,
      contextType: "driver",
      contextDriverId: 42,
      lease: { state: "switching", fromDriver: true, updatedAt: 0 },
    });
    expect(plan).toEqual({ action: "acquire_and_publish", driverId: 42 });
  });
});
