import { afterEach, describe, expect, it } from "@jest/globals";
import {
  getDriverAvailabilityActive,
  resetDriverAvailabilityBridgeForTests,
  setDriverAvailabilityActive,
} from "./driverAvailabilityBridge";
import { resolveTrackingEligibility } from "../tracking/trackingEligibility";

describe("driverAvailabilityBridge", () => {
  afterEach(() => {
    resetDriverAvailabilityBridgeForTests();
  });

  it("cold start : UNKNOWN, pas éligible PRESENCE/LIVE, pas hors service", () => {
    expect(getDriverAvailabilityActive()).toBeNull();
    const r = resolveTrackingEligibility({
      driverAvailable: getDriverAvailabilityActive(),
      appForeground: true,
      presenceDisclosureAccepted: true,
      permissionsReady: true,
      hasActiveMission: false,
    });
    expect(r.availabilityPending).toBe(true);
    expect(r.trackingEligible).toBe(false);
    expect(r.blocked).toBe(false);
    expect(r.mode).toBe("OFF");
  });

  it("cache is_available=false → UNAVAILABLE → OFF", () => {
    setDriverAvailabilityActive(false);
    expect(getDriverAvailabilityActive()).toBe(false);
    const r = resolveTrackingEligibility({
      driverAvailable: getDriverAvailabilityActive(),
      appForeground: true,
      presenceDisclosureAccepted: true,
      permissionsReady: true,
      hasActiveMission: true,
    });
    expect(r.mode).toBe("OFF");
    expect(r.availabilityPending).toBe(false);
    expect(r.trackingEligible).toBe(false);
  });

  it("logout/reset → retour UNKNOWN", () => {
    setDriverAvailabilityActive(true);
    resetDriverAvailabilityBridgeForTests();
    expect(getDriverAvailabilityActive()).toBeNull();
  });
});
