import { describe, expect, it } from "@jest/globals";
import { isPresenceEngineEligible } from "./PresenceTrackingEngine";

describe("PresenceTrackingEngine", () => {
  it("éligible hors ancienne fenêtre 07–19 si disponible + permissions", () => {
    expect(
      isPresenceEngineEligible({
        driverAvailable: true,
        presenceWindowOpen: false,
        appForeground: true,
        presenceDisclosureAccepted: true,
        permissionsReady: true,
        hasActiveMission: false,
      })
    ).toBe(true);
  });

  it("éligible en FG si disponible + permissions", () => {
    expect(
      isPresenceEngineEligible({
        driverAvailable: true,
        presenceWindowOpen: true,
        appForeground: true,
        presenceDisclosureAccepted: true,
        permissionsReady: true,
        hasActiveMission: false,
      })
    ).toBe(true);
  });

  it("inéligible en BG sans permissionsReady", () => {
    expect(
      isPresenceEngineEligible({
        driverAvailable: true,
        presenceWindowOpen: true,
        appForeground: false,
        presenceDisclosureAccepted: true,
        permissionsReady: false,
        hasActiveMission: false,
      })
    ).toBe(false);
  });
});
