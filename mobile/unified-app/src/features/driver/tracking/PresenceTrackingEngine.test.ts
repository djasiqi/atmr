import { describe, expect, it } from "@jest/globals";
import { isPresenceEngineEligible } from "./PresenceTrackingEngine";

describe("PresenceTrackingEngine", () => {
  it("éligible en FG hors fenêtre si disponible + disclosure", () => {
    expect(
      isPresenceEngineEligible({
        driverAvailable: true,
        presenceWindowOpen: false,
        appForeground: true,
        presenceDisclosureAccepted: true,
        hasActiveMission: false,
      })
    ).toBe(true);
  });

  it("inéligible en BG hors fenêtre", () => {
    expect(
      isPresenceEngineEligible({
        driverAvailable: true,
        presenceWindowOpen: false,
        appForeground: false,
        presenceDisclosureAccepted: true,
        hasActiveMission: false,
      })
    ).toBe(false);
  });
});
