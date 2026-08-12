import { describe, expect, it } from "@jest/globals";
import { isPresenceEngineEligible } from "./PresenceTrackingEngine";

describe("PresenceTrackingEngine", () => {
  it("inéligible en FG hors fenêtre même disponible + disclosure (P0-F TIME)", () => {
    expect(
      isPresenceEngineEligible({
        driverAvailable: true,
        presenceWindowOpen: false,
        appForeground: true,
        presenceDisclosureAccepted: true,
        hasActiveMission: false,
      })
    ).toBe(false);
  });

  it("éligible en FG dans la fenêtre si disponible + disclosure", () => {
    expect(
      isPresenceEngineEligible({
        driverAvailable: true,
        presenceWindowOpen: true,
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
