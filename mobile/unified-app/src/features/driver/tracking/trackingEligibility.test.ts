import { describe, expect, it } from "@jest/globals";
import {
  resolvePresenceGpsAccuracy,
  resolveTrackingEligibility,
} from "./trackingEligibility";

describe("resolveTrackingEligibility", () => {
  const base = {
    driverAvailable: true,
    presenceWindowOpen: false,
    appForeground: true,
    presenceDisclosureAccepted: true,
    hasActiveMission: false,
  };

  it("1. available + FG + 03:40 hors fenêtre => eligible PRESENCE_FG", () => {
    const r = resolveTrackingEligibility(base);
    expect(r.trackingEligible).toBe(true);
    expect(r.mode).toBe("PRESENCE_FG");
    expect(r.foregroundPresenceEligible).toBe(true);
    expect(r.backgroundPresenceEligible).toBe(false);
  });

  it("2. FG → BG à 03:40 hors fenêtre => ineligible", () => {
    const r = resolveTrackingEligibility({
      ...base,
      appForeground: false,
      presenceWindowOpen: false,
    });
    expect(r.trackingEligible).toBe(false);
    expect(r.mode).toBe("OFF");
  });

  it("3. BG 03:40 → FG => redémarre PRESENCE_FG", () => {
    const r = resolveTrackingEligibility({
      ...base,
      appForeground: true,
      presenceWindowOpen: false,
    });
    expect(r.trackingEligible).toBe(true);
    expect(r.mode).toBe("PRESENCE_FG");
  });

  it("4. available + 10:00 + FG → BG => PRESENCE_BG", () => {
    const fg = resolveTrackingEligibility({
      ...base,
      presenceWindowOpen: true,
      appForeground: true,
    });
    expect(fg.mode).toBe("PRESENCE_FG");
    const bg = resolveTrackingEligibility({
      ...base,
      presenceWindowOpen: true,
      appForeground: false,
    });
    expect(bg.trackingEligible).toBe(true);
    expect(bg.mode).toBe("PRESENCE_BG");
  });

  it("5. mission + 03:40 + BG => MISSION (fenêtre sans effet)", () => {
    const r = resolveTrackingEligibility({
      ...base,
      hasActiveMission: true,
      appForeground: false,
      presenceWindowOpen: false,
    });
    expect(r.trackingEligible).toBe(true);
    expect(r.mode).toBe("MISSION");
  });

  it("disclosure refusée => pas de présence même disponible FG", () => {
    const r = resolveTrackingEligibility({
      ...base,
      presenceDisclosureAccepted: false,
    });
    expect(r.trackingEligible).toBe(false);
  });
});

describe("resolvePresenceGpsAccuracy", () => {
  it("mission => high", () => {
    expect(
      resolvePresenceGpsAccuracy({ hasActiveMission: true, appForeground: false })
    ).toBe("high");
  });

  it("présence FG => high", () => {
    expect(
      resolvePresenceGpsAccuracy({ hasActiveMission: false, appForeground: true })
    ).toBe("high");
  });

  it("présence BG => balanced", () => {
    expect(
      resolvePresenceGpsAccuracy({ hasActiveMission: false, appForeground: false })
    ).toBe("balanced");
  });
});
