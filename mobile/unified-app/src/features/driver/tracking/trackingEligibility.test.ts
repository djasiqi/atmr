import {
  resolveTrackingEligibility,
  resolvePresenceGpsAccuracy,
} from "./trackingEligibility";

describe("resolveTrackingEligibility", () => {
  const base = {
    driverAvailable: true,
    presenceWindowOpen: false,
    appForeground: true,
    presenceDisclosureAccepted: true,
    hasActiveMission: false,
  };

  it("1. available + FG hors ancienne fenêtre 07–19 => PRESENCE_FG (fenêtre ignorée)", () => {
    const r = resolveTrackingEligibility(base);
    expect(r.mode).toBe("PRESENCE_FG");
    expect(r.trackingEligible).toBe(true);
    expect(r.blocked).toBe(false);
  });

  it("2. unavailable => OFF", () => {
    const r = resolveTrackingEligibility({
      ...base,
      driverAvailable: false,
    });
    expect(r.mode).toBe("OFF");
    expect(r.trackingEligible).toBe(false);
  });

  it("3. available + disclosure + FG => PRESENCE_FG", () => {
    const r = resolveTrackingEligibility({
      ...base,
      presenceWindowOpen: true,
    });
    expect(r.mode).toBe("PRESENCE_FG");
  });

  it("4. available + disclosure + FG → BG => PRESENCE_BG", () => {
    const bg = resolveTrackingEligibility({
      ...base,
      appForeground: false,
      presenceWindowOpen: true,
    });
    expect(bg.mode).toBe("PRESENCE_BG");
  });

  it("5. mission + hors fenêtre + BG => MISSION (fenêtre sans effet)", () => {
    const r = resolveTrackingEligibility({
      ...base,
      hasActiveMission: true,
      appForeground: false,
      presenceWindowOpen: false,
    });
    expect(r.mode).toBe("MISSION");
  });

  it("6. disclosure refusée => BLOCKED (pas OFF) si disponible", () => {
    const r = resolveTrackingEligibility({
      ...base,
      presenceDisclosureAccepted: false,
    });
    expect(r.mode).toBe("BLOCKED");
    expect(r.blocked).toBe(true);
    expect(r.trackingEligible).toBe(false);
  });

  it("7. permissionsReady=false explicite => BLOCKED", () => {
    const r = resolveTrackingEligibility({
      ...base,
      presenceDisclosureAccepted: true,
      permissionsReady: false,
    });
    expect(r.mode).toBe("BLOCKED");
  });
});

describe("resolvePresenceGpsAccuracy", () => {
  it("mission ou FG => high", () => {
    expect(
      resolvePresenceGpsAccuracy({ hasActiveMission: true, appForeground: false })
    ).toBe("high");
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
