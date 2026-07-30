import {
  batteryActionLabel,
  computeTrackingReady,
  locationActionLabel,
  resolveBatteryReadinessStatus,
  resolveLocationReadinessAction,
  shouldApplyRefreshSequence,
  shouldShowOemGuidance,
} from "./trackingReadinessModel";

describe("resolveBatteryReadinessStatus", () => {
  it("iOS => not_applicable", () => {
    expect(
      resolveBatteryReadinessStatus({
        platformOs: "ios",
        checked: false,
        isIgnoring: null,
      })
    ).toBe("not_applicable");
  });

  it("Android checked + ignoring => exempt", () => {
    expect(
      resolveBatteryReadinessStatus({
        platformOs: "android",
        checked: true,
        isIgnoring: true,
      })
    ).toBe("exempt");
  });

  it("Android checked + not ignoring => restricted", () => {
    expect(
      resolveBatteryReadinessStatus({
        platformOs: "android",
        checked: true,
        isIgnoring: false,
      })
    ).toBe("restricted");
  });

  it("Android unchecked => unknown (pas exempt)", () => {
    expect(
      resolveBatteryReadinessStatus({
        platformOs: "android",
        checked: false,
        isIgnoring: null,
      })
    ).toBe("unknown");
  });
});

describe("computeTrackingReady", () => {
  const base = {
    fgPermissionGranted: true,
    bgPermissionGranted: true,
    locationAccuracy: "precise" as const,
    gpsEnabled: true,
    notificationsGranted: true,
    batteryStatus: "exempt" as const,
  };

  it("tout OK => ready", () => {
    expect(computeTrackingReady(base)).toBe(true);
  });

  it("approximate => non prêt", () => {
    expect(computeTrackingReady({ ...base, locationAccuracy: "approximate" })).toBe(false);
  });

  it("accuracy unknown => non prêt (fail-closed)", () => {
    expect(computeTrackingReady({ ...base, locationAccuracy: "unknown" })).toBe(false);
  });

  it("batterie unknown => ready (historique)", () => {
    expect(computeTrackingReady({ ...base, batteryStatus: "unknown" })).toBe(true);
  });

  it("batterie restricted => non prêt", () => {
    expect(computeTrackingReady({ ...base, batteryStatus: "restricted" })).toBe(false);
  });

  it("GPS off => non prêt", () => {
    expect(computeTrackingReady({ ...base, gpsEnabled: false })).toBe(false);
  });
});

describe("resolveLocationReadinessAction", () => {
  it("FG absent => foreground", () => {
    expect(
      resolveLocationReadinessAction({
        fgPermissionGranted: false,
        bgPermissionGranted: false,
        locationAccuracy: "unknown",
      })
    ).toBe("foreground");
  });

  it("FG OK, approximate => enable_precise", () => {
    expect(
      resolveLocationReadinessAction({
        fgPermissionGranted: true,
        bgPermissionGranted: false,
        locationAccuracy: "approximate",
      })
    ).toBe("enable_precise");
  });

  it("FG OK, accuracy unknown => verify_accuracy", () => {
    expect(
      resolveLocationReadinessAction({
        fgPermissionGranted: true,
        bgPermissionGranted: true,
        locationAccuracy: "unknown",
      })
    ).toBe("verify_accuracy");
  });

  it("FG+precise, BG absent => background uniquement", () => {
    expect(
      resolveLocationReadinessAction({
        fgPermissionGranted: true,
        bgPermissionGranted: false,
        locationAccuracy: "precise",
      })
    ).toBe("background");
  });

  it("tout OK => null (aucun bouton localisation)", () => {
    expect(
      resolveLocationReadinessAction({
        fgPermissionGranted: true,
        bgPermissionGranted: true,
        locationAccuracy: "precise",
      })
    ).toBeNull();
  });
});

describe("shouldShowOemGuidance", () => {
  it("masqué si batterie exempt", () => {
    expect(
      shouldShowOemGuidance({
        platformOs: "android",
        hasOemSettings: true,
        oemGuidanceAcknowledged: false,
        batteryStatus: "exempt",
      })
    ).toBe(false);
  });

  it("affiché si restricted et non acquitté", () => {
    expect(
      shouldShowOemGuidance({
        platformOs: "android",
        hasOemSettings: true,
        oemGuidanceAcknowledged: false,
        batteryStatus: "restricted",
      })
    ).toBe(true);
  });

  it("masqué après acquittement", () => {
    expect(
      shouldShowOemGuidance({
        platformOs: "android",
        hasOemSettings: true,
        oemGuidanceAcknowledged: true,
        batteryStatus: "restricted",
      })
    ).toBe(false);
  });
});

describe("labels + refresh sequence", () => {
  it("labels localisation / batterie", () => {
    expect(locationActionLabel("background")).toBe("Autoriser toujours");
    expect(locationActionLabel("verify_accuracy")).toBe("Vérifier la précision");
    expect(batteryActionLabel("restricted")).toBe("Batterie");
    expect(batteryActionLabel("unknown")).toBe("Vérifier la batterie");
    expect(batteryActionLabel("exempt")).toBeNull();
  });

  it("refresh stale : A après B n’écrase pas B", () => {
    let latest = 0;
    const applyA = ++latest; // 1
    const applyB = ++latest; // 2
    // B termine en premier
    expect(shouldApplyRefreshSequence(applyB, latest)).toBe(true);
    // A termine ensuite — obsolète
    expect(shouldApplyRefreshSequence(applyA, latest)).toBe(false);
  });
});
