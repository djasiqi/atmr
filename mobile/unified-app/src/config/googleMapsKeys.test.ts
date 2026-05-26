import {
  diagnoseGoogleMapsWebKeyIssue,
  formatGoogleMapsWebKeyHelpMessage,
  resolveFleetMapsLibraryList,
} from "./googleMapsKeys";

describe("diagnoseGoogleMapsWebKeyIssue", () => {
  const prevWeb = process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY;
  const prevAndroid = process.env.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY;

  afterEach(() => {
    if (prevWeb === undefined) delete process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY;
    else process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY = prevWeb;
    if (prevAndroid === undefined) delete process.env.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY;
    else process.env.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY = prevAndroid;
  });

  it("signale une config Android seule sans clé web", () => {
    delete process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY;
    process.env.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY = "mock_android_maps_key_for_unit_tests_only";
    expect(diagnoseGoogleMapsWebKeyIssue()).toBe("android_only_configured");
    expect(formatGoogleMapsWebKeyHelpMessage("android_only_configured")).toMatch(/EXPO_PUBLIC_GOOGLE_MAPS_API_KEY/);
  });
});

describe("resolveFleetMapsLibraryList", () => {
  const prevFleet = process.env.EXPO_PUBLIC_GOOGLE_MAPS_FLEET_LIBRARIES;
  const prevAll = process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES;

  afterEach(() => {
    if (prevFleet === undefined) delete process.env.EXPO_PUBLIC_GOOGLE_MAPS_FLEET_LIBRARIES;
    else process.env.EXPO_PUBLIC_GOOGLE_MAPS_FLEET_LIBRARIES = prevFleet;
    if (prevAll === undefined) delete process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES;
    else process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES = prevAll;
  });

  it("retire places de la liste globale", () => {
    process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES = "maps,marker,places";
    expect(resolveFleetMapsLibraryList()).toEqual(["maps", "marker"]);
  });

  it("utilise la liste dédiée flotte si définie", () => {
    process.env.EXPO_PUBLIC_GOOGLE_MAPS_FLEET_LIBRARIES = "marker";
    process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES = "maps,marker,places";
    expect(resolveFleetMapsLibraryList()).toEqual(["marker"]);
  });
});
