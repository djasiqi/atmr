import { Asset } from "expo-asset";

import {
  clearMetroAssetResolveCacheForTests,
  resolveMetroAssetSource,
} from "./resolveMetroAssetSource";
import {
  DEFAULT_LIRIE_DRIVER_MARKER_MODULE,
  resolveLirieDriverMarkerModule,
} from "./fleetLirieDriverMarkerModules";

jest.mock("expo-asset", () => ({
  Asset: {
    fromModule: jest.fn(),
  },
}));

describe("resolveMetroAssetSource", () => {
  beforeEach(() => {
    clearMetroAssetResolveCacheForTests();
    jest.clearAllMocks();
  });

  it("resout un PNG embarque via expo-asset Asset.fromModule", () => {
    (Asset.fromModule as jest.Mock).mockReturnValue({
      uri: "file:///marker.png",
      localUri: "file:///marker.png",
      width: 18,
      height: 28,
    });
    const moduleId = resolveLirieDriverMarkerModule("available");
    const resolved = resolveMetroAssetSource(moduleId);
    expect(resolved).not.toBeNull();
    expect(resolved!.uri.length).toBeGreaterThan(0);
  });

  it("retourne null sans throw si uri absente", () => {
    (Asset.fromModule as jest.Mock).mockReturnValue({
      uri: "",
      localUri: undefined,
      width: 0,
      height: 0,
    });
    const resolved = resolveMetroAssetSource(999_001);
    expect(resolved).toBeNull();
  });

  it("utilise DEFAULT_LIRIE_DRIVER_MARKER_MODULE pour available", () => {
    expect(resolveLirieDriverMarkerModule("available")).toBe(
      DEFAULT_LIRIE_DRIVER_MARKER_MODULE
    );
  });
});
