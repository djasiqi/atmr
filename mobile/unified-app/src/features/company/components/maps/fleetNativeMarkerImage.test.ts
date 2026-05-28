import { Asset } from "expo-asset";

import {
  buildFleetDriverMarkerImageSource,
  FLEET_NATIVE_DRIVER_MARKER_SIZE_PX,
} from "./fleetNativeMarkerImage";
import { usesLirieDriverMarkerRasterPng } from "./fleetLirieDriverMarkerAssets";
import { clearMetroAssetResolveCacheForTests } from "./resolveMetroAssetSource";
import type { FleetOperationalStatus } from "./mapStatusTheme";

jest.mock("expo-asset", () => ({
  Asset: {
    fromModule: jest.fn(),
  },
}));

const ALL_STATUSES: FleetOperationalStatus[] = [
  "available",
  "on_mission",
  "break",
  "delayed",
  "incident",
  "offline",
];

describe("fleetNativeMarkerImage", () => {
  beforeEach(() => {
    clearMetroAssetResolveCacheForTests();
    jest.clearAllMocks();
  });

  it("retourne une source PNG Lirie sur mobile", () => {
    (Asset.fromModule as jest.Mock).mockReturnValue({
      uri: "file:///marker.png",
      localUri: "file:///marker.png",
      width: 18,
      height: 28,
    });
    const src = buildFleetDriverMarkerImageSource("available", false);
    expect(src.uri.length).toBeGreaterThan(0);
    expect(src.width).toBe(FLEET_NATIVE_DRIVER_MARKER_SIZE_PX);
    expect(src.height).toBeGreaterThan(0);
    if (usesLirieDriverMarkerRasterPng()) {
      expect(src.assetModule).toBeDefined();
    }
  });

  it("retourne toujours uri non vide pour tous les statuts si Asset vide", () => {
    (Asset.fromModule as jest.Mock).mockReturnValue({
      uri: "",
      localUri: undefined,
      width: 0,
      height: 0,
    });
    for (const status of ALL_STATUSES) {
      const src = buildFleetDriverMarkerImageSource(status, false);
      expect(src.uri.length).toBeGreaterThan(0);
      expect(src.width).toBeGreaterThan(0);
      expect(src.height).toBeGreaterThan(0);
    }
  });

  it("produit des URIs distinctes pour delayed et incident", () => {
    (Asset.fromModule as jest.Mock).mockReturnValue({
      uri: "file:///marker.png",
      localUri: "file:///marker.png",
      width: 18,
      height: 28,
    });
    const delayed = buildFleetDriverMarkerImageSource("delayed", false);
    const incident = buildFleetDriverMarkerImageSource("incident", false);
    expect(delayed.uri).toBeTruthy();
    expect(incident.uri).toBeTruthy();
  });
});
