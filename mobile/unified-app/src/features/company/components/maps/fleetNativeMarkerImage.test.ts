import { Platform } from "react-native";

import {
  buildFleetClusterCountBadgeImageSource,
  buildFleetDriverMarkerImageSource,
  buildFleetEtaBadgeImageSource,
  buildMissionAnchorImageSource,
  FLEET_NATIVE_DRIVER_MARKER_SIZE_PX,
} from "./fleetNativeMarkerImage";
import { usesLirieDriverMarkerRasterPng } from "./fleetLirieDriverMarkerAssets";

describe("fleetNativeMarkerImage", () => {
  it("utilise le marqueur Lirie embarqué pour un chauffeur disponible", () => {
    const src = buildFleetDriverMarkerImageSource("available", false);
    expect(src.uri).not.toMatch(/^data:image\/svg\+xml/);
    expect(src.uri.length).toBeGreaterThan(0);
    expect(src.assetModule).toBeDefined();
    expect(src.width).toBe(FLEET_NATIVE_DRIVER_MARKER_SIZE_PX);
    expect(src.height).toBeGreaterThan(src.width);
    if (usesLirieDriverMarkerRasterPng()) {
      expect(Platform.OS === "ios" || Platform.OS === "android").toBe(true);
    }
  });

  it("réutilise le pin critique pour retard et incident", () => {
    const delayed = buildFleetDriverMarkerImageSource("delayed", false);
    const incident = buildFleetDriverMarkerImageSource("incident", false);
    expect(delayed.uri).toBe(incident.uri);
  });

  it("produit une pastille compteur pour cluster", () => {
    const badge = buildFleetClusterCountBadgeImageSource(3);
    expect(badge.uri).toMatch(/^data:image\/svg\+xml/);
    expect(decodeURIComponent(badge.uri)).toContain(">3<");
  });

  it("produit une pastille ETA raster", () => {
    const src = buildFleetEtaBadgeImageSource("+12 min");
    expect(src.uri).toMatch(/^data:image\/svg\+xml/);
    expect(src.width).toBeGreaterThanOrEqual(58);
    expect(src.height).toBe(26);
  });

  it("produit une pastille mission sans pin Google", () => {
    const src = buildMissionAnchorImageSource({
      role: "pickup",
      fill: "#00796B",
      stroke: "#ffffff",
      radius: 7,
      opacity: 1,
      zIndex: 50,
    });
    expect(src.uri).toMatch(/^data:image\/svg\+xml/);
    expect(src.width).toBeGreaterThanOrEqual(18);
    expect(src.width).toBeLessThanOrEqual(24);
  });
});
