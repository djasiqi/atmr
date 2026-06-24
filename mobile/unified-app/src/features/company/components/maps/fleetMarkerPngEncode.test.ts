import { describe, expect, it } from "@jest/globals";

import {
  buildClusterMarkerPngUri,
  buildDriverMarkerPngUri,
  buildEtaBadgePngUri,
  buildMissionAnchorPngUri,
  clearFleetMarkerPngCache,
} from "./fleetMarkerPngEncode";

describe("fleetMarkerPngEncode", () => {
  it("produit des PNG base64 (pas SVG) pour Android", () => {
    clearFleetMarkerPngCache();
    const uri = buildDriverMarkerPngUri({
      fill: "#4ade80",
      opacity: 1,
      label: "JD",
      sizePx: 42,
    });
    expect(uri.startsWith("data:image/png;base64,")).toBe(true);
    expect(uri.length).toBeGreaterThan(120);
  });

  it("met en cache les encodages identiques", () => {
    clearFleetMarkerPngCache();
    const a = buildMissionAnchorPngUri({
      fill: "#EF4444",
      stroke: "#ffffff",
      radiusPx: 7,
      selected: false,
      halo: true,
    });
    const b = buildMissionAnchorPngUri({
      fill: "#EF4444",
      stroke: "#ffffff",
      radiusPx: 7,
      selected: false,
      halo: true,
    });
    expect(a).toBe(b);
  });

  it("encode cluster et ETA", () => {
    clearFleetMarkerPngCache();
    const cluster = buildClusterMarkerPngUri(12, 46, "#ef4444");
    expect(cluster.startsWith("data:image/png;base64,")).toBe(true);
    const eta = buildEtaBadgePngUri("+8 min");
    expect(eta.uri.startsWith("data:image/png;base64,")).toBe(true);
    expect(eta.width).toBeGreaterThanOrEqual(58);
  });
});
