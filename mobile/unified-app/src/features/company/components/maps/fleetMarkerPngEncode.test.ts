import { describe, expect, it } from "@jest/globals";

import {
  buildClusterMarkerPngUri,
  buildDriverMarkerPngUri,
  buildEtaBadgePngUri,
  buildMissionAnchorPngUri,
  clearFleetMarkerPngCache,
  resolveDriverMarkerLabelBitmapLayout,
} from "./fleetMarkerPngEncode";

describe("fleetMarkerPngEncode", () => {
  it("produit des PNG base64 (pas SVG) pour Android", () => {
    clearFleetMarkerPngCache();
    const uri = buildDriverMarkerPngUri({
      fill: "#4ade80",
      opacity: 1,
      label: "JD",
      sizePx: 58,
    });
    expect(uri.startsWith("data:image/png;base64,")).toBe(true);
    expect(uri.length).toBeGreaterThan(120);
  });

  it("cale les initiales dans le disque (2 lettres)", () => {
    const layout = resolveDriverMarkerLabelBitmapLayout(58, "KA");
    expect(layout.trimmed).toBe("KA");
    expect(layout.textScale).toBeLessThanOrEqual(3);
    expect(layout.glyphAdvance).toBe(5);
    const textW = layout.trimmed.length * layout.glyphAdvance * layout.textScale;
    expect(textW).toBeLessThanOrEqual(58 * 0.72);
  });

  it("encode les initiales alphabétiques (Android)", () => {
    clearFleetMarkerPngCache();
    const withLabel = buildDriverMarkerPngUri({
      fill: "#4ade80",
      opacity: 1,
      label: "KA",
      sizePx: 58,
    });
    const withoutLabel = buildDriverMarkerPngUri({
      fill: "#4ade80",
      opacity: 1,
      label: "",
      sizePx: 58,
    });
    expect(withLabel).not.toBe(withoutLabel);
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
