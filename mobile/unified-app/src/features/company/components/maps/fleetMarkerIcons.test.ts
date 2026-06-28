import {
  buildFleetCircleMarkerSvgMarkup,
  buildFleetClusterMarkerSvgMarkup,
  makeFleetCircleMarkerDataUrl,
  resolveDriverMarkerInitials,
} from "./fleetMarkerIcons";
import { FLEET_WEB_STATUS_COLORS } from "./fleetMapStatusContract";

describe("fleetMarkerIcons", () => {
  it("génère un SVG 58×58 avec initiales et anneau blanc", () => {
    const svg = buildFleetCircleMarkerSvgMarkup("#4ade80", 58, 1, { label: "AB" });
    expect(svg).toContain('width="58"');
    expect(svg).toContain('stroke="#ffffff"');
    expect(svg).toContain(">AB<");
    expect(svg).not.toContain("M18.92 6.01");
  });

  it("produit une data URL encodée pour cercle", () => {
    const url = makeFleetCircleMarkerDataUrl("#00796B", 58, 1, { label: "CD" });
    expect(url.startsWith("data:image/svg+xml;charset=UTF-8,")).toBe(true);
    expect(decodeURIComponent(url.split(",")[1] ?? "")).toContain("#00796B");
  });

  it("initiales limitées à 2 caractères (prénom + nom)", () => {
    expect(resolveDriverMarkerInitials("Jean Dupont")).toBe("JD");
  });

  it("cluster affiche le compteur avec couleur dominante", () => {
    const svg = buildFleetClusterMarkerSvgMarkup(4, FLEET_WEB_STATUS_COLORS.emergency, 40);
    expect(svg).toContain(">4<");
    expect(svg).toContain('viewBox="0 0 40 40"');
    expect(svg).toContain(FLEET_WEB_STATUS_COLORS.emergency);
    expect(svg).toContain('font-weight="600"');
  });
});
