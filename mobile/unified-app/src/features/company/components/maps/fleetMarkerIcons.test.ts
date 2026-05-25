import {
  buildFleetClusterMarkerSvgMarkup,
  buildFleetVehicleMarkerSvgMarkup,
  makeFleetEtaBadgeMarkerDataUrl,
  makeFleetVehicleMarkerDataUrl,
  resolveFleetMarkerSizePx,
} from "./fleetMarkerIcons";

describe("fleetMarkerIcons", () => {
  it("génère un SVG 56×56 avec voiture centrée et ombre intégrée", () => {
    const svg = buildFleetVehicleMarkerSvgMarkup("#00796B", { sizePx: 56 });
    expect(svg).toContain('viewBox="0 0 56 56"');
    expect(svg).toContain('fill="#00796B"');
    expect(svg).toContain("M18.92 6.01");
    expect(svg).toContain("feDropShadow");
    expect(svg).not.toContain('stroke="#ffffff"');
    expect(svg).not.toContain("rgba(255,255,255,0.98)");
  });

  it("garde la même taille lorsque le marqueur est sélectionné", () => {
    expect(resolveFleetMarkerSizePx({ sizePx: 56, selected: true })).toBe(56);
  });

  it("agrandit le disque interne quand sélectionné (sans anneau externe)", () => {
    const base = buildFleetVehicleMarkerSvgMarkup("#00796B", { selected: false });
    const selected = buildFleetVehicleMarkerSvgMarkup("#00796B", { selected: true });
    expect(selected).not.toContain('stroke="#0B2B2A"');
    expect(selected.length).toBeGreaterThan(base.length);
  });

  it("produit une data URL encodée", () => {
    const url = makeFleetVehicleMarkerDataUrl("#3498DB");
    expect(url.startsWith("data:image/svg+xml;charset=UTF-8,")).toBe(true);
    expect(decodeURIComponent(url.split(",")[1] ?? "")).toContain("#3498DB");
  });

  it("encode une pastille ETA en data-uri", () => {
    const badge = makeFleetEtaBadgeMarkerDataUrl("~8 min");
    expect(badge.uri).toMatch(/^data:image\/svg\+xml/);
    expect(badge.width).toBeGreaterThanOrEqual(58);
  });

  it("cluster affiche le compteur discret (2, 3, 4…)", () => {
    const svg = buildFleetClusterMarkerSvgMarkup(4, 40);
    expect(svg).toContain(">4<");
    expect(svg).toContain('viewBox="0 0 40 40"');
    expect(svg).toContain("#00796B");
    expect(svg).toContain('font-weight="600"');
  });
});
