/**
 * Régénère les PNG mdpi de base depuis des SVG (optionnel).
 * Les variantes Android (hdpi, xhdpi…) et iOS (@2x, @3x) sont fournies dans assets/images/markers.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Resvg } from "@resvg/resvg-js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const markersDir = join(__dirname, "../assets/images/markers");

const SVG_FILES = [
  "driver_lirie_available.svg",
  "driver_lirie_assigned.svg",
  "driver_lirie_warning.svg",
  "driver_lirie_critical.svg",
  "driver_lirie_offline.svg",
];

/** Aligné sur LIRIE_DRIVER_MARKER_PNG_WIDTH_PX (fleetLirieMarkerSizing.ts). */
const OUT_WIDTH_PX = 48;

for (const svgFile of SVG_FILES) {
  const svgPath = join(markersDir, svgFile);
  if (!existsSync(svgPath)) {
    console.log(`Skip ${svgFile} (fichier absent — utilisez les PNG du dossier markers).`);
    continue;
  }
  const pngFile = svgFile.replace(/\.svg$/i, ".png");
  const pngPath = join(markersDir, pngFile);

  const svgContent = readFileSync(svgPath, "utf8");
  const resvg = new Resvg(svgContent, {
    fitTo: { mode: "width", value: OUT_WIDTH_PX },
    background: "transparent",
  });
  const rendered = resvg.render();
  const png = rendered.asPng();
  writeFileSync(pngPath, png);
  console.log(`OK ${pngFile} (${rendered.width}×${rendered.height}, ${png.length} bytes)`);
}
