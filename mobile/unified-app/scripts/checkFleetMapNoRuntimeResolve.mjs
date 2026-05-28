/**
 * Bloque resolveMetroAssetSource dans les composants carte (résolution uniquement dans les builders).
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(fileURLToPath(new URL(".", import.meta.url)), "..");
const mapsDir = join(root, "src/features/company/components/maps");

const ALLOWLIST = new Set([
  "resolveMetroAssetSource.ts",
  "fleetLirieDriverMarkerAssets.ts",
  "fleetNativeMarkerImage.ts",
]);

const PATTERN = /resolveMetroAssetSource/;

function walk(dir, files = []) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      walk(full, files);
      continue;
    }
    if (/\.(tsx?|jsx?)$/.test(name) && !name.endsWith(".test.ts") && !name.endsWith(".test.tsx")) {
      files.push(full);
    }
  }
  return files;
}

let failed = false;
for (const file of walk(mapsDir)) {
  const base = file.split(/[/\\]/).pop() ?? "";
  if (ALLOWLIST.has(base)) continue;
  const content = readFileSync(file, "utf8");
  if (PATTERN.test(content)) {
    console.error(
      `❌ ${relative(root, file)}: resolveMetroAssetSource interdit hors builders (allowlist: ${[...ALLOWLIST].join(", ")})`
    );
    failed = true;
  }
}

if (failed) {
  process.exit(1);
}
console.log("✅ check:fleet-map-no-runtime-resolve OK");
