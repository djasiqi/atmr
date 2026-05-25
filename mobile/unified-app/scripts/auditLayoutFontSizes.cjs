/**
 * Audit literal StyleSheet font sizes — unified-app.
 * Usage: node ./scripts/auditLayoutFontSizes.cjs
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const SCAN_ROOTS = [path.join(ROOT, "app"), path.join(ROOT, "src")];
const OUT_MD = path.join(ROOT, "docs", "layout", "font-sizes-report.md");
const OUT_JSON = path.join(ROOT, "docs", "layout", "font-sizes-report.json");
const FILE_EXT = new Set([".ts", ".tsx"]);
const SKIP_DIR = new Set(["node_modules", "dist-web-test", ".expo-tmp-bundle-test"]);

function listFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP_DIR.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...listFiles(full));
      continue;
    }
    if (!FILE_EXT.has(path.extname(entry.name))) continue;
    if (entry.name.includes(".test.")) continue;
    out.push(full);
  }
  return out;
}

function stripComments(content) {
  return content
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "");
}

function countFile(file) {
  const rel = path.relative(ROOT, file).replace(/\\/g, "/");
  const content = stripComments(fs.readFileSync(file, "utf8"));
  const matches = [...content.matchAll(/\bfontSize\s*:\s*(-?\d+(?:\.\d+)?)/g)];
  return {
    file: rel,
    count: matches.length,
    values: matches.map((m) => Number(m[1])),
  };
}

const rows = SCAN_ROOTS.flatMap(listFiles)
  .map(countFile)
  .filter((row) => row.count > 0)
  .sort((a, b) => b.count - a.count || a.file.localeCompare(b.file));

const total = rows.reduce((sum, row) => sum + row.count, 0);
const top = rows.slice(0, 20);

fs.mkdirSync(path.dirname(OUT_MD), { recursive: true });
fs.writeFileSync(OUT_JSON, JSON.stringify({ generatedAt: new Date().toISOString(), total, rows }, null, 2));

const md = [];
md.push("# Font sizes report — unified-app");
md.push("");
md.push(`Généré : ${new Date().toISOString()}`);
md.push("");
md.push("| KPI | Valeur actuelle |");
md.push("|-----|-----------------|");
md.push(`| fontSize littéraux (StyleSheet / inline style) | ${total} |`);
md.push(`| Fichiers concernés | ${rows.length} |`);
md.push("");
md.push("## Top fichiers");
md.push("");
md.push("| Fichier | Occurrences |");
md.push("|---------|-------------|");
for (const row of top) {
  md.push(`| \`${row.file}\` | ${row.count} |`);
}
md.push("");
md.push("Cible Sprint 3 : < 100 occurrences. Migrer par volume décroissant vers `useResponsiveTokens()` / composants texte (`AppText`).");
md.push("");

fs.writeFileSync(OUT_MD, md.join("\n"));
console.log(`audit:layout-font-sizes: wrote ${path.relative(ROOT, OUT_MD)}`);
console.log(`  fontSize literals: ${total}, files: ${rows.length}`);
