/**
 * One-off migration helper: replace `fontSize: N` literals with `FONT_SIZE.pxN` tokens.
 *
 * Run from `mobile/unified-app`:
 *   node ./scripts/migrateLayoutFontSizeTokens.cjs
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const TOKEN_MODULE = path.join(ROOT, "src", "design", "responsive", "typographyTokens");
const SCAN_ROOTS = [path.join(ROOT, "app"), path.join(ROOT, "src")];
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

function tokenName(value) {
  return `px${String(value).replace("-", "neg").replace(".", "_")}`;
}

function importPathFor(file) {
  let rel = path.relative(path.dirname(file), TOKEN_MODULE).replace(/\\/g, "/");
  if (!rel.startsWith(".")) rel = `./${rel}`;
  return rel;
}

function ensureImport(content, file) {
  if (/FONT_SIZE/.test(content)) return content;
  const importLine = `import { FONT_SIZE } from "${importPathFor(file)}";\n`;
  if (/^import[\s\S]*?;\r?\n/.test(content)) {
    return content.replace(/((?:import[\s\S]*?;\r?\n)+)/, `$1${importLine}`);
  }
  return `${importLine}${content}`;
}

let changed = 0;
for (const file of SCAN_ROOTS.flatMap(listFiles)) {
  const before = fs.readFileSync(file, "utf8");
  if (!/\bfontSize\s*:\s*-?\d/.test(stripComments(before))) continue;

  let next = ensureImport(before, file);
  next = next.replace(/\bfontSize\s*:\s*(-?\d+(?:\.\d+)?)/g, (_m, value) => {
    return `fontSize: FONT_SIZE.${tokenName(value)}`;
  });

  if (next !== before) {
    fs.writeFileSync(file, next);
    changed += 1;
  }
}

console.log(`migrate:layout-font-size-tokens: changed ${changed} files`);
