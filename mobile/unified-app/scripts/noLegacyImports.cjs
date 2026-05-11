const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const SCAN_ROOTS = [path.join(ROOT, "app"), path.join(ROOT, "src")];
/** Dépendances interdites : uniquement des chemins de modules (import / require / import()), pas les mentions dans les commentaires. */
const FORBIDDEN_PATTERNS = [
  /\bfrom\s+["'][^"']*operations-app[^"']*["']/i,
  /\brequire\s*\(\s*["'][^"']*operations-app[^"']*["']\s*\)/i,
  /\bimport\s*\(\s*["'][^"']*operations-app[^"']*["']\s*\)/i,
  /\bexport\s+[^;{]*\bfrom\s+["'][^"']*operations-app[^"']*["']/i,
];
const FILE_EXT = new Set([".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]);

function listFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...listFiles(full));
      continue;
    }
    if (FILE_EXT.has(path.extname(entry.name))) {
      files.push(full);
    }
  }
  return files;
}

const violations = [];
for (const root of SCAN_ROOTS) {
  for (const file of listFiles(root)) {
    const content = fs.readFileSync(file, "utf8");
    for (const pattern of FORBIDDEN_PATTERNS) {
      if (pattern.test(content)) {
        violations.push({
          file: path.relative(ROOT, file),
          pattern: pattern.toString(),
        });
        break;
      }
    }
  }
}

if (violations.length > 0) {
  console.error("no-legacy-imports: forbidden operations-app module references found:");
  for (const violation of violations) {
    console.error(` - ${violation.file} (${violation.pattern})`);
  }
  process.exit(1);
}

console.log("no-legacy-imports: OK");
