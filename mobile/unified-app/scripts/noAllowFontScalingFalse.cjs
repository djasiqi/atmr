/**
 * Interdit `allowFontScaling={false}` hors allowlist cartes/marqueurs.
 * Analyse AST TypeScript (ignore commentaires et chaînes documentaires).
 */
const fs = require("fs");
const path = require("path");
const ts = require("typescript");

const ROOT = path.resolve(__dirname, "..");
const SCAN_ROOTS = [path.join(ROOT, "app"), path.join(ROOT, "src")];
const FILE_EXT = new Set([".ts", ".tsx", ".js", ".jsx"]);

/** Chemins relatifs (posix) autorisés — marqueurs carte uniquement. */
const ALLOWLIST = new Set([
  "src/features/company/components/maps/ClusterCountBadgeMarker.tsx",
]);

function toPosix(p) {
  return p.split(path.sep).join("/");
}

function listFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "node_modules" || entry.name === "dist") continue;
      files.push(...listFiles(full));
      continue;
    }
    if (FILE_EXT.has(path.extname(entry.name))) files.push(full);
  }
  return files;
}

function isFalseLiteral(node) {
  return (
    node.kind === ts.SyntaxKind.FalseKeyword ||
    (ts.isLiteralExpression(node) && node.text === "false")
  );
}

function findViolations(filePath, sourceText) {
  const kind = filePath.endsWith(".tsx") || filePath.endsWith(".jsx")
    ? ts.ScriptKind.TSX
    : ts.ScriptKind.TS;
  const sf = ts.createSourceFile(filePath, sourceText, ts.ScriptTarget.Latest, true, kind);
  const hits = [];

  function visit(node) {
    if (ts.isJsxAttribute(node) && node.name && node.name.getText(sf) === "allowFontScaling") {
      const init = node.initializer;
      if (!init) {
        // allowFontScaling seul ≈ true en JSX booléen shorthand — OK
      } else if (ts.isJsxExpression(init) && init.expression && isFalseLiteral(init.expression)) {
        const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
        hits.push(line + 1);
      } else if (ts.isStringLiteral(init) && init.text === "false") {
        const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
        hits.push(line + 1);
      }
    }
    if (
      ts.isPropertyAssignment(node) &&
      ((ts.isIdentifier(node.name) && node.name.text === "allowFontScaling") ||
        (ts.isStringLiteral(node.name) && node.name.text === "allowFontScaling"))
    ) {
      if (isFalseLiteral(node.initializer)) {
        const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
        hits.push(line + 1);
      }
    }
    ts.forEachChild(node, visit);
  }

  visit(sf);
  return hits;
}

const violations = [];
for (const root of SCAN_ROOTS) {
  for (const file of listFiles(root)) {
    const rel = toPosix(path.relative(ROOT, file));
    if (ALLOWLIST.has(rel)) continue;
    const content = fs.readFileSync(file, "utf8");
    const lines = findViolations(file, content);
    for (const line of lines) {
      violations.push({ file: rel, line });
    }
  }
}

if (violations.length > 0) {
  console.error("no-allow-font-scaling-false: usages interdits (hors allowlist marqueurs) :");
  for (const v of violations) {
    console.error(` - ${v.file}:${v.line}`);
  }
  process.exit(1);
}

console.log("no-allow-font-scaling-false: OK");
