/**
 * Audit layout magic numbers — unified-app
 * Usage: node ./scripts/auditMagicLayoutNumbers.cjs
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const SCAN_ROOTS = [path.join(ROOT, "app"), path.join(ROOT, "src")];
const OUT_MD = path.join(ROOT, "docs", "layout", "magic-numbers-report.md");
const OUT_JSON = path.join(ROOT, "docs", "layout", "magic-numbers-report.json");
const FILE_EXT = new Set([".ts", ".tsx"]);
const SKIP_DIR = new Set(["node_modules", "dist-web-test", ".expo-tmp-bundle-test"]);

const STYLE_PROPS = [
  "height",
  "maxHeight",
  "minHeight",
  "bottom",
  "top",
  "left",
  "right",
  "paddingBottom",
  "marginBottom",
  "paddingTop",
  "marginTop",
  "translateX",
  "translateY",
];

const ACCEPTABLE_NAME_RE =
  /(_ICON_|_CLIP_|borderWidth|minTouch|LIRIE_GOOGLE|COMPACT_MISSION|FLOATING_TAB_PILL|CLIENT_FLOATING_BAR_BASE)/i;

const WINDOW_DIMENSION_WHITELIST_RE =
  /useAppViewport|useKeyboardLayout|scrollAnchorAboveKeyboard|ChatConversationShell|LegacyModal/;

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
    const ext = path.extname(entry.name);
    if (!FILE_EXT.has(ext)) continue;
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

function parseNumeric(val) {
  if (typeof val !== "string") return null;
  const t = val.trim();
  if (/^\d+(\.\d+)?%$/.test(t)) return { num: parseFloat(t), isPercent: true };
  const m = t.match(/^(-?\d+(?:\.\d+)?)$/);
  if (m) return { num: parseFloat(m[1]), isPercent: false };
  return null;
}

function classify(prop, value, lineText, fileRel) {
  const abs = Math.abs(value);
  // Exception explicite (revue humaine, ex. debug widget, fixture).
  if (/MAGIC-NUMBER:allow/.test(lineText)) {
    return "Acceptable";
  }
  if (lineText.includes("Math.max(260") || /keyboardScrollPaddingMin|keyboardDidShow/.test(lineText)) {
    return "Critical";
  }
  // Skeletons / dimensions cosmétiques très petites — acceptables même dans modales
  if (
    /Skeleton|skeleton/.test(fileRel) ||
    /shimmer|placeholder|divider/i.test(lineText)
  ) {
    if (abs <= 24 && (prop === "height" || prop === "width" || prop === "maxHeight")) {
      return "Acceptable";
    }
  }
  // maxHeight très petits (icones/badges/lignes-skeleton) — acceptables
  if (prop === "maxHeight" && abs > 0 && abs <= 48) {
    return "Acceptable";
  }
  if (prop === "maxHeight" && abs > 48 && abs < 400 && !lineText.includes("usableHeight")) {
    return "Critical";
  }
  if (/%/.test(lineText) && (prop === "maxHeight" || prop === "height")) {
    return "Critical";
  }
  if (ACCEPTABLE_NAME_RE.test(lineText)) {
    return "Acceptable";
  }
  if (
    (prop === "height" || prop === "width") &&
    abs > 0 &&
    abs <= 48 &&
    !/padding|margin/.test(prop)
  ) {
    return "Acceptable";
  }
  if (prop === "borderWidth" || (abs <= 4 && /spacing|gap/i.test(lineText))) {
    return "Acceptable";
  }
  if (
    (prop.startsWith("padding") || prop.startsWith("margin")) &&
    abs >= 21 &&
    abs <= 80
  ) {
    return "À tokeniser";
  }
  if (
    abs >= 80 ||
    prop === "bottom" ||
    prop === "top" ||
    prop === "translateY" ||
    prop === "translateX"
  ) {
    if (abs >= 80 && (prop === "bottom" || prop === "top" || prop.startsWith("translate"))) {
      return "Dangereux";
    }
    if (["height", "maxHeight", "paddingBottom", "marginBottom"].includes(prop) && abs >= 80) {
      return "Dangereux";
    }
  }
  if (abs > 48 && abs < 80) {
    return "À tokeniser";
  }
  return abs <= 48 ? "Acceptable" : "Dangereux";
}

function scanFile(filePath) {
  const content = fs.readFileSync(filePath, "utf8");
  const lines = content.split("\n");
  const rel = path.relative(ROOT, filePath).replace(/\\/g, "/");
  const findings = [];

  const propAlt = STYLE_PROPS.join("|");
  const stylePropRe = new RegExp(`\\b(${propAlt})\\s*:\\s*([^,}\\n]+)`, "g");

  lines.forEach((line, idx) => {
    const lineNo = idx + 1;
    let m;
    stylePropRe.lastIndex = 0;
    while ((m = stylePropRe.exec(line)) !== null) {
      const prop = m[1];
      const rawVal = m[2].trim();
      const parsed = parseNumeric(rawVal);
      if (parsed == null) continue;
      if (parsed.isPercent) {
        findings.push({
          file: rel,
          line: lineNo,
          prop,
          value: rawVal,
          tier: classify(prop, parsed.num, line, rel),
        });
        continue;
      }
      findings.push({
        file: rel,
        line: lineNo,
        prop,
        value: parsed.num,
        tier: classify(prop, parsed.num, line, rel),
      });
    }

    const transformRe = /translate([XY])\s*:\s*(-?\d+(?:\.\d+)?)/g;
    let tm;
    while ((tm = transformRe.exec(line)) !== null) {
      const prop = `translate${tm[1]}`;
      const num = parseFloat(tm[2]);
      findings.push({
        file: rel,
        line: lineNo,
        prop,
        value: num,
        tier: classify(prop, num, line, rel),
      });
    }
  });

  return findings;
}

function countKpis(allFiles) {
  let windowDims = 0;
  let fontSize = 0;
  let keyboardDup = 0;
  const keyboardFiles = new Set();

  for (const file of allFiles) {
    const rel = path.relative(ROOT, file).replace(/\\/g, "/");
    const content = stripComments(fs.readFileSync(file, "utf8"));
    if (/useWindowDimensions|Dimensions\.get\s*\(\s*['"]window['"]/.test(content)) {
      if (!WINDOW_DIMENSION_WHITELIST_RE.test(rel)) {
        windowDims += 1;
      }
    }
    if (/fontSize\s*:\s*\d/.test(content)) {
      const matches = content.match(/fontSize\s*:\s*\d/g);
      fontSize += matches ? matches.length : 0;
    }
    if (
      rel.startsWith("app/(public)/") &&
      /Keyboard\.addListener\s*\(\s*["']keyboard(?:Did|Will)Show/.test(content)
    ) {
      keyboardDup += 1;
      keyboardFiles.add(rel);
    }
  }

  return { windowDims, fontSize, keyboardDup, keyboardFiles: [...keyboardFiles] };
}

const allFindings = [];
for (const root of SCAN_ROOTS) {
  for (const file of listFiles(root)) {
    allFindings.push(...scanFile(file));
  }
}

const byTier = { Acceptable: [], "À tokeniser": [], Dangereux: [], Critical: [] };
for (const f of allFindings) {
  const tier = byTier[f.tier] ? f.tier : "Dangereux";
  byTier[tier].push(f);
}

const allScanFiles = SCAN_ROOTS.flatMap((r) => listFiles(r));
const kpis = countKpis(allScanFiles);

const criticalCount = byTier.Critical.length;
const dangerousCount = byTier.Dangereux.length;

const md = [];
md.push("# Magic numbers report — unified-app");
md.push("");
md.push(`Généré : ${new Date().toISOString()}`);
md.push("");
md.push("## Layout KPI");
md.push("");
md.push("| KPI | Valeur actuelle |");
md.push("|-----|-----------------|");
md.push(`| Critical magic numbers (total) | ${criticalCount} |`);
md.push(`| Dangereux | ${dangerousCount} |`);
md.push(`| Acceptable | ${byTier.Acceptable.length} |`);
md.push(`| Fichiers useWindowDimensions/Dimensions.get hors whitelist | ${kpis.windowDims} |`);
md.push(`| fontSize littéraux (occurrences) | ${kpis.fontSize} |`);
md.push(`| Écrans publics keyboardDidShow dupliqués | ${kpis.keyboardDup} |`);
md.push("");
md.push("Regénérer : `npm run audit:layout-magic-numbers`");
md.push("");

for (const tier of ["Critical", "Dangereux", "À tokeniser"]) {
  const items = byTier[tier];
  if (items.length === 0) continue;
  md.push(`## ${tier} (${items.length})`);
  md.push("");
  const sorted = [...items].sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line);
  for (const item of sorted.slice(0, 200)) {
    md.push(
      `- \`${item.prop}: ${item.value}\` → \`${item.file}:${item.line}\` → **${item.tier}**`
    );
  }
  if (sorted.length > 200) {
    md.push(`- … et ${sorted.length - 200} autres (voir JSON)`);
  }
  md.push("");
}

fs.mkdirSync(path.dirname(OUT_MD), { recursive: true });
fs.writeFileSync(OUT_MD, md.join("\n"));
fs.writeFileSync(
  OUT_JSON,
  JSON.stringify({ generatedAt: new Date().toISOString(), kpis, byTier, total: allFindings.length }, null, 2)
);

console.log(`audit:layout-magic-numbers: wrote ${path.relative(ROOT, OUT_MD)}`);
console.log(`  Critical: ${criticalCount}, Dangereux: ${dangerousCount}, total: ${allFindings.length}`);
