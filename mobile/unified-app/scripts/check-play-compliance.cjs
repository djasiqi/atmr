#!/usr/bin/env node
/**
 * Échoue si requestPermissionsAsync apparaît ailleurs que
 * requestNotificationOsPermissions.ts (prod).
 * Parcourt le filesystem (inclut fichiers non trackés par git).
 */
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "..");
const SEARCH_ROOTS = ["src", "app"];
const CODE_EXT = new Set([".ts", ".tsx"]);

function walk(dir, out) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "node_modules" || entry.name === "dist") continue;
      walk(full, out);
      continue;
    }
    if (!CODE_EXT.has(path.extname(entry.name))) continue;
    out.push(full);
  }
}

function grepProd() {
  const files = [];
  for (const rel of SEARCH_ROOTS) {
    walk(path.join(root, rel), files);
  }
  const matches = [];
  for (const file of files) {
    const rel = path.relative(root, file).replace(/\\/g, "/");
    if (rel.includes(".test.") || /\/tests?\//.test(rel)) continue;
    const text = fs.readFileSync(file, "utf8");
    const lines = text.split(/\r?\n/);
    lines.forEach((line, idx) => {
      if (line.includes("requestPermissionsAsync(")) {
        matches.push(`${rel}:${idx + 1}:${line.trim()}`);
      }
    });
  }
  return matches;
}

const matches = grepProd();
const count = matches.length;

if (count !== 1) {
  console.error(`FAIL: attendu 1 occurrence prod, trouvé ${count}`);
  matches.forEach((m) => console.error(m));
  process.exit(1);
}

if (!matches[0].includes("requestNotificationOsPermissions.ts")) {
  console.error(
    "FAIL: requestPermissionsAsync doit être dans requestNotificationOsPermissions.ts"
  );
  console.error(matches[0]);
  process.exit(1);
}

console.log(
  "OK: requestPermissionsAsync — 1 site prod (requestNotificationOsPermissions.ts)"
);
