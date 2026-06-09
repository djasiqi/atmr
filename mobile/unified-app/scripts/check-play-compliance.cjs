#!/usr/bin/env node
/**
 * Échoue si requestPermissionsAsync apparaît ailleurs que registerPushToken.ts (prod).
 */
const { execSync } = require("node:child_process");
const path = require("node:path");

const root = path.join(__dirname, "..");

function grepProd() {
  const cmd =
    process.platform === "win32"
      ? `git grep -n "requestPermissionsAsync(" -- "src" "app" "*.ts" "*.tsx"`
      : `grep -R "requestPermissionsAsync(" src app --include="*.ts" --include="*.tsx"`;
  try {
    const out = execSync(cmd, { cwd: root, encoding: "utf8" });
    return out
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .filter((line) => !line.includes(".test."))
      .filter((line) => !/\/test[s]?\//.test(line));
  } catch (err) {
    if (err.status === 1) return [];
    throw err;
  }
}

const matches = grepProd();
const count = matches.length;

if (count !== 1) {
  console.error(`FAIL: attendu 1 occurrence prod, trouvé ${count}`);
  matches.forEach((m) => console.error(m));
  process.exit(1);
}

if (!matches[0].includes("registerPushToken.ts")) {
  console.error("FAIL: requestPermissionsAsync doit être dans registerPushToken.ts");
  console.error(matches[0]);
  process.exit(1);
}

console.log("OK: requestPermissionsAsync — 1 site prod (registerPushToken.ts)");
