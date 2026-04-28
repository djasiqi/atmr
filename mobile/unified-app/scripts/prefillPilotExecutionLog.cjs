#!/usr/bin/env node

const fs = require("fs");
const { execSync } = require("child_process");
const path = require("path");

const LOG_PATH = path.resolve(__dirname, "../docs/PILOT_EXECUTION_LOG.md");
const DRY_RUN = process.argv.includes("--dry-run");

function safeExec(cmd) {
  try {
    return execSync(cmd, { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
  } catch {
    return "unknown";
  }
}

function getBranch() {
  return safeExec("git rev-parse --abbrev-ref HEAD");
}

function getCommit() {
  return safeExec("git rev-parse HEAD");
}

function getDate() {
  return new Date().toISOString().slice(0, 10);
}

function checkScript(scriptName, fallback = "unknown") {
  try {
    execSync(`npm run ${scriptName}`, { stdio: "ignore" });
    return "passed";
  } catch {
    return fallback;
  }
}

function replaceToken(content, token, value) {
  return content.replace(new RegExp(token, "g"), value);
}

function replaceYamlField(content, fieldRegex, value) {
  if (!fieldRegex.test(content)) return content;
  return content.replace(fieldRegex, `$1${value}`);
}

function updateLog(content) {
  const values = {
    branch: getBranch(),
    commit: getCommit(),
    date: getDate(),
    lint: checkScript("lint", "failed"),
    tests: checkScript("test -- --watchAll=false", "failed"),
    instrumentation: checkScript("check-instrumentation-events", "failed"),
    boundaries: checkScript("no-legacy-imports", "failed"),
  };

  let updated = content;

  // Placeholder compatibility
  updated = replaceToken(updated, "__AUTO_BRANCH__", values.branch);
  updated = replaceToken(updated, "__AUTO_COMMIT__", values.commit);
  updated = replaceToken(updated, "__AUTO_DATE__", values.date);
  updated = replaceToken(updated, "__AUTO_LINT__", values.lint);
  updated = replaceToken(updated, "__AUTO_TESTS__", values.tests);
  updated = replaceToken(updated, "__AUTO_INSTRUMENTATION__", values.instrumentation);
  updated = replaceToken(updated, "__AUTO_BOUNDARIES__", values.boundaries);

  // Idempotent updates on already-prefilled files
  updated = replaceYamlField(updated, /(^\s*branch:\s*).*/m, values.branch);
  updated = replaceYamlField(updated, /(^\s*commit:\s*).*/m, values.commit);
  updated = replaceYamlField(updated, /(^\s*start_date:\s*).*/m, values.date);
  updated = replaceYamlField(updated, /(^\s*lint:\s*).*/m, values.lint);
  updated = replaceYamlField(updated, /(^\s*jest_tests:\s*).*/m, values.tests);
  updated = replaceYamlField(
    updated,
    /(^\s*instrumentation_check:\s*).*/m,
    values.instrumentation
  );
  updated = replaceYamlField(updated, /(^\s*boundary_checks:\s*).*/m, values.boundaries);

  return { updated, values };
}

function main() {
  if (!fs.existsSync(LOG_PATH)) {
    console.error("PILOT_EXECUTION_LOG.md not found");
    process.exit(1);
  }

  const content = fs.readFileSync(LOG_PATH, "utf8");
  const { updated, values } = updateLog(content);

  if (DRY_RUN) {
    console.log("Pilot execution log dry-run values:");
    console.log(JSON.stringify(values, null, 2));
    console.log("No file has been modified.");
    return;
  }

  fs.writeFileSync(LOG_PATH, updated);

  console.log("Pilot execution log prefilled successfully.");
}

main();
