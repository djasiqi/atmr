#!/usr/bin/env node
/* eslint-env node */
/* global __dirname */

const fs = require("fs");
const path = require("path");

const STRICT = process.argv.includes("--strict");
const DRY_RUN = process.argv.includes("--dry-run");
const TODAY = new Date().toISOString();

const ROOT = path.resolve(__dirname, "..");

const REQUIRED_FILES = [
  "docs/runtime/notification_contract.md",
  "docs/contracts/deep_link_contract_v1.md",
  "docs/runtime/ota_policy.md",
  "docs/runtime/runtime_navigation_contract.md",
  "docs/runtime/quick_actions_contract.md",
  "docs/runtime/chat_attachment_contract.ts",
  "docs/runtime/phase2_runtime_rollback.md",
  "docs/runtime/phase2_runtime_observability.md",
  "docs/runtime/cold_start_routing_matrix.md",
  "docs/migration/PHASE2_DEVICE_PROOF_MATRIX.md",
];

const REQUIRED_DEEPLINK_SNIPPETS = [
  "atmr://mission/{mission_id}",
  "atmr://chat/{thread_id}",
  "atmr://transfer/{ride_id}",
  "atmr://dashboard",
  "atmr://rides?filter=urgent",
];

const REQUIRED_MATRIX_ROWS = [
  "mission notification",
  "silent refresh",
  "chat attachment",
  "deep link routing",
  "mission bar",
  "transfer flow",
];

function readUtf8(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function exists(relativePath) {
  return fs.existsSync(path.join(ROOT, relativePath));
}

function parseMarkdownTable(markdown) {
  const lines = markdown
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const tableLines = lines.filter((line) => line.startsWith("|") && line.endsWith("|"));
  if (tableLines.length < 3) return [];
  const headers = tableLines[0]
    .slice(1, -1)
    .split("|")
    .map((cell) => cell.trim().toLowerCase());
  return tableLines.slice(2).map((line) => {
    const cells = line
      .slice(1, -1)
      .split("|")
      .map((cell) => cell.trim());
    const row = {};
    headers.forEach((header, index) => {
      row[header] = cells[index] ?? "";
    });
    return row;
  });
}

function rowStatus(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase();
}

function evaluate() {
  const missingFiles = REQUIRED_FILES.filter((relativePath) => !exists(relativePath));
  const fileChecks = REQUIRED_FILES.map((relativePath) => ({
    file: relativePath,
    exists: exists(relativePath),
  }));

  const deepLinkContract = exists("docs/contracts/deep_link_contract_v1.md")
    ? readUtf8(path.join(ROOT, "docs/contracts/deep_link_contract_v1.md"))
    : "";
  const missingDeepLinkRoutes = REQUIRED_DEEPLINK_SNIPPETS.filter(
    (snippet) => !deepLinkContract.includes(snippet)
  );

  const matrixPath = path.join(ROOT, "docs/migration/PHASE2_DEVICE_PROOF_MATRIX.md");
  const matrix = exists("docs/migration/PHASE2_DEVICE_PROOF_MATRIX.md")
    ? parseMarkdownTable(readUtf8(matrixPath))
    : [];

  const matrixChecks = REQUIRED_MATRIX_ROWS.map((feature) => {
    const row = matrix.find((entry) =>
      rowStatus(entry.feature || entry["feature"]) === feature
    );
    if (!row) {
      return {
        feature,
        exists: false,
        android: "missing",
        ios: "missing",
        valid: false,
      };
    }
    const android = rowStatus(row.android);
    const ios = rowStatus(row.ios);
    const valid = android !== "pending" && ios !== "pending" && android !== "" && ios !== "";
    return { feature, exists: true, android, ios, valid };
  });

  const hasMatrixBlocking = matrixChecks.some((check) => !check.valid);
  const hasMissingContractRoute = missingDeepLinkRoutes.length > 0;
  const hasMissingFiles = missingFiles.length > 0;

  const blocking = hasMissingFiles || hasMissingContractRoute || (STRICT && hasMatrixBlocking);
  const status = blocking ? "HOLD_PENDING_CERTIFICATION" : "READY_FOR_CLOSE_AUTH";

  return {
    status,
    missingFiles,
    missingDeepLinkRoutes,
    matrixChecks,
    fileChecks,
    strict: STRICT,
    blocking,
  };
}

function renderReport(result) {
  const lines = [];
  lines.push("# Phase 2 Close Authorization");
  lines.push("");
  lines.push(`Generated at: ${TODAY}`);
  lines.push(`Mode strict: ${result.strict ? "true" : "false"}`);
  lines.push(`Status: ${result.status}`);
  lines.push("");
  lines.push("## Required files");
  lines.push("");
  lines.push("| File | Exists |");
  lines.push("|---|---|");
  result.fileChecks.forEach((check) => {
    lines.push(`| ${check.file} | ${check.exists ? "yes" : "no"} |`);
  });
  lines.push("");
  lines.push("## Deep link contract");
  lines.push("");
  if (result.missingDeepLinkRoutes.length === 0) {
    lines.push("- all required canonical routes found");
  } else {
    result.missingDeepLinkRoutes.forEach((route) => lines.push(`- missing route: ${route}`));
  }
  lines.push("");
  lines.push("## Device proof matrix");
  lines.push("");
  lines.push("| Feature | Android | iOS | Valid |");
  lines.push("|---|---|---|---|");
  result.matrixChecks.forEach((check) => {
    lines.push(
      `| ${check.feature} | ${check.android} | ${check.ios} | ${check.valid ? "yes" : "no"} |`
    );
  });
  lines.push("");
  lines.push("## Decision");
  lines.push("");
  lines.push(`- Close authorization status: ${result.status}`);
  lines.push(
    `- Blocking reason present: ${result.blocking ? "yes" : "no"}`
  );
  return `${lines.join("\n")}\n`;
}

function main() {
  const result = evaluate();
  const report = renderReport(result);
  const reportPath = path.join(ROOT, "docs/migration/PHASE2_CLOSE_AUTHORIZATION.md");

  if (DRY_RUN) {
    console.log(report);
  } else {
    fs.writeFileSync(reportPath, report, "utf8");
    console.log(`Phase 2 close authorization report updated: ${reportPath}`);
  }

  if (result.blocking) {
    console.error("Phase 2 close authorization is blocked.");
    process.exit(1);
  }
}

main();
