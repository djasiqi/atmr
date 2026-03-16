#!/usr/bin/env node

const { spawnSync } = require("child_process");
const path = require("path");

const platform = String(process.env.EAS_BUILD_PLATFORM || "").toLowerCase();
const profile = String(process.env.EAS_BUILD_PROFILE || "").toLowerCase();

const shouldRunBuildCheck = platform === "android" && profile === "production";

if (!shouldRunBuildCheck) {
  console.log(
    `[eas-preinstall-check] Skip check-build-ready (platform=${platform || "unknown"}, profile=${profile || "unknown"})`
  );
  process.exit(0);
}

console.log("[eas-preinstall-check] Run check-build-ready for android production");

const scriptPath = path.join(__dirname, "check-build-ready.js");
const result = spawnSync(process.execPath, [scriptPath], {
  stdio: "inherit",
  env: process.env,
});

if (result.error) {
  console.error("[eas-preinstall-check] Failed to run check-build-ready:", result.error);
  process.exit(1);
}

process.exit(result.status ?? 1);

