/**
 * EAS build helper: write google-services.json from env secret.
 *
 * Usage:
 * - Set EAS secret: GOOGLE_SERVICES_JSON_B64 (base64 of google-services.json)
 * - EAS will run `npm run eas-build-pre-install` automatically if present.
 */
/* eslint-disable no-console */

const fs = require("fs");
const path = require("path");

const OUT_PATH = path.join(__dirname, "..", "google-services.json");

function main() {
  if (fs.existsSync(OUT_PATH)) {
    console.log("[write-google-services] google-services.json already exists, skip.");
    return;
  }

  // Prefer raw JSON from EAS "secret" env var (masked in UI)
  const raw = process.env.GOOGLE_SERVICES_JSON;
  const b64 = process.env.GOOGLE_SERVICES_JSON_B64;

  if (!raw && !b64) {
    console.log(
      "[write-google-services] GOOGLE_SERVICES_JSON(_B64) not set; leaving google-services.json absent."
    );
    return;
  }

  const jsonText = raw ? String(raw) : Buffer.from(String(b64), "base64").toString("utf8");

  // Validate it looks like JSON before writing.
  try {
    JSON.parse(jsonText);
  } catch (e) {
    throw new Error(
      "[write-google-services] Decoded GOOGLE_SERVICES_JSON_B64 is not valid JSON."
    );
  }

  fs.writeFileSync(OUT_PATH, jsonText, { encoding: "utf8" });
  console.log("[write-google-services] Wrote google-services.json to:", OUT_PATH);
}

main();

