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

const OUT_ANDROID_JSON_PATH = path.join(__dirname, "..", "google-services.json");
const OUT_IOS_PLIST_PATH = path.join(__dirname, "..", "GoogleService-Info.plist");

function main() {
  if (!fs.existsSync(OUT_ANDROID_JSON_PATH)) {
    // Prefer raw JSON from EAS "secret" env var (masked in UI)
    const rawAndroid = process.env.GOOGLE_SERVICES_JSON;
    const b64Android = process.env.GOOGLE_SERVICES_JSON_B64;

    if (rawAndroid || b64Android) {
      const jsonText = rawAndroid
        ? String(rawAndroid)
        : Buffer.from(String(b64Android), "base64").toString("utf8");

      // Validate it looks like JSON before writing.
      try {
        JSON.parse(jsonText);
      } catch (e) {
        throw new Error(
          "[write-google-services] Decoded GOOGLE_SERVICES_JSON(_B64) is not valid JSON."
        );
      }

      fs.writeFileSync(OUT_ANDROID_JSON_PATH, jsonText, { encoding: "utf8" });
      console.log(
        "[write-google-services] Wrote google-services.json to:",
        OUT_ANDROID_JSON_PATH
      );
    } else {
      console.log(
        "[write-google-services] GOOGLE_SERVICES_JSON(_B64) not set; leaving google-services.json absent."
      );
    }
  } else {
    console.log("[write-google-services] google-services.json already exists, skip.");
  }

  // iOS: GoogleService-Info.plist (Firebase config)
  if (!fs.existsSync(OUT_IOS_PLIST_PATH)) {
    const rawPlist = process.env.GOOGLE_SERVICES_PLIST;
    if (rawPlist) {
      const plistText = String(rawPlist);
      fs.writeFileSync(OUT_IOS_PLIST_PATH, plistText, { encoding: "utf8" });
      console.log(
        "[write-google-services] Wrote GoogleService-Info.plist to:",
        OUT_IOS_PLIST_PATH
      );
    } else {
      console.log(
        "[write-google-services] GOOGLE_SERVICES_PLIST not set; leaving GoogleService-Info.plist absent."
      );
    }
  } else {
    console.log("[write-google-services] GoogleService-Info.plist already exists, skip.");
  }
}

main();

