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

function looksLikeJson(s) {
  const t = String(s || "").trim();
  return t.startsWith("{") || t.startsWith("[");
}

function looksLikePlist(s) {
  const t = String(s || "").trim();
  return t.startsWith("<?xml") || t.includes("<plist");
}

function decodeBase64ToUtf8(b64) {
  return Buffer.from(String(b64), "base64").toString("utf8");
}

function main() {
  if (!fs.existsSync(OUT_ANDROID_JSON_PATH)) {
    // Prefer raw JSON from EAS "secret" env var (masked in UI)
    const rawAndroid = process.env.GOOGLE_SERVICES_JSON;
    const b64Android = process.env.GOOGLE_SERVICES_JSON_B64;

    if (rawAndroid || b64Android) {
      // Note: EAS secrets of type "file" are injected as base64 (FILE_BASE64).
      // So GOOGLE_SERVICES_JSON may actually be base64, not raw JSON.
      let jsonText = null;
      if (rawAndroid) {
        const raw = String(rawAndroid);
        jsonText = looksLikeJson(raw) ? raw : decodeBase64ToUtf8(raw);
      } else {
        jsonText = decodeBase64ToUtf8(b64Android);
      }

      // Validate it looks like JSON before writing.
      try {
        JSON.parse(jsonText);
      } catch (e) {
        throw new Error(
          "[write-google-services] GOOGLE_SERVICES_JSON(_B64) is not valid JSON (raw or base64)."
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
      // Note: EAS secrets of type "file" are injected as base64 (FILE_BASE64).
      // So GOOGLE_SERVICES_PLIST is usually base64, not raw plist.
      const raw = String(rawPlist);
      const plistText = looksLikePlist(raw) ? raw : decodeBase64ToUtf8(raw);

      if (!looksLikePlist(plistText)) {
        throw new Error(
          "[write-google-services] GOOGLE_SERVICES_PLIST does not look like a plist (raw or base64)."
        );
      }

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

