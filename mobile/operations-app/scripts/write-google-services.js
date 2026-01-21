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

function readMaybeFilePath(value) {
  const v = String(value || "").trim();
  if (!v) return null;
  try {
    if (fs.existsSync(v) && fs.statSync(v).isFile()) {
      return fs.readFileSync(v, "utf8");
    }
  } catch {
    // ignore
  }
  return null;
}

function resolveAndroidJsonText() {
  const rawAndroid = process.env.GOOGLE_SERVICES_JSON;
  const b64Android = process.env.GOOGLE_SERVICES_JSON_B64;

  // EAS env vars with `type=file` are injected as a FILE PATH on the builder.
  // So GOOGLE_SERVICES_JSON is usually a path, not raw JSON/base64.
  if (rawAndroid) {
    const fromPath = readMaybeFilePath(rawAndroid);
    if (fromPath) return fromPath;

    const raw = String(rawAndroid);
    if (looksLikeJson(raw)) return raw;

    try {
      const decoded = decodeBase64ToUtf8(raw);
      if (looksLikeJson(decoded)) return decoded;
    } catch {
      // ignore
    }
  }

  if (b64Android) {
    try {
      const decoded = decodeBase64ToUtf8(b64Android);
      if (looksLikeJson(decoded)) return decoded;
    } catch {
      // ignore
    }
  }

  return null;
}

function resolveIosPlistText() {
  const rawPlist = process.env.GOOGLE_SERVICES_PLIST;

  // EAS env vars with `type=file` are injected as a FILE PATH on the builder.
  // So GOOGLE_SERVICES_PLIST is usually a path, not raw plist/base64.
  if (rawPlist) {
    const fromPath = readMaybeFilePath(rawPlist);
    if (fromPath) return fromPath;

    const raw = String(rawPlist);
    if (looksLikePlist(raw)) return raw;

    try {
      const decoded = decodeBase64ToUtf8(raw);
      if (looksLikePlist(decoded)) return decoded;
    } catch {
      // ignore
    }
  }

  return null;
}

function main() {
  if (!fs.existsSync(OUT_ANDROID_JSON_PATH)) {
    const jsonText = resolveAndroidJsonText();
    if (!jsonText) {
      console.log(
        "[write-google-services] No usable GOOGLE_SERVICES_JSON(_B64); leaving google-services.json absent."
      );
    } else {
      try {
        JSON.parse(jsonText);
        fs.writeFileSync(OUT_ANDROID_JSON_PATH, jsonText, { encoding: "utf8" });
        console.log(
          "[write-google-services] Wrote google-services.json to:",
          OUT_ANDROID_JSON_PATH
        );
      } catch (e) {
        console.warn(
          "[write-google-services] google-services.json content is not valid JSON; skipping write."
        );
      }
    }
  } else {
    console.log("[write-google-services] google-services.json already exists, skip.");
  }

  // iOS: GoogleService-Info.plist (Firebase config)
  if (!fs.existsSync(OUT_IOS_PLIST_PATH)) {
    const plistText = resolveIosPlistText();
    if (!plistText) {
      console.log(
        "[write-google-services] No usable GOOGLE_SERVICES_PLIST; leaving GoogleService-Info.plist absent."
      );
    } else if (!looksLikePlist(plistText)) {
      console.warn(
        "[write-google-services] GoogleService-Info.plist content does not look like plist; skipping write."
      );
    } else {
      fs.writeFileSync(OUT_IOS_PLIST_PATH, plistText, { encoding: "utf8" });
      console.log(
        "[write-google-services] Wrote GoogleService-Info.plist to:",
        OUT_IOS_PLIST_PATH
      );
    }
  } else {
    console.log("[write-google-services] GoogleService-Info.plist already exists, skip.");
  }
}

main();

