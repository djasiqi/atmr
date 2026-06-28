#!/usr/bin/env node
/**
 * Diagnostic clé Google Maps Android (build local debug / expo run:android).
 * Carte blanche = clé souvent présente dans l'APK mais refusée par Google Cloud (package / SHA-1).
 */
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const ROOT = path.join(__dirname, "..");
const MANIFEST = path.join(ROOT, "android", "app", "src", "main", "AndroidManifest.xml");
const DEBUG_KEYSTORE = path.join(ROOT, "android", "app", "debug.keystore");
const DEBUG_APK = path.join(
  ROOT,
  "android",
  "app",
  "build",
  "outputs",
  "apk",
  "debug",
  "app-debug.apk"
);

function readEnvKey() {
  const candidates = [".env.development", ".env.local-dev", ".env"];
  for (const name of candidates) {
    const file = path.join(ROOT, name);
    if (!fs.existsSync(file)) continue;
    const line = fs
      .readFileSync(file, "utf8")
      .split(/\r?\n/)
      .find((l) => l.startsWith("EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY="));
    if (line) {
      const value = line.slice(line.indexOf("=") + 1).trim();
      if (value) return { file: name, value };
    }
  }
  return { file: null, value: process.env.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY?.trim() ?? "" };
}

function maskKey(key) {
  if (!key) return "(absent)";
  if (key.length <= 12) return key;
  return `${key.slice(0, 8)}… (${key.length} car.)`;
}

function readManifestKey() {
  if (!fs.existsSync(MANIFEST)) return null;
  const xml = fs.readFileSync(MANIFEST, "utf8");
  const match = xml.match(
    /android:name="com\.google\.android\.geo\.API_KEY"[^>]*android:value="([^"]+)"/
  );
  return match?.[1] ?? null;
}

function readApkKey() {
  if (!fs.existsSync(DEBUG_APK)) return null;
  try {
    const aapt = execSync(
      'where aapt 2>nul || ls "$LOCALAPPDATA/Android/Sdk/build-tools"/*/aapt.exe 2>/dev/null | tail -1',
      { encoding: "utf8", shell: true }
    ).trim();
    if (!aapt) return null;
    const out = execSync(`"${aapt}" dump xmltree "${DEBUG_APK}" AndroidManifest.xml`, {
      encoding: "utf8",
    });
    const match = out.match(/com\.google\.android\.geo\.API_KEY[\s\S]*?android:value.*?="([^"]+)"/);
    return match?.[1] ?? null;
  } catch {
    return null;
  }
}

function readDebugSha1() {
  if (!fs.existsSync(DEBUG_KEYSTORE)) return null;
  try {
    const out = execSync(
      `keytool -list -v -keystore "${DEBUG_KEYSTORE}" -alias androiddebugkey -storepass android -keypass android`,
      { encoding: "utf8" }
    );
    const match = out.match(/SHA\s*1:\s*([0A-F:]+)/i);
    return match?.[1] ?? null;
  } catch {
    return null;
  }
}

const env = readEnvKey();
const manifestKey = readManifestKey();
const apkKey = readApkKey();
const sha1 = readDebugSha1();

console.log("=== Google Maps Android (local debug) ===\n");
console.log(`Package Android : ch.liri.operations`);
console.log(`Clé .env (${env.file ?? "process.env"}) : ${maskKey(env.value)}`);
console.log(`Clé AndroidManifest.xml : ${maskKey(manifestKey ?? "")}`);
console.log(`Clé dans app-debug.apk : ${maskKey(apkKey ?? "")}`);
console.log(`SHA-1 debug.keystore : ${sha1 ?? "(indisponible — keytool / keystore)"}`);

console.log("\n--- Si la carte est blanche mais la clé est dans l'APK ---");
console.log("1. Google Cloud Console → APIs & Services → Credentials → votre clé Android");
console.log("2. Restrictions « Applications Android » :");
console.log("   - Nom du package : ch.liri.operations");
console.log(`   - Empreinte SHA-1 : ${sha1 ?? "5E:8F:16:06:2E:A3:CD:2C:4A:0D:54:78:76:BA:A6:F3:8C:AB:F6:25 (debug standard)"}`);
console.log("3. APIs activées : Maps SDK for Android (+ Directions si itinéraires)");
console.log("4. Facturation GCP activée sur le projet");
console.log("\n--- Rebuild local après changement .env / prebuild ---");
console.log("   npx expo prebuild --platform android");
console.log("   npx expo run:android");

if (!manifestKey && !apkKey) {
  console.error("\n❌ Aucune clé détectée dans le manifest / APK — lancer prebuild avec .env.development.");
  process.exit(1);
}

if (env.value && manifestKey && env.value !== manifestKey) {
  console.warn("\n⚠️ Clé .env ≠ AndroidManifest — relancer prebuild ou expo run:android.");
}
