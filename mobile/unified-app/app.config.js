/**
 * Config Expo : injecte les clés Google Maps natives (Android / iOS) depuis .env / EAS.
 * URLs API prod : alignées sur operations-app (garde-fous EAS + repli legacy EXPO_PUBLIC_API_URL).
 */
// eslint-disable-next-line @typescript-eslint/no-require-imports
const fs = require("fs");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const path = require("path");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const appJson = require("./app.json");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const pkg = require("./package.json");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const {
  trimEnv,
  resolveApiBaseUrlFromEnv,
  resolveDriverSocketUrlFromEnv,
  resolveProdApiBaseUrlForEas,
  resolveProdDriverSocketUrlForEas,
} = require("./config/publicApiEnv.cjs");

const APP_VARIANT = process.env.APP_VARIANT || "prod";
const isDevVariant = APP_VARIANT === "dev";
const isProdVariant = APP_VARIANT === "prod";
const isCiBuild = process.env.CI === "true" || process.env.EAS_BUILD === "true";
const shouldEnforceProdEnv = isProdVariant && isCiBuild;
const displayName = isDevVariant ? "Lirie Dev" : "Lirie";

const REQUIRED_ICON_ASSETS = [
  "assets/images/icon.png",
  "assets/images/adaptive-foreground.png",
  "assets/images/favicon.png",
  "assets/images/apple-touch-icon.png",
];

let apiBaseUrl = resolveApiBaseUrlFromEnv();
let driverSocketUrl = resolveDriverSocketUrlFromEnv();

if (shouldEnforceProdEnv) {
  // `expo config` peut s'exécuter avant l'injection eas.json / avec un .env LAN encore présent.
  apiBaseUrl = resolveProdApiBaseUrlForEas(apiBaseUrl);
  driverSocketUrl = resolveProdDriverSocketUrlForEas(driverSocketUrl);
  process.env.EXPO_PUBLIC_API_BASE_URL = apiBaseUrl;
  process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = driverSocketUrl;
  for (const relativePath of REQUIRED_ICON_ASSETS) {
    const absolutePath = path.join(__dirname, relativePath);
    if (!fs.existsSync(absolutePath)) {
      throw new Error(
        `[app.config] Missing ${relativePath} on EAS builder — icône Lirie requise (fichier non versionné git ?)`
      );
    }
  }
}

function envOrExistingFile(envValue, relativePath) {
  if (envValue) return envValue;
  try {
    return fs.existsSync(relativePath) ? relativePath : undefined;
  } catch {
    return undefined;
  }
}

function resolveAndroidMapsApiKey() {
  return trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY") || trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_API_KEY");
}

function resolveIosMapsApiKey() {
  return trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_IOS_API_KEY") || trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_API_KEY");
}

const INVALID_ANDROID_MAPS_KEYS = new Set(["", "test-android-key", "your-android-maps-api-key"]);

function assertAndroidMapsApiKeyForEasProdBuild(apiKey) {
  const normalized = typeof apiKey === "string" ? apiKey.trim() : "";
  if (!normalized || INVALID_ANDROID_MAPS_KEYS.has(normalized)) {
    throw new Error(
      "[app.config] EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY manquante ou placeholder sur build EAS prod. " +
        "Ajoutez la clé dans eas.json (preview/production env) ou EAS Environment, puis relancez avec --clear-cache."
    );
  }
  if (!normalized.startsWith("AIza")) {
    throw new Error(
      `[app.config] EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY invalide sur build EAS prod (attendu préfixe AIza).`
    );
  }
}

/** @type {import('expo/config').ExpoConfig} */
module.exports = ({ config }) => {
  const base = config ?? appJson.expo;
  const androidMapsApiKey = resolveAndroidMapsApiKey();
  const iosMapsApiKey = resolveIosMapsApiKey();

  if (isCiBuild) {
    const mapsEnv = process.env.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY;
    console.log(
      "[app.config] EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY =",
      typeof mapsEnv === "string" && mapsEnv.trim().length > 0
        ? `${mapsEnv.trim().slice(0, 8)}… (${mapsEnv.trim().length} chars)`
        : mapsEnv ?? "(undefined)"
    );
    console.log(
      "[app.config] android.config.googleMaps.apiKey resolved =",
      androidMapsApiKey
        ? `${androidMapsApiKey.slice(0, 8)}… (${androidMapsApiKey.length} chars)`
        : "(empty)"
    );
  }

  if (shouldEnforceProdEnv) {
    assertAndroidMapsApiKeyForEasProdBuild(androidMapsApiKey);
  }

  // Correlation canary / health (release_sha) — EAS injecte EAS_BUILD_GIT_COMMIT_HASH.
  const releaseShaRaw = (
    process.env.EXPO_PUBLIC_RELEASE_SHA ||
    process.env.EAS_BUILD_GIT_COMMIT_HASH ||
    ""
  ).trim();
  const releaseSha = releaseShaRaw ? releaseShaRaw.slice(0, 64) : null;

  return {
    ...base,
    name: displayName,
    runtimeVersion: base.version ?? pkg.version,
    extra: {
      ...(base.extra ?? {}),
      APP_VARIANT,
      productionApiUrl: "https://api.lirie.ch",
      ...(apiBaseUrl ? { apiBaseUrl } : {}),
      ...(driverSocketUrl ? { driverSocketUrl } : {}),
      ...(releaseSha ? { releaseSha } : {}),
    },
    android: {
      ...base.android,
      config: {
        ...(base.android?.config ?? {}),
        googleMaps: {
          ...(base.android?.config?.googleMaps ?? {}),
          apiKey: androidMapsApiKey,
        },
      },
      googleServicesFile: envOrExistingFile(
        process.env.GOOGLE_SERVICES_JSON,
        base.android?.googleServicesFile ?? "./google-services.json"
      ),
    },
    ios: {
      ...base.ios,
      infoPlist: {
        ...(base.ios?.infoPlist ?? {}),
        CFBundleDisplayName: displayName,
      },
      googleServicesFile: envOrExistingFile(
        process.env.GOOGLE_SERVICES_PLIST,
        base.ios?.googleServicesFile ?? "./GoogleService-Info.plist"
      ),
      config: {
        ...(base.ios?.config ?? {}),
        googleMapsApiKey: iosMapsApiKey,
        usesNonExemptEncryption: false,
      },
    },
  };
};
