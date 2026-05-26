/**
 * Injecte / remplace com.google.android.geo.API_KEY dans AndroidManifest.xml
 * au moment du prebuild (après résolution app.config.js).
 *
 * Repli si app.config n'a pas reçu la variable (cache EAS, timing env) :
 * lit EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY puis EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.
 */
const { withAndroidManifest } = require("expo/config-plugins");

const META_API_KEY = "com.google.android.geo.API_KEY";
const PLACEHOLDER_KEYS = new Set(["", "test-android-key", "your-android-maps-api-key"]);

function resolveAndroidMapsApiKeyFromEnv() {
  const candidates = [
    process.env.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY,
    process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY,
  ];
  for (const raw of candidates) {
    if (typeof raw !== "string") continue;
    const trimmed = raw.trim();
    if (trimmed.length > 0 && !PLACEHOLDER_KEYS.has(trimmed)) {
      return trimmed;
    }
  }
  return null;
}

function upsertGoogleMapsMetaData(application, apiKey) {
  if (!application["meta-data"]) {
    application["meta-data"] = [];
  }
  const existingKeyIndex = application["meta-data"].findIndex(
    (meta) => meta.$?.["android:name"] === META_API_KEY
  );
  const googleMapsMetaData = {
    $: {
      "android:name": META_API_KEY,
      "android:value": apiKey,
    },
  };
  if (existingKeyIndex >= 0) {
    application["meta-data"][existingKeyIndex] = googleMapsMetaData;
  } else {
    application["meta-data"].push(googleMapsMetaData);
  }
}

module.exports = function withAndroidGoogleMapsKey(config) {
  return withAndroidManifest(config, (config) => {
    const fromConfig = config.android?.config?.googleMaps?.apiKey;
    const configKey =
      typeof fromConfig === "string" && fromConfig.trim().length > 0 && !PLACEHOLDER_KEYS.has(fromConfig.trim())
        ? fromConfig.trim()
        : null;
    const apiKey = configKey ?? resolveAndroidMapsApiKeyFromEnv();

    if (!apiKey) {
      console.warn(
        "[withAndroidGoogleMapsKey] Aucune clé Maps Android valide — " +
          "définir EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY (eas.json ou EAS Environment)."
      );
      return config;
    }

    const androidManifest = config.modResults;
    if (!androidManifest.manifest.application) {
      androidManifest.manifest.application = [{}];
    }
    const application = androidManifest.manifest.application[0];
    upsertGoogleMapsMetaData(application, apiKey);

    return config;
  });
};
