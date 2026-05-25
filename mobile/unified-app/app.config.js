/**
 * Config Expo : injecte les clés Google Maps natives (Android / iOS) depuis .env / EAS.
 * Le web utilise toujours EXPO_PUBLIC_GOOGLE_MAPS_API_KEY (Maps JavaScript API).
 */
// eslint-disable-next-line @typescript-eslint/no-require-imports
const appJson = require("./app.json");

function trimEnv(name) {
  const v = process.env[name];
  return typeof v === "string" ? v.trim() : "";
}

function resolveAndroidMapsApiKey() {
  return trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY") || trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_API_KEY");
}

function resolveIosMapsApiKey() {
  return trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_IOS_API_KEY") || trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_API_KEY");
}

/** @type {import('expo/config').ExpoConfig} */
module.exports = ({ config }) => {
  const base = config ?? appJson.expo;
  const androidMapsApiKey = resolveAndroidMapsApiKey();
  const iosMapsApiKey = resolveIosMapsApiKey();
  const apiBaseUrl = trimEnv("EXPO_PUBLIC_API_BASE_URL");
  const driverSocketUrl = trimEnv("EXPO_PUBLIC_DRIVER_SOCKET_URL");
  const devCleartext =
    apiBaseUrl.startsWith("http://") || driverSocketUrl.startsWith("http://");

  return {
    ...base,
    extra: {
      ...(base.extra ?? {}),
      ...(apiBaseUrl ? { apiBaseUrl } : {}),
    },
    android: {
      ...base.android,
      ...(devCleartext ? { usesCleartextTraffic: true } : {}),
      config: {
        ...(base.android?.config ?? {}),
        googleMaps: {
          ...(base.android?.config?.googleMaps ?? {}),
          apiKey: androidMapsApiKey,
        },
      },
    },
    ios: {
      ...base.ios,
      config: {
        ...(base.ios?.config ?? {}),
        googleMapsApiKey: iosMapsApiKey,
      },
    },
  };
};
