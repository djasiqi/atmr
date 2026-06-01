/**
 * Expo config plugin — force foregroundServiceType=location on expo-location FGS (Android 14+).
 */
const { withAndroidManifest } = require("@expo/config-plugins");

const LOCATION_FGS_TYPE = "location";
const EXPO_LOCATION_SERVICE_CLASS = "expo.modules.location.services.LocationTaskService";

function ensureForegroundServiceType(service) {
  if (!service.$) {
    service.$ = {};
  }
  const existing = String(service.$["android:foregroundServiceType"] || "");
  const types = new Set(
    existing
      .split("|")
      .map((t) => t.trim())
      .filter(Boolean)
  );
  types.add(LOCATION_FGS_TYPE);
  service.$["android:foregroundServiceType"] = Array.from(types).join("|");
}

function withAndroidTrackingForegroundService(config) {
  return withAndroidManifest(config, (cfg) => {
    const manifest = cfg.modResults.manifest;
    const application = manifest.application?.[0];
    if (!application?.service) {
      return cfg;
    }

    let matched = 0;
    for (const service of application.service) {
      const name = String(service.$?.["android:name"] || "");
      if (
        name === EXPO_LOCATION_SERVICE_CLASS ||
        name.endsWith(".LocationTaskService")
      ) {
        ensureForegroundServiceType(service);
        matched += 1;
      }
    }

    if (matched === 0) {
      // eslint-disable-next-line no-console
      console.warn(
        "[withAndroidTrackingForegroundService] Aucun service expo-location LocationTaskService trouvé dans AndroidManifest."
      );
    }

    return cfg;
  });
}

module.exports = withAndroidTrackingForegroundService;
