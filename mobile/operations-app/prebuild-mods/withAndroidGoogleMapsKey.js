/**
 * Mod Prebuild pour injecter la clé Google Maps API dans AndroidManifest.xml
 * depuis la variable d'environnement EXPO_PUBLIC_ANDROID_MAPS_API_KEY
 */
const { withAndroidManifest } = require('expo/config-plugins');

module.exports = function withAndroidGoogleMapsKey(config) {
  return withAndroidManifest(config, async (config) => {
    const androidManifest = config.modResults;
    const apiKey = process.env.EXPO_PUBLIC_ANDROID_MAPS_API_KEY;

    if (!apiKey) {
      console.warn(
        '⚠️  EXPO_PUBLIC_ANDROID_MAPS_API_KEY n\'est pas définie. ' +
        'La clé Google Maps ne sera pas injectée dans AndroidManifest.xml.'
      );
      return config;
    }

    // Trouver ou créer la section application
    if (!androidManifest.manifest.application) {
      androidManifest.manifest.application = [{}];
    }

    const application = androidManifest.manifest.application[0];

    // Initialiser 'meta-data' si nécessaire
    if (!application['meta-data']) {
      application['meta-data'] = [];
    }

    // Chercher la clé Google Maps existante
    const existingKeyIndex = application['meta-data'].findIndex(
      (meta) => meta.$['android:name'] === 'com.google.android.geo.API_KEY'
    );

    const googleMapsMetaData = {
      $: {
        'android:name': 'com.google.android.geo.API_KEY',
        'android:value': apiKey,
      },
    };

    if (existingKeyIndex >= 0) {
      // Remplacer la clé existante
      application['meta-data'][existingKeyIndex] = googleMapsMetaData;
    } else {
      // Ajouter la nouvelle clé
      application['meta-data'].push(googleMapsMetaData);
    }

    return config;
  });
};

