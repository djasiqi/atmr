// app.config.js
module.exports = () => ({
  name: "Liri Opérations",
  slug: "operations-app",
  version: "1.0.1",
  // sdkVersion: "53.0.0", // Supprimé : n'est plus nécessaire avec les SDKs récents
  scheme: "liri",
  orientation: "portrait",
  userInterfaceStyle: "automatic",

  icon: "./assets/images/icon.png",
  splash: {
    image: "./assets/images/splash-icon.png",
    imageWidth: 200,
    resizeMode: "contain",
    backgroundColor: "#ffffff",
  },

  ios: {
    supportsTablet: true,
    bundleIdentifier: "ch.liri.operations",
    buildNumber: "1.0.0",
    // googleServicesFile: process.env.GOOGLE_SERVICES_PLIST ?? "./GoogleService-Info.plist",
  },

  android: {
    package: "ch.liri.operations",
    googleServicesFile:
      process.env.GOOGLE_SERVICES_JSON ?? "./google-services.json",
    versionCode: 2,
    adaptiveIcon: {
      foregroundImage: "./assets/images/adaptive-icon.png",
      backgroundColor: "#ffffff",
    },
    permissions: [
      "android.permission.POST_NOTIFICATIONS",
      "android.permission.ACCESS_BACKGROUND_LOCATION",
      "android.permission.FOREGROUND_SERVICE",
      "android.permission.FOREGROUND_SERVICE_LOCATION",
    ],
    config: {
      googleMaps: {
        apiKey: process.env.EXPO_PUBLIC_ANDROID_MAPS_API_KEY,
      },
    },
  },

  web: {
    bundler: "metro",
    output: "static",
    favicon: "./assets/images/favicon.png",
  },

  plugins: [
    "expo-router",
    "expo-font",
    "expo-web-browser",
    "expo-secure-store",
    "sentry-expo",
    [
      "expo-notifications",
      {
        icon: "./assets/icons/notification-icon.png",
        color: "#ffffff",
        // sounds: [] // `sounds` est vide, peut être omis si vous utilisez le son par défaut
      },
    ],
    [
      "expo-location",
      {
        // ✅ AMÉLIORATION : Configuration moderne pour la géolocalisation en arrière-plan
        foregroundService: {
          notificationTitle: "Liri Opérations est active",
          notificationBody:
            "Suivi de la localisation en cours pour vos opérations.",
          notificationColor: "#ffffff",
        },
      },
    ],
    [
      "expo-build-properties",
      {
        android: {
          classpath: "com.google.gms:google-services:4.4.2",
          gradlePluginVersion: "8.5.2",
          gradleVersion: "8.5.2",
          kotlinVersion: "2.0.21",
        },
      },
    ],
  ],

  experiments: { typedRoutes: true },

  extra: {
    productionApiUrl: "https://api.monapp.com",
    publicApiUrl:
      process.env.EXPO_PUBLIC_API_URL || "http://192.168.1.216:5000",
    backendPort: 5000,
    // 👇 NOUVEAU : configure l’hôte/port Socket via env ou fallback
    SOCKET_HOST: process.env.SOCKET_HOST || "192.168.1.216",
    SOCKET_PORT: process.env.SOCKET_PORT || "5000",
    router: {},
    eas: { projectId: "3be107c7-29d2-4987-91a0-8d7c31604891" },
  },

  owner: "drinjasiqi",
});
