// app.config.js
require('dotenv-flow').config();
const pkg = require('./package.json');
const withAndroidBackButtonMod = require('./prebuild-mods/withAndroidBackButtonMod');

const APP_VARIANT = process.env.APP_VARIANT || "prod";
const isDevVariant = APP_VARIANT === "dev";
const runtimeBase = pkg.version || "1.0.0";

module.exports = withAndroidBackButtonMod(() => ({
  name: isDevVariant ? "Liri Opérations Dev" : "Liri Opérations",
  slug: "operations-app",
  runtimeVersion: isDevVariant ? `${runtimeBase}-dev` : runtimeBase,
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
    supportsTablet: false,
    bundleIdentifier: isDevVariant
      ? "ch.liri.operations.dev"
      : "ch.liri.operations",
    buildNumber: "1.0.0",
    // googleServicesFile: process.env.GOOGLE_SERVICES_PLIST ?? "./GoogleService-Info.plist",
  },

  android: {
    enableTablet: false,
    package: isDevVariant ? "ch.liri.operations.dev" : "ch.liri.operations",
    googleServicesFile:
      process.env.GOOGLE_SERVICES_JSON ?? "./google-services.json",
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
          // Pin Kotlin and AGP so EAS Prebuild generates matching native config
          kotlinVersion: "2.0.21",
          gradlePluginVersion: "8.11.0",
          // Let EAS choose Gradle version (logs show 8.14.3); don't override here
          // Keep Google Services via plugin in app/build.gradle
          classpath: "com.google.gms:google-services:4.4.2",
        },
      },
    ],
  ],

  experiments: { typedRoutes: true },

  extra: {
    APP_VARIANT: APP_VARIANT, // Passer APP_VARIANT pour détection runtime
    productionApiUrl: "https://api.lirie.ch",
    publicApiUrl: process.env.EXPO_PUBLIC_API_URL || "http://localhost:5000",
    backendPort: 5000,
    SOCKET_HOST: process.env.SOCKET_HOST || "api.lirie.ch",
    SOCKET_PORT: process.env.SOCKET_PORT || "5000",
    router: {},
    eas: { projectId: "3be107c7-29d2-4987-91a0-8d7c31604891" },
  },

  owner: "drinjasiqi",
}));
