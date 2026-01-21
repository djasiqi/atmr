// app.config.js
require('dotenv-flow').config();
const pkg = require('./package.json');
const withAndroidBackButtonMod = require('./prebuild-mods/withAndroidBackButtonMod');
const withAndroidGoogleMapsKey = require('./prebuild-mods/withAndroidGoogleMapsKey');
const withAndroidImmersiveMode = require('./prebuild-mods/withAndroidImmersiveMode');
const withAndroidR8Enabled = require('./prebuild-mods/withAndroidR8Enabled');

const APP_VARIANT = process.env.APP_VARIANT || "prod";
const isDevVariant = APP_VARIANT === "dev";
const runtimeBase = pkg.version || "1.0.0";

module.exports = withAndroidR8Enabled(
  withAndroidImmersiveMode(
    withAndroidGoogleMapsKey(
      withAndroidBackButtonMod(() => ({
  name: isDevVariant ? "Liri Opérations Dev" : "Liri Opérations",
  slug: "operations-app",
  runtimeVersion: isDevVariant ? `${runtimeBase}-dev` : runtimeBase,
  // ✅ Configuration EAS Update
  updates: {
    url: "https://u.expo.dev/3be107c7-29d2-4987-91a0-8d7c31604891"
  },
  // sdkVersion: "53.0.0", // Supprimé : n'est plus nécessaire avec les SDKs récents
  scheme: "atmr",
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
    buildNumber: process.env.IOS_BUILD_NUMBER || "1",
    version: pkg.version,
    // Le fichier est généré au build via `eas-build-pre-install` depuis un secret EAS.
    googleServicesFile: "./GoogleService-Info.plist",
    // ✅ Background modes pour notifications silencieuses + background fetch
    // (nécessite un rebuild natif iOS)
    infoPlist: {
      UIBackgroundModes: ["fetch", "remote-notification"],
    },
    config: {
      usesNonExemptEncryption: false, // À définir selon vos besoins de conformité
    },
  },

  android: {
    enableTablet: false,
    package: isDevVariant ? "ch.liri.operations.dev" : "ch.liri.operations",
    versionCode: parseInt(process.env.ANDROID_VERSION_CODE || "1", 10),
    version: pkg.version,
    // Le fichier est généré au build via `eas-build-pre-install` depuis un secret EAS.
    googleServicesFile: "./google-services.json",
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
    "expo-local-authentication", // ✅ PHASE 2 : Authentification biométrique
    "sentry-expo",
    "expo-task-manager", // ✅ Nécessaire pour le tracking en arrière-plan
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
      "expo-image-picker",
      {
        photosPermission: "L'application a besoin d'accéder à vos photos pour envoyer des images dans le chat.",
        cameraPermission: "L'application a besoin d'accéder à votre caméra pour prendre des photos."
      }
    ],
    "expo-document-picker",
    [
      "expo-build-properties",
      {
        android: {
          // Pin Kotlin and AGP so EAS Prebuild generates matching native config
          kotlinVersion: "2.1.20",
          gradlePluginVersion: "8.13.1",
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
    // ✅ Socket.IO URL depuis variable d'environnement dédiée
    // Dev: http://localhost:5000 ou http://10.0.2.2:5000 (Android emulator)
    // Prod: https://api.lirie.ch (REQUIS)
    socketUrl: process.env.EXPO_PUBLIC_SOCKET_URL || (isDevVariant ? "http://localhost:5000" : undefined),
    router: {},
    eas: { projectId: "3be107c7-29d2-4987-91a0-8d7c31604891" },
  },

  owner: "drinjasiqi",
}))
    )
  )
);
