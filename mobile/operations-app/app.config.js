// app.config.js
require("dotenv-flow").config({ silent: true });
const fs = require("fs");
const pkg = require('./package.json');
const withAndroidBackButtonMod = require('./prebuild-mods/withAndroidBackButtonMod');
const withAndroidGoogleMapsKey = require('./prebuild-mods/withAndroidGoogleMapsKey');
const withAndroidImmersiveMode = require('./prebuild-mods/withAndroidImmersiveMode');
const withAndroidR8Enabled = require('./prebuild-mods/withAndroidR8Enabled');
const withAndroidIconBackground = require('./prebuild-mods/withAndroidIconBackground');
const withAndroidNotificationColorFix = require('./prebuild-mods/withAndroidNotificationColorFix');
const withIosFirebaseModularHeaders = require('./prebuild-mods/withIosFirebaseModularHeaders');

const APP_VARIANT = process.env.APP_VARIANT || "prod";
const isDevVariant = APP_VARIANT === "dev";
const runtimeBase = pkg.version || "1.0.0";
const isProdVariant = APP_VARIANT === "prod";
const isCiBuild = process.env.CI === "true" || process.env.EAS_BUILD === "true";
const shouldEnforceProdEnv = isProdVariant && isCiBuild;

function assertProdHttpsEnv(name) {
  const value = process.env[name];
  if (!shouldEnforceProdEnv) return value;
  if (!value || typeof value !== "string") {
    throw new Error(`[app.config] Missing required ${name} for APP_VARIANT=prod`);
  }
  const normalized = value.trim();
  if (!normalized.startsWith("https://")) {
    throw new Error(`[app.config] ${name} must be HTTPS in production`);
  }
  if (normalized.includes("localhost") || normalized.includes("127.0.0.1")) {
    throw new Error(`[app.config] ${name} must not target localhost in production`);
  }
  return normalized;
}

const requiredProdApiUrl = assertProdHttpsEnv("EXPO_PUBLIC_API_URL");
const requiredProdSocketUrl = assertProdHttpsEnv("EXPO_PUBLIC_SOCKET_URL");

function envOrExistingFile(envValue, relativePath) {
  // En local, `eas build` exécute `expo config` avant d'injecter les secrets EAS.
  // Donc si le fichier n'existe pas localement, il faut éviter de le référencer,
  // sinon `expo config --type introspect` échoue avec ENOENT.
  if (envValue) return envValue;
  try {
    return fs.existsSync(relativePath) ? relativePath : undefined;
  } catch {
    return undefined;
  }
}

module.exports = withAndroidR8Enabled(
  withAndroidImmersiveMode(
    withAndroidIconBackground(
      withAndroidGoogleMapsKey(
        withAndroidBackButtonMod(() => ({
  name: isDevVariant ? "Lirie Dev" : "Lirie",
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
    image: "./assets/images/splash.png",
    resizeMode: "cover",
    backgroundColor: "#0D7F72",
  },

  ios: {
    supportsTablet: false,
    bundleIdentifier: isDevVariant
      ? "ch.liri.operations.dev"
      : "ch.liri.operations",
    buildNumber: process.env.IOS_BUILD_NUMBER || "1",
    version: pkg.version,
    // EAS injecte les variables `type=file` comme chemin vers un fichier sur le builder.
    // Fallback local: fichier à la racine du projet.
    googleServicesFile: envOrExistingFile(process.env.GOOGLE_SERVICES_PLIST, "./GoogleService-Info.plist"),
    // ✅ Background modes : fetch, remote-notification, location (tracking chauffeur mission-active)
    // (nécessite un rebuild natif iOS)
    infoPlist: {
      UIBackgroundModes: ["fetch", "remote-notification", "location"],
      LSApplicationQueriesSchemes: ["tel", "comgooglemaps"],
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
    // EAS injecte les variables `type=file` comme chemin vers un fichier sur le builder.
    // Fallback local: fichier à la racine du projet.
    googleServicesFile: envOrExistingFile(process.env.GOOGLE_SERVICES_JSON, "./google-services.json"),
    adaptiveIcon: {
      foregroundImage: "./assets/images/adaptive-foreground.png",
      backgroundColor: "#FFFFFF",
    },
    permissions: [
      "android.permission.POST_NOTIFICATIONS",
      "android.permission.FOREGROUND_SERVICE",
      "android.permission.FOREGROUND_SERVICE_LOCATION",
      "android.permission.FOREGROUND_SERVICE_DATA_SYNC",
    ],
    // Durcissement permissions: on bloque explicitement les permissions
    // non utilisées et à risque qui peuvent être injectées transitive-ment
    // par certaines dépendances Android.
    blockedPermissions: [
      "android.permission.SYSTEM_ALERT_WINDOW",
      "android.permission.RECORD_AUDIO",
      "android.permission.READ_EXTERNAL_STORAGE",
      "android.permission.WRITE_EXTERNAL_STORAGE",
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
    [
      "expo-build-properties",
      {
        android: {
          kotlinVersion: "2.1.20",
          gradlePluginVersion: "8.13.1",
          classpath: "com.google.gms:google-services:4.4.2",
        },
        ios: {
          forceStaticLinking: [
            "RNFBApp",
            "RNFBMessaging",
          ],
        },
      },
    ],
    "@react-native-firebase/app",
    "@react-native-firebase/messaging",
    "expo-router",
    "expo-font",
    "expo-web-browser",
    "expo-secure-store",
    "expo-local-authentication",
    "sentry-expo",
    "expo-task-manager",
    [
      "expo-notifications",
      {
        icon: "./assets/icons/notification.png",
        color: "#0A7F59",
      },
    ],
    withAndroidNotificationColorFix,
    withIosFirebaseModularHeaders,
    [
      "expo-location",
      {
        // ✅ AMÉLIORATION : Configuration moderne pour la géolocalisation en arrière-plan
        isAndroidBackgroundLocationEnabled: true,
        foregroundService: {
          notificationTitle: "Liri Opérations est active",
          notificationBody:
            "Suivi de la localisation en cours pour vos opérations.",
          notificationColor: "#0A7F59",
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
  ],

  experiments: { typedRoutes: false }, // Désactivé: @expo/router-server requiert Expo 55+

  extra: {
    APP_VARIANT: APP_VARIANT, // Passer APP_VARIANT pour détection runtime
    productionApiUrl: "https://api.lirie.ch",
    publicApiUrl: isProdVariant
      ? requiredProdApiUrl
      : process.env.EXPO_PUBLIC_API_URL || "http://localhost:5000",
    backendPort: 5000,
    // ✅ Socket.IO URL depuis variable d'environnement dédiée
    // Dev: http://localhost:5000 ou http://10.0.2.2:5000 (Android emulator)
    // Prod: https://api.lirie.ch (REQUIS)
    socketUrl: isProdVariant
      ? requiredProdSocketUrl
      : process.env.EXPO_PUBLIC_SOCKET_URL || (isDevVariant ? "http://localhost:5000" : undefined),
    router: {},
    eas: { projectId: "3be107c7-29d2-4987-91a0-8d7c31604891" },
  },

  owner: "drinjasiqi",
}))
      )
    )
  )
);
