#!/usr/bin/env node

/**
 * Vérifications avant build EAS production (aligné operations-app).
 * EAS n'embarque que les fichiers versionnés git → icônes non commitées = icône Android par défaut.
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const errors = [];
const warnings = [];
const checks = [];

function log(message) {
  console.log(message);
}

function checkFile(relativePath, label, required = true) {
  const fullPath = path.join(ROOT, relativePath);
  if (fs.existsSync(fullPath)) {
    checks.push(`✅ ${label}`);
    return true;
  }
  const msg = `${required ? "❌" : "⚠️"} ${label}${required ? " (requis)" : ""}`;
  if (required) errors.push(msg);
  else checks.push(msg);
  return false;
}

function checkGitTracked(relativePath, label) {
  if (!checkFile(relativePath, label, true)) return;
  try {
    execSync(`git ls-files --error-unmatch "${relativePath.replace(/\\/g, "/")}"`, {
      cwd: ROOT,
      stdio: "pipe",
    });
    checks.push(`✅ ${label} versionné git (inclus dans EAS Build)`);
  } catch {
    errors.push(
      `❌ ${label} présent localement mais NON versionné git — EAS Build affichera l'icône Android par défaut. Lancez: git add ${relativePath}`
    );
  }
}

function checkDisplayName() {
  const appConfigPath = path.join(ROOT, "app.config.js");
  if (!fs.existsSync(appConfigPath)) return;
  try {
    const prevVariant = process.env.APP_VARIANT;
    const prevEasBuild = process.env.EAS_BUILD;
    const prevCi = process.env.CI;
    process.env.APP_VARIANT = "prod";
    delete process.env.EAS_BUILD;
    delete process.env.CI;
    delete require.cache[require.resolve(appConfigPath)];
    const appJson = require(path.join(ROOT, "app.json"));
    const configFn = require(appConfigPath);
    const resolved = configFn({ config: appJson.expo });
    if (resolved.name === "Lirie") {
      checks.push('✅ Nom affiché configuré: "Lirie"');
    } else {
      errors.push(`❌ Nom affiché incorrect: "${resolved.name ?? "(vide)"}" (attendu: Lirie)`);
    }
    if (prevVariant === undefined) delete process.env.APP_VARIANT;
    else process.env.APP_VARIANT = prevVariant;
    if (prevEasBuild === undefined) delete process.env.EAS_BUILD;
    else process.env.EAS_BUILD = prevEasBuild;
    if (prevCi === undefined) delete process.env.CI;
    else process.env.CI = prevCi;
  } catch (error) {
    warnings.push(
      `⚠️ Impossible de valider le nom affiché: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

function checkGpsProductionReadiness(prodEnv) {
  const requiredTrackingFlags = [
    ["EXPO_PUBLIC_ENABLE_BG_LOCATION", "1"],
    ["EXPO_PUBLIC_ENABLE_DRIVER_SOCKET", "1"],
    ["EXPO_PUBLIC_ENABLE_TRACKING_PERSISTENT_QUEUE", "1"],
    ["EXPO_PUBLIC_ENABLE_TRACKING_HTTP_FALLBACK", "1"],
    ["EXPO_PUBLIC_ENABLE_TRACKING_PRESENCE_MODE", "1"],
    ["EXPO_PUBLIC_ENABLE_TRACKING_SELF_HEAL_WATCH", "1"],
  ];

  for (const [key, expected] of requiredTrackingFlags) {
    if (prodEnv[key] === expected) {
      checks.push(`✅ ${key}=${expected} (production GPS)`);
    } else {
      errors.push(`❌ ${key} doit être "${expected}" dans eas.json (production) — tracking GPS désactivé au build`);
    }
  }

  if (prodEnv.EXPO_PUBLIC_DRIVER_SOCKET_BATCH_MIN_INTERVAL_MS === "5000") {
    checks.push("✅ EXPO_PUBLIC_DRIVER_SOCKET_BATCH_MIN_INTERVAL_MS=5000 (aligné rate limiter WS)");
  } else {
    errors.push(
      "❌ EXPO_PUBLIC_DRIVER_SOCKET_BATCH_MIN_INTERVAL_MS doit être \"5000\" dans eas.json (production)"
    );
  }

  const fgsTitle = prodEnv.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_TITLE?.trim();
  const fgsMission = prodEnv.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_BODY?.trim();
  const fgsPresence = prodEnv.EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_BODY_PRESENCE?.trim();
  if (fgsTitle && fgsMission && fgsPresence) {
    checks.push("✅ Textes FGS fr-CH définis dans eas.json (production)");
    if (fgsTitle.includes("Unified")) {
      warnings.push("⚠️ EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_TITLE contient encore « Unified »");
    }
  } else {
    errors.push(
      "❌ Textes FGS manquants dans eas.json (EXPO_PUBLIC_DRIVER_BG_NOTIFICATION_TITLE/BODY/BODY_PRESENCE)"
    );
  }

  const appJsonPath = path.join(ROOT, "app.json");
  if (fs.existsSync(appJsonPath)) {
    const app = JSON.parse(fs.readFileSync(appJsonPath, "utf8"));
    const perms = app.expo?.android?.permissions ?? [];
    const needs = [
      "android.permission.ACCESS_BACKGROUND_LOCATION",
      "android.permission.FOREGROUND_SERVICE_LOCATION",
      "android.permission.RECORD_AUDIO",
    ];
    for (const perm of needs) {
      if (perms.includes(perm)) {
        checks.push(`✅ app.json permission ${perm}`);
      } else {
        errors.push(`❌ app.json : permission manquante ${perm}`);
      }
    }

    const iosLoc = app.expo?.ios?.infoPlist?.NSLocationWhenInUseUsageDescription ?? "";
    if (iosLoc.startsWith("La localisation")) {
      checks.push("✅ app.json permissions iOS localisation en français");
    } else {
      warnings.push("⚠️ app.json : NSLocation* iOS pas en français — aligner avec modales disclosure");
    }

    const patchPath = path.join(ROOT, "patches", "expo-location+19.0.8.patch");
    if (fs.existsSync(patchPath)) {
      checks.push("✅ Patch natif expo-location Android 16 présent");
    } else {
      errors.push("❌ patches/expo-location+19.0.8.patch manquant — BG Android 16 non couvert");
    }
  }
}

function checkEasProduction() {
  const easPath = path.join(ROOT, "eas.json");
  if (!fs.existsSync(easPath)) {
    errors.push("❌ eas.json introuvable");
    return;
  }
  const eas = JSON.parse(fs.readFileSync(easPath, "utf8"));
  if (!eas.build?.production) {
    errors.push("❌ Profil build production manquant dans eas.json");
    return;
  }
  checks.push("✅ Profil build production présent");
  if (!eas.submit?.production) {
    errors.push("❌ Profil submit production manquant dans eas.json");
  } else {
    checks.push("✅ Profil submit production présent");
  }
  const prodEnv = eas.build.production.env ?? {};
  if (prodEnv.EXPO_PUBLIC_API_BASE_URL?.startsWith("https://")) {
    checks.push("✅ EXPO_PUBLIC_API_BASE_URL prod dans eas.json");
  } else {
    errors.push("❌ EXPO_PUBLIC_API_BASE_URL HTTPS manquant dans eas.json (production.env)");
  }
  const androidMapsKey = prodEnv.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY?.trim();
  const mapsFromShell = process.env.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY?.trim();
  if (androidMapsKey && androidMapsKey.startsWith("AIza") && androidMapsKey !== "test-android-key") {
    checks.push("✅ EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY prod dans eas.json");
  } else if (mapsFromShell && mapsFromShell.startsWith("AIza")) {
    checks.push("✅ EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY présente dans l'environnement shell");
  } else {
    warnings.push(
      "⚠️ EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY absente de eas.json — OK si définie dans EAS Environment (production) sur expo.dev"
    );
  }
  const prodPrebuild = eas.build.production.prebuildCommand ?? "";
  if (prodPrebuild.includes("--clean")) {
    checks.push("✅ prebuildCommand --clean (production) — évite le skip prebuild EAS");
  } else {
    errors.push(
      "❌ prebuildCommand avec --clean manquant dans eas.json (production) — EAS peut ignorer le prebuild si android/ existe"
    );
  }
  const otaAutoReload = prodEnv.EXPO_PUBLIC_OTA_AUTO_RELOAD_ENABLED;
  if (otaAutoReload === "1" || otaAutoReload === "true") {
    checks.push("✅ EXPO_PUBLIC_OTA_AUTO_RELOAD_ENABLED actif (production)");
  } else {
    errors.push(
      "❌ EXPO_PUBLIC_OTA_AUTO_RELOAD_ENABLED doit être \"1\" dans eas.json (production) pour les builds store"
    );
  }
  const appJsonPath = path.join(ROOT, "app.json");
  if (fs.existsSync(appJsonPath)) {
    const app = JSON.parse(fs.readFileSync(appJsonPath, "utf8"));
    const version = app.expo?.version;
    const runtimeVersion = app.expo?.runtimeVersion;
    if (version && runtimeVersion && version === runtimeVersion) {
      checks.push(`✅ version / runtimeVersion alignés (${version})`);
    } else {
      errors.push(
        `❌ version (${version ?? "?"}) et runtimeVersion (${runtimeVersion ?? "?"}) doivent être identiques pour ce release store`
      );
    }
  }
  checkGpsProductionReadiness(prodEnv);
}

function checkNativeDirsAbsentForEas() {
  for (const dir of ["android", "ios"]) {
    const fullPath = path.join(ROOT, dir);
    if (fs.existsSync(fullPath)) {
      warnings.push(
        `⚠️ Dossier ${dir}/ présent localement — supprimez-le avant eas build (npm run prebuild le régénère). .easignore l'exclut mais --clean côté serveur est plus sûr.`
      );
    }
  }
}

log("\n🔍 Vérification build production unified-app\n");

checkEasProduction();
checkNativeDirsAbsentForEas();
checkDisplayName();
checkGitTracked("assets/images/icon.png", "Icône store (512, icon.png)");
checkGitTracked("assets/images/adaptive-foreground.png", "Adaptive Android (1024, zone sûre ~66 %, adaptive-foreground.png)");
checkGitTracked("assets/images/apple-touch-icon.png", "Apple touch web (180, apple-touch-icon.png)");
checkGitTracked("assets/images/splash-solid.png", "Splash couleur unie (#EAF3F1, splash-solid.png)");
checkGitTracked("assets/images/favicon.png", "Favicon Expo web (96, favicon.png)");
checkFile("assets/images/lirie-logo-color.png", "Logo UI (écrans login/signup, lirie-logo-color.png)", true);
checkFile("app.config.js", "app.config.js", true);
checkFile("eas.json", "eas.json", true);

if (checks.length) {
  log("\n" + checks.join("\n"));
}

if (warnings.length) {
  log("\n⚠️ Avertissements:\n" + warnings.join("\n"));
}

if (errors.length) {
  log("\n❌ Erreurs:\n" + errors.join("\n"));
  log("\nCorrigez puis relancez: npm run check-build-ready\n");
  process.exit(1);
}

log("\n✅ Prêt pour: eas build --profile production\n");
process.exit(0);
