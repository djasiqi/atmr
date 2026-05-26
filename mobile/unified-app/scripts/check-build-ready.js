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
}

log("\n🔍 Vérification build production unified-app\n");

checkEasProduction();
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
