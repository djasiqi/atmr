#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-require-imports */
/**
 * Wrapper strict autour de `eas update --branch production`.
 *
 * Objectif : empêcher qu'un `.env` local dev (URLs LAN/HTTP) ne contamine
 * un bundle OTA prod. `eas update` exécute Metro localement, qui charge `.env`
 * via Expo CLI ; il n'y a aucun garde-fou natif (contrairement à `eas build`
 * qui passe par eas.json > env > production).
 *
 * Règles :
 *   1) Si `.env` contient une variable `EXPO_PUBLIC_*` dont la valeur est :
 *      - en `http://` (non HTTPS) OU
 *      - un host LAN/localhost (192.168.x, 10.x, 172.16-31.x, localhost, 127.0.0.1)
 *      → REFUS, le script s'arrête (exit 2) avec un message de remédiation.
 *   2) Sinon, on force `EXPO_PUBLIC_API_BASE_URL` et `EXPO_PUBLIC_DRIVER_SOCKET_URL`
 *      aux valeurs prod hardcodées (PROD_API_BASE_URL / PROD_DRIVER_SOCKET_URL)
 *      dans `process.env`. `dotenv` respecte `override: false` → les valeurs
 *      du `.env` n'écraseront pas celles déjà présentes.
 *   3) `APP_VARIANT=prod` et `EXPO_PUBLIC_APP_ENV=production` sont aussi forcés.
 *
 * Usage : node ./scripts/easUpdateProd.cjs --platform <ios|android|all> [--message "..."]
 */

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");
const {
  isPrivateOrLocalHost,
  PROD_API_BASE_URL,
  PROD_DRIVER_SOCKET_URL,
} = require("../config/publicApiEnv.cjs");

function parseDotEnv(filePath) {
  if (!fs.existsSync(filePath)) return {};
  const out = {};
  const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}

function detectLeaks(envMap) {
  const offenders = [];
  for (const [key, value] of Object.entries(envMap)) {
    if (!key.startsWith("EXPO_PUBLIC_")) continue;
    if (typeof value !== "string" || value.length === 0) continue;
    if (!/^https?:\/\//i.test(value)) continue;
    const isPlainHttp = value.toLowerCase().startsWith("http://");
    const isLan = isPrivateOrLocalHost(value);
    if (isPlainHttp || isLan) {
      offenders.push({ key, value, reasons: [isPlainHttp ? "http" : null, isLan ? "lan/localhost" : null].filter(Boolean) });
    }
  }
  return offenders;
}

function fail(message, code = 2) {
  console.error(`\n[easUpdateProd] ${message}\n`);
  process.exit(code);
}

const argv = process.argv.slice(2);
const platformIdx = argv.indexOf("--platform");
const branchIdx = argv.indexOf("--branch");

if (branchIdx >= 0 && argv[branchIdx + 1] !== "production") {
  fail(
    `REFUS : ce wrapper est dédié à --branch production (reçu: ${argv[branchIdx + 1]}). ` +
      `Utilise \`npx eas update\` directement pour les autres branches.`
  );
}

const platform = platformIdx >= 0 ? argv[platformIdx + 1] : "all";
if (!["ios", "android", "all"].includes(platform)) {
  fail(`REFUS : --platform doit être ios | android | all (reçu: ${platform}).`);
}

const dotenvPath = path.join(process.cwd(), ".env");
const dotenvMap = parseDotEnv(dotenvPath);
const offenders = detectLeaks(dotenvMap);

if (offenders.length > 0) {
  console.error("\n[easUpdateProd] REFUS : `.env` contient des URLs LAN/HTTP qui contamineraient l'OTA prod.\n");
  for (const { key, value, reasons } of offenders) {
    console.error(`  - ${key}=${value}   [${reasons.join(", ")}]`);
  }
  console.error("\nRemédiation (au choix) :");
  console.error("  1. Renomme `.env` en `.env.local-dev` (Expo ne le charge pas automatiquement).");
  console.error("  2. Commente les lignes EXPO_PUBLIC_* offensantes dans `.env`.");
  console.error("  3. Supprime temporairement `.env` (il est gitignored).");
  console.error("\nPuis relance : npm run update:prod:" + platform + "\n");
  process.exit(2);
}

process.env.EXPO_PUBLIC_API_BASE_URL = PROD_API_BASE_URL;
process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL = PROD_DRIVER_SOCKET_URL;
process.env.EXPO_PUBLIC_APP_ENV = process.env.EXPO_PUBLIC_APP_ENV || "production";
process.env.APP_VARIANT = process.env.APP_VARIANT || "prod";

console.log("[easUpdateProd] Garde-fou OK — lancement OTA prod avec :");
console.log(`  EXPO_PUBLIC_API_BASE_URL      = ${process.env.EXPO_PUBLIC_API_BASE_URL}`);
console.log(`  EXPO_PUBLIC_DRIVER_SOCKET_URL = ${process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL}`);
console.log(`  EXPO_PUBLIC_APP_ENV           = ${process.env.EXPO_PUBLIC_APP_ENV}`);
console.log(`  APP_VARIANT                   = ${process.env.APP_VARIANT}`);
console.log(`  Platform                      = ${platform}`);
console.log("");

const easArgs = ["update", "--branch", "production"];
if (platformIdx < 0) {
  easArgs.push("--platform", platform);
}
for (const arg of argv) easArgs.push(arg);

function quoteForShell(arg) {
  if (arg === "" || /[\s"'`$\\!()]/.test(arg)) {
    return `"${String(arg).replace(/(["\\$`])/g, "\\$1")}"`;
  }
  return arg;
}

const easBinary = process.platform === "win32" ? "eas.cmd" : "eas";
const fullCommand = [easBinary, ...easArgs.map(quoteForShell)].join(" ");

const result = spawnSync(fullCommand, {
  stdio: "inherit",
  shell: true,
  env: process.env,
});

process.exit(result.status ?? 1);
