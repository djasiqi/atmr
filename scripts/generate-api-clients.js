#!/usr/bin/env node
// scripts/generate-api-clients.js
// ✅ Tâche 2: Script Node.js pour générer les clients TypeScript depuis la spec OpenAPI

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const SPEC_FILE = path.join(PROJECT_ROOT, "backend", "docs", "openapi.json");
const FRONTEND_OUTPUT = path.join(PROJECT_ROOT, "frontend", "src", "generated", "api");
const MOBILE_OUTPUT = path.join(PROJECT_ROOT, "mobile", "operations-app", "src", "generated", "api");

// Vérifier que la spec existe
if (!fs.existsSync(SPEC_FILE)) {
    console.error("❌ Erreur: " + SPEC_FILE + " introuvable");
    console.error("   Exécutez d'abord: npm run api:spec");
    process.exit(1);
}

// Vérifier que openapi-generator est installé
try {
    execSync("openapi-generator-cli version", { stdio: "ignore" });
} catch (error) {
    console.log("⚠️  openapi-generator-cli non trouvé. Installation...");
    execSync("npm install -g @openapitools/openapi-generator-cli", { stdio: "inherit" });
}

console.log("📦 Génération des clients TypeScript depuis " + SPEC_FILE + "...");

// Générer le client pour le frontend web
console.log("🔧 Génération client frontend web...");
if (!fs.existsSync(FRONTEND_OUTPUT)) {
    fs.mkdirSync(FRONTEND_OUTPUT, { recursive: true });
}
execSync(
    `openapi-generator-cli generate ` +
    `-i "${SPEC_FILE}" ` +
    `-g typescript-axios ` +
    `-o "${FRONTEND_OUTPUT}" ` +
    `--additional-properties=supportsES6=true,withInterfaces=true,typescriptThreePlus=true`,
    { stdio: "inherit" }
);

// Générer le client pour le mobile
console.log("🔧 Génération client mobile...");
if (!fs.existsSync(MOBILE_OUTPUT)) {
    fs.mkdirSync(MOBILE_OUTPUT, { recursive: true });
}
execSync(
    `openapi-generator-cli generate ` +
    `-i "${SPEC_FILE}" ` +
    `-g typescript-axios ` +
    `-o "${MOBILE_OUTPUT}" ` +
    `--additional-properties=supportsES6=true,withInterfaces=true,typescriptThreePlus=true`,
    { stdio: "inherit" }
);

console.log("✅ Clients TypeScript générés:");
console.log("   - Frontend: " + FRONTEND_OUTPUT);
console.log("   - Mobile: " + MOBILE_OUTPUT);

