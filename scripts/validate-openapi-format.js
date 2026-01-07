#!/usr/bin/env node
// scripts/validate-openapi-format.js
// ✅ D8: Script pour valider le format de la spec (Swagger 2.0 vs OpenAPI 3)

const fs = require("fs");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const SPEC_FILE = path.join(PROJECT_ROOT, "backend", "docs", "openapi.json");

console.log("🔍 Validation du format de la spec OpenAPI/Swagger...");

if (!fs.existsSync(SPEC_FILE)) {
    console.error("❌ Erreur: " + SPEC_FILE + " introuvable");
    console.error("   Exécutez d'abord: npm run api:spec");
    process.exit(1);
}

try {
    const specContent = fs.readFileSync(SPEC_FILE, "utf8");
    const spec = JSON.parse(specContent);
    
    // Vérifier le format
    if (spec.swagger) {
        console.log("✅ Format détecté: Swagger " + spec.swagger);
        if (spec.swagger !== "2.0") {
            console.warn("⚠️  Version Swagger: " + spec.swagger + " (attendu: 2.0)");
        }
    } else if (spec.openapi) {
        console.log("✅ Format détecté: OpenAPI " + spec.openapi);
        if (!spec.openapi.startsWith("3.")) {
            console.warn("⚠️  Version OpenAPI: " + spec.openapi + " (attendu: 3.x)");
        }
    } else {
        console.error("❌ Format inconnu: ni 'swagger' ni 'openapi' trouvé");
        process.exit(1);
    }
    
    // Vérifier les champs requis
    if (!spec.paths || Object.keys(spec.paths).length === 0) {
        console.error("❌ Aucun endpoint défini dans 'paths'");
        process.exit(1);
    }
    
    console.log("✅ La spec est valide!");
    console.log("   - Format: " + (spec.swagger ? "Swagger " + spec.swagger : "OpenAPI " + spec.openapi));
    console.log("   - Endpoints: " + Object.keys(spec.paths).length);
    console.log("   - Base path: " + (spec.basePath || spec.servers?.[0]?.url || "N/A"));
    
    process.exit(0);
} catch (error) {
    console.error("❌ Erreur lors de la validation:", error.message);
    if (error instanceof SyntaxError) {
        console.error("   La spec JSON est invalide");
    }
    process.exit(1);
}

