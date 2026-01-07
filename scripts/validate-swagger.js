#!/usr/bin/env node
// scripts/validate-swagger.js
// ✅ Tâche 3: Script pour valider la spec Swagger/OpenAPI

const fs = require("fs");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const SPEC_FILE = path.join(PROJECT_ROOT, "backend", "docs", "openapi.json");

// Vérifier que la spec existe
if (!fs.existsSync(SPEC_FILE)) {
    console.error("❌ Erreur: " + SPEC_FILE + " introuvable");
    console.error("   Exécutez d'abord: npm run api:spec");
    process.exit(1);
}

console.log("🔍 Validation de la spec Swagger/OpenAPI...");

try {
    const specContent = fs.readFileSync(SPEC_FILE, "utf8");
    const spec = JSON.parse(specContent);
    
    const errors = [];
    const warnings = [];
    
    // Vérifier la structure de base
    if (!spec.swagger && !spec.openapi) {
        errors.push("La spec doit être Swagger 2.0 ou OpenAPI 3.x");
    }
    
    if (spec.swagger && spec.swagger !== "2.0") {
        warnings.push(`Version Swagger: ${spec.swagger} (attendu: 2.0)`);
    }
    
    if (!spec.paths || Object.keys(spec.paths).length === 0) {
        errors.push("Aucun endpoint défini dans 'paths'");
    }
    
    // Vérifier que chaque endpoint a au moins une méthode HTTP
    for (const [path, methods] of Object.entries(spec.paths || {})) {
        if (typeof methods !== "object" || Object.keys(methods).length === 0) {
            errors.push(`Endpoint '${path}' n'a aucune méthode HTTP définie`);
        }
        
        // Vérifier que chaque méthode a des réponses
        for (const [method, operation] of Object.entries(methods)) {
            if (["get", "post", "put", "delete", "patch"].includes(method.toLowerCase())) {
                if (!operation.responses || Object.keys(operation.responses).length === 0) {
                    warnings.push(`Endpoint '${path}' (${method.toUpperCase()}) n'a aucune réponse définie`);
                }
                
                // Vérifier la présence de schémas pour les réponses 200
                if (operation.responses && operation.responses["200"]) {
                    const response200 = operation.responses["200"];
                    if (!response200.schema && !response200.content) {
                        warnings.push(`Endpoint '${path}' (${method.toUpperCase()}) - réponse 200 sans schéma`);
                    }
                }
            }
        }
    }
    
    // Afficher les résultats
    if (errors.length > 0) {
        console.error("❌ Erreurs de validation:");
        errors.forEach((err) => console.error("   - " + err));
    }
    
    if (warnings.length > 0) {
        console.warn("⚠️  Avertissements:");
        warnings.forEach((warn) => console.warn("   - " + warn));
    }
    
    if (errors.length === 0) {
        console.log("✅ La spec Swagger/OpenAPI est valide!");
        console.log(`   - ${Object.keys(spec.paths || {}).length} endpoints`);
        process.exit(0);
    } else {
        process.exit(1);
    }
} catch (error) {
    console.error("❌ Erreur lors de la validation:", error.message);
    if (error instanceof SyntaxError) {
        console.error("   La spec JSON est invalide");
    }
    process.exit(1);
}

