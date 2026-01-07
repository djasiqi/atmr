#!/usr/bin/env node
// scripts/detect-breaking-changes.js
// ✅ Tâche 3: Script pour détecter les breaking changes dans la spec OpenAPI

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const SPEC_FILE = path.join(PROJECT_ROOT, "backend", "docs", "openapi.json");
const SPEC_BASELINE = path.join(PROJECT_ROOT, "backend", "docs", "openapi.baseline.json");

console.log("🔍 Détection des breaking changes dans la spec OpenAPI...");

// Charger la spec actuelle
if (!fs.existsSync(SPEC_FILE)) {
    console.error("❌ Erreur: " + SPEC_FILE + " introuvable");
    console.error("   Exécutez d'abord: npm run api:spec");
    process.exit(1);
}

const currentSpec = JSON.parse(fs.readFileSync(SPEC_FILE, "utf8"));

// Si pas de baseline, créer une baseline avec la spec actuelle
if (!fs.existsSync(SPEC_BASELINE)) {
    console.log("📝 Création de la baseline (première exécution)...");
    fs.writeFileSync(SPEC_BASELINE, JSON.stringify(currentSpec, null, 2));
    console.log("✅ Baseline créée: " + SPEC_BASELINE);
    console.log("   Les prochaines exécutions détecteront les breaking changes");
    process.exit(0);
}

// Charger la baseline
const baselineSpec = JSON.parse(fs.readFileSync(SPEC_BASELINE, "utf8"));

const breakingChanges = [];
const warnings = [];

// Comparer les endpoints
const currentPaths = Object.keys(currentSpec.paths || {});
const baselinePaths = Object.keys(baselineSpec.paths || {});

// Endpoints supprimés
const removedPaths = baselinePaths.filter((p) => !currentPaths.includes(p));
removedPaths.forEach((path) => {
    breakingChanges.push(`Endpoint supprimé: ${path}`);
});

// Endpoints modifiés
for (const path of currentPaths) {
    if (!baselineSpec.paths[path]) {
        // Nouvel endpoint (pas un breaking change)
        continue;
    }
    
    const currentMethods = currentSpec.paths[path];
    const baselineMethods = baselineSpec.paths[path];
    
    // Méthodes supprimées
    const removedMethods = Object.keys(baselineMethods).filter(
        (m) => !currentMethods[m]
    );
    removedMethods.forEach((method) => {
        breakingChanges.push(`Méthode supprimée: ${method.toUpperCase()} ${path}`);
    });
    
    // Vérifier les changements de type/enum dans les paramètres et réponses
    for (const [method, operation] of Object.entries(currentMethods)) {
        if (!baselineMethods[method]) {
            continue; // Nouvelle méthode
        }
        
        const baselineOperation = baselineMethods[method];
        
        // Comparer les paramètres
        const currentParams = operation.parameters || [];
        const baselineParams = baselineOperation.parameters || [];
        
        for (const param of baselineParams) {
            const currentParam = currentParams.find((p) => p.name === param.name && p.in === param.in);
            if (!currentParam) {
                if (param.required) {
                    breakingChanges.push(`Paramètre requis supprimé: ${param.name} dans ${method.toUpperCase()} ${path}`);
                } else {
                    warnings.push(`Paramètre optionnel supprimé: ${param.name} dans ${method.toUpperCase()} ${path}`);
                }
            } else if (param.required && !currentParam.required) {
                warnings.push(`Paramètre requis devenu optionnel: ${param.name} dans ${method.toUpperCase()} ${path}`);
            } else if (!param.required && currentParam.required) {
                breakingChanges.push(`Paramètre optionnel devenu requis: ${param.name} dans ${method.toUpperCase()} ${path}`);
            }
            
            // Vérifier les changements de type
            if (currentParam && param.type !== currentParam.type) {
                breakingChanges.push(`Type de paramètre changé: ${param.name} (${param.type} → ${currentParam.type}) dans ${method.toUpperCase()} ${path}`);
            }
        }
        
        // Comparer les réponses
        const currentResponses = operation.responses || {};
        const baselineResponses = baselineOperation.responses || {};
        
        // Réponses supprimées
        for (const [status, response] of Object.entries(baselineResponses)) {
            if (!currentResponses[status]) {
                if (status === "200" || status === "201") {
                    breakingChanges.push(`Réponse ${status} supprimée dans ${method.toUpperCase()} ${path}`);
                } else {
                    warnings.push(`Réponse ${status} supprimée dans ${method.toUpperCase()} ${path}`);
                }
            }
        }
    }
}

// Afficher les résultats
if (breakingChanges.length > 0) {
    console.error("❌ Breaking changes détectés:");
    breakingChanges.forEach((change) => console.error("   - " + change));
    console.error("\n⚠️  Ces changements peuvent casser les clients existants!");
    process.exit(1);
}

if (warnings.length > 0) {
    console.warn("⚠️  Avertissements (changements non-breaking):");
    warnings.forEach((warn) => console.warn("   - " + warn));
}

if (breakingChanges.length === 0) {
    console.log("✅ Aucun breaking change détecté!");
    console.log(`   - ${currentPaths.length} endpoints (${baselinePaths.length} dans la baseline)`);
}

