#!/usr/bin/env node
// scripts/api-coverage-matrix.js
// ✅ Tâche 5: Script pour générer une matrice de couverture "calls web/mobile" vs "swagger paths"

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const SPEC_FILE = path.join(PROJECT_ROOT, "backend", "docs", "openapi.json");
const FRONTEND_DIR = path.join(PROJECT_ROOT, "frontend", "src");
const MOBILE_DIR = path.join(PROJECT_ROOT, "mobile", "operations-app");

console.log("📊 Génération de la matrice de couverture API...");

// Charger la spec OpenAPI
if (!fs.existsSync(SPEC_FILE)) {
    console.error("❌ Erreur: " + SPEC_FILE + " introuvable");
    console.error("   Exécutez d'abord: npm run api:spec");
    process.exit(1);
}

const spec = JSON.parse(fs.readFileSync(SPEC_FILE, "utf8"));

// Extraire tous les endpoints de la spec
const swaggerPaths = new Set();
for (const [path, methods] of Object.entries(spec.paths || {})) {
    for (const method of Object.keys(methods)) {
        if (["get", "post", "put", "delete", "patch"].includes(method.toLowerCase())) {
            swaggerPaths.add(`${method.toUpperCase()} ${path}`);
        }
    }
}

// Extraire les appels API depuis le frontend web
const frontendCalls = new Set();
const extractApiCalls = (dir, prefix = "") => {
    if (!fs.existsSync(dir)) {
        return;
    }
    
    const files = fs.readdirSync(dir, { withFileTypes: true });
    for (const file of files) {
        const fullPath = path.join(dir, file.name);
        
        if (file.isDirectory()) {
            // Ignorer node_modules et autres répertoires à ignorer
            if (!file.name.startsWith(".") && file.name !== "node_modules" && file.name !== "generated") {
                extractApiCalls(fullPath, prefix);
            }
        } else if (file.isFile() && (file.name.endsWith(".js") || file.name.endsWith(".jsx") || file.name.endsWith(".ts") || file.name.endsWith(".tsx"))) {
            try {
                const content = fs.readFileSync(fullPath, "utf8");
                
                // Chercher les patterns d'appels API
                // Pattern 1: axios.get('/api/v1/...')
                const axiosPattern = /axios\.(get|post|put|delete|patch)\(['"`]([^'"`]+)['"`]/gi;
                let match;
                while ((match = axiosPattern.exec(content)) !== null) {
                    const method = match[1].toUpperCase();
                    const url = match[2];
                    if (url.startsWith("/api/v1/")) {
                        frontendCalls.add(`${method} ${url}`);
                    }
                }
                
                // Pattern 2: apiClient.get('/api/v1/...')
                const apiClientPattern = /apiClient\.(get|post|put|delete|patch)\(['"`]([^'"`]+)['"`]/gi;
                while ((match = apiClientPattern.exec(content)) !== null) {
                    const method = match[1].toUpperCase();
                    const url = match[2];
                    if (url.startsWith("/api/v1/")) {
                        frontendCalls.add(`${method} ${url}`);
                    }
                }
                
                // Pattern 3: fetch('/api/v1/...', { method: 'GET' })
                const fetchPattern = /fetch\(['"`]([^'"`]+)['"`]\s*,\s*\{[^}]*method:\s*['"`]([^'"`]+)['"`]/gi;
                while ((match = fetchPattern.exec(content)) !== null) {
                    const method = match[2].toUpperCase();
                    const url = match[1];
                    if (url.startsWith("/api/v1/")) {
                        frontendCalls.add(`${method} ${url}`);
                    }
                }
            } catch (error) {
                // Ignorer les erreurs de lecture
            }
        }
    }
};

console.log("🔍 Analyse du frontend web...");
extractApiCalls(FRONTEND_DIR);

// Extraire les appels API depuis le mobile
const mobileCalls = new Set();
console.log("🔍 Analyse du mobile...");
extractApiCalls(MOBILE_DIR);

// Normaliser les chemins pour la comparaison
const normalizePath = (pathStr) => {
    // Retirer les paramètres de requête
    let normalized = pathStr.split("?")[0];
    // Retirer les paramètres de chemin {id} -> :id
    normalized = normalized.replace(/\{([^}]+)\}/g, ":$1");
    // Normaliser les slashes
    normalized = normalized.replace(/\/+/g, "/");
    return normalized;
};

// Comparer les appels avec les endpoints de la spec
const normalizeSwaggerPath = (pathStr) => {
    // Retirer le préfixe /api/v1 si présent
    let normalized = pathStr.replace(/^\/api\/v1/, "");
    // Normaliser les paramètres {id} -> :id
    normalized = normalized.replace(/\{([^}]+)\}/g, ":$1");
    // Normaliser les slashes
    normalized = normalized.replace(/\/+/g, "/");
    return normalized;
};

// Créer la matrice
const matrix = {
    swaggerOnly: [],
    frontendOnly: [],
    mobileOnly: [],
    both: [],
    covered: [],
};

// Endpoints dans la spec mais non utilisés
for (const swaggerPath of swaggerPaths) {
    const [method, path] = swaggerPath.split(" ", 2);
    const normalizedSwagger = normalizeSwaggerPath(path);
    
    const inFrontend = Array.from(frontendCalls).some((call) => {
        const [callMethod, callPath] = call.split(" ", 2);
        return callMethod === method && normalizePath(callPath.replace("/api/v1", "")) === normalizedSwagger;
    });
    
    const inMobile = Array.from(mobileCalls).some((call) => {
        const [callMethod, callPath] = call.split(" ", 2);
        return callMethod === method && normalizePath(callPath.replace("/api/v1", "")) === normalizedSwagger;
    });
    
    if (!inFrontend && !inMobile) {
        matrix.swaggerOnly.push(swaggerPath);
    } else if (inFrontend && inMobile) {
        matrix.both.push(swaggerPath);
        matrix.covered.push(swaggerPath);
    } else if (inFrontend) {
        matrix.covered.push(swaggerPath);
    } else if (inMobile) {
        matrix.covered.push(swaggerPath);
    }
}

// Appels dans le frontend mais absents de la spec
for (const call of frontendCalls) {
    const [method, callPath] = call.split(" ", 2);
    const normalizedCall = normalizePath(callPath.replace("/api/v1", ""));
    
    const inSwagger = Array.from(swaggerPaths).some((swaggerPath) => {
        const [swaggerMethod, swaggerPathStr] = swaggerPath.split(" ", 2);
        return swaggerMethod === method && normalizeSwaggerPath(swaggerPathStr) === normalizedCall;
    });
    
    if (!inSwagger) {
        matrix.frontendOnly.push(call);
    }
}

// Appels dans le mobile mais absents de la spec
for (const call of mobileCalls) {
    const [method, callPath] = call.split(" ", 2);
    const normalizedCall = normalizePath(callPath.replace("/api/v1", ""));
    
    const inSwagger = Array.from(swaggerPaths).some((swaggerPath) => {
        const [swaggerMethod, swaggerPathStr] = swaggerPath.split(" ", 2);
        return swaggerMethod === method && normalizeSwaggerPath(swaggerPathStr) === normalizedCall;
    });
    
    if (!inSwagger) {
        matrix.mobileOnly.push(call);
    }
}

// Générer le rapport
const reportPath = path.join(PROJECT_ROOT, "docs", "API_COVERAGE_MATRIX.md");
const reportDir = path.dirname(reportPath);
if (!fs.existsSync(reportDir)) {
    fs.mkdirSync(reportDir, { recursive: true });
}

const coveragePercent = ((matrix.covered.length / swaggerPaths.size) * 100).toFixed(1);

const report = `# Matrice de Couverture API

Généré le: ${new Date().toISOString()}

## Résumé

- **Endpoints Swagger**: ${swaggerPaths.size}
- **Endpoints couverts**: ${matrix.covered.length} (${coveragePercent}%)
- **Endpoints non utilisés**: ${matrix.swaggerOnly.length}
- **Appels frontend non documentés**: ${matrix.frontendOnly.length}
- **Appels mobile non documentés**: ${matrix.mobileOnly.length}
- **Endpoints utilisés par les deux**: ${matrix.both.length}

## Endpoints Swagger non utilisés (${matrix.swaggerOnly.length})

${matrix.swaggerOnly.length > 0 ? matrix.swaggerOnly.map((p) => `- \`${p}\``).join("\n") : "*Aucun*"}

## Appels Frontend non documentés (${matrix.frontendOnly.length})

${matrix.frontendOnly.length > 0 ? matrix.frontendOnly.map((p) => `- \`${p}\``).join("\n") : "*Aucun*"}

## Appels Mobile non documentés (${matrix.mobileOnly.length})

${matrix.mobileOnly.length > 0 ? matrix.mobileOnly.map((p) => `- \`${p}\``).join("\n") : "*Aucun*"}

## Endpoints utilisés par Frontend ET Mobile (${matrix.both.length})

${matrix.both.length > 0 ? matrix.both.map((p) => `- \`${p}\``).join("\n") : "*Aucun*"}

## Recommandations

1. **Endpoints non utilisés**: Vérifier s'ils sont encore nécessaires ou s'ils peuvent être supprimés
2. **Appels non documentés**: Ajouter ces endpoints à la spec Swagger
3. **Endpoints utilisés par les deux**: S'assurer qu'ils sont bien testés et documentés
`;

fs.writeFileSync(reportPath, report, "utf8");

console.log("✅ Matrice de couverture générée:");
console.log(`   - Fichier: ${reportPath}`);
console.log(`   - Couverture: ${coveragePercent}%`);
console.log(`   - Endpoints non utilisés: ${matrix.swaggerOnly.length}`);
console.log(`   - Appels non documentés: ${matrix.frontendOnly.length + matrix.mobileOnly.length}`);

