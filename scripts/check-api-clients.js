#!/usr/bin/env node
// scripts/check-api-clients.js
// ✅ Tâche 2: Script pour vérifier que les clients générés sont à jour

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

// Générer les clients dans un répertoire temporaire
const TEMP_DIR = path.join(PROJECT_ROOT, ".tmp", "api-clients-check");
const FRONTEND_TEMP = path.join(TEMP_DIR, "frontend");
const MOBILE_TEMP = path.join(TEMP_DIR, "mobile");

console.log("🔍 Vérification que les clients générés sont à jour...");

// Nettoyer le répertoire temporaire
if (fs.existsSync(TEMP_DIR)) {
    fs.rmSync(TEMP_DIR, { recursive: true, force: true });
}
fs.mkdirSync(TEMP_DIR, { recursive: true });

try {
    // Générer les clients dans le répertoire temporaire
    const { execSync } = require("child_process");
    
    console.log("📦 Génération temporaire des clients pour comparaison...");
    
    // Frontend
    execSync(
        `openapi-generator-cli generate ` +
        `-i "${SPEC_FILE}" ` +
        `-g typescript-axios ` +
        `-o "${FRONTEND_TEMP}" ` +
        `--additional-properties=supportsES6=true,withInterfaces=true,typescriptThreePlus=true`,
        { stdio: "pipe" }
    );
    
    // Mobile
    execSync(
        `openapi-generator-cli generate ` +
        `-i "${SPEC_FILE}" ` +
        `-g typescript-axios ` +
        `-o "${MOBILE_TEMP}" ` +
        `--additional-properties=supportsES6=true,withInterfaces=true,typescriptThreePlus=true`,
        { stdio: "pipe" }
    );
    
    // Comparer les fichiers générés avec ceux existants
    const compareDirectories = (dir1, dir2, name) => {
        if (!fs.existsSync(dir1)) {
            console.error(`❌ ${name}: Répertoire généré introuvable: ${dir1}`);
            return false;
        }
        if (!fs.existsSync(dir2)) {
            console.error(`❌ ${name}: Répertoire existant introuvable: ${dir2}`);
            console.error(`   Exécutez: npm run api:generate`);
            return false;
        }
        
        const files1 = getAllFiles(dir1);
        const files2 = getAllFiles(dir2);
        
        if (files1.length !== files2.length) {
            console.error(`❌ ${name}: Nombre de fichiers différent (${files1.length} vs ${files2.length})`);
            return false;
        }
        
        let hasDiff = false;
        for (const file of files1) {
            const relPath = path.relative(dir1, file);
            const file2 = path.join(dir2, relPath);
            
            if (!fs.existsSync(file2)) {
                console.error(`❌ ${name}: Fichier manquant: ${relPath}`);
                hasDiff = true;
                continue;
            }
            
            const content1 = fs.readFileSync(file, "utf8");
            const content2 = fs.readFileSync(file2, "utf8");
            
            if (content1 !== content2) {
                console.error(`❌ ${name}: Fichier modifié: ${relPath}`);
                hasDiff = true;
            }
        }
        
        return !hasDiff;
    };
    
    const getAllFiles = (dir) => {
        const files = [];
        const items = fs.readdirSync(dir);
        for (const item of items) {
            const fullPath = path.join(dir, item);
            const stat = fs.statSync(fullPath);
            if (stat.isDirectory()) {
                files.push(...getAllFiles(fullPath));
            } else {
                files.push(fullPath);
            }
        }
        return files;
    };
    
    const frontendOk = compareDirectories(FRONTEND_TEMP, FRONTEND_OUTPUT, "Frontend");
    const mobileOk = compareDirectories(MOBILE_TEMP, MOBILE_OUTPUT, "Mobile");
    
    // Nettoyer le répertoire temporaire
    fs.rmSync(TEMP_DIR, { recursive: true, force: true });
    
    if (frontendOk && mobileOk) {
        console.log("✅ Les clients générés sont à jour!");
        process.exit(0);
    } else {
        console.error("❌ Les clients générés ne sont pas à jour!");
        console.error("   Exécutez: npm run api:generate");
        process.exit(1);
    }
} catch (error) {
    // Nettoyer le répertoire temporaire en cas d'erreur
    if (fs.existsSync(TEMP_DIR)) {
        fs.rmSync(TEMP_DIR, { recursive: true, force: true });
    }
    console.error("❌ Erreur lors de la vérification:", error.message);
    process.exit(1);
}

