#!/usr/bin/env node

/**
 * Script pour mettre à jour glob vers la version sécurisée (>= 11.1.0)
 * et vérifier que toutes les dépendances utilisent la version corrigée.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔒 Mise à jour de glob pour corriger la vulnérabilité GHSA-ww39-953v-wcq6\n');

try {
  // Vérifier que package.json contient l'override
  const packageJsonPath = path.join(__dirname, '..', 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  
  if (!packageJson.overrides || !packageJson.overrides.glob) {
    console.error('❌ Erreur: L\'override pour glob n\'est pas configuré dans package.json');
    process.exit(1);
  }
  
  const globVersion = packageJson.overrides.glob;
  console.log(`✓ Override configuré: glob@${globVersion}\n`);
  
  // Installer les dépendances pour appliquer l'override
  console.log('📦 Installation des dépendances avec l\'override...');
  execSync('npm install', { 
    stdio: 'inherit',
    cwd: path.join(__dirname, '..')
  });
  
  // Vérifier les versions installées
  console.log('\n🔍 Vérification des versions de glob installées...');
  const output = execSync('npm list glob --depth=10', { 
    encoding: 'utf8',
    cwd: path.join(__dirname, '..')
  });
  
  // Vérifier s'il y a des versions vulnérables
  const vulnerableVersions = output.match(/glob@(10\.[0-4]\.|11\.0\.)/g);
  if (vulnerableVersions) {
    console.warn('\n⚠️  Attention: Des versions vulnérables de glob sont encore présentes:');
    vulnerableVersions.forEach(v => console.warn(`   - ${v}`));
    console.warn('\n💡 Essayez de supprimer node_modules et package-lock.json, puis réexécutez npm install');
  } else {
    console.log('\n✅ Toutes les versions de glob sont sécurisées (>= 10.5.0 ou >= 11.1.0)');
  }
  
  // Audit de sécurité
  console.log('\n🔒 Exécution de l\'audit de sécurité npm...');
  try {
    execSync('npm audit --audit-level=moderate', { 
      stdio: 'inherit',
      cwd: path.join(__dirname, '..')
    });
  } catch (error) {
    // npm audit peut échouer s'il y a des vulnérabilités, c'est normal
    console.log('\n💡 Utilisez "npm audit fix" pour corriger automatiquement les vulnérabilités');
  }
  
  console.log('\n✅ Mise à jour terminée!');
  
} catch (error) {
  console.error('\n❌ Erreur lors de la mise à jour:', error.message);
  process.exit(1);
}

