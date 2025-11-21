#!/usr/bin/env node

/**
 * Script de vérification avant un build de production EAS
 * Vérifie que tous les éléments nécessaires sont en place
 */

const fs = require('fs');
const path = require('path');

const errors = [];
const warnings = [];
const checks = [];

// Couleurs pour la console
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  blue: '\x1b[34m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function checkFile(filePath, description, required = true) {
  const fullPath = path.join(__dirname, '..', filePath);
  const exists = fs.existsSync(fullPath);
  
  if (exists) {
    checks.push(`✅ ${description}`);
    return true;
  } else {
    const message = `${required ? '❌' : '⚠️'} ${description} ${required ? '(requis)' : '(optionnel)'}`;
    if (required) {
      errors.push(message);
    } else {
      warnings.push(message);
    }
    return false;
  }
}

function checkPackageJson() {
  const pkgPath = path.join(__dirname, '..', 'package.json');
  if (!fs.existsSync(pkgPath)) {
    errors.push('❌ package.json introuvable');
    return;
  }

  const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  
  if (!pkg.version || pkg.version === '0.0.0') {
    warnings.push('⚠️ Version dans package.json non définie ou invalide');
  } else {
    checks.push(`✅ Version de l'application: ${pkg.version}`);
  }

  if (!pkg.scripts || !pkg.scripts['build:prod']) {
    warnings.push('⚠️ Script build:prod non trouvé dans package.json');
  } else {
    checks.push('✅ Script build:prod présent');
  }
}

function checkEasConfig() {
  const easPath = path.join(__dirname, '..', 'eas.json');
  if (!fs.existsSync(easPath)) {
    errors.push('❌ eas.json introuvable');
    return;
  }

  const eas = JSON.parse(fs.readFileSync(easPath, 'utf8'));
  
  if (!eas.build || !eas.build.production) {
    errors.push('❌ Profil de production non trouvé dans eas.json');
  } else {
    checks.push('✅ Profil de production configuré dans eas.json');
    
    if (eas.build.production.android && eas.build.production.android.buildType === 'app-bundle') {
      checks.push('✅ Build Android configuré pour App Bundle');
    }
    
    if (eas.build.production.ios) {
      checks.push('✅ Configuration iOS présente');
    }
  }
}

function checkAppConfig() {
  const appConfigPath = path.join(__dirname, '..', 'app.config.js');
  if (!fs.existsSync(appConfigPath)) {
    errors.push('❌ app.config.js introuvable');
    return;
  }

  checks.push('✅ app.config.js présent');
  
  // Vérifier que les assets existent
  const configContent = fs.readFileSync(appConfigPath, 'utf8');
  
  if (configContent.includes('./assets/images/icon.png')) {
    checkFile('assets/images/icon.png', 'Icône de l\'application', true);
  }
  
  if (configContent.includes('./assets/images/splash-icon.png')) {
    checkFile('assets/images/splash-icon.png', 'Image de splash screen', true);
  }
}

function checkEnvExample() {
  const envExamplePath = path.join(__dirname, '..', 'env.example');
  if (fs.existsSync(envExamplePath)) {
    checks.push('✅ env.example présent');
    
    const envContent = fs.readFileSync(envExamplePath, 'utf8');
    const requiredVars = [
      'APP_VARIANT',
      'EXPO_PUBLIC_API_URL',
      'EXPO_PUBLIC_ANDROID_MAPS_API_KEY',
    ];
    
    requiredVars.forEach(varName => {
      if (envContent.includes(varName)) {
        checks.push(`✅ Variable ${varName} documentée dans env.example`);
      } else {
        warnings.push(`⚠️ Variable ${varName} non documentée dans env.example`);
      }
    });
  } else {
    warnings.push('⚠️ env.example introuvable');
  }
}

// Exécution des vérifications
log('\n🔍 Vérification de la préparation au build de production EAS\n', 'blue');

checkPackageJson();
checkEasConfig();
checkAppConfig();
checkEnvExample();

// Vérifications de fichiers
checkFile('eas.json', 'Fichier eas.json', true);
checkFile('app.config.js', 'Fichier app.config.js', true);
checkFile('package.json', 'Fichier package.json', true);
checkFile('google-services.json', 'Fichier google-services.json (Android)', false);
checkFile('GoogleService-Info.plist', 'Fichier GoogleService-Info.plist (iOS)', false);

// Affichage des résultats
log('\n📋 Résultats des vérifications:\n', 'blue');

if (checks.length > 0) {
  checks.forEach(check => log(check, 'green'));
}

if (warnings.length > 0) {
  log('\n⚠️ Avertissements:', 'yellow');
  warnings.forEach(warning => log(warning, 'yellow'));
}

if (errors.length > 0) {
  log('\n❌ Erreurs:', 'red');
  errors.forEach(error => log(error, 'red'));
  log('\n❌ Le build ne peut pas être effectué. Veuillez corriger les erreurs ci-dessus.\n', 'red');
  process.exit(1);
} else {
  log('\n✅ Toutes les vérifications critiques sont passées !', 'green');
  log('💡 N\'oubliez pas de configurer les secrets EAS avant de lancer le build:\n', 'blue');
  log('   eas secret:create --scope project --name EXPO_PUBLIC_API_URL --value https://api.lirie.ch');
  log('   eas secret:create --scope project --name EXPO_PUBLIC_ANDROID_MAPS_API_KEY --value YOUR_KEY');
  log('   eas secret:create --scope project --name APP_VARIANT --value prod');
  log('\n🚀 Vous pouvez maintenant lancer: eas build --platform android --profile production\n', 'green');
  process.exit(0);
}

