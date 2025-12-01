const { withGradleProperties } = require('@expo/config-plugins');

/**
 * Plugin Expo pour activer R8/ProGuard dans les builds de production
 * Ajoute android.enableMinifyInReleaseBuilds=true dans gradle.properties
 * 
 * Ce plugin garantit que R8 (remplaçant moderne de ProGuard) est activé
 * pour réduire la taille de l'application et générer le fichier de mapping
 * nécessaire pour désobscurcir les stack traces dans Google Play Console.
 */
function withAndroidR8Enabled(config) {
  return withGradleProperties(config, (config) => {
    const properties = config.modResults;
    
    // Trouver ou créer la propriété android.enableMinifyInReleaseBuilds
    const existingIndex = properties.findIndex(
      (prop) => prop.key === 'android.enableMinifyInReleaseBuilds'
    );
    
    if (existingIndex >= 0) {
      // Mettre à jour la propriété existante
      properties[existingIndex].value = 'true';
    } else {
      // Ajouter la propriété
      properties.push({
        type: 'property',
        key: 'android.enableMinifyInReleaseBuilds',
        value: 'true',
      });
    }
    
    return config;
  });
}

module.exports = withAndroidR8Enabled;

