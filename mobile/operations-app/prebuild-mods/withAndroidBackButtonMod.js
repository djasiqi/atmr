/**
 * Mod Prebuild pour préserver le comportement personnalisé du bouton retour
 * dans MainActivity.kt. Ce mod ajoute la méthode invokeDefaultOnBackPressed()
 * qui gère correctement le comportement du bouton retour sur Android S et versions antérieures.
 */
const { withMainActivity, withDangerousMod } = require('expo/config-plugins');
const fs = require('fs');
const path = require('path');

module.exports = function withAndroidBackButtonMod(config) {
  // Utiliser withDangerousMod pour modifier le fichier MainActivity.kt après Prebuild
  config = withDangerousMod(config, [
    'android',
    async (config) => {
      const mainActivityPath = path.join(
        config.modRequest.platformProjectRoot,
        'app',
        'src',
        'main',
        'java',
        config.android?.package?.replace(/\./g, '/') || 'ch/liri/operations',
        'MainActivity.kt'
      );

      // Vérifier si le fichier existe
      if (fs.existsSync(mainActivityPath)) {
        let contents = fs.readFileSync(mainActivityPath, 'utf-8');

        // Si la personnalisation n'existe pas déjà, l'ajouter
        if (!contents.includes('invokeDefaultOnBackPressed')) {
          // Ajouter la méthode personnalisée avant la dernière accolade fermante
          const backButtonMethod = `
  /**
    * Align the back button behavior with Android S
    * where moving root activities to background instead of finishing activities.
    * @see <a href="https://developer.android.com/reference/android/app/Activity#onBackPressed()">onBackPressed</a>
    */
  override fun invokeDefaultOnBackPressed() {
      if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.R) {
          if (!moveTaskToBack(false)) {
              // For non-root activities, use the default implementation to finish them.
              super.invokeDefaultOnBackPressed()
          }
          return
      }

      // Use the default back button implementation on Android S
      // because it's doing more than [Activity.moveTaskToBack] in fact.
      super.invokeDefaultOnBackPressed()
  }`;

          // Trouver la fin de la classe (avant la dernière accolade fermante)
          const lastBraceIndex = contents.lastIndexOf('}');
          contents =
            contents.substring(0, lastBraceIndex) +
            backButtonMethod +
            '\n' +
            contents.substring(lastBraceIndex);

          fs.writeFileSync(mainActivityPath, contents, 'utf-8');
        }
      }

      return config;
    },
  ]);

  return config;
};
