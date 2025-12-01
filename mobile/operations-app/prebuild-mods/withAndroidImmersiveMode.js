/**
 * Mod Prebuild pour activer le mode immersif sur Android
 * Cache la barre de navigation système pour éviter qu'elle recouvre la tabBar
 */
const { withMainActivity, withDangerousMod } = require('expo/config-plugins');
const fs = require('fs');
const path = require('path');

module.exports = function withAndroidImmersiveMode(config) {
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

        // Ajouter les imports nécessaires si pas déjà présents
        if (!contents.includes('android.view.View') && !contents.includes('View.SYSTEM_UI_FLAG_')) {
          // Trouver où insérer les imports (après le dernier import ou après package)
          let insertIndex = contents.lastIndexOf('import ');
          if (insertIndex === -1) {
            // Si pas d'imports, insérer après package
            const packageIndex = contents.indexOf('package ');
            if (packageIndex !== -1) {
              insertIndex = contents.indexOf('\n', packageIndex);
            }
          } else {
            // Trouver la fin de la dernière ligne d'import
            insertIndex = contents.indexOf('\n', insertIndex);
          }
          
          if (insertIndex !== -1) {
            const importStatement = `import android.view.View\nimport android.os.Build\n`;
            contents = 
              contents.substring(0, insertIndex + 1) +
              importStatement +
              contents.substring(insertIndex + 1);
          }
        }

        // Ajouter la méthode pour activer le mode immersif si pas déjà présente
        if (!contents.includes('enableImmersiveMode')) {
          const immersiveMethod = `
  /**
   * Active le mode immersif pour cacher la barre de navigation système
   * et éviter qu'elle recouvre la tabBar de l'application
   */
  private fun enableImmersiveMode() {
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
            or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
            or View.SYSTEM_UI_FLAG_FULLSCREEN
            or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        )
      }
  }

  override fun onResume() {
      super.onResume()
      enableImmersiveMode()
  }

  override fun onWindowFocusChanged(hasFocus: Boolean) {
      super.onWindowFocusChanged(hasFocus)
      if (hasFocus) {
          enableImmersiveMode()
      }
  }`;

          // Trouver la fin de la classe (avant la dernière accolade fermante)
          const lastBraceIndex = contents.lastIndexOf('}');
          contents =
            contents.substring(0, lastBraceIndex) +
            immersiveMethod +
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

