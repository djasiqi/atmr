/**
 * Mod Prebuild pour ajouter la couleur iconBackground dans colors.xml.
 * Requise par les icônes adaptatives (ic_launcher.xml, ic_launcher_round.xml).
 */
const { withDangerousMod } = require("expo/config-plugins");
const fs = require("fs");
const path = require("path");

function withAndroidIconBackground(config) {
  return withDangerousMod(config, [
    "android",
    async (config) => {
      const colorsPath = path.join(
        config.modRequest.platformProjectRoot,
        "app/src/main/res/values/colors.xml"
      );

      if (!fs.existsSync(colorsPath)) {
        console.warn("[withAndroidIconBackground] colors.xml not found, skipping");
        return config;
      }

      let contents = fs.readFileSync(colorsPath, "utf8");

      if (contents.includes('name="iconBackground"')) {
        return config;
      }

      const insertAfter = "<resources>";
      const colorEntry = '  <color name="iconBackground">#FFFFFF</color>\n';
      contents = contents.replace(insertAfter, insertAfter + "\n" + colorEntry);

      fs.writeFileSync(colorsPath, contents);
      return config;
    },
  ]);
}

module.exports = withAndroidIconBackground;
