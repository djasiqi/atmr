const { withDangerousMod } = require("expo/config-plugins");
const fs = require("fs");
const path = require("path");

/**
 * Adds :modular_headers => true for GoogleUtilities and FirebaseCoreInternal
 * in the Podfile. Required because FirebaseCoreInternal (Swift) depends on
 * GoogleUtilities (ObjC) which doesn't generate module maps by default when
 * RNFB pods are force-linked statically (expo-build-properties).
 */
module.exports = function withIosFirebaseModularHeaders(config) {
  return withDangerousMod(config, [
    "ios",
    async (config) => {
      const podfilePath = path.join(config.modRequest.platformProjectRoot, "Podfile");
      let content = fs.readFileSync(podfilePath, "utf-8");

      const marker = "# [Firebase] modular headers for Swift interop";
      if (content.includes(marker)) return config;

      const patch = [
        `  ${marker}`,
        "  pod 'GoogleUtilities', :modular_headers => true",
        "  pod 'FirebaseCoreInternal', :modular_headers => true",
      ].join("\n");

      content = content.replace(/(use_expo_modules!\n)/, `$1${patch}\n`);
      fs.writeFileSync(podfilePath, content);
      return config;
    },
  ]);
};
