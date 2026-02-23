const { withAppBuildGradle } = require("expo/config-plugins");

const MARKER_BEGIN = "// --- BEGIN LIRI MANIFEST PATCH (auto) ---";
const MARKER_END = "// --- END LIRI MANIFEST PATCH (auto) ---";

const GRADLE_SNIPPET = `
${MARKER_BEGIN}
def patchAndroidManifestForToolsReplace = { File manifestFile ->
    if (!manifestFile.exists()) {
        println("[liri-manifest-patch] Manifest not found: " + manifestFile)
        return
    }

    def text = manifestFile.getText("UTF-8")
    def changed = false

    if (!text.contains("xmlns:tools=\\"http://schemas.android.com/tools\\"")) {
        text = text.replaceFirst(/<manifest\\b/, "<manifest xmlns:tools=\\"http://schemas.android.com/tools\\"")
        changed = true
    }

    def META_NAME = "com.google.firebase.messaging.default_notification_color"
    def metaDataPattern = /<meta-data\\b([^>]*\\bandroid:name\\s*=\\s*\\"${"$"}{META_NAME}\\"[^>]*)>/
    def matcher = (text =~ metaDataPattern)
    if (matcher.find()) {
        def fullStartTag = matcher.group(0)
        if (!fullStartTag.contains("tools:replace=\\"android:resource\\"")) {
            def patchedStartTag = fullStartTag.replaceFirst(/<meta-data\\b/, "<meta-data tools:replace=\\"android:resource\\"")
            text = text.replace(fullStartTag, patchedStartTag)
            changed = true
        }
    } else {
        println("[liri-manifest-patch] meta-data not found for name=" + META_NAME)
    }

    if (changed) {
        manifestFile.write(text, "UTF-8")
        println("[liri-manifest-patch] Manifest patched: " + manifestFile)
    } else {
        println("[liri-manifest-patch] No changes needed.")
    }
}

afterEvaluate {
    tasks.matching { t ->
        t.name.startsWith("process") && t.name.endsWith("MainManifest")
    }.configureEach { task ->
        task.doFirst {
            def manifestFile = file("${"$"}projectDir/src/main/AndroidManifest.xml")
            patchAndroidManifestForToolsReplace(manifestFile)
        }
    }
}
${MARKER_END}
`;

function withAndroidNotificationColorFix(config) {
  return withAppBuildGradle(config, (config) => {
    if (config.modResults.language !== "groovy") {
      throw new Error(
        "Only Groovy build.gradle is supported by this plugin snippet.",
      );
    }

    let contents = config.modResults.contents;

    const beginIdx = contents.indexOf(MARKER_BEGIN);
    const endIdx = contents.indexOf(MARKER_END);
    if (beginIdx !== -1 && endIdx !== -1 && endIdx > beginIdx) {
      contents =
        contents.slice(0, beginIdx) +
        contents.slice(endIdx + MARKER_END.length);
    }

    contents = `${contents}\n\n${GRADLE_SNIPPET}\n`;
    config.modResults.contents = contents;
    return config;
  });
}

module.exports = withAndroidNotificationColorFix;
