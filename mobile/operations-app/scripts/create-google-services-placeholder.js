/**
 * Crée un google-services.json minimal pour le prebuild local.
 * À remplacer par le fichier réel depuis Firebase Console pour les push notifications.
 *
 * Usage: node scripts/create-google-services-placeholder.js
 */
const fs = require("fs");
const path = require("path");

const OUT_PATH = path.join(__dirname, "..", "google-services.json");

const PLACEHOLDER = {
  project_info: {
    project_number: "000000000000",
    project_id: "liri-operations-placeholder",
    storage_bucket: "liri-operations-placeholder.appspot.com",
  },
  client: [
    {
      client_info: {
        mobilesdk_app_id: "1:000000000000:android:0000000000000000000000",
        android_client_info: {
          package_name: "ch.liri.operations.dev",
        },
      },
      oauth_client: [],
      api_key: [{ current_key: "placeholder-key-replace-with-real-from-firebase" }],
      services: { appinvite_service: { other_platform_oauth_client: [] } },
    },
    {
      client_info: {
        mobilesdk_app_id: "1:000000000000:android:0000000000000000000001",
        android_client_info: {
          package_name: "ch.liri.operations",
        },
      },
      oauth_client: [],
      api_key: [{ current_key: "placeholder-key-replace-with-real-from-firebase" }],
      services: { appinvite_service: { other_platform_oauth_client: [] } },
    },
  ],
};

function main() {
  if (fs.existsSync(OUT_PATH)) {
    console.log("google-services.json existe déjà. Supprimez-le pour régénérer le placeholder.");
    return;
  }
  fs.writeFileSync(OUT_PATH, JSON.stringify(PLACEHOLDER, null, 2), "utf8");
  console.log("✅ google-services.json créé (placeholder).");
  console.log("   Pour les push notifications, remplacez par le fichier depuis Firebase Console :");
  console.log("   https://console.firebase.google.com → Votre projet → Paramètres → Comptes de service");
}

main();
