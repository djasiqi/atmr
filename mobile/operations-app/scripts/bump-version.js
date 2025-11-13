const fs = require("fs");
const path = require("path");

const CONFIG_PATH = path.join(__dirname, "..", "app.config.js");
const PACKAGE_PATH = path.join(__dirname, "..", "package.json");

function bumpSemanticVersion(version) {
  const parts = version.split(".").map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) {
    throw new Error(`Version invalide: ${version}`);
  }
  parts[2] += 1; // Incrément patch
  return parts.join(".");
}

function updateAppConfig(newVersion, newVersionCode) {
  let content = fs.readFileSync(CONFIG_PATH, "utf8");

  const versionRegex = /(version:\s*")([0-9]+\.[0-9]+\.[0-9]+)(")/;
  if (!versionRegex.test(content)) {
    throw new Error("Impossible de trouver version dans app.config.js");
  }
  content = content.replace(versionRegex, `$1${newVersion}$3`);

  const versionCodeRegex = /(versionCode:\s*)([0-9]+)/;
  if (!versionCodeRegex.test(content)) {
    throw new Error("Impossible de trouver versionCode dans app.config.js");
  }
  content = content.replace(versionCodeRegex, `$1${newVersionCode}`);

  fs.writeFileSync(CONFIG_PATH, content);
}

function updatePackageJson(newVersion) {
  const pkg = JSON.parse(fs.readFileSync(PACKAGE_PATH, "utf8"));
  pkg.version = newVersion;
  fs.writeFileSync(PACKAGE_PATH, JSON.stringify(pkg, null, 2) + "\n");
}

function main() {
  const config = require(CONFIG_PATH)();
  const currentVersion = config.version;
  const currentVersionCode = config.android?.versionCode;

  if (typeof currentVersionCode !== "number") {
    throw new Error("versionCode invalide dans app.config.js");
  }

  const newVersion = bumpSemanticVersion(currentVersion);
  const newVersionCode = currentVersionCode + 1;

  updateAppConfig(newVersion, newVersionCode);
  updatePackageJson(newVersion);

  console.log(`Version mise à jour: ${currentVersion} → ${newVersion}`);
  console.log(`versionCode mis à jour: ${currentVersionCode} → ${newVersionCode}`);
}

main();
