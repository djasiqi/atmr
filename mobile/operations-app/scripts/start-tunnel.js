/**
 * Démarre Expo en mode tunnel (contourne pare-feu/réseau).
 * Réinitialise ADB avant Expo : le tunnel appelle `adb reverse` / `emu avd name` pour chaque
 * série connue du serveur ; un `emulator-5554` fantôme (port 5554 fermé) fait échouer tout le démarrage.
 * Un `kill-server` vide la liste ; les appareils USB / réseau se reconnectent tout seuls au prochain `adb`.
 */
const { spawn, execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

function getAdbPath() {
  const fromEnv = process.env.ANDROID_HOME || process.env.ANDROID_SDK_ROOT;
  if (fromEnv) {
    const p = path.join(fromEnv, "platform-tools", process.platform === "win32" ? "adb.exe" : "adb");
    if (fs.existsSync(p)) return p;
  }
  const winDefault = path.join(process.env.LOCALAPPDATA || "", "Android", "Sdk", "platform-tools", "adb.exe");
  if (fs.existsSync(winDefault)) return winDefault;
  return null;
}

function resetAdbBeforeExpoTunnel() {
  if (process.env.EXPO_SKIP_ADB_RESET === "1") return;

  const adb = getAdbPath();
  if (!adb) return;

  const tryExec = (args, opts = {}) => {
    try {
      execSync(`"${adb}" ${args}`, { stdio: "ignore", ...opts });
    } catch {
      /* ignore */
    }
  };

  tryExec("disconnect emulator-5554");
  tryExec("disconnect 127.0.0.1:5554");
  tryExec("kill-server");
  tryExec("start-server");
}

resetAdbBeforeExpoTunnel();
console.log("[start:tunnel] Mode tunnel — le QR code fonctionnera même si le réseau local bloque.\n");

const child = spawn(
  "npx",
  ["expo", "start", "--tunnel", "--clear"],
  {
    stdio: "inherit",
    shell: true,
    env: {
      ...process.env,
      NODE_OPTIONS: "--max-old-space-size=8192",
    },
  }
);

child.on("exit", (code) => process.exit(code ?? 0));
