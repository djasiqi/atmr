/**
 * Démarre Expo en mode tunnel (contourne pare-feu/réseau).
 * Déconnecte d'abord l'émulateur fantôme pour éviter l'erreur ADB.
 */
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

function disconnectGhostEmulator() {
  const adb = path.join(process.env.LOCALAPPDATA || "", "Android", "Sdk", "platform-tools", "adb.exe");
  if (fs.existsSync(adb)) {
    try {
      require("child_process").execSync(`"${adb}" disconnect emulator-5554`, { stdio: "ignore" });
    } catch {}
  }
}

disconnectGhostEmulator();
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
      // Prevent ADB from auto-registering stale emulator-5554 entries on Windows.
      ADB_LOCAL_TRANSPORT_MAX_PORT: "0",
    },
  }
);

child.on("exit", (code) => process.exit(code ?? 0));
