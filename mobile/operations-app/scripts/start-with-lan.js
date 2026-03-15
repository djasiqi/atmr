/**
 * Démarre Expo avec l'IP LAN pour que le QR code soit scannable depuis le téléphone.
 * Préfère 192.168.x.x et 10.x.x.x (Wi‑Fi/Ethernet) et ignore 172.16-31.x.x (Docker/WSL).
 * Déconnecte l'émulateur fantôme (emulator-5554 offline) pour éviter les erreurs ADB.
 */
const { spawn, execSync } = require("child_process");
const os = require("os");
const path = require("path");
const fs = require("fs");

function disconnectGhostEmulator() {
  const adb = process.platform === "win32"
    ? path.join(process.env.LOCALAPPDATA || "", "Android", "Sdk", "platform-tools", "adb.exe")
    : "adb";
  try {
    if (fs.existsSync(adb)) {
      execSync(`"${adb}" disconnect emulator-5554`, { stdio: "ignore" });
    }
  } catch {}
}

// Charger .env.development pour récupérer REACT_NATIVE_PACKAGER_HOSTNAME si défini
function loadEnvHost() {
  try {
    const envPath = path.join(__dirname, "..", ".env.development");
    if (fs.existsSync(envPath)) {
      const content = fs.readFileSync(envPath, "utf8");
      const match = content.match(/REACT_NATIVE_PACKAGER_HOSTNAME=(.+)/);
      if (match) {
        const val = match[1].trim().split("#")[0].trim();
        if (val && /^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(val)) return val;
      }
    }
  } catch {}
  return null;
}

function getLANIP() {
  const fromEnv = loadEnvHost();
  if (fromEnv) return fromEnv;

  const candidates = [];
  const nets = os.networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family !== "IPv4" || net.internal) continue;
      const a = net.address;
      const m = a.match(/^172\.(\d+)\./);
      const isDockerWsl = m && parseInt(m[1], 10) >= 16 && parseInt(m[1], 10) <= 31;
      if (a.startsWith("192.168.") || a.startsWith("10.")) {
        candidates.unshift(a);
      } else if (!isDockerWsl) {
        candidates.push(a);
      }
    }
  }
  return candidates[0] || null;
}

disconnectGhostEmulator();

const ip = getLANIP();
if (ip) {
  process.env.REACT_NATIVE_PACKAGER_HOSTNAME = ip;
  console.log(`[start] Using LAN IP for QR code: ${ip}`);
} else {
  console.warn("[start] No LAN IP found. Set REACT_NATIVE_PACKAGER_HOSTNAME in .env.development");
}

const child = spawn(
  "npx",
  ["expo", "start", "--host", "lan", ...process.argv.slice(2)],
  {
    stdio: "inherit",
    shell: true,
    env: { ...process.env, REACT_NATIVE_PACKAGER_HOSTNAME: ip || process.env.REACT_NATIVE_PACKAGER_HOSTNAME || "localhost" },
  }
);

child.on("exit", (code) => process.exit(code ?? 0));
