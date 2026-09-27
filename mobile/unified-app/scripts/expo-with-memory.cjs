/**
 * Relance Expo avec un tas Node plus grand.
 * Le bundle web entreprise dépasse la limite par défaut (~2 Go) et tue Metro.
 */
const { spawn } = require("node:child_process");
const path = require("node:path");

const HEAP_FLAG = "--max-old-space-size=8192";
const existing = process.env.NODE_OPTIONS ?? "";
const nodeOptions = existing.includes("max-old-space-size")
  ? existing
  : [existing, HEAP_FLAG].filter(Boolean).join(" ");

const expoCli = path.join(__dirname, "..", "node_modules", "expo", "bin", "cli");
const child = spawn(process.execPath, [expoCli, ...process.argv.slice(2)], {
  stdio: "inherit",
  env: { ...process.env, NODE_OPTIONS: nodeOptions },
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 1);
});
