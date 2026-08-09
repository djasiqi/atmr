/**
 * Garde typecheck sur la surface GPS / file / bridge / carte flotte.
 * Le typecheck global a une dette hors scope ; ce script échoue uniquement
 * si les fichiers tracking introduisent des erreurs TS.
 */
const { spawnSync } = require("child_process");
const path = require("path");

/** Surface GPS P0 — erreurs bloquantes uniquement sur ces chemins. */
const SURFACE =
  /src\/features\/driver\/services\/(driverTrackingQueue|trackingQueueStore|driverRealtimeBridge|socketBatchPacing)|src\/features\/company\/(realtime\/useCompanyDriverLiveTracking|components\/maps\/fleetMapStale|utils\/localDriverLocationFreshness)|src\/core\/featureFlags\/registry\.ts/;

const result = spawnSync("npx", ["tsc", "--noEmit", "--pretty", "false"], {
  cwd: path.resolve(__dirname, ".."),
  encoding: "utf8",
  shell: true,
});

const output = `${result.stdout || ""}\n${result.stderr || ""}`;
const lines = output.split(/\r?\n/).filter((line) => line.includes("error TS"));
const surfaceErrors = lines.filter((line) => SURFACE.test(line.replace(/\\/g, "/")));

if (surfaceErrors.length > 0) {
  console.error("Erreurs TypeScript sur la surface tracking GPS :");
  for (const line of surfaceErrors) {
    console.error(line);
  }
  process.exit(1);
}

console.log(
  `Typecheck surface tracking OK (${lines.length} erreur(s) hors surface tolérée(s) temporairement).`
);
process.exit(0);
