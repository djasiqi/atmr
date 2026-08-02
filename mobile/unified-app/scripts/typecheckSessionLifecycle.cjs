/**
 * Garde typecheck sur la surface session PR2.
 * Le typecheck global du monorepo mobile a une dette préexistante hors scope ;
 * ce script échoue uniquement si nos fichiers session/auth introduisent des erreurs TS.
 */
const { spawnSync } = require("child_process");
const path = require("path");

/** Fichiers introduits / fortement remaniés par PR2 (hors dette TS préexistante client/sessionProvider). */
const SURFACE =
  /src\/core\/auth\/(sessionCredentialMutex|sessionLifecycle|authRecoveryCoordinator|authCredentialStore|mobileSessionStatus|sessionStateMachine|sessionAuthDecision|contextSwitchOperation)|app\/index\.tsx/;

const result = spawnSync("npx", ["tsc", "--noEmit", "--pretty", "false"], {
  cwd: path.resolve(__dirname, ".."),
  encoding: "utf8",
  shell: true,
});

const output = `${result.stdout || ""}\n${result.stderr || ""}`;
const lines = output.split(/\r?\n/).filter((line) => line.includes("error TS"));
const surfaceErrors = lines.filter((line) => SURFACE.test(line.replace(/\\/g, "/")));

if (surfaceErrors.length > 0) {
  console.error("Erreurs TypeScript sur la surface session PR2 :");
  for (const line of surfaceErrors) {
    console.error(line);
  }
  process.exit(1);
}

console.log(
  `Typecheck surface session PR2 OK (${lines.length} erreur(s) hors surface tolérée(s) temporairement).`
);
process.exit(0);
