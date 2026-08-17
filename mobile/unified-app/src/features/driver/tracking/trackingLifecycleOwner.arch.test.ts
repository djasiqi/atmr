/**
 * D5 T8 — garde architecturale : aucun caller direct de
 * `stopBackgroundLocationTask` hors lifecycle owner / module task.
 */
import { describe, expect, it } from "@jest/globals";
import * as fs from "fs";
import * as path from "path";

const DRIVER_ROOT = path.resolve(__dirname, "..");

const ALLOWED_STOP_BACKGROUND = new Set([
  path.normalize("services/driverTrackingBridge.ts"),
  path.normalize("services/backgroundLocationTask.ts"),
]);

function walkTsFiles(dir: string, out: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (
      entry.name === "node_modules" ||
      entry.name === "dist" ||
      entry.name === "dist-web-test"
    ) {
      continue;
    }
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkTsFiles(full, out);
    } else if (/\.(ts|tsx)$/.test(entry.name) && !entry.name.endsWith(".d.ts")) {
      out.push(full);
    }
  }
  return out;
}

describe("D5 T8 — ownership STOP natif", () => {
  it("aucun import/appel stopBackgroundLocationTask hors allowlist", () => {
    const files = walkTsFiles(DRIVER_ROOT);
    const offenders: string[] = [];

    for (const file of files) {
      const rel = path.normalize(path.relative(DRIVER_ROOT, file));
      if (rel.includes(".test.")) continue;
      if (ALLOWED_STOP_BACKGROUND.has(rel)) continue;

      const text = fs.readFileSync(file, "utf8");
      if (text.includes("stopBackgroundLocationTask")) {
        offenders.push(rel);
      }
    }

    expect(offenders).toEqual([]);
  });

  it("Location.stopLocationUpdatesAsync uniquement dans backgroundLocationTask", () => {
    const files = walkTsFiles(DRIVER_ROOT);
    const offenders: string[] = [];
    for (const file of files) {
      const rel = path.normalize(path.relative(DRIVER_ROOT, file));
      if (rel.includes(".test.")) continue;
      if (rel === path.normalize("services/backgroundLocationTask.ts")) continue;
      const text = fs.readFileSync(file, "utf8");
      if (text.includes("stopLocationUpdatesAsync")) {
        offenders.push(rel);
      }
    }
    expect(offenders).toEqual([]);
  });

  it("T12 : owner_version_mismatch n'appelle plus stopNative…Safely directement", () => {
    const file = path.join(DRIVER_ROOT, "services", "backgroundLocationTask.ts");
    const text = fs.readFileSync(file, "utf8");
    // Plus de stopNative…Safely(`${reason}:owner_version_mismatch`)
    expect(text).not.toMatch(
      /stopNativeBackgroundLocationUpdatesSafely\(\s*[`'"]\$\{reason\}:owner_version_mismatch[`'"]/
    );
    expect(text).not.toMatch(
      /stopNativeBackgroundLocationUpdatesSafely\(\s*`\$\{reason\}:owner_version_mismatch`/
    );
    // Pas d'Unlocked direct avec ce reason (hors corps générique de la fonction)
    expect(text).not.toMatch(
      /stopNativeBackgroundLocationUpdatesUnlocked\(\s*[`'"].*owner_version_mismatch/
    );
    // Doit passer par la politique + requestOwnedStop
    expect(text).toContain("decideOwnerVersionMismatchAction");
    expect(text).toContain("requestOwnedStopForOwnerMismatch");
  });
});
