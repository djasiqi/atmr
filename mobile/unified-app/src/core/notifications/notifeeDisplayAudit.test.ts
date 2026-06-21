/** Vérifie qu'un seul appel notifee.displayNotification existe hors tests (chauffeur). */
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "@jest/globals";

const TARGET = "notifee.displayNotification";

function walkSourceFiles(dir: string, acc: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "dist-web-test" || entry.name === "node_modules") continue;
      walkSourceFiles(fullPath, acc);
      continue;
    }
    if (!/\.(ts|tsx)$/.test(entry.name)) continue;
    if (/\.test\.(ts|tsx)$/.test(entry.name)) continue;
    acc.push(fullPath);
  }
  return acc;
}

describe("audit Notifee chauffeur", () => {
  it("n'autorise qu'un seul notifee.displayNotification dans pushLocalDisplay.ts", () => {
    const srcRoot = path.resolve(__dirname, "../..");
    const hits = walkSourceFiles(srcRoot).flatMap((filePath) => {
      const content = fs.readFileSync(filePath, "utf8");
      if (!content.includes(TARGET)) return [];
      return [path.relative(srcRoot, filePath)];
    });

    expect(hits).toHaveLength(1);
    expect(hits[0]?.replace(/\\/g, "/")).toBe("core/notifications/pushLocalDisplay.ts");
  });
});
