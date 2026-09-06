import { readFileSync } from "fs";
import { join } from "path";
import { describe, expect, it } from "@jest/globals";
import { MaybeCompanyRidesEngineActions } from "./MaybeCompanyRidesEngineActions";

describe("MaybeCompanyRidesEngineActions", () => {
  it("ne monte aucune branche moteur tant que le LOCK est OFF", () => {
    const node = MaybeCompanyRidesEngineActions({
      contextId: "company:42",
      selectedDate: "2026-09-05",
      onRan: async () => undefined,
    });
    expect(node).toBeNull();
  });

  it("retire les CTA et appels moteur de l'écran Courses", () => {
    const src = readFileSync(
      join(__dirname, "../../../../../app/(app)/(company)/rides.tsx"),
      "utf8"
    );
    expect(src).not.toContain("Lancer le dispatch");
    expect(src).not.toContain("Lancer l’optimiseur");
    expect(src).not.toContain("getCompanyDispatchModes");
    expect(src).not.toContain("runCompanyDispatch");
    expect(src).not.toContain("runCompanyOptimizer");
    expect(src).not.toContain("loadDispatchMode");
  });
});
