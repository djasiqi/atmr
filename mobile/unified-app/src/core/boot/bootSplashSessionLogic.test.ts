import { describe, expect, it } from "@jest/globals";

import { resolveBootSplashSessionBlocksOverlay } from "./bootSplashSessionLogic";

describe("resolveBootSplashSessionBlocksOverlay", () => {
  it("bloque au premier idle ou bootstrapping", () => {
    expect(resolveBootSplashSessionBlocksOverlay("idle", false, true)).toBe(true);
    expect(resolveBootSplashSessionBlocksOverlay("bootstrapping", false, true)).toBe(true);
  });

  it("ne bloque plus idle/bootstrapping après le premier boot", () => {
    expect(resolveBootSplashSessionBlocksOverlay("idle", true, true)).toBe(false);
    expect(resolveBootSplashSessionBlocksOverlay("bootstrapping", true, true)).toBe(false);
  });

  it("bloque encore sur ready tant que l'intro Lottie n'est pas terminée", () => {
    expect(resolveBootSplashSessionBlocksOverlay("ready", true, false)).toBe(true);
    expect(resolveBootSplashSessionBlocksOverlay("ready", true, true)).toBe(false);
  });

  it("n'affiche pas le splash en cas d'erreur de session", () => {
    expect(resolveBootSplashSessionBlocksOverlay("error", true, false)).toBe(false);
  });
});
