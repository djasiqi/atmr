import {
  shouldReportBootSplashFallback,
  SPLASH_BOOT_SENTRY_MAX_OVERLAY_MS,
  SPLASH_BOOT_SENTRY_MAX_SESSION_MS,
} from "./bootSplashFallbackLogic";

describe("shouldReportBootSplashFallback", () => {
  it("accepte un fallback dans la fenêtre boot normale", () => {
    expect(
      shouldReportBootSplashFallback({
        elapsedSinceOverlayMs: 4_000,
        elapsedSinceSessionMs: 5_000,
      }),
    ).toBe(true);
  });

  it("refuse un fallback retardé par session longue (timer background)", () => {
    expect(
      shouldReportBootSplashFallback({
        elapsedSinceOverlayMs: 4_000,
        elapsedSinceSessionMs: 4_098_414,
      }),
    ).toBe(false);
  });

  it("refuse un fallback longtemps après montage overlay", () => {
    expect(
      shouldReportBootSplashFallback({
        elapsedSinceOverlayMs: SPLASH_BOOT_SENTRY_MAX_OVERLAY_MS + 1,
        elapsedSinceSessionMs: 20_000,
      }),
    ).toBe(false);
  });

  it("refuse au-delà du plafond session même si overlay récent", () => {
    expect(
      shouldReportBootSplashFallback({
        elapsedSinceOverlayMs: 2_000,
        elapsedSinceSessionMs: SPLASH_BOOT_SENTRY_MAX_SESSION_MS + 1,
      }),
    ).toBe(false);
  });
});
