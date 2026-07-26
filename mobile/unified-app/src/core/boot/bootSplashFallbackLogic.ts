/** Fenêtre max depuis l'affichage overlay pour considérer un fallback splash comme boot réel. */
export const SPLASH_BOOT_SENTRY_MAX_OVERLAY_MS = 15_000;

/** Fenêtre max depuis le début de session pour remonter le fallback splash à Sentry. */
export const SPLASH_BOOT_SENTRY_MAX_SESSION_MS = 60_000;

/**
 * Filtre les faux positifs Sentry : timers JS retardés en arrière-plan (Android)
 * ou overlay remonté longtemps après le cold start (elapsedMs >> 4 s attendus).
 */
export function shouldReportBootSplashFallback(args: {
  elapsedSinceOverlayMs: number;
  elapsedSinceSessionMs: number;
}): boolean {
  return (
    args.elapsedSinceOverlayMs >= 0 &&
    args.elapsedSinceOverlayMs <= SPLASH_BOOT_SENTRY_MAX_OVERLAY_MS &&
    args.elapsedSinceSessionMs >= 0 &&
    args.elapsedSinceSessionMs <= SPLASH_BOOT_SENTRY_MAX_SESSION_MS
  );
}
