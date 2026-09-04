/**
 * Harness P1 optionnel (diagnostics API / deep-link).
 * Module requis par `api/client.ts` — no-op hors __DEV__.
 */
export function emitP1HarnessLog(
  event: string,
  payload: Record<string, unknown> = {}
): void {
  if (typeof __DEV__ !== "undefined" && __DEV__) {
    // eslint-disable-next-line no-console -- harness debug uniquement
    console.debug(`[p1-harness] ${event}`, payload);
  }
}
