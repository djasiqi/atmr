/**
 * Instrumentation observationnelle P0-E A4b (JS task → enqueue).
 * Tag logcat : `ATMR_JS_J` (adb logcat -s ReactNativeJS:V | grep ATMR_JS_J)
 * Aucun changement de décision métier.
 */
type DiagFields = Record<string, string | number | boolean | null | undefined>;

export function atmrJsDiag(step: string, fields: DiagFields = {}): void {
  try {
    const parts = Object.entries(fields)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => `${k}=${v === null ? "null" : String(v)}`)
      .join(" ");
    // eslint-disable-next-line no-console
    console.log(`ATMR_JS_J ${step}${parts ? ` ${parts}` : ""}`);
  } catch {
    /* noop */
  }
}
