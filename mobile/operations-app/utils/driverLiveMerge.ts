/**
 * Alignement web (`mergeDriverLiveUpdate`) — garde observabilité-only sur la carte entreprise.
 */

function parseIsoMs(s: unknown): number | null {
  if (s == null || s === "") return null;
  const t = Date.parse(String(s));
  return Number.isFinite(t) ? t : null;
}

/**
 * Priorité : `received_at` > `recorded_at` > `timestamp` / `ts` (spec interne, alignée frontend).
 */
export function canonicalTimeMs(o: Record<string, unknown> | null | undefined): number | null {
  if (!o || typeof o !== "object") return null;
  const v = o.received_at ?? o.recorded_at ?? o.timestamp ?? o.ts;
  return parseIsoMs(v);
}

/**
 * Ne pas appliquer une position « observabilité seule » si elle est plus ancienne que l’état courant.
 */
export function shouldIgnoreObservabilityRegression(
  payload: Record<string, unknown>,
  existingMarker: Record<string, unknown> | null | undefined
): boolean {
  if (payload.accept_status !== "accepted_observability_only") return false;
  const inc = canonicalTimeMs(payload);
  const cur = existingMarker ? canonicalTimeMs(existingMarker) : null;
  if (inc != null && cur != null && inc < cur) return true;
  return false;
}
