/**
 * Corrélation lignes dispatch (id / booking_id) ↔ map retards (booking_id).
 * @param {unknown} raw — id réservation ou booking_id
 * @returns {number|string|null}
 */
export function normalizeDispatchDelayMapKey(raw) {
  if (raw == null || raw === '') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : String(raw);
}

/**
 * @param {Record<PropertyKey, any>|undefined|null} delayMap
 * @param {{ id?: unknown; booking_id?: unknown }} row
 */
export function getDispatchRowDelayInfo(delayMap, row) {
  if (!delayMap || !row) return undefined;
  const raw = row.booking_id ?? row.id;
  const k = normalizeDispatchDelayMapKey(raw);
  return k != null ? delayMap[k] : undefined;
}
