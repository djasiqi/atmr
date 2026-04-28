/**
 * Aperçu admin / tests — même logique que le serveur
 * (services.client_surface.indicative_fare + round Chf 5 c.).
 * Ne pas mélanger avec le preview de réservation (compute_price).
 */

function roundChfToFiveRappen(value) {
  const x = Number(value);
  if (!Number.isFinite(x)) return x;
  return Math.round((x + Number.EPSILON) * 20) / 20;
}

/**
 * @param {object} cfg - min_fare_chf, base_chf, per_minute_chf, ref_km, ref_min (nombres)
 * @param {number} distanceM
 * @param {number} durationS
 * @returns {number|null}
 */
export function computeIndicativeFromConfigChf(cfg, distanceM, durationS) {
  if (!cfg || distanceM == null || Number.isNaN(distanceM) || distanceM <= 0) return null;
  const refKm = Number(cfg.ref_km);
  if (!Number.isFinite(refKm) || refKm <= 0) return null;
  const perKm = (Number(cfg.min_fare_chf) - Number(cfg.base_chf) - Number(cfg.ref_min) * Number(cfg.per_minute_chf)) / refKm;
  const km = distanceM / 1000;
  const min = (durationS != null && !Number.isNaN(Number(durationS)) ? Number(durationS) : 0) / 60;
  const raw = Number(cfg.base_chf) + perKm * km + Number(cfg.per_minute_chf) * min;
  const clamped = Math.max(raw, Number(cfg.min_fare_chf));
  return roundChfToFiveRappen(clamped);
}
