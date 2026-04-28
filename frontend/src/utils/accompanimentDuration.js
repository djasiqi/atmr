/**
 * Durée d'accompagnement → heures (décimal) pour facturation = taux_h × heures.
 * Formes acceptées : 2, 0.5, 1.5, 1h, 1h30, 1:30, 30min, 90min, …
 */
export function parseAccompanimentDurationToHours(raw) {
  if (raw === undefined || raw === null) return null;
  const t = String(raw)
    .trim()
    .toLowerCase()
    .replace(/,/g, '.')
    .replace(/\s+/g, ' ');
  if (!t) return null;

  let m = t.match(/^(\d+(?:\.\d+)?)\s*min(?:ute)?s?$/);
  if (m) {
    const min = Math.max(0, parseFloat(m[1]));
    if (Number.isNaN(min)) return null;
    return min / 60;
  }

  m = t.match(/^(\d+):(\d{1,2})$/);
  if (m) {
    const h = parseInt(m[1], 10);
    const min = parseInt(m[2], 10);
    if (Number.isNaN(h) || Number.isNaN(min) || min >= 60) return null;
    return h + min / 60;
  }

  m = t.match(/^(\d+)h(\d{1,2})$/i);
  if (m) {
    const h = parseInt(m[1], 10);
    const min = parseInt(m[2], 10);
    if (Number.isNaN(h) || Number.isNaN(min) || min >= 60) return null;
    return h + min / 60;
  }

  m = t.match(/^(\d+)\s*h\s*(\d{1,2})\s*m?$/i);
  if (m) {
    const h = parseInt(m[1], 10);
    const min = parseInt(m[2], 10);
    if (Number.isNaN(h) || Number.isNaN(min) || min >= 60) return null;
    return h + min / 60;
  }

  m = t.match(/^(\d+(?:\.\d+)?)\s*h(?:r|eure)?s?$/i);
  if (m) {
    const v = parseFloat(m[1]);
    return Number.isNaN(v) || v < 0 ? null : v;
  }

  m = t.match(/^(\d+(?:\.\d+)?)$/);
  if (m) {
    const v = parseFloat(m[1]);
    return Number.isNaN(v) || v <= 0 ? null : v;
  }

  return null;
}

/**
 * Aperçu : arrondi affichage (2 déc.); le backend arrondit en 5 ct.
 */
export function computeAccompanimentLineTotal(rateChfPerHour, rawDuration) {
  const rate = Number(String(rateChfPerHour).replace(',', '.'));
  const h = parseAccompanimentDurationToHours(String(rawDuration));
  if (!Number.isFinite(rate) || rate <= 0) return null;
  if (h == null || h <= 0) return null;
  return rate * h;
}
