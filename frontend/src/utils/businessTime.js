/**
 * Fuseau métier plateforme (P0-F TIME).
 * Source unique pour l’affichage d’instants techniques en heure Genève.
 */

export const BUSINESS_TZ = 'Europe/Zurich';

/**
 * Date calendaire YYYY-MM-DD en Europe/Zurich pour un instant absolu.
 * @param {Date|string|number} value
 * @returns {string}
 */
export function getBusinessCalendarDate(value) {
  const d = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(d.getTime())) return '';
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: BUSINESS_TZ,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(d);
  const y = parts.find((p) => p.type === 'year')?.value;
  const m = parts.find((p) => p.type === 'month')?.value;
  const day = parts.find((p) => p.type === 'day')?.value;
  return y && m && day ? `${y}-${m}-${day}` : '';
}

/**
 * Heure murale HH:MM en Europe/Zurich.
 * @param {Date|string|number} value
 * @returns {string}
 */
export function formatBusinessClockTime(value) {
  const d = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleTimeString('fr-CH', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: BUSINESS_TZ,
  });
}

/**
 * Calendrier Zurich J-1 (YYYY-MM-DD) à partir d’une date calendaire Zurich.
 * @param {string} ymd
 * @returns {string}
 */
function zurichCalendarYesterday(ymd) {
  const [y, m, d] = ymd.split('-').map(Number);
  if (!y || !m || !d) return '';
  // Midi UTC du jour civil Y-M-D ; −24h puis relecture calendaire Zurich (DST-safe).
  const noonUtc = new Date(Date.UTC(y, m - 1, d, 12, 0, 0));
  return getBusinessCalendarDate(new Date(noonUtc.getTime() - 24 * 60 * 60 * 1000));
}

/**
 * Libellé relatif jour : aujourd'hui / hier / date courte (calendrier Zurich).
 * @param {Date|string|number} recordedAt
 * @param {Date} [now]
 * @returns {string} ex. "aujourd'hui à 20:00" ou "hier à 01:30" ou "12 août à 01:30"
 */
export function formatBusinessAbsoluteDayTime(recordedAt, now = new Date()) {
  const d = recordedAt instanceof Date ? recordedAt : new Date(recordedAt);
  if (Number.isNaN(d.getTime())) return '';
  const clock = formatBusinessClockTime(d);
  if (!clock) return '';

  const recordedDay = getBusinessCalendarDate(d);
  const today = getBusinessCalendarDate(now);
  if (!recordedDay || !today) {
    return clock;
  }

  if (recordedDay === today) {
    return `aujourd'hui à ${clock}`;
  }

  if (recordedDay === zurichCalendarYesterday(today)) {
    return `hier à ${clock}`;
  }

  const dateShort = d.toLocaleDateString('fr-CH', {
    day: 'numeric',
    month: 'long',
    timeZone: BUSINESS_TZ,
  });
  return `${dateShort} à ${clock}`;
}
