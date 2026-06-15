/**
 * Heures mission institution — fuseau métier Europe/Zurich (heure murale).
 * Évite les décalages liés au fuseau du navigateur.
 */

export const BUSINESS_TZ = 'Europe/Zurich';

const NAIVE_ISO_RE = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})/;
const HAS_TZ_RE = /(Z|[+-]\d{2}:\d{2})$/;

/** True si la chaîne ISO représente une heure murale naïve (sans fuseau). */
export const isNaiveMissionIso = (value) => {
  if (!value) return false;
  const raw = String(value).trim();
  return NAIVE_ISO_RE.test(raw) && !HAS_TZ_RE.test(raw);
};

/** Extrait YYYY-MM-DD depuis une valeur API (naïf ou instant absolu). */
export const extractWallClockDate = (value) => {
  if (!value) return '';
  const raw = String(value).trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  const naive = raw.match(NAIVE_ISO_RE);
  if (naive && !HAS_TZ_RE.test(raw)) return naive[1];
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw.slice(0, 10);
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
};

/** Extrait HH:MM en heure murale Genève. */
export const extractWallClockTime = (value) => {
  if (!value) return '';
  const raw = String(value).trim();
  const naive = raw.match(NAIVE_ISO_RE);
  if (naive && !HAS_TZ_RE.test(raw)) {
    return `${naive[2]}:${naive[3]}`;
  }
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleTimeString('fr-CH', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: BUSINESS_TZ,
  });
};

/** Formate une date courte (ex. « 15 juin ») en fuseau métier. */
export const formatWallClockDateShort = (value) => {
  if (!value) return '';
  const raw = String(value).trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    const d = new Date(`${raw}T12:00:00`);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleDateString('fr-CH', { day: '2-digit', month: 'short' });
    }
  }
  const naive = raw.match(NAIVE_ISO_RE);
  if (naive && !HAS_TZ_RE.test(raw)) {
    const d = new Date(`${naive[1]}T12:00:00`);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleDateString('fr-CH', { day: '2-digit', month: 'short' });
    }
  }
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 10);
  return d.toLocaleDateString('fr-CH', {
    day: '2-digit',
    month: 'short',
    timeZone: BUSINESS_TZ,
  });
};

/** Valeur pour input datetime-local (YYYY-MM-DDTHH:MM) en heure murale Genève. */
export const toDatetimeLocalGeneva = (value) => {
  const datePart = extractWallClockDate(value);
  const timePart = extractWallClockTime(value);
  if (!datePart || !timePart) return '';
  return `${datePart}T${timePart}`;
};

/** Formate date + heure murales (ex. « 15.06.2026 · 18:00 »). */
export const formatWallClockDateTime = (value) => {
  if (!value) return { date: '—', time: '' };
  const dateIso = extractWallClockDate(value);
  const time = extractWallClockTime(value);
  if (!dateIso) return { date: '—', time: time || '' };
  const [y, m, d] = dateIso.split('-');
  return {
    date: `${d}.${m}.${y}`,
    time: time || '',
  };
};

/**
 * Combine date mission + HH:MM en ISO naïf (heure murale Genève, sans conversion navigateur).
 */
export const combineMissionDateTimeNaive = (missionDate, timeHHMM) => {
  const normalizedDate = String(missionDate || '').trim();
  const time = String(timeHHMM || '').trim();
  if (!normalizedDate || !time) return null;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(normalizedDate)) return null;
  if (!/^\d{2}:\d{2}$/.test(time)) return null;
  return `${normalizedDate}T${time}:00`;
};
