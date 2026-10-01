import { getBusinessCalendarDate } from '../../../../utils/businessTime';

function pad(value) {
  return String(value).padStart(2, '0');
}

/** Vrai si from–to est exactement un mois civil (1er → dernier jour). */
export function isFullCalendarMonth(from, to) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(from || '') || !/^\d{4}-\d{2}-\d{2}$/.test(to || '')) {
    return false;
  }
  const [year, month, day] = from.split('-').map(Number);
  if (day !== 1) return false;
  const last = new Date(Date.UTC(year, month, 0)).getUTCDate();
  return to === `${year}-${pad(month)}-${pad(last)}`;
}

export function monthName(ymd) {
  const [year, month] = ymd.split('-').map(Number);
  return new Intl.DateTimeFormat('fr-CH', { month: 'long' }).format(
    new Date(year, month - 1, 1),
  );
}

/**
 * La clôture n'existe que dans la vue « Ce mois », pour un mois civil déjà terminé.
 * Le lendemain du dernier jour est le premier jour où elle devient possible.
 */
export function monthClosureState(preset, range, now = new Date()) {
  const hidden = { visible: false, closable: false, availableOn: '', monthName: '' };
  if (preset !== 'month' || !isFullCalendarMonth(range?.from, range?.to)) return hidden;
  const [year, month] = range.from.split('-').map(Number);
  const nextMonth = month === 12 ? 1 : month + 1;
  const nextYear = month === 12 ? year + 1 : year;
  const availableOn = `${nextYear}-${pad(nextMonth)}-01`;
  const today = getBusinessCalendarDate(now);
  return {
    visible: true,
    closable: today > range.to,
    availableOn,
    monthName: monthName(range.from),
  };
}
