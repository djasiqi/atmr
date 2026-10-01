import { getBusinessCalendarDate } from '../../../../utils/businessTime';

function addDays(ymd, days) {
  const [year, month, day] = ymd.split('-').map(Number);
  const shifted = new Date(Date.UTC(year, month - 1, day + days));
  return shifted.toISOString().slice(0, 10);
}

/** Bornes inclusives YYYY-MM-DD en calendrier Europe/Zurich, sans conversion UTC du jour. */
export function periodRange(preset, customFrom, customTo, now = new Date()) {
  const today = getBusinessCalendarDate(now);
  if (preset === 'today') return { from: today, to: today };
  if (preset === 'week') {
    const [year, month, day] = today.split('-').map(Number);
    const noon = new Date(Date.UTC(year, month - 1, day, 12));
    const mondayOffset = (noon.getUTCDay() + 6) % 7;
    const from = addDays(today, -mondayOffset);
    return { from, to: addDays(from, 6) };
  }
  if (preset === 'month') {
    const [year, month] = today.split('-');
    const last = new Date(Date.UTC(Number(year), Number(month), 0)).getUTCDate();
    return { from: `${year}-${month}-01`, to: `${year}-${month}-${String(last).padStart(2, '0')}` };
  }
  if (preset === 'year') {
    const year = today.slice(0, 4);
    return { from: `${year}-01-01`, to: `${year}-12-31` };
  }
  const from = customFrom || today;
  const to = customTo || from;
  return from <= to ? { from, to } : { from: to, to: from };
}

/** Décale la période affichée d’un cran (jour, semaine, mois ou fenêtre personnalisée). */
export function shiftPeriod(preset, range, direction) {
  const step = direction < 0 ? -1 : 1;
  if (preset === 'month') {
    const [year, month] = range.from.split('-').map(Number);
    const cursor = new Date(Date.UTC(year, month - 1 + step, 1));
    const from = cursor.toISOString().slice(0, 10);
    const last = new Date(Date.UTC(cursor.getUTCFullYear(), cursor.getUTCMonth() + 1, 0)).getUTCDate();
    return {
      from,
      to: `${from.slice(0, 8)}${String(last).padStart(2, '0')}`,
      anchor: from,
    };
  }
  if (preset === 'week') {
    const from = addDays(range.from, 7 * step);
    return { from, to: addDays(from, 6), anchor: from };
  }
  if (preset === 'today') {
    const from = addDays(range.from, step);
    return { from, to: from, anchor: from };
  }
  const [fromYear, fromMonth, fromDay] = range.from.split('-').map(Number);
  const [toYear, toMonth, toDay] = range.to.split('-').map(Number);
  const length = Math.round(
    (Date.UTC(toYear, toMonth - 1, toDay) - Date.UTC(fromYear, fromMonth - 1, fromDay)) / 86400000
  ) + 1;
  const from = addDays(range.from, length * step);
  return { from, to: addDays(from, length - 1), anchor: from };
}
