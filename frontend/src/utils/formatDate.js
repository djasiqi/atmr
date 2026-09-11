// src/utils/formatDate.js (mode local naïf)

import { extractWallClockDate, extractWallClockTime } from './missionTimeDisplay';

/**
 * Formate une chaîne datetime **naïve locale** sans conversions.
 * Accepte:
 *  - "YYYY-MM-DD HH:MM[:SS]"
 *  - "YYYY-MM-DDTHH:MM[:SS]"
 *  - Date (utilisée telle quelle, sans TZ)
 */
function formatLocalNaive(dateInput) {
  if (!dateInput) return 'Non spécifié';
  try {
    let dateObj;

    if (dateInput instanceof Date) {
      dateObj = dateInput;
    } else if (typeof dateInput === 'string') {
      // Parser la chaîne ISO ou autre format
      const s = dateInput.trim().replace(' ', 'T');
      dateObj = new Date(s);
    } else {
      return 'Non spécifié';
    }

    if (isNaN(dateObj.getTime())) {
      return 'Date invalide';
    }

    // Format suisse : dd.MM.yyyy • HH:mm
    const pad = (n) => String(n).padStart(2, '0');
    const day = pad(dateObj.getDate());
    const month = pad(dateObj.getMonth() + 1);
    const year = dateObj.getFullYear();
    const hours = pad(dateObj.getHours());
    const minutes = pad(dateObj.getMinutes());

    return `${day}.${month}.${year} • ${hours}:${minutes}`;
  } catch (e) {
    console.error('Error formatting local naive date:', e);
    return 'Date invalide';
  }
}

function formatDateOnly(value) {
  if (!value || typeof value !== 'string') return null;
  const raw = value.trim();
  if (!raw) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(raw);
  if (m) return `${m[3]}.${m[2]}.${m[1]}`;
  const d = new Date(raw.replace(' ', 'T'));
  if (Number.isNaN(d.getTime())) return null;
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`;
}

function formatSwissDateFromYmd(ymd) {
  if (!ymd || !/^\d{4}-\d{2}-\d{2}$/.test(ymd)) return '';
  const [y, m, d] = ymd.split('-');
  return `${d}.${m}.${y}`;
}

/** Libellé horaire à partir de l'heure murale + confirmation (évite un scheduling périmé). */
export function buildScheduleDisplay({
  scheduledTime,
  timeConfirmed,
  isReturn = false,
} = {}) {
  const ymd = extractWallClockDate(scheduledTime);
  const hhmm = extractWallClockTime(scheduledTime);
  const dateLabel = formatSwissDateFromYmd(ymd);
  const confirmed = timeConfirmed === true;
  if (!hhmm) {
    return {
      scheduled_time: scheduledTime || null,
      time_confirmed: confirmed,
      time_defined: confirmed,
      time_scheduled: false,
      display_time: 'À définir',
      display_datetime: dateLabel || 'À définir',
    };
  }
  let displayTime = hhmm;
  if (timeConfirmed === false) {
    displayTime = isReturn ? 'À confirmer' : `${hhmm} (non confirmé)`;
  }
  return {
    scheduled_time: scheduledTime,
    time_confirmed: timeConfirmed !== false,
    time_defined: timeConfirmed !== false,
    time_scheduled: true,
    display_time: displayTime,
    display_datetime: dateLabel ? `${dateLabel} • ${displayTime}` : displayTime,
  };
}

/**
 * Formate la date d'une réservation.
 * `time_confirmed` + `scheduled_time` priment sur `scheduling.display_datetime`
 * (souvent périmé après une reconfirmation de départ).
 * @param {object} booking - L'objet réservation du backend.
 * @returns {string}
 */
export function renderBookingDateTime(booking) {
  if (!booking) return 'Non spécifié';

  const timeConfirmed = booking.time_confirmed;
  const iso = booking.scheduled_time || booking.scheduling?.scheduled_time;
  if (timeConfirmed === true || timeConfirmed === false) {
    return buildScheduleDisplay({
      scheduledTime: iso,
      timeConfirmed,
      isReturn: Boolean(
        booking.is_return
        || booking.trip_flags?.return_leg
        || Number(booking.route_sequence_number) > 1
      ),
    }).display_datetime;
  }

  const scheduling = booking.scheduling;
  if (scheduling?.display_datetime) {
    return scheduling.display_datetime;
  }
  if (scheduling && scheduling.time_defined === false) {
    return scheduling.display_time || 'À définir';
  }

  if (!iso) {
    const returnDateLabel =
      formatDateOnly(booking.return_date) ||
      formatDateOnly(booking.scheduled_date) ||
      formatDateOnly(booking.date) ||
      null;
    return returnDateLabel ? `${returnDateLabel} • À définir` : 'À définir';
  }

  return formatLocalNaive(iso);
}

export { formatLocalNaive };
