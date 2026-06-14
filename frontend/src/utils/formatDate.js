// src/utils/formatDate.js (mode local naïf)

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

/**
 * Formate la date d'une réservation, en utilisant les champs pré-formatés
 * du backend si disponibles, sinon en forçant le fuseau horaire de Zurich.
 * @param {object} booking - L'objet réservation du backend.
 * @returns {string}
 */
export function renderBookingDateTime(booking) {
  if (!booking) return 'Non spécifié';

  const scheduling = booking.scheduling;
  if (scheduling?.display_datetime) {
    return scheduling.display_datetime;
  }
  if (scheduling && scheduling.time_defined === false) {
    return scheduling.display_time || 'À définir';
  }

  const timeConfirmed = booking.time_confirmed;
  if (timeConfirmed === false) {
    const returnDateLabel =
      formatDateOnly(booking.return_date) ||
      formatDateOnly(booking.scheduled_date) ||
      formatDateOnly(booking.date) ||
      null;
    return returnDateLabel ? `${returnDateLabel} • À définir` : 'À définir';
  }

  if (!booking.scheduled_time) {
    return 'À définir';
  }

  return formatLocalNaive(booking.scheduled_time);
}

export { formatLocalNaive };
