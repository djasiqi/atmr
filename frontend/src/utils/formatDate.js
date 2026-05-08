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

  const isReturn = booking.is_return;
  const scheduledTime = booking.scheduled_time;
  const timeConfirmed = booking.time_confirmed;

  // Si c'est un retour avec heure non confirmée (time_confirmed = false)
  if (isReturn && scheduledTime && timeConfirmed !== true) {
    const date = new Date(scheduledTime);
    const pad = (n) => String(n).padStart(2, '0');
    const day = pad(date.getDate());
    const month = pad(date.getMonth() + 1);
    const year = date.getFullYear();
    return `${day}.${month}.${year} • À définir`;
  }

  // Si c'est un retour sans scheduled_time du tout
  if (isReturn && !scheduledTime) {
    const returnDateLabel =
      formatDateOnly(booking.return_date) ||
      formatDateOnly(booking.scheduled_date) ||
      formatDateOnly(booking.date) ||
      null;
    return returnDateLabel ? `${returnDateLabel} • À définir` : 'À définir';
  }

  // 🔍 Détecter les heures à 00:00 (heure par défaut à confirmer)
  if (scheduledTime) {
    const date = new Date(scheduledTime);
    const hours = date.getHours();
    const minutes = date.getMinutes();

    // Si l'heure est exactement 00:00, c'est probablement une heure à confirmer
    if (hours === 0 && minutes === 0) {
      const pad = (n) => String(n).padStart(2, '0');
      const day = pad(date.getDate());
      const month = pad(date.getMonth() + 1);
      const year = date.getFullYear();
      return `${day}.${month}.${year} • À définir`;
    }
  }

  // ✅ Toujours utiliser le format suisse (dd.MM.yyyy) même si date_formatted est présent
  // Ignorer date_formatted du backend pour uniformiser l'affichage
  // Utiliser directement scheduled_time pour garantir le format suisse
  return formatLocalNaive(booking.scheduled_time);
}

export { formatLocalNaive };
