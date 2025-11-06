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
  if (isReturn && scheduledTime && timeConfirmed === false) {
    const date = new Date(scheduledTime);
    const pad = (n) => String(n).padStart(2, '0');
    const day = pad(date.getDate());
    const month = pad(date.getMonth() + 1);
    const year = date.getFullYear();
    return `${day}.${month}.${year} • ⏱️`;
  }

  // Si c'est un retour sans scheduled_time du tout
  if (isReturn && !scheduledTime) {
    return '⏱️';
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
      return `${day}.${month}.${year} • ⏱️`;
    }
  }

  // Priorité aux champs déjà formatés par le backend (qu'on suppose **locaux naïfs**)
  if (booking.date_formatted) {
    const timeFormatted = booking.time_formatted ? ` • ${booking.time_formatted}` : '';
    return `${booking.date_formatted}${timeFormatted}`;
  }
  // Sinon, on affiche la chaîne naïve telle quelle
  return formatLocalNaive(booking.scheduled_time);
}

export { formatLocalNaive };
