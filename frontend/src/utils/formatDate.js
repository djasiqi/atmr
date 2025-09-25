// src/utils/formatDate.js (mode local naïf)

/**
 * Formate une chaîne datetime **naïve locale** sans conversions.
 * Accepte:
 *  - "YYYY-MM-DD HH:MM[:SS]"
 *  - "YYYY-MM-DDTHH:MM[:SS]"
 *  - Date (utilisée telle quelle, sans TZ)
 */
function formatLocalNaive(dateInput) {
  if (!dateInput) return "Non spécifié";
  try {
    let s = typeof dateInput === "string" ? dateInput.trim() : "";
    if (!s && dateInput instanceof Date) {
      // ATTENTION: éviter toute conversion; on reconstruit depuis composants locaux
      const pad = (n) => String(n).padStart(2, "0");
      const d = dateInput; // supposé déjà local
      s = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }
    // Normaliser séparateur
    s = s.replace(" ", "T");
    const [datePart, timePartFull = ""] = s.split("T");
    const [hh = "00", mm = "00"] = timePartFull.split(":");
    const timePart = `${hh.padStart(2, "0")}:${mm.padStart(2, "0")}`;
    return `${datePart} • ${timePart}`;
  } catch (e) {
    console.error("Error formatting local naive date:", e);
    return "Date invalide";
  }
}

/**
  * Formate la date d'une réservation, en utilisant les champs pré-formatés
  * du backend si disponibles, sinon en forçant le fuseau horaire de Zurich.
  * @param {object} booking - L'objet réservation du backend.
  * @returns {string}
  */
 export function renderBookingDateTime(booking) {
  if (!booking) return "Non spécifié";
  // Priorité aux champs déjà formatés par le backend (qu’on suppose **locaux naïfs**)
  if (booking.date_formatted) {
    const timeFormatted = booking.time_formatted ? ` • ${booking.time_formatted}` : "";
    return `${booking.date_formatted}${timeFormatted}`;
  }
  // Sinon, on affiche la chaîne naïve telle quelle
  return formatLocalNaive(booking.scheduled_time);
}

export { formatLocalNaive };