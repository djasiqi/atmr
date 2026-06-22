/** Nom affichable du chauffeur d'une réservation (aligné tableau dispatch). */

const INVALID_DRIVER_NAME = /^(none(\s+none)?|null(\s+null)?|undefined)$/i;

function isUsableDriverLabel(value) {
  const text = String(value ?? '').trim();
  return text.length > 0 && !INVALID_DRIVER_NAME.test(text);
}

/**
 * Résout le nom chauffeur depuis les champs liste / détail / assignation.
 * @param {Record<string, unknown> | null | undefined} reservation
 * @returns {string | null}
 */
export function resolveBookingDriverName(reservation) {
  if (!reservation) return null;

  const candidates = [
    reservation.driver?.full_name,
    reservation.driver?.name,
    reservation.driver?.username,
    reservation.assignment?.driver?.full_name,
    reservation.assignment?.driver?.name,
    reservation.driver_name,
    reservation.driver_id ? `Chauffeur #${reservation.driver_id}` : null,
  ];

  for (const candidate of candidates) {
    if (isUsableDriverLabel(candidate)) {
      return String(candidate).trim();
    }
  }
  return null;
}
