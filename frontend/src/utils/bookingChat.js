/**
 * Indique si le mini-chat booking doit être en lecture seule (transport terminé).
 * En aller-retour, le fil reste ouvert tant que le segment retour n'est pas terminal.
 */

const TERMINAL_STATUSES = new Set([
  'COMPLETED',
  'RETURN_COMPLETED',
  'CANCELED',
  'CANCELLED',
]);

function normalizeStatus(raw) {
  const s = String(raw ?? '').trim().toUpperCase();
  return s === 'CANCELLED' ? 'CANCELED' : s;
}

function isTerminalStatus(raw) {
  return TERMINAL_STATUSES.has(normalizeStatus(raw));
}

/**
 * @param {object|null|undefined} booking — réservation entreprise, booking client ou booking_summary institution
 * @returns {boolean}
 */
export function isBookingChatClosed(booking) {
  if (!booking) return false;

  const outboundStatus = normalizeStatus(booking.status ?? booking.booking_status);
  const returnBooking = booking.return_booking;

  if (returnBooking) {
    return isTerminalStatus(returnBooking.status);
  }

  const overall = String(booking.overall_status ?? '').toLowerCase();
  if (overall === 'completed' || overall === 'cancelled') return true;
  if (overall === 'outbound_completed' || overall === 'in_progress' || overall === 'planned') {
    return false;
  }

  const isRoundTrip = Boolean(
    booking.is_round_trip || booking.has_return || booking.round_trip
  );
  if (isRoundTrip) {
    if (outboundStatus === 'RETURN_COMPLETED') return true;
    if (outboundStatus === 'CANCELED') return true;
    // Aller terminé, retour encore actif ou à planifier
    if (outboundStatus === 'COMPLETED') return false;
    return isTerminalStatus(outboundStatus);
  }

  return isTerminalStatus(outboundStatus);
}
