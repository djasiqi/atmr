/**
 * Indique si le client doit régler en ligne (Saferpay) pour cette réservation.
 * Hors scope : facturation différée / tiers payeur (clinique, assurance) — pas de checkout client.
 *
 * @param {Record<string, unknown> | null | undefined} bookingLike
 * @returns {boolean}
 */
export function requiresPrivateOnlinePaymentAtBooking(bookingLike) {
  if (!bookingLike || typeof bookingLike !== 'object') {
    return true;
  }
  const raw =
    bookingLike.billing?.billed_to_type ?? bookingLike.billed_to_type ?? 'patient';
  const bt = String(raw).trim().toLowerCase();
  return bt === 'patient';
}
