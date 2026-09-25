/**
 * Indique si le client doit régler en ligne (Saferpay) pour cette réservation.
 * Hors scope :
 * - facturation différée / tiers payeur (clinique, assurance) — pas de checkout client ;
 * - compte ``PORTAL`` — l’entreprise facture après prestation (pas d’encaissement LIRIE).
 *
 * @param {Record<string, unknown> | null | undefined} bookingLike
 * @returns {boolean}
 */
export function requiresPrivateOnlinePaymentAtBooking(bookingLike) {
  if (!bookingLike || typeof bookingLike !== 'object') {
    return true;
  }
  const clientType = String(
    bookingLike.client?.client_type ?? bookingLike.client_type ?? ''
  )
    .trim()
    .toUpperCase();
  if (clientType === 'PORTAL') {
    return false;
  }
  const flow = String(bookingLike.portal_contract_flow || '')
    .trim()
    .toLowerCase();
  if (flow === 'double_validation_v2' || flow === 'conditional_order_v1' || flow === 'legacy') {
    // Flux portail privé : même règle que le backend (pas de Saferpay).
    return false;
  }
  const raw =
    bookingLike.billing?.billed_to_type ?? bookingLike.billed_to_type ?? 'patient';
  const bt = String(raw).trim().toLowerCase();
  return bt === 'patient';
}
