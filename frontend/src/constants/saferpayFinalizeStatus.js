/**
 * Doit rester aligné avec backend/services/saferpay/assert_response_status.py (§11.2).
 */
export const SAFERPAY_FINALIZE_ALREADY_COMPLETED = 'already_completed';
export const SAFERPAY_FINALIZE_COMPLETED = 'completed';
export const SAFERPAY_FINALIZE_PAYMENT_FAILED = 'payment_failed';
export const SAFERPAY_FINALIZE_ASSERT_FAILED = 'assert_failed';
export const SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS = 'unexpected_tx_status';
export const SAFERPAY_FINALIZE_ASSERT_TRANSIENT = 'assert_transient';
export const SAFERPAY_FINALIZE_CAPTURE_FAILED = 'capture_failed';

export const SAFERPAY_FINALIZE_RESPONSE_STATUSES = [
  SAFERPAY_FINALIZE_ALREADY_COMPLETED,
  SAFERPAY_FINALIZE_COMPLETED,
  SAFERPAY_FINALIZE_PAYMENT_FAILED,
  SAFERPAY_FINALIZE_ASSERT_FAILED,
  SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
  SAFERPAY_FINALIZE_ASSERT_TRANSIENT,
  SAFERPAY_FINALIZE_CAPTURE_FAILED,
];

/** Message court après réponse 200 de POST assert (complète le statut booking). */
export function userMessageForSaferpayFinalizeStatus(status) {
  switch (status) {
    case SAFERPAY_FINALIZE_CAPTURE_FAILED:
      return 'La banque a accepté le paiement mais la confirmation finale a échoué. Vous pouvez réessayer ou contacter le support en indiquant le numéro de réservation.';
    case SAFERPAY_FINALIZE_ASSERT_TRANSIENT:
      return 'Connexion temporaire avec le prestataire de paiement interrompue. Réessayez dans quelques instants.';
    case SAFERPAY_FINALIZE_ASSERT_FAILED:
    case SAFERPAY_FINALIZE_PAYMENT_FAILED:
    case SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS:
      return 'Le paiement n’a pas pu être confirmé. Le statut ci-dessous correspond à l’enregistrement côté serveur.';
    case SAFERPAY_FINALIZE_ALREADY_COMPLETED:
    case SAFERPAY_FINALIZE_COMPLETED:
      return null;
    default:
      return null;
  }
}
