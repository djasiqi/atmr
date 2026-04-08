import apiClient from '../utils/apiClient';
import { getApiErrorMessage } from '../utils/apiErrorMessage';

/**
 * Démarre un paiement Worldline (MyCheckout) pour une réservation.
 * Redirige le navigateur vers les pages hébergées Worldline.
 *
 * @param {number} bookingId
 * @param {{ returnUrl?: string }} [options]
 */
export async function startWorldlineHostedCheckout(bookingId, options = {}) {
  const body = {};
  if (options.returnUrl) {
    body.return_url = options.returnUrl;
  }
  let data;
  try {
    const res = await apiClient.post(
      `/bookings/${bookingId}/worldline/hosted-checkout`,
      body
    );
    data = res.data;
  } catch (e) {
    throw new Error(
      getApiErrorMessage(
        e,
        "Le paiement en ligne n'est pas disponible pour le moment."
      )
    );
  }
  const payload = data?.data ?? data;
  const url = payload?.redirect_url;
  if (!url) {
    throw new Error('Réponse de paiement incomplète (redirect_url manquant)');
  }
  window.location.assign(url);
}
