import apiClient from '../utils/apiClient';
import { getApiErrorMessage } from '../utils/apiErrorMessage';

/** Assert + Capture : le backend enchaîne 2 appels Saferpay (timeout 60s chacun). > timeout axios par défaut (30s). */
const SAFERPAY_ASSERT_CLIENT_TIMEOUT_MS = 150000;

/**
 * Démarre un paiement Saferpay (Payment Page) pour une réservation.
 * Redirige le navigateur vers l’URL hébergée Saferpay.
 *
 * @param {number} bookingId
 * @param {{ returnUrl?: string }} [options]
 */
export async function startSaferpayHostedCheckout(bookingId, options = {}) {
  const body = {};
  if (options.returnUrl) {
    body.return_url = options.returnUrl;
  }
  let data;
  try {
    const res = await apiClient.post(
      `/bookings/${bookingId}/saferpay/initialize`,
      body
    );
    data = res.data;
  } catch (e) {
    const msg = getApiErrorMessage(
      e,
      "Le paiement en ligne n'est pas disponible pour le moment."
    );
    const err = new Error(msg);
    err.httpStatus = e?.response?.status;
    throw err;
  }
  const payload = data?.data ?? data;
  const url = payload?.redirect_url;
  if (!url) {
    throw new Error('Réponse de paiement incomplète (redirect_url manquant)');
  }
  window.location.assign(url);
}

/**
 * Finalise le paiement (Assert + Capture) après retour Saferpay.
 *
 * @param {number} bookingId
 * @param {number} paymentId
 */
/**
 * @returns {Promise<Record<string, unknown>>} Corps métier (ex. ``status`` §11.2) sous ``data`` ou racine.
 */
export async function assertSaferpayCheckout(bookingId, paymentId) {
  try {
    const res = await apiClient.post(
      `/bookings/${bookingId}/saferpay/assert`,
      { payment_id: paymentId },
      { timeout: SAFERPAY_ASSERT_CLIENT_TIMEOUT_MS }
    );
    const body = res.data;
    return body?.data ?? body ?? {};
  } catch (e) {
    const msg = getApiErrorMessage(
      e,
      "Impossible de finaliser le paiement pour le moment."
    );
    const err = new Error(msg);
    err.httpStatus = e?.response?.status;
    err.response = e?.response;
    throw err;
  }
}
