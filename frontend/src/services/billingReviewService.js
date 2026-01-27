// frontend/src/services/billingReviewService.js
import apiClient from '../utils/apiClient';

/**
 * Service pour le contrôle facturation (P5)
 */

/**
 * Récupère la liste des bookings pour le contrôle facturation mensuel
 * @param {Object} params - Paramètres (company_id, year, month, status, billing_party_id, clinic_id)
 * @returns {Promise} Liste des bookings avec informations de facturation
 */
export const fetchMonthlyReview = async (params) => {
  const response = await apiClient.get('/billing/monthly-review', {
    params,
  });
  return response.data;
};

/**
 * Modifie le payeur d'un booking
 * @param {number} bookingId - ID du booking
 * @param {Object} data - Données (billed_to_type, billing_party_id?, billed_to_company_id?, reason)
 * @returns {Promise} Booking modifié
 */
export const setBookingPayer = async (bookingId, data) => {
  const response = await apiClient.post(
    `/billing/bookings/${bookingId}/set-payer`,
    data
  );
  return response.data;
};

/**
 * Verrouille un booking
 * @param {number} bookingId - ID du booking
 * @param {Object} data - Données (reason)
 * @returns {Promise} Booking verrouillé
 */
export const lockBooking = async (bookingId, data) => {
  const response = await apiClient.post(
    `/billing/bookings/${bookingId}/lock`,
    data
  );
  return response.data;
};

/**
 * Déverrouille un booking (admin uniquement)
 * @param {number} bookingId - ID du booking
 * @param {Object} data - Données (reason)
 * @returns {Promise} Booking déverrouillé
 */
export const unlockBooking = async (bookingId, data) => {
  const response = await apiClient.post(
    `/billing/bookings/${bookingId}/unlock`,
    data
  );
  return response.data;
};

/**
 * Modifie le payeur de plusieurs bookings en batch
 * @param {Object} data - Données (booking_ids[], billed_to_type, billing_party_id?, billed_to_company_id?, reason)
 * @returns {Promise} Résultat de la modification batch
 */
export const batchSetBookingPayer = async (data) => {
  const response = await apiClient.post('/billing/bookings/batch-set-payer', data);
  return response.data;
};
