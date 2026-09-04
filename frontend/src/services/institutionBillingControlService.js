/**
 * API contrôle facturation institution — couche transport HTTP uniquement.
 */

import { apiClient } from '../utils/apiClient';

const BASE_PATH = '/institutions/billing';

export const listBillingControlBookings = async (params = {}) => {
  const response = await apiClient.get(`${BASE_PATH}/control/bookings`, { params });
  return response.data;
};

export const getBillingControlBooking = async (bookingId) => {
  const response = await apiClient.get(`${BASE_PATH}/control/bookings/${bookingId}`);
  return response.data;
};

export const validateBillingControlBooking = async (bookingId, body = {}) => {
  const response = await apiClient.post(
    `${BASE_PATH}/control/bookings/${bookingId}/validate`,
    body,
  );
  return response.data;
};

export const markBillingControlAnomaly = async (bookingId, body = {}) => {
  const response = await apiClient.post(
    `${BASE_PATH}/control/bookings/${bookingId}/anomaly`,
    body,
  );
  return response.data;
};

export const reopenBillingControlBooking = async (bookingId, body = {}) => {
  const response = await apiClient.post(
    `${BASE_PATH}/control/bookings/${bookingId}/reopen`,
    body,
  );
  return response.data;
};

export const changeBillingControlPayer = async (bookingId, data) => {
  const response = await apiClient.put(`${BASE_PATH}/bookings/${bookingId}`, data);
  return response.data;
};

export const getBillingControlDispute = async (bookingId) => {
  const response = await apiClient.get(`${BASE_PATH}/bookings/${bookingId}/dispute`);
  return response.data;
};

export const decideBillingControlDispute = async (bookingId, body) => {
  const response = await apiClient.post(
    `${BASE_PATH}/bookings/${bookingId}/dispute/decide`,
    body,
  );
  return response.data;
};

const institutionBillingControlService = {
  listBillingControlBookings,
  getBillingControlBooking,
  validateBillingControlBooking,
  markBillingControlAnomaly,
  reopenBillingControlBooking,
  changeBillingControlPayer,
  getBillingControlDispute,
  decideBillingControlDispute,
};

export default institutionBillingControlService;
