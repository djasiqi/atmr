// frontend/src/services/transportVoucherService.js
import apiClient from '../utils/apiClient';

/**
 * Service pour gérer les bons de transport (P3)
 */

/**
 * Récupère la liste des bons de transport avec filtres optionnels
 * @param {Object} filters - Filtres (client_id, booking_id, status, type)
 * @returns {Promise} Liste des bons
 */
export const fetchTransportVouchers = async (filters = {}) => {
  const response = await apiClient.get('/transport-vouchers', {
    params: filters,
  });
  return response.data;
};

/**
 * Récupère un bon de transport par son ID
 * @param {number} voucherId - ID du bon
 * @returns {Promise} Détails du bon
 */
export const fetchTransportVoucher = async (voucherId) => {
  const response = await apiClient.get(`/transport-vouchers/${voucherId}`);
  return response.data;
};

/**
 * Crée un nouveau bon de transport
 * @param {Object} voucherData - Données du bon
 * @returns {Promise} Bon créé
 */
export const createTransportVoucher = async (voucherData) => {
  const response = await apiClient.post('/transport-vouchers', voucherData);
  return response.data;
};

/**
 * Met à jour un bon de transport
 * @param {number} voucherId - ID du bon
 * @param {Object} voucherData - Données à mettre à jour
 * @returns {Promise} Bon mis à jour
 */
export const updateTransportVoucher = async (voucherId, voucherData) => {
  const response = await apiClient.patch(`/transport-vouchers/${voucherId}`, voucherData);
  return response.data;
};

/**
 * Supprime un bon de transport (uniquement si draft)
 * @param {number} voucherId - ID du bon
 * @returns {Promise}
 */
export const deleteTransportVoucher = async (voucherId) => {
  const response = await apiClient.delete(`/transport-vouchers/${voucherId}`);
  return response.data;
};

/**
 * Valide un bon de transport (backoffice)
 * @param {number} voucherId - ID du bon
 * @param {Object} data - Données de validation (billing_party_id, notes)
 * @returns {Promise} Bon validé
 */
export const validateTransportVoucher = async (voucherId, data = {}) => {
  const response = await apiClient.post(`/transport-vouchers/${voucherId}/validate`, data);
  return response.data;
};

/**
 * Rejette un bon de transport (backoffice)
 * @param {number} voucherId - ID du bon
 * @param {Object} data - Données de rejet (reason, notes)
 * @returns {Promise} Bon rejeté
 */
export const rejectTransportVoucher = async (voucherId, data) => {
  const response = await apiClient.post(`/transport-vouchers/${voucherId}/reject`, data);
  return response.data;
};

/**
 * Upload un fichier pour un bon de transport
 * @param {number} voucherId - ID du bon
 * @param {File} file - Fichier à uploader
 * @returns {Promise} Fichier uploadé
 */
export const uploadTransportVoucherFile = async (voucherId, file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post(`/transport-vouchers/${voucherId}/files`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Supprime un fichier attaché à un bon
 * @param {number} voucherId - ID du bon
 * @param {number} fileId - ID du fichier
 * @returns {Promise}
 */
export const deleteTransportVoucherFile = async (voucherId, fileId) => {
  const response = await apiClient.delete(`/transport-vouchers/${voucherId}/files`, {
    params: { file_id: fileId },
  });
  return response.data;
};
