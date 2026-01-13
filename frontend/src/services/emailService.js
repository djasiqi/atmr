/**
 * Service pour la configuration des emails transactionnels (Brevo)
 */

import apiClient from '../utils/apiClient';

/**
 * Récupère la configuration email actuelle de l'entreprise
 * @returns {Promise<Object>} Configuration email
 */
export const getEmailConfig = async () => {
  const response = await apiClient.get(`/email/config`);
  return response.data;
};

/**
 * Configure l'adresse email d'envoi et récupère les DNS à configurer
 * @param {Object} data - Configuration email
 * @param {string} data.from_email - Adresse email d'envoi (ex: noreply@entreprise.ch)
 * @param {string} data.from_name - Nom d'expéditeur (ex: "Entreprise SA")
 * @returns {Promise<Object>} Résultat de la configuration avec les DNS records
 */
export const setupEmailDomain = async (data) => {
  const response = await apiClient.post(`/email/domain/setup`, data);
  return response.data;
};

/**
 * Vérifie que le domaine est validé dans Brevo (SPF/DKIM configurés)
 * @returns {Promise<Object>} Résultat de la vérification
 */
export const verifyEmailDomain = async () => {
  const response = await apiClient.post(`/email/domain/verify`);
  return response.data;
};

/**
 * Effectue un diagnostic complet du domaine (mode debug)
 * @returns {Promise<Object>} Diagnostic détaillé avec réponse Brevo complète
 */
export const diagnosticEmailDomain = async () => {
  const response = await apiClient.post(`/email/domain/diagnostic`);
  return response.data;
};

const emailService = {
  getEmailConfig,
  setupEmailDomain,
  verifyEmailDomain,
  diagnosticEmailDomain,
};

export default emailService;
