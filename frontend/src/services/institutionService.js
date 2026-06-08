// services/institutionService.js
/**
 * ÉTAPE 6: Service API pour le portail Institution
 * 
 * Consomme les endpoints /api/v1/institutions/*
 */

import { apiClient } from '../utils/apiClient';

const BASE_PATH = '/institutions';

// ============================================================================
// Institution Info
// ============================================================================

/**
 * Récupère les informations de l'institution connectée
 */
export const getMe = async () => {
  const response = await apiClient.get(`${BASE_PATH}/me`);
  return response.data;
};

/**
 * Met à jour le profil de l'institution (admin only)
 * @param {Object} data - { name?, address?, contact_email?, contact_phone?, notes? }
 */
export const updateInstitution = async (data) => {
  const response = await apiClient.put(`${BASE_PATH}/me`, data);
  return response.data;
};

/**
 * Récupère les entreprises éligibles pour les préférences transport
 */
export const getEligibleCompanies = async () => {
  const response = await apiClient.get(`${BASE_PATH}/settings/eligible-companies`);
  return response.data;
};

// ============================================================================
// Transport Requests
// ============================================================================

/**
 * Liste les demandes de transport
 * @param {Object} params - Filtres (status, date_from, date_to, query, page, per_page)
 */
export const listRequests = async (params = {}) => {
  const response = await apiClient.get(`${BASE_PATH}/requests`, { params });
  return response.data;
};

/**
 * Récupère une demande par son ID
 */
export const getRequest = async (requestId) => {
  const response = await apiClient.get(`${BASE_PATH}/requests/${requestId}`);
  return response.data;
};

/**
 * Crée une nouvelle demande de transport
 */
export const createRequest = async (data) => {
  const response = await apiClient.post(`${BASE_PATH}/requests`, data);
  return response.data;
};

/**
 * Met à jour une demande existante
 */
export const updateRequest = async (requestId, data) => {
  const response = await apiClient.put(`${BASE_PATH}/requests/${requestId}`, data);
  return response.data;
};

/**
 * Envoie une demande aux transporteurs
 * @param {number} requestId - ID de la demande
 * @param {Object} options - Options d'envoi (mode, timeout_minutes, etc.)
 */
export const sendRequest = async (requestId, options = {}) => {
  const response = await apiClient.post(`${BASE_PATH}/requests/${requestId}/send`, options);
  return response.data;
};

/**
 * Annule une demande
 * @param {number} requestId - ID de la demande
 * @param {string} reason - Raison de l'annulation (optionnel)
 */
export const cancelRequest = async (requestId, reason = '') => {
  const response = await apiClient.post(`${BASE_PATH}/requests/${requestId}/cancel`, { reason });
  return response.data;
};

/**
 * Récupère une demande par sa référence externe
 */
export const getRequestByReference = async (externalReference) => {
  const response = await apiClient.get(`${BASE_PATH}/requests/by-reference/${externalReference}`);
  return response.data;
};

// ============================================================================
// Patients
// ============================================================================

/**
 * Liste les patients de l'institution
 */
export const listPatients = async (params = {}) => {
  const response = await apiClient.get(`${BASE_PATH}/patients`, { params });
  return response.data;
};

/**
 * Récupère un patient par son ID
 */
export const getPatient = async (patientId) => {
  const response = await apiClient.get(`${BASE_PATH}/patients/${patientId}`);
  return response.data;
};

/**
 * Crée un nouveau patient
 */
export const createPatient = async (data) => {
  const response = await apiClient.post(`${BASE_PATH}/patients`, data);
  return response.data;
};

/**
 * Met à jour un patient
 */
export const updatePatient = async (patientId, data) => {
  const response = await apiClient.put(`${BASE_PATH}/patients/${patientId}`, data);
  return response.data;
};

// ============================================================================
// Settings - Institution Settings (billing, notifications, timeouts)
// ============================================================================

/**
 * Récupère les paramètres institution (billing + notifications + timeouts)
 * @returns {{ institution: Object, settings: Object }}
 */
export const getInstitutionSettings = async () => {
  const response = await apiClient.get(`${BASE_PATH}/settings`);
  return response.data;
};

/**
 * Met à jour les paramètres institution (admin only)
 * @param {Object} data - Champs à mettre à jour (update partiel)
 */
export const updateInstitutionSettings = async (data) => {
  const response = await apiClient.put(`${BASE_PATH}/settings`, data);
  return response.data;
};

// ============================================================================
// Settings - Transport Preferences
// ============================================================================

/**
 * Récupère les préférences de transport
 */
export const getTransportPreferences = async () => {
  const response = await apiClient.get(`${BASE_PATH}/settings/transport-preferences`);
  return response.data;
};

/**
 * Met à jour les préférences de transport (ordre des transporteurs)
 */
export const updateTransportPreferences = async (data) => {
  const response = await apiClient.put(`${BASE_PATH}/settings/transport-preferences`, data);
  return response.data;
};

// ============================================================================
// API Keys Management
// ============================================================================

/**
 * Liste les clés API de l'institution
 */
export const listApiKeys = async () => {
  const response = await apiClient.get(`${BASE_PATH}/api-keys`);
  return response.data;
};

/**
 * Crée une nouvelle clé API
 * @param {Object} data - { name, scopes: string[] }
 * @returns {Object} - Contient raw_key (à afficher une seule fois)
 */
export const createApiKey = async (data) => {
  const response = await apiClient.post(`${BASE_PATH}/api-keys`, data);
  return response.data;
};

/**
 * Révoque une clé API
 */
export const revokeApiKey = async (keyId) => {
  const response = await apiClient.post(`${BASE_PATH}/api-keys/${keyId}/revoke`);
  return response.data;
};

// ============================================================================
// Billing
// ============================================================================

/**
 * Met à jour les infos de facturation d'une demande
 * Requiert rôle: billing ou admin
 */
export const updateRequestBilling = async (requestId, data) => {
  const response = await apiClient.put(`${BASE_PATH}/billing/requests/${requestId}`, data);
  return response.data;
};

/**
 * Met à jour les infos de facturation d'un booking
 * Requiert rôle: billing ou admin
 */
export const updateBookingBilling = async (bookingId, data) => {
  const response = await apiClient.put(`${BASE_PATH}/billing/bookings/${bookingId}`, data);
  return response.data;
};

/**
 * Modification opérationnelle d'un booking institution (avant boarded_at)
 */
export const patchInstitutionBooking = async (bookingId, data) => {
  const response = await apiClient.patch(`${BASE_PATH}/bookings/${bookingId}`, data);
  return response.data;
};

/**
 * Annulation booking institution (demande CONVERTED)
 */
export const cancelInstitutionBooking = async (bookingId, data) => {
  const response = await apiClient.post(`${BASE_PATH}/bookings/${bookingId}/cancel`, data);
  return response.data;
};

/**
 * Historique audit / activité booking
 */
export const fetchBookingChangeEvents = async (bookingId) => {
  const response = await apiClient.get(`${BASE_PATH}/bookings/${bookingId}/change-events`);
  return response.data;
};

// ============================================================================
// Users Management
// ============================================================================

/**
 * Liste les utilisateurs de l'institution
 */
export const listInstitutionUsers = async () => {
  const response = await apiClient.get(`${BASE_PATH}/users`);
  return response.data;
};

/**
 * Ajoute/invite un utilisateur dans l'institution
 */
export const inviteInstitutionUser = async (data) => {
  const response = await apiClient.post(`${BASE_PATH}/users`, data);
  return response.data;
};

/**
 * Met à jour le rôle d'un utilisateur
 */
export const updateUserRole = async ({ userId, institution_role }) => {
  const response = await apiClient.put(`${BASE_PATH}/users/${userId}/role`, { institution_role });
  return response.data;
};

/**
 * Met à jour les champs descriptifs d'un utilisateur (fonction/métier).
 * Donnée organisationnelle sans impact sur les permissions.
 */
export const updateUserProfile = async ({ userId, job_title }) => {
  const response = await apiClient.patch(`${BASE_PATH}/users/${userId}`, { job_title });
  return response.data;
};

/**
 * Retire un utilisateur de l'institution
 */
export const removeInstitutionUser = async (userId) => {
  const response = await apiClient.delete(`${BASE_PATH}/users/${userId}`);
  return response.data;
};

/**
 * Renvoie une invitation par email
 */
export const resendInvite = async (userId) => {
  const response = await apiClient.post(`${BASE_PATH}/users/${userId}/resend-invite`);
  return response.data;
};

/**
 * Désactive un utilisateur de l'institution
 */
export const disableInstitutionUser = async (userId) => {
  const response = await apiClient.post(`${BASE_PATH}/users/${userId}/disable`);
  return response.data;
};

/**
 * Liste les collaborateurs en attente d'activation
 */
export const getPendingActivationUsers = async () => {
  const response = await apiClient.get(`${BASE_PATH}/users/pending-activation`);
  return response.data;
};

/**
 * Réinitialise le mot de passe temporaire (Mode B — identifiant)
 */
export const resetInstitutionUserPassword = async (userId) => {
  const response = await apiClient.post(`${BASE_PATH}/users/${userId}/reset-password`);
  return response.data;
};

// ============================================================================
// Notifications
// ============================================================================

/**
 * Liste paginée des notifications de l'institution
 * @param {Object} params - { limit, offset }
 */
export const getNotifications = async (params = {}) => {
  const response = await apiClient.get(`${BASE_PATH}/notifications`, { params });
  return response.data;
};

/**
 * Marque une notification comme lue
 * @param {number} notificationId
 */
export const markNotificationRead = async (notificationId) => {
  const response = await apiClient.put(`${BASE_PATH}/notifications/${notificationId}/read`);
  return response.data;
};

/**
 * Marque toutes les notifications comme lues
 */
export const markAllNotificationsRead = async () => {
  const response = await apiClient.put(`${BASE_PATH}/notifications/read-all`);
  return response.data;
};

// ============================================================================
// Mon profil (personnel)
// ============================================================================

/**
 * Met à jour le profil personnel de l'utilisateur connecté
 * @param {Object} data - { first_name?, last_name?, phone? }
 */
export const updateMyProfile = async (data) => {
  const response = await apiClient.put(`${BASE_PATH}/me/profile`, data);
  return response.data;
};

// ============================================================================
// Demandes de droits
// ============================================================================

/**
 * Crée une demande de droits auprès de l'admin
 * @param {Object} data - { requested_role, message }
 */
export const createPermissionRequest = async (data) => {
  const response = await apiClient.post(`${BASE_PATH}/permission-requests`, data);
  return response.data;
};

/**
 * Liste les demandes de droits (admin: toutes, autres: les siennes)
 */
export const listPermissionRequests = async () => {
  const response = await apiClient.get(`${BASE_PATH}/permission-requests`);
  return response.data;
};

/**
 * Résout une demande de droits (admin: approve/deny)
 * @param {number} requestId
 * @param {string} action - 'approve' ou 'deny'
 */
export const resolvePermissionRequest = async (requestId, action) => {
  const response = await apiClient.put(
    `${BASE_PATH}/permission-requests/${requestId}/resolve`,
    { action }
  );
  return response.data;
};

// ============================================================================
// Curator Teams (curatelle)
// ============================================================================

export const listTeams = async () => {
  const response = await apiClient.get(`${BASE_PATH}/teams`);
  return response.data;
};

export const createTeam = async (data) => {
  const response = await apiClient.post(`${BASE_PATH}/teams`, data);
  return response.data;
};

export const updateTeam = async (teamId, data) => {
  const response = await apiClient.put(`${BASE_PATH}/teams/${teamId}`, data);
  return response.data;
};

export const deleteTeam = async (teamId) => {
  const response = await apiClient.delete(`${BASE_PATH}/teams/${teamId}`);
  return response.data;
};

export const addTeamMember = async (teamId, userId) => {
  const response = await apiClient.post(`${BASE_PATH}/teams/${teamId}/members`, { user_id: userId });
  return response.data;
};

export const removeTeamMember = async (teamId, userId) => {
  const response = await apiClient.delete(`${BASE_PATH}/teams/${teamId}/members/${userId}`);
  return response.data;
};

export const assignPatientTeam = async (patientId, teamId) => {
  const response = await apiClient.put(`${BASE_PATH}/teams/assign-patient/${patientId}`, { team_id: teamId });
  return response.data;
};

// ============================================================================
// Patient Identity / Sync / Matching
// ============================================================================

export const getPatientIdentity = async (patientId) => {
  const response = await apiClient.get(`${BASE_PATH}/patients/${patientId}/identity`);
  return response.data;
};

export const getPatientSyncStatus = async (patientId) => {
  const response = await apiClient.get(`${BASE_PATH}/patients/${patientId}/sync-status`);
  return response.data;
};

export const getPatientMatches = async (patientId) => {
  const response = await apiClient.get(`${BASE_PATH}/patients/${patientId}/matches`);
  return response.data;
};

export const confirmPatientMatch = async (patientId, identityId) => {
  const response = await apiClient.post(`${BASE_PATH}/patients/${patientId}/matches/${identityId}/confirm`);
  return response.data;
};

export const rejectPatientMatch = async (patientId, identityId) => {
  const response = await apiClient.post(`${BASE_PATH}/patients/${patientId}/matches/${identityId}/reject`);
  return response.data;
};

export const detachPatientIdentity = async (patientId, reason = '') => {
  const response = await apiClient.put(`${BASE_PATH}/patients/${patientId}/identity/detach`, { reason });
  return response.data;
};

// Link Suggestions
export const getPatientSuggestions = async (patientId) => {
  const response = await apiClient.get(`${BASE_PATH}/patients/${patientId}/suggestions`);
  return response.data;
};

export const confirmPatientSuggestion = async (patientId, suggestionId) => {
  const response = await apiClient.post(`${BASE_PATH}/patients/${patientId}/suggestions/${suggestionId}/confirm`);
  return response.data;
};

export const rejectPatientSuggestion = async (patientId, suggestionId) => {
  const response = await apiClient.post(`${BASE_PATH}/patients/${patientId}/suggestions/${suggestionId}/reject`);
  return response.data;
};

// ============================================================================
// Booking Messages (mini-canal de communication)
// ============================================================================

/**
 * Récupère les messages d'un booking (paginé)
 * @param {number} bookingId
 * @param {Object} opts - { limit, beforeId }
 */
export const fetchBookingMessages = async (bookingId, { limit, beforeId } = {}) => {
  const params = {};
  if (limit) params.limit = limit;
  if (beforeId) params.before_id = beforeId;
  const response = await apiClient.get(`/bookings/${bookingId}/messages`, { params });
  return response.data;
};

/**
 * Envoie un message dans le mini-chat du booking
 * @param {number} bookingId
 * @param {string} content
 */
export const sendBookingMessage = async (bookingId, content) => {
  const response = await apiClient.post(`/bookings/${bookingId}/messages`, { content });
  return response.data;
};

// ============================================================================
// Export default pour import simplifié
// ============================================================================

const institutionService = {
  // Institution
  getMe,
  updateInstitution,
  getEligibleCompanies,
  // Requests
  listRequests,
  getRequest,
  createRequest,
  updateRequest,
  sendRequest,
  cancelRequest,
  getRequestByReference,
  // Patients
  listPatients,
  getPatient,
  createPatient,
  updatePatient,
  // Settings
  getInstitutionSettings,
  updateInstitutionSettings,
  getTransportPreferences,
  updateTransportPreferences,
  // API Keys
  listApiKeys,
  createApiKey,
  revokeApiKey,
  // Billing
  updateRequestBilling,
  updateBookingBilling,
  patchInstitutionBooking,
  cancelInstitutionBooking,
  fetchBookingChangeEvents,
  // Users
  listInstitutionUsers,
  inviteInstitutionUser,
  updateUserRole,
  updateUserProfile,
  removeInstitutionUser,
  resendInvite,
  disableInstitutionUser,
  getPendingActivationUsers,
  resetInstitutionUserPassword,
  // Notifications
  getNotifications,
  markNotificationRead,
  markAllNotificationsRead,
  // Mon profil
  updateMyProfile,
  // Permission requests
  createPermissionRequest,
  listPermissionRequests,
  resolvePermissionRequest,
  // Curator Teams
  listTeams,
  createTeam,
  updateTeam,
  deleteTeam,
  addTeamMember,
  removeTeamMember,
  assignPatientTeam,
  // Patient Identity / Sync / Matching
  getPatientIdentity,
  getPatientSyncStatus,
  getPatientMatches,
  confirmPatientMatch,
  rejectPatientMatch,
  detachPatientIdentity,
  // Link Suggestions
  getPatientSuggestions,
  confirmPatientSuggestion,
  rejectPatientSuggestion,
  // Booking Messages
  fetchBookingMessages,
  sendBookingMessage,
};

export default institutionService;
