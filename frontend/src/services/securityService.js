import apiClient from '../utils/apiClient';

export const fetchSessions = () => apiClient.get('/auth/sessions');

export const revokeSession = (id) => apiClient.delete(`/auth/sessions/${id}`);

export const revokeOtherSessions = () =>
  apiClient.post('/auth/sessions/revoke-others');

export const fetchAuditLogs = (params) =>
  apiClient.get('/company-settings/security/audit-logs', { params });

export const exportAuditLogs = (params) =>
  apiClient.get('/company-settings/security/audit-logs/export', {
    params,
    responseType: 'blob',
  });

export const AUDIT_ACTION_LABELS = {
  login_success: 'Connexion',
  user_login: 'Connexion',
  login_failed: 'Tentative de connexion échouée',
  user_login_failed: 'Tentative de connexion échouée',
  user_logout: 'Déconnexion',
  session_revoked: 'Session révoquée',
  sessions_bulk_revoked: 'Sessions révoquées en masse',
  settings_updated: 'Paramètres modifiés',
  billing_settings_updated: 'Paramètres de facturation modifiés',
  booking_cancelled: 'Réservation annulée',
  booking_created: 'Réservation créée',
  booking_modified: 'Réservation modifiée',
  booking_created_from_request: 'Réservation créée (depuis demande)',
  dispatch_complete: 'Dispatch terminé',
  audit_log_exported: 'Journal exporté',
  client_created: 'Client ajouté',
  client_modified: 'Client modifié',
  driver_assigned: 'Chauffeur assigné',
  invoice_generated: 'Facture générée',
  totp_enabled: '2FA activée',
  totp_disabled: '2FA désactivée',
  totp_challenge_failed: 'Tentative 2FA échouée',
  security_policy_updated: 'Politique de sécurité modifiée',
  password_changed: 'Mot de passe modifié',
  data_access: 'Accès aux données',
  user_invited: 'Utilisateur invité',
  recovery_codes_regenerated: 'Codes de secours régénérés',
  request_converted: 'Demande convertie en réservation',
  offer_accepted: 'Offre acceptée',
  offer_rejected: 'Offre refusée',
  transport_request_created: 'Demande de transport créée',
  transport_request_cancelled: 'Demande de transport annulée',
  patient_created: 'Patient ajouté',
  patient_updated: 'Patient modifié',
};

// ─── TOTP 2FA ────────────────────────────────────────────
export const fetchTotpStatus = () => apiClient.get('/auth/totp/status');

export const setupTotp = () => apiClient.post('/auth/totp/setup');

export const verifyTotp = (code) =>
  apiClient.post('/auth/totp/verify', { code });

export const disableTotp = (password) =>
  apiClient.post('/auth/totp/disable', { password });

export const regenerateRecoveryCodes = (code) =>
  apiClient.post('/auth/totp/recovery-codes', { code });

// ─── Security Policy ────────────────────────────────────
export const fetchSecurityPolicy = () =>
  apiClient.get('/company-settings/security/policy');

export const updateSecurityPolicy = (policy) =>
  apiClient.put('/company-settings/security/policy', policy);

// ─── Security Alerts ────────────────────────────────────
export const fetchSecurityAlerts = () =>
  apiClient.get('/company-settings/security/alerts');

export const fetchAlertPreferences = () =>
  apiClient.get('/company-settings/security/alerts/preferences');

export const updateAlertPreferences = (prefs) =>
  apiClient.put('/company-settings/security/alerts/preferences', prefs);

export const AUDIT_CATEGORY_LABELS = {
  all: 'Tous',
  security: 'Sécurité',
  dispatch: 'Dispatch',
  billing: 'Facturation',
  data: 'Données',
  operations: 'Opérations',
  settings: 'Paramètres',
  institution: 'Institution',
};
