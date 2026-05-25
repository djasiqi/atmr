// utils/institutionPermissions.js
/**
 * ÉTAPE 6: Gestion des permissions UI pour le portail Institution
 * 
 * Rôles institution_role:
 * - institution_admin: Tous droits
 * - institution_requester: Créer/modifier/envoyer/annuler les demandes
 * - institution_reader: Lecture seule
 * - institution_billing: Modifier les infos de facturation
 * - institution_curator: Curateur (curatelle) — gère demandes + facturation pour ses protégés
 */

// ============================================================================
// Actions disponibles
// ============================================================================

export const INSTITUTION_ACTIONS = {
  // Demandes
  CREATE_REQUEST: 'create_request',
  EDIT_REQUEST: 'edit_request',
  SEND_REQUEST: 'send_request',
  CANCEL_REQUEST: 'cancel_request',
  VIEW_REQUEST: 'view_request',
  
  // Patients
  CREATE_PATIENT: 'create_patient',
  EDIT_PATIENT: 'edit_patient',
  VIEW_PATIENT: 'view_patient',
  
  // Settings
  VIEW_SETTINGS: 'view_settings',
  EDIT_PREFERENCES: 'edit_preferences',
  MANAGE_API_KEYS: 'manage_api_keys',
  
  // Billing
  EDIT_BILLING: 'edit_billing',
  EDIT_REQUEST_BILLING: 'edit_request_billing', // Changer facturation institution/patient sur une demande

  // Données administratives sensibles (AVS, assurance, curatelle)
  VIEW_ADMIN_DATA: 'view_admin_data',
  EDIT_ADMIN_DATA: 'edit_admin_data',

  // Édition partielle patient (coordonnées, assurance, curatelle) — rôle billing
  EDIT_PATIENT_BILLING_DATA: 'edit_patient_billing_data',
};

// ============================================================================
// Mapping rôle -> actions autorisées
// ============================================================================

const ROLE_PERMISSIONS = {
  institution_admin: [
    // Tous les droits
    INSTITUTION_ACTIONS.CREATE_REQUEST,
    INSTITUTION_ACTIONS.EDIT_REQUEST,
    INSTITUTION_ACTIONS.SEND_REQUEST,
    INSTITUTION_ACTIONS.CANCEL_REQUEST,
    INSTITUTION_ACTIONS.VIEW_REQUEST,
    INSTITUTION_ACTIONS.CREATE_PATIENT,
    INSTITUTION_ACTIONS.EDIT_PATIENT,
    INSTITUTION_ACTIONS.VIEW_PATIENT,
    INSTITUTION_ACTIONS.VIEW_SETTINGS,
    INSTITUTION_ACTIONS.EDIT_PREFERENCES,
    INSTITUTION_ACTIONS.MANAGE_API_KEYS,
    INSTITUTION_ACTIONS.EDIT_BILLING,
    INSTITUTION_ACTIONS.EDIT_REQUEST_BILLING,
    INSTITUTION_ACTIONS.VIEW_ADMIN_DATA,
    INSTITUTION_ACTIONS.EDIT_ADMIN_DATA,
    INSTITUTION_ACTIONS.EDIT_PATIENT_BILLING_DATA,
  ],
  
  institution_requester: [
    // CRUD demandes + patients + voir settings
    INSTITUTION_ACTIONS.CREATE_REQUEST,
    INSTITUTION_ACTIONS.EDIT_REQUEST,
    INSTITUTION_ACTIONS.SEND_REQUEST,
    INSTITUTION_ACTIONS.CANCEL_REQUEST,
    INSTITUTION_ACTIONS.VIEW_REQUEST,
    INSTITUTION_ACTIONS.CREATE_PATIENT,
    INSTITUTION_ACTIONS.EDIT_PATIENT,
    INSTITUTION_ACTIONS.VIEW_PATIENT,
    INSTITUTION_ACTIONS.VIEW_SETTINGS,
  ],
  
  institution_reader: [
    // Lecture seule + voir settings
    INSTITUTION_ACTIONS.VIEW_REQUEST,
    INSTITUTION_ACTIONS.VIEW_PATIENT,
    INSTITUTION_ACTIONS.VIEW_SETTINGS,
  ],
  
  institution_billing: [
    // Lecture + facturation complète + changer facturation sur demandes + données admin patient
    INSTITUTION_ACTIONS.VIEW_REQUEST,
    INSTITUTION_ACTIONS.VIEW_PATIENT,
    INSTITUTION_ACTIONS.EDIT_BILLING,
    INSTITUTION_ACTIONS.EDIT_REQUEST_BILLING,
    INSTITUTION_ACTIONS.VIEW_SETTINGS,
    INSTITUTION_ACTIONS.VIEW_ADMIN_DATA,
    INSTITUTION_ACTIONS.EDIT_ADMIN_DATA,
    INSTITUTION_ACTIONS.EDIT_PATIENT_BILLING_DATA,
  ],

  institution_curator: [
    // Demandes + patients + facturation, scope filtré par équipe (backend)
    INSTITUTION_ACTIONS.CREATE_REQUEST,
    INSTITUTION_ACTIONS.EDIT_REQUEST,
    INSTITUTION_ACTIONS.SEND_REQUEST,
    INSTITUTION_ACTIONS.CANCEL_REQUEST,
    INSTITUTION_ACTIONS.VIEW_REQUEST,
    INSTITUTION_ACTIONS.CREATE_PATIENT,
    INSTITUTION_ACTIONS.EDIT_PATIENT,
    INSTITUTION_ACTIONS.VIEW_PATIENT,
    INSTITUTION_ACTIONS.EDIT_BILLING,
    INSTITUTION_ACTIONS.EDIT_REQUEST_BILLING,
    INSTITUTION_ACTIONS.VIEW_SETTINGS,
    INSTITUTION_ACTIONS.VIEW_ADMIN_DATA,
    INSTITUTION_ACTIONS.EDIT_ADMIN_DATA,
    INSTITUTION_ACTIONS.EDIT_PATIENT_BILLING_DATA,
  ],
};

// ============================================================================
// Fonctions de vérification
// ============================================================================

/**
 * Vérifie si un utilisateur peut effectuer une action
 * @param {string} institutionRole - Rôle de l'utilisateur (ex: "institution_admin")
 * @param {string} action - Action à vérifier (ex: INSTITUTION_ACTIONS.CREATE_REQUEST)
 * @returns {boolean}
 */
export function can(institutionRole, action) {
  if (!institutionRole) return false;
  
  // Normaliser le rôle (lowercase, sans "institution_" prefix si présent)
  const normalizedRole = institutionRole.toLowerCase();
  
  const permissions = ROLE_PERMISSIONS[normalizedRole];
  if (!permissions) return false;
  
  return permissions.includes(action);
}

/**
 * Vérifie si l'utilisateur est admin
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function isAdmin(institutionRole) {
  return institutionRole?.toLowerCase() === 'institution_admin';
}

/**
 * Vérifie si l'utilisateur est demandeur (requester)
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function isRequester(institutionRole) {
  return institutionRole?.toLowerCase() === 'institution_requester';
}

/**
 * Vérifie si l'utilisateur est curateur
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function isCurator(institutionRole) {
  return institutionRole?.toLowerCase() === 'institution_curator';
}

/**
 * Vérifie si l'utilisateur peut créer/modifier des demandes
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function canManageRequests(institutionRole) {
  return can(institutionRole, INSTITUTION_ACTIONS.CREATE_REQUEST);
}

/**
 * Vérifie si l'utilisateur peut modifier la facturation
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function canEditBilling(institutionRole) {
  return can(institutionRole, INSTITUTION_ACTIONS.EDIT_BILLING);
}

/** Montants visibles : admin, billing, curator uniquement */
export function canViewFinancialAmounts(institutionRole) {
  const role = institutionRole?.toLowerCase();
  return role === 'institution_admin'
    || role === 'institution_billing'
    || role === 'institution_curator';
}

/** Bloc facturation visible (édition ou lecture) : admin + billing */
export function canViewBillingSection(institutionRole) {
  return canViewFinancialAmounts(institutionRole);
}

/**
 * Vérifie si l'utilisateur peut changer la facturation (institution/patient) sur une demande
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function canEditRequestBilling(institutionRole) {
  return can(institutionRole, INSTITUTION_ACTIONS.EDIT_REQUEST_BILLING);
}

/**
 * Vérifie si l'utilisateur peut voir les données administratives sensibles (AVS, assurance, curatelle)
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function canViewAdminData(institutionRole) {
  return can(institutionRole, INSTITUTION_ACTIONS.VIEW_ADMIN_DATA);
}

/**
 * Vérifie si l'utilisateur peut modifier les données administratives sensibles
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function canEditAdminData(institutionRole) {
  return can(institutionRole, INSTITUTION_ACTIONS.EDIT_ADMIN_DATA);
}

/**
 * Vérifie si l'utilisateur peut modifier les données billing du patient
 * (coordonnées, assurance, curatelle) — rôle billing + admin + requester
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function canEditPatientBillingData(institutionRole) {
  return can(institutionRole, INSTITUTION_ACTIONS.EDIT_PATIENT_BILLING_DATA);
}

/**
 * Vérifie si l'utilisateur peut accéder aux settings
 * @param {string} institutionRole
 * @returns {boolean}
 */
export function canViewSettings(institutionRole) {
  return isAdmin(institutionRole) || can(institutionRole, INSTITUTION_ACTIONS.VIEW_SETTINGS);
}

/**
 * Retourne le libellé français du rôle
 * @param {string} institutionRole
 * @returns {string}
 */
export function getRoleLabel(institutionRole) {
  const labels = {
    institution_admin: 'Administrateur',
    institution_requester: 'Demandeur',
    institution_reader: 'Lecteur',
    institution_billing: 'Facturation',
    institution_curator: 'Curateur',
  };
  return labels[institutionRole?.toLowerCase()] || institutionRole || 'Inconnu';
}

/**
 * Retourne la couleur du badge du rôle
 * @param {string} institutionRole
 * @returns {string}
 */
export function getRoleBadgeColor(institutionRole) {
  // Colors aligned with Lirie brand palette — see docs/brand/lirie-brand-guidelines.md
  const colors = {
    institution_admin: '#00796B',  // Brand teal (primary)
    institution_requester: '#059669', // Brand success green
    institution_reader: '#94A3B8',   // Brand text-muted
    institution_billing: '#0A88EF',  // Brand accent blue
    institution_curator: '#7C3AED',  // Curator purple
  };
  return colors[institutionRole?.toLowerCase()] || '#94A3B8';
}

const institutionPermissions = {
  INSTITUTION_ACTIONS,
  can,
  isAdmin,
  isRequester,
  isCurator,
  canManageRequests,
  canEditBilling,
  canViewFinancialAmounts,
  canViewBillingSection,
  canEditRequestBilling,
  canViewAdminData,
  canEditAdminData,
  canEditPatientBillingData,
  canViewSettings,
  getRoleLabel,
  getRoleBadgeColor,
};

export default institutionPermissions;
