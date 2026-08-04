// src/services/adminService.js
import apiClient from '../utils/apiClient';
import { getActiveAccessToken } from '../utils/webAuthSession';
/**
 * Récupère le token JWT stocké en local (si disponible).
 * ✅ Si pas de token, on compte sur les cookies httpOnly (apiClient gère automatiquement).
 */
const getAuthToken = () => {
  const token = getActiveAccessToken({ allowLegacy: true });
  // ✅ Si pas de token, on retourne null et apiClient utilisera les cookies httpOnly
  return token || null;
};

/**
 * Récupère les statistiques pour l'admin.
 */
export const fetchAdminStats = async () => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/admin/stats', { headers });
    return response.data;
  } catch (error) {
    console.error('❌ Erreur chargement stats admin :', error.response?.data || error.message);
    throw error;
  }
};

/**
 * Agrégat léger pour le tableau de bord admin (priorités, KPI, santé plateforme, tendances, activité).
 */
export const fetchAdminDashboardSummary = async () => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/admin/dashboard-summary', { headers });
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur chargement résumé dashboard admin :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Récupère les réservations récentes.
 */
export const fetchRecentBookings = async () => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/admin/recent-bookings', { headers });
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur chargement des courses récentes :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Liste paginée réservations admin plateforme (filtres, synthèse).
 * @param {Record<string, string|number|boolean|undefined>} params query string params
 */
export const fetchAdminBookings = async (params = {}) => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/admin/bookings', {
      headers,
      params,
      // Liste admin : synthèse + pagination ; éviter timeout si la base est volumineuse
      timeout: 90000,
    });
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur chargement réservations admin :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Détail réservation admin (timeline, liens).
 */
export const fetchAdminBookingDetail = async (bookingId) => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get(`/admin/bookings/${bookingId}`, { headers });
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur détail réservation admin :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Télécharge l'export CSV (mêmes filtres que la liste).
 */
export const downloadAdminBookingsExport = async (params = {}) => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.get('/admin/bookings/export', {
    headers,
    params,
    responseType: 'blob',
    timeout: 120000,
  });
  const disposition = response.headers['content-disposition'];
  let filename = 'bookings_export.csv';
  if (disposition && disposition.includes('filename=')) {
    const m = disposition.match(/filename="?([^";]+)"?/);
    if (m && m[1]) filename = m[1];
  }
  const url = window.URL.createObjectURL(new Blob([response.data], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

/**
 * Récupère les utilisateurs récents.
 */
export const fetchRecentUsers = async () => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/admin/recent-users', { headers });
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur chargement des utilisateurs récents :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Récupère la liste des utilisateurs (avec pagination/filtrage).
 */
export const fetchUsers = async (params = {}) => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const shouldPaginate = params.paginate !== false;
    const queryParams = shouldPaginate
      ? {
          paginate: 'true',
          page: params.page || 1,
          per_page: params.per_page || 50,
          search: params.search || '',
          role: params.role || '',
          sort_by: params.sort_by || 'created_at',
          sort_order: params.sort_order || 'desc',
          include_synthetic: params.include_synthetic ? 'true' : 'false',
          ...(params.company_id != null && params.company_id !== ''
            ? { company_id: params.company_id }
            : {}),
        }
      : {
          paginate: false,
          include_synthetic: params.include_synthetic ? 'true' : 'false',
          ...(params.company_id != null && params.company_id !== ''
            ? { company_id: params.company_id }
            : {}),
        };

    const response = await apiClient.get('/admin/users', {
      headers,
      params: queryParams,
    });

    const payload = response.data || {};
    const users = payload.users || [];
    const resolvedPerPage = Number(payload.per_page ?? params.per_page ?? users.length ?? 50) || 50;
    const safePerPage = Math.max(1, resolvedPerPage);
    const resolvedTotal = Number(payload.total ?? users.length ?? 0) || 0;
    const resolvedPage = Number(payload.page ?? params.page ?? 1) || 1;
    const resolvedTotalPages =
      Number(payload.total_pages) ||
      Math.max(Math.ceil(resolvedTotal / safePerPage), 1);

    return {
      users,
      total: resolvedTotal,
      page: resolvedPage,
      per_page: safePerPage,
      total_pages: resolvedTotalPages,
      role_counts: payload.role_counts || null,
    };
  } catch (error) {
    console.error(
      '❌ Erreur récupération des utilisateurs :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Liste unifiée des organisations partenaires (CP-PR1).
 */
export const fetchPartnerOrganizations = async (params = {}) => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/admin/partners/organizations', {
      headers,
      params: {
        page: params.page || 1,
        per_page: params.per_page || 50,
        include_synthetic: params.include_synthetic ? 'true' : 'false',
        organization_type: params.organization_type || undefined,
        configuration_status: params.configuration_status || undefined,
        lifecycle_status: params.lifecycle_status || undefined,
        search: params.search || undefined,
      },
    });
    return response.data || {};
  } catch (error) {
    console.error(
      '❌ Erreur organisations partenaires :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Fiche organisation control plane (public_id).
 */
export const fetchOrganizationDetail = async (publicId) => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get(`/admin/organizations/${publicId}`, {
      headers,
    });
    return response.data || {};
  } catch (error) {
    console.error(
      '❌ Erreur détail organisation :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Anomalies control plane persistées.
 */
export const fetchControlPlaneAnomalies = async (params = {}) => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/admin/control-plane/anomalies', {
      headers,
      params: {
        page: params.page || 1,
        per_page: params.per_page || 50,
        entity_type: params.entity_type || undefined,
        severity: params.severity || undefined,
        code: params.code || undefined,
        unresolved_only: params.unresolved_only === false ? 'false' : 'true',
      },
    });
    return response.data || {};
  } catch (error) {
    console.error(
      '❌ Erreur anomalies control plane :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Diagnostic d'intégrité d'un compte (lecture seule).
 */
export const fetchAccountIntegrity = async (userId) => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get(
      `/admin/partners/accounts/${userId}/integrity`,
      { headers }
    );
    return response.data || {};
  } catch (error) {
    console.error(
      '❌ Erreur diagnostic compte :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Récupère la liste de toutes les entreprises.
 * Utilise GET /companies qui liste toutes les companies (admin uniquement).
 */
export const fetchCompanies = async () => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/admin/companies', { headers });
    console.log('📌 Données reçues de /admin/companies :', response.data);
    // La réponse peut être un array ou un objet { companies: [...] }
    return response.data?.companies ?? (Array.isArray(response.data) ? response.data : []);
  } catch (error) {
    console.error(
      '❌ Erreur lors de la récupération des entreprises :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Récupère la liste de toutes les institutions (cliniques, EMS, hôpitaux).
 * Utilise GET /admin/institutions (admin uniquement).
 */
export const fetchInstitutions = async () => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/admin/institutions', { headers });
    console.log('📌 Données reçues de /admin/institutions :', response.data);
    return response.data?.institutions ?? (Array.isArray(response.data) ? response.data : []);
  } catch (error) {
    console.error(
      '❌ Erreur lors de la récupération des institutions :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Met à jour le rôle d'un utilisateur.
 * Si le rôle 'driver' est sélectionné sans fournir de company_id,
 * affiche la liste des entreprises et demande à l'admin de choisir.
 */
export const updateUserRole = async (userId, updatedData) => {
  try {
    const token = getAuthToken();

    if (!updatedData.role) {
      throw new Error("Le champ 'role' est requis.");
    }

    if (updatedData.role.toLowerCase() === 'driver' && !updatedData.company_id) {
      throw new Error("Un company_id est requis pour le rôle 'driver'.");
    }

    const payload = {
      ...updatedData,
      role: String(updatedData.role).toLowerCase(),
    };
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.put(`/admin/users/${userId}/role`, payload, { headers });
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur mise à jour du rôle utilisateur :',
      error.response?.data?.message || error.message
    );
    throw error;
  }
};

/**
 * Prévisualise une transition de rôle.
 */
export const previewUserRoleTransition = async (userId, payload) => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.post(
    `/admin/users/${userId}/role-transition/preview`,
    { ...payload, role: String(payload.role || '').toLowerCase() },
    { headers }
  );
  return response.data;
};

/**
 * Contexte drawer gestion compte.
 */
export const fetchAccountManageContext = async (userId) => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.get(`/admin/accounts/${userId}/manage-context`, {
    headers,
  });
  return response.data || {};
};

/**
 * Réinitialise le mot de passe d'un utilisateur.
 * @param {number} userId
 * @param {{ reason: string }} payload
 */
export const resetUserPassword = async (userId, payload = {}) => {
  if (!userId) {
    console.error('❌ Erreur : userId est undefined dans resetUserPassword !');
    return;
  }
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.post(
      `/admin/users/${userId}/reset-password`,
      { reason: payload.reason },
      { headers }
    );
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur lors de la réinitialisation du mot de passe :',
      error.response?.data?.message || error.message
    );
    throw error;
  }
};

/**
 * Soft-disable / réactivation chauffeur (sans changer le rôle).
 * @param {number} userId
 * @param {{ is_active: boolean, reason: string, expected_is_active?: boolean }} payload
 */
export const updateDriverStatus = async (userId, payload) => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.put(
    `/admin/users/${userId}/driver-status`,
    payload,
    { headers }
  );
  return response.data;
};

/**
 * Révoque toutes les sessions d'un compte (sans reset MDP).
 * @param {number} userId
 * @param {{ reason: string }} payload
 */
export const revokeUserSessions = async (userId, payload = {}) => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.post(
    `/admin/users/${userId}/revoke-sessions`,
    { reason: payload.reason },
    { headers }
  );
  return response.data;
};

/**
 * Supprime un utilisateur (backend gelé hors TESTING).
 */
export const deleteUser = async (userId) => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.delete(`/admin/users/${userId}`, { headers });
    return response.data;
  } catch (error) {
    console.error(
      "❌ Erreur lors de la suppression de l'utilisateur :",
      error.response?.data?.message || error.message
    );
    throw error;
  }
};

/**
 * Déclenche l'optimisation Optuna pour les hyperparamètres DQN.
 * @param {Object} config - Configuration de l'optimisation
 * @param {number} [config.company_id] - ID de l'entreprise (optionnel)
 * @param {string} [config.data_period] - Période de données: 'day', 'week', 'month', 'custom'
 * @param {number} [config.n_trials] - Nombre de trials Optuna
 * @param {number} [config.training_episodes] - Épisodes d'entraînement par trial
 * @param {number} [config.eval_episodes] - Épisodes d'évaluation par trial
 * @param {number} [config.custom_days] - Nombre de jours si data_period='custom'
 */
export const runOptunaOptimization = async (config = {}) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const token = getAuthToken();
    console.log('🚀 Démarrage de l\'optimisation Optuna...', config);

    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.post(
      '/admin/optuna/optimize',
      config,
      { headers }
    );

    console.log('✅ Optimisation Optuna démarrée :', response.data);
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur lors du démarrage de l\'optimisation Optuna :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Entraîne un modèle DQN complet avec les hyperparamètres optimaux.
 * @param {Object} config - Configuration de l'entraînement
 * @param {string} [config.config_path] - Chemin vers optimal_config.json
 * @param {string} [config.study_name] - Nom de l'étude Optuna
 * @param {string} [config.model_output_path] - Chemin de sortie pour le modèle
 * @param {number} [config.training_episodes] - Nombre d'épisodes d'entraînement
 * @param {number} [config.eval_episodes] - Nombre d'épisodes d'évaluation
 * @param {number} [config.company_id] - ID de l'entreprise
 */
export const trainModelWithOptimalParams = async (config = {}) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const token = getAuthToken();
    console.log('🎓 Démarrage de l\'entraînement du modèle avec hyperparamètres optimaux...', config);

    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.post(
      '/admin/optuna/train',
      config,
      { headers }
    );

    console.log('✅ Entraînement du modèle démarré :', response.data);
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur lors du démarrage de l\'entraînement du modèle :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Agrégat plateforme (Admin Ops) — lecture seule.
 * GET /platform/status
 */
export const fetchPlatformStatus = async () => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/platform/status', { headers });
    return response.data;
  } catch (error) {
    console.error(
      'Erreur chargement platform/status :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Métriques runtime (hors hot path) — lecture seule.
 * GET /platform/runtime — pas de polling automatique côté UI par défaut.
 */
export const fetchPlatformRuntime = async () => {
  try {
    const token = getAuthToken();
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.get('/platform/runtime', { headers });
    return response.data;
  } catch (error) {
    console.error(
      'Erreur chargement platform/runtime :',
      error.response?.data || error.message
    );
    throw error;
  }
};

const platformHeaders = () => {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

/** GET /platform/me */
export const fetchPlatformMe = async () => {
  const response = await apiClient.get('/platform/me', { headers: platformHeaders() });
  return response.data;
};

/** GET /admin/capabilities — capacités admin.* + flag enforcement (PR2bis) */
export const fetchAdminCapabilities = async () => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.get('/admin/capabilities', { headers });
  return response.data;
};

/** POST /platform/policies/evaluate */
export const postPlatformPoliciesEvaluate = async (body) => {
  const response = await apiClient.post('/platform/policies/evaluate', body, {
    headers: platformHeaders(),
  });
  return response.data;
};

/** GET /platform/tenants */
export const fetchPlatformTenants = async (params = {}) => {
  const response = await apiClient.get('/platform/tenants', {
    headers: platformHeaders(),
    params,
  });
  return response.data;
};

/** GET /platform/tenants/:id */
export const fetchPlatformTenant = async (tenantId) => {
  const response = await apiClient.get(`/platform/tenants/${tenantId}`, {
    headers: platformHeaders(),
  });
  return response.data;
};

/** POST /platform/tenants/:id/suspend/preview */
export const postPlatformTenantSuspendPreview = async (tenantId, body = {}) => {
  const response = await apiClient.post(`/platform/tenants/${tenantId}/suspend/preview`, body, {
    headers: platformHeaders(),
  });
  return response.data;
};

/** POST /platform/tenants/:id/suspend */
export const postPlatformTenantSuspend = async (tenantId, body) => {
  const response = await apiClient.post(`/platform/tenants/${tenantId}/suspend`, body, {
    headers: platformHeaders(),
  });
  return response.data;
};

/** GET /platform/runbooks */
export const fetchPlatformRunbooks = async () => {
  const response = await apiClient.get('/platform/runbooks', { headers: platformHeaders() });
  return response.data;
};

/** POST /platform/runbooks/:id/executions */
export const postPlatformRunbookExecution = async (runbookId, body) => {
  const response = await apiClient.post(`/platform/runbooks/${runbookId}/executions`, body, {
    headers: platformHeaders(),
  });
  return response.data;
};

/** POST /platform/search */
export const postPlatformSearch = async (body) => {
  const response = await apiClient.post('/platform/search', body, { headers: platformHeaders() });
  return response.data;
};

/** GET /platform/reconciliation */
export const fetchPlatformReconciliation = async (tenantId) => {
  const response = await apiClient.get('/platform/reconciliation', {
    headers: platformHeaders(),
    params: { tenant_id: tenantId },
  });
  return response.data;
};

/** GET /platform/audit-events */
export const fetchPlatformAuditEvents = async (params = {}) => {
  const response = await apiClient.get('/platform/audit-events', {
    headers: platformHeaders(),
    params,
  });
  return response.data;
};

/** GET /platform/audit-events/replay — timeline ordonnée (réponse API telle quelle) */
export const fetchPlatformAuditReplay = async (correlationId) => {
  const response = await apiClient.get('/platform/audit-events/replay', {
    headers: platformHeaders(),
    params: { correlation_id: correlationId },
  });
  return response.data;
};

/** POST /platform/runbooks/executions/:executionId/rollback */
export const postPlatformRunbookRollback = async (executionId) => {
  const response = await apiClient.post(
    `/platform/runbooks/executions/${executionId}/rollback`,
    {},
    { headers: platformHeaders() }
  );
  return response.data;
};

// ─── Facturation plateforme LIRIE — GET/POST /admin/platform-billing/* ───

const _adminAuthHeaders = () => {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

/** Liste des périodes de facturation plateforme. */
export const fetchPlatformBillingPeriods = async () => {
  const response = await apiClient.get('/admin/platform-billing/periods', {
    headers: _adminAuthHeaders(),
    timeout: 60000,
  });
  return response.data;
};

/** Crée ou récupère une période (draft). */
export const createPlatformBillingPeriod = async (billingYear, billingMonth) => {
  const response = await apiClient.post(
    '/admin/platform-billing/periods',
    { billing_year: billingYear, billing_month: billingMonth },
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Recalcule les brouillons pour une période draft. */
export const recalculatePlatformBillingPeriod = async (periodId) => {
  const response = await apiClient.post(
    `/admin/platform-billing/periods/${periodId}/recalculate`,
    {},
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Verrouille une période. */
export const lockPlatformBillingPeriod = async (periodId) => {
  const response = await apiClient.post(
    `/admin/platform-billing/periods/${periodId}/lock`,
    {},
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Relevés pour une période. */
export const fetchPlatformBillingPeriodInvoices = async (periodId) => {
  const response = await apiClient.get(`/admin/platform-billing/periods/${periodId}/invoices`, {
    headers: _adminAuthHeaders(),
    timeout: 120000,
  });
  return response.data;
};

/** Détail d'un relevé + lignes. */
export const fetchPlatformBillingInvoice = async (invoiceId) => {
  const response = await apiClient.get(`/admin/platform-billing/invoices/${invoiceId}`, {
    headers: _adminAuthHeaders(),
  });
  return response.data;
};

/** Export CSV UTF-8 d'une période. */
export const downloadPlatformBillingPeriodExport = async (periodId) => {
  const response = await apiClient.get(`/admin/platform-billing/periods/${periodId}/export`, {
    headers: _adminAuthHeaders(),
    responseType: 'blob',
    timeout: 120000,
  });
  const disposition = response.headers['content-disposition'];
  let filename = `platform-billing-period-${periodId}.csv`;
  if (disposition && disposition.includes('filename=')) {
    const m = disposition.match(/filename="?([^";]+)"?/);
    if (m && m[1]) filename = m[1];
  }
  const url = window.URL.createObjectURL(
    new Blob([response.data], { type: 'text/csv;charset=utf-8' })
  );
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

/** Liste entreprises + dernière config facturation plateforme (paramètres transporteurs). */
export const fetchPlatformBillingCompaniesConfig = async (params = {}) => {
  const response = await apiClient.get('/admin/platform-billing/companies/config', {
    headers: _adminAuthHeaders(),
    params,
    timeout: 60000,
  });
  return response.data;
};

/** Grille d'abonnement globale (lecture). */
export const fetchPlatformSubscriptionPricing = async () => {
  const response = await apiClient.get('/admin/platform-billing/subscription-pricing', {
    headers: _adminAuthHeaders(),
  });
  return response.data;
};

/** Config facturation plateforme d'une entreprise (lecture). */
export const fetchPlatformBillingCompanyConfig = async (companyId) => {
  const response = await apiClient.get(
    `/admin/platform-billing/companies/${companyId}/config`,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** PUT config facturation plateforme pour une entreprise. */
export const putPlatformBillingCompanyConfig = async (companyId, payload) => {
  const response = await apiClient.put(
    `/admin/platform-billing/companies/${companyId}/config`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Feature flags facturation plateforme. */
export const fetchPlatformBillingFeatureFlags = async () => {
  const response = await apiClient.get('/admin/platform-billing/feature-flags', {
    headers: _adminAuthHeaders(),
  });
  return response.data;
};

export const fetchPlatformBillingContracts = async (companyId) => {
  const response = await apiClient.get(
    `/admin/platform-billing/companies/${companyId}/billing-contracts`,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const createPlatformBillingContract = async (companyId, payload) => {
  const response = await apiClient.post(
    `/admin/platform-billing/companies/${companyId}/billing-contracts`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const closePlatformBillingContract = async (contractId, payload = {}) => {
  const response = await apiClient.post(
    `/admin/platform-billing/billing-contracts/${contractId}/close`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const fetchPlatformBillingCreditor = async () => {
  const response = await apiClient.get('/admin/platform-billing/creditor', {
    headers: _adminAuthHeaders(),
  });
  return response.data;
};

export const putPlatformBillingCreditor = async (payload) => {
  const response = await apiClient.put('/admin/platform-billing/creditor', payload, {
    headers: _adminAuthHeaders(),
  });
  return response.data;
};

/** Adresse de facturation débiteur (transporteur) pour QR-facture plateforme. */
export const putPlatformBillingDebtorAddress = async (companyId, payload) => {
  const response = await apiClient.put(
    `/admin/platform-billing/companies/${companyId}/debtor-address`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Génère / régénère le DOCX d'accord partenaire pour une version commerciale. */
export const generatePartnerAgreement = async (contractId) => {
  const response = await apiClient.post(
    `/admin/platform-billing/billing-contracts/${contractId}/agreements`,
    {},
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const fetchPartnerAgreements = async (contractId) => {
  const response = await apiClient.get(
    `/admin/platform-billing/billing-contracts/${contractId}/agreements`,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const markPartnerAgreementSent = async (agreementId) => {
  const response = await apiClient.post(
    `/admin/platform-billing/agreements/${agreementId}/mark-sent`,
    {},
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const voidPartnerAgreement = async (agreementId, reason) => {
  const response = await apiClient.post(
    `/admin/platform-billing/agreements/${agreementId}/void`,
    { reason },
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const uploadPartnerAgreementSigned = async (
  agreementId,
  file,
  agreementSignedOn
) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('agreement_signed_on', agreementSignedOn);
  const response = await apiClient.post(
    `/admin/platform-billing/agreements/${agreementId}/upload-signed`,
    formData,
    {
      headers: {
        ..._adminAuthHeaders(),
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
};

export const downloadPartnerAgreementDocxUrl = (agreementId) =>
  `/admin/platform-billing/agreements/${agreementId}/docx`;

export const downloadPartnerAgreementSignedUrl = (agreementId) =>
  `/admin/platform-billing/agreements/${agreementId}/signed`;

/** Téléchargement authentifié (blob) d'un accord. */
export const downloadPartnerAgreementFile = async (urlPath, filename) => {
  const response = await apiClient.get(urlPath, {
    headers: _adminAuthHeaders(),
    responseType: 'blob',
  });
  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || 'document';
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

export const fetchPlatformPricingGrids = async () => {
  const response = await apiClient.get('/admin/platform-billing/pricing-grids', {
    headers: _adminAuthHeaders(),
  });
  return response.data;
};

export const validatePlatformBillingInvoice = async (invoiceId) => {
  const response = await apiClient.post(
    `/admin/platform-billing/invoices/${invoiceId}/validate`,
    {},
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Réouvre un relevé validé/verrouillé (annule facture non envoyée) pour correction. */
export const reopenPlatformBillingInvoice = async (invoiceId) => {
  const response = await apiClient.post(
    `/admin/platform-billing/invoices/${invoiceId}/reopen`,
    {},
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const issuePlatformBillingInvoice = async (invoiceId) => {
  const response = await apiClient.post(
    `/admin/platform-billing/invoices/${invoiceId}/issue`,
    {},
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Télécharge le PDF d'une facture plateforme émise. */
export const downloadPlatformIssuedInvoicePdf = async (issuedId) => {
  const response = await apiClient.get(
    `/admin/platform-billing/issued-invoices/${issuedId}/pdf`,
    {
      headers: _adminAuthHeaders(),
      responseType: 'blob',
      timeout: 120000,
    }
  );
  const disposition = response.headers['content-disposition'];
  let filename = `facture-lirie-${issuedId}.pdf`;
  if (disposition && disposition.includes('filename=')) {
    const m = disposition.match(/filename="?([^";]+)"?/);
    if (m && m[1]) filename = m[1];
  }
  const url = window.URL.createObjectURL(
    new Blob([response.data], { type: 'application/pdf' })
  );
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

export const fetchPlatformBillingInvoiceReadiness = async (invoiceId) => {
  const response = await apiClient.get(
    `/admin/platform-billing/invoices/${invoiceId}/readiness`,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const fetchPlatformBillingStatementItems = async (invoiceId) => {
  const response = await apiClient.get(
    `/admin/platform-billing/invoices/${invoiceId}/statement-items`,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Liste des heures de support plateforme (optionnellement filtrées par entreprise). */
export const fetchPlatformSupportEntries = async (companyId) => {
  const response = await apiClient.get('/admin/platform-billing/support-entries', {
    headers: _adminAuthHeaders(),
    params: companyId != null ? { company_id: companyId } : undefined,
  });
  return response.data;
};

/**
 * Saisie d'heures de support plateforme.
 * @param {object} payload
 * @param {number} payload.company_id
 * @param {number|string} [payload.duration_hours]
 * @param {number} [payload.duration_minutes]
 * @param {string} [payload.hourly_rate_snapshot]
 * @param {string} [payload.description]
 * @param {string} [payload.category]
 * @param {number} [payload.billing_period_id]
 * @param {boolean} [payload.auto_validate]
 * @param {boolean} [payload.recalculate_period]
 */
export const createPlatformSupportEntry = async (payload) => {
  const response = await apiClient.post(
    '/admin/platform-billing/support-entries',
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Rectifie une entrée support (durée, catégorie, tarif…). */
export const updatePlatformSupportEntry = async (entryId, payload) => {
  const response = await apiClient.patch(
    `/admin/platform-billing/support-entries/${entryId}`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Supprime une entrée support et recalcule le relevé si demandé. */
export const deletePlatformSupportEntry = async (entryId, params = {}) => {
  const response = await apiClient.delete(
    `/admin/platform-billing/support-entries/${entryId}`,
    {
      headers: _adminAuthHeaders(),
      params,
    }
  );
  return response.data;
};

export const sendPlatformIssuedInvoice = async (issuedId) => {
  const response = await apiClient.post(
    `/admin/platform-billing/issued-invoices/${issuedId}/send`,
    {},
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

export const payPlatformIssuedInvoice = async (issuedId, payload) => {
  const response = await apiClient.post(
    `/admin/platform-billing/issued-invoices/${issuedId}/payments`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/**
 * Registre des factures LIRIE émises (liste + stats + pagination + filtres).
 * @param {object} [params] q, status, payment_state, year, month, with_balance,
 *   overdue_only, with_dunning, document_type, page, per_page, sort_by, sort_order
 */
export const fetchPlatformIssuedInvoices = async (params = {}) => {
  const response = await apiClient.get('/admin/platform-billing/issued-invoices', {
    headers: _adminAuthHeaders(),
    params,
    timeout: 60000,
  });
  return response.data;
};

/** Détail d'une facture LIRIE émise (lignes, paiements, historique échéance, relance). */
export const fetchPlatformIssuedInvoice = async (issuedId) => {
  const response = await apiClient.get(
    `/admin/platform-billing/issued-invoices/${issuedId}`,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Export CSV UTF-8 du registre des factures LIRIE émises (mêmes filtres que la liste). */
export const exportPlatformIssuedInvoices = async (params = {}) => {
  const response = await apiClient.get(
    '/admin/platform-billing/issued-invoices/export',
    {
      headers: _adminAuthHeaders(),
      params,
      responseType: 'blob',
      timeout: 120000,
    }
  );
  const disposition = response.headers['content-disposition'];
  let filename = 'factures-lirie-emises.csv';
  if (disposition && disposition.includes('filename=')) {
    const m = disposition.match(/filename="?([^";]+)"?/);
    if (m && m[1]) filename = m[1];
  }
  const url = window.URL.createObjectURL(
    new Blob([response.data], { type: 'text/csv;charset=utf-8' })
  );
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

/** Modifie l'échéance d'une facture LIRIE émise (motif obligatoire). */
export const updatePlatformIssuedDueDate = async (issuedId, payload) => {
  const response = await apiClient.patch(
    `/admin/platform-billing/issued-invoices/${issuedId}/due-date`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Contre-passe (annule) un paiement enregistré sur une facture LIRIE émise. */
export const reversePlatformIssuedPayment = async (issuedId, paymentId, payload = {}) => {
  const response = await apiClient.post(
    `/admin/platform-billing/issued-invoices/${issuedId}/payments/${paymentId}/reverse`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Annule une facture LIRIE émise (uniquement avant envoi, sans paiement). */
export const cancelPlatformIssuedInvoice = async (issuedId) => {
  const response = await apiClient.post(
    `/admin/platform-billing/issued-invoices/${issuedId}/cancel`,
    {},
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Crée une note de crédit (avoir) totale pour une facture LIRIE émise (motif obligatoire). */
export const createPlatformIssuedCreditNote = async (issuedId, payload) => {
  const response = await apiClient.post(
    `/admin/platform-billing/issued-invoices/${issuedId}/credit-note`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/**
 * Met à jour l'état d'accès commercial billing d'une entreprise (active|partial|full).
 * Distinct de platform_suspended (gouvernance).
 */
export const setCompanyBillingAccess = async (companyId, payload) => {
  const response = await apiClient.put(
    `/admin/platform-billing/companies/${companyId}/billing-access`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/**
 * Met en pause le recouvrement automatique (dunning) pour une entreprise.
 * Payload : { paused_until?: ISO, days?: number, reason }
 */
export const pauseCompanyDunning = async (companyId, payload = {}) => {
  const response = await apiClient.post(
    `/admin/platform-billing/companies/${companyId}/dunning/pause`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Reprend immédiatement le recouvrement automatique. */
export const resumeCompanyDunning = async (companyId, payload = {}) => {
  const response = await apiClient.post(
    `/admin/platform-billing/companies/${companyId}/dunning/resume`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Approbation plateforme Company (indépendante du dispatch). */
export const setCompanyApproval = async (companyId, payload) => {
  const response = await apiClient.put(
    `/admin/companies/${companyId}/approval`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Statut dispatch Company. */
export const setCompanyDispatchStatus = async (companyId, payload) => {
  const response = await apiClient.put(
    `/admin/companies/${companyId}/dispatch-status`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};

/** Preview d'impact avant désactivation dispatch. */
export const fetchCompanyDispatchDisablePreview = async (companyId) => {
  const response = await apiClient.get(
    `/admin/companies/${companyId}/dispatch-disable-preview`,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};
