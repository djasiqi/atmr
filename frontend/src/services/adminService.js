// src/services/adminService.js
import apiClient from '../utils/apiClient';
/**
 * Récupère le token JWT stocké en local (si disponible).
 * ✅ Si pas de token, on compte sur les cookies httpOnly (apiClient gère automatiquement).
 */
const getAuthToken = () => {
  const token = localStorage.getItem('authToken');
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
 * Pilotage billing plateforme — synthèse KPIs.
 */
export const fetchBillingPilotageSummary = async (params = {}) => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.get('/admin/billing/pilotage/summary', {
    headers,
    params,
    timeout: 120000,
  });
  return response.data;
};

/**
 * Pilotage billing — tableau entreprises.
 */
export const fetchBillingPilotageCompanies = async (params = {}) => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.get('/admin/billing/pilotage/companies', {
    headers,
    params,
    timeout: 120000,
  });
  return response.data;
};

/**
 * Pilotage billing — détail entreprise.
 */
export const fetchBillingPilotageCompanyDetail = async (companyId, params = {}) => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.get(`/admin/billing/pilotage/companies/${companyId}`, {
    headers,
    params,
    timeout: 120000,
  });
  return response.data;
};

/**
 * Export CSV pilotage billing.
 */
export const downloadBillingPilotageExport = async (params = {}) => {
  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.get('/admin/billing/pilotage/export', {
    headers,
    params,
    responseType: 'blob',
    timeout: 120000,
  });
  const disposition = response.headers['content-disposition'];
  let filename = 'pilotage_export.csv';
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
        }
      : {
          paginate: false,
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
    return {
      users: [],
      total: 0,
      page: 1,
      per_page: 50,
      total_pages: 1,
      role_counts: null,
    };
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
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const token = getAuthToken();

    if (!updatedData.role) {
      throw new Error("Le champ 'role' est requis.");
    }

    // Si le rôle est 'driver' et qu'aucun company_id n'est fourni,
    // on renvoie une erreur pour signaler à l'interface de demander la sélection.
    if (updatedData.role.toLowerCase() === 'driver' && !updatedData.company_id) {
      throw new Error("Un company_id est requis pour le rôle 'driver'.");
    }

    const payload = {
      ...updatedData,
      role: String(updatedData.role).toLowerCase(), // <-- normalisation
    };
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.put(`/admin/users/${userId}/role`, payload, { headers });
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur mise à jour du rôle utilisateur :',
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Supprime un utilisateur.
 * @param {number} userId - L'ID de l'utilisateur.
 */
export const deleteUser = async (userId) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const token = getAuthToken();
    console.log(`📌 Tentative de suppression de l'utilisateur ID: ${userId}`);
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.delete(`/admin/users/${userId}`, { headers });
    console.log('✅ Utilisateur supprimé avec succès :', response.data);
    return response.data;
  } catch (error) {
    console.error(
      "❌ Erreur lors de la suppression de l'utilisateur :",
      error.response?.data || error.message
    );
    throw error;
  }
};

/**
 * Réinitialise le mot de passe d'un utilisateur.
 * @param {number} userId - L'ID de l'utilisateur.
 */
export const resetUserPassword = async (userId) => {
  if (!userId) {
    console.error('❌ Erreur : userId est undefined dans resetUserPassword !');
    return;
  }
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const token = getAuthToken();
    console.log(`🔄 Réinitialisation du mot de passe pour user ID: ${userId}`);
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await apiClient.post(
      `/admin/users/${userId}/reset-password`,
      {},
      { headers }
    );
    console.log('✅ Mot de passe réinitialisé :', response.data);
    return response.data;
  } catch (error) {
    console.error(
      '❌ Erreur lors de la réinitialisation du mot de passe :',
      error.response?.data || error.message
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

/** PUT config facturation plateforme pour une entreprise. */
export const putPlatformBillingCompanyConfig = async (companyId, payload) => {
  const response = await apiClient.put(
    `/admin/platform-billing/companies/${companyId}/config`,
    payload,
    { headers: _adminAuthHeaders() }
  );
  return response.data;
};
