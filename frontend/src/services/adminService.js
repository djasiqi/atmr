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
