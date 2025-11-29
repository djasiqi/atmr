// src/services/adminService.js
import apiClient from '../utils/apiClient';
/**
 * Récupère le token JWT stocké en local.
 */
const getAuthToken = () => {
  const token = localStorage.getItem('authToken');
  if (!token) {
    console.error("🚨 Erreur : Aucun token JWT trouvé. L'utilisateur doit être connecté.");
  }
  return token;
};

/**
 * Récupère les statistiques pour l'admin.
 */
export const fetchAdminStats = async () => {
  try {
    const token = getAuthToken();
    const response = await apiClient.get('/admin/stats', {
      headers: { Authorization: `Bearer ${token}` },
    });
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
    const token = getAuthToken();
    const response = await apiClient.get('/admin/recent-bookings', {
      headers: { Authorization: `Bearer ${token}` },
    });
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
    const token = getAuthToken();
    const response = await apiClient.get('/admin/recent-users', {
      headers: { Authorization: `Bearer ${token}` },
    });
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
 * Récupère la liste de tous les utilisateurs.
 */
export const fetchUsers = async () => {
  try {
    const token = getAuthToken();
    console.log('📡 Envoi de la requête GET /admin/users...');

    const response = await apiClient.get('/admin/users', {
      headers: { Authorization: `Bearer ${token}` },
    });

    console.log('📌 Données reçues de /admin/users :', response.data);

    // Vérifie si "users" existe bien dans la réponse JSON
    if (!response.data || !response.data.users) {
      console.warn('⚠️ Aucune donnée utilisateur reçue !');
      return [];
    }

    return response.data.users;
  } catch (error) {
    console.error(
      '❌ Erreur récupération des utilisateurs :',
      error.response?.data || error.message
    );
    return [];
  }
};

/**
 * Récupère la liste de toutes les entreprises.
 * Utilise GET /companies qui liste toutes les companies (admin uniquement).
 */
export const fetchCompanies = async () => {
  try {
    const token = getAuthToken();
    const response = await apiClient.get('/companies', {
      headers: { Authorization: `Bearer ${token}` },
    });
    console.log('📌 Données reçues de /companies :', response.data);
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

    // Si le rôle est 'driver' et qu'aucun company_id n'est fourni,
    // on renvoie une erreur pour signaler à l'interface de demander la sélection.
    if (updatedData.role.toLowerCase() === 'driver' && !updatedData.company_id) {
      throw new Error("Un company_id est requis pour le rôle 'driver'.");
    }

    const payload = {
      ...updatedData,
      role: String(updatedData.role).toLowerCase(), // <-- normalisation
    };
    const response = await apiClient.put(`/admin/users/${userId}/role`, payload, {
      headers: { Authorization: `Bearer ${token}` },
    });
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
    const token = getAuthToken();
    if (!token) return;
    console.log(`📌 Tentative de suppression de l'utilisateur ID: ${userId}`);
    const response = await apiClient.delete(`/admin/users/${userId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
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
    const token = getAuthToken();
    if (!token) {
      console.error("❌ Erreur : Aucun token JWT trouvé. L'utilisateur doit être connecté.");
      return;
    }
    console.log(`🔄 Réinitialisation du mot de passe pour user ID: ${userId}`);
    const response = await apiClient.post(
      `/admin/users/${userId}/reset-password`,
      {},
      {
        headers: { Authorization: `Bearer ${token}` },
      }
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
    const token = getAuthToken();
    if (!token) {
      throw new Error("Aucun token JWT trouvé. L'utilisateur doit être connecté.");
    }

    console.log('🚀 Démarrage de l\'optimisation Optuna...', config);

    const response = await apiClient.post(
      '/admin/optuna/optimize',
      config,
      {
        headers: { Authorization: `Bearer ${token}` },
      }
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
