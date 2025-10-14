// frontend/src/services/dispatchMonitoringService.js
/**
 * Service pour le monitoring temps réel du dispatch
 * Interactions avec les endpoints de retards et d'optimisation
 */

import apiClient from "../utils/apiClient";

/**
 * Récupère les retards pour une date donnée
 * @param {string} date - Date au format YYYY-MM-DD
 * @returns {Promise} Liste des retards avec suggestions
 */
export const getDelays = async (date) => {
  try {
    const response = await apiClient.get("/company_dispatch/delays", {
      params: { date },
    });
    return response.data;
  } catch (error) {
    console.error("[DispatchMonitoring] Error fetching delays:", error);
    throw error;
  }
};

/**
 * Récupère les retards en temps réel avec recalcul des ETAs
 * @param {string} date - Date au format YYYY-MM-DD
 * @returns {Promise} Retards temps réel avec suggestions et impacts cascade
 */
export const getLiveDelays = async (date) => {
  try {
    const response = await apiClient.get("/company_dispatch/delays/live", {
      params: { date },
    });
    return response.data;
  } catch (error) {
    console.error("[DispatchMonitoring] Error fetching live delays:", error);
    throw error;
  }
};

/**
 * Démarre le monitoring automatique en temps réel
 * @param {number} checkIntervalSeconds - Intervalle de vérification en secondes (défaut: 120)
 * @returns {Promise} Statut du monitoring
 */
export const startRealTimeOptimizer = async (checkIntervalSeconds = 120) => {
  try {
    const response = await apiClient.post("/company_dispatch/optimizer/start", {
      check_interval_seconds: checkIntervalSeconds,
    });
    return response.data;
  } catch (error) {
    console.error("[DispatchMonitoring] Error starting optimizer:", error);
    throw error;
  }
};

/**
 * Arrête le monitoring automatique
 * @returns {Promise} Confirmation d'arrêt
 */
export const stopRealTimeOptimizer = async () => {
  try {
    const response = await apiClient.post("/company_dispatch/optimizer/stop");
    return response.data;
  } catch (error) {
    console.error("[DispatchMonitoring] Error stopping optimizer:", error);
    throw error;
  }
};

/**
 * Récupère le statut du monitoring automatique
 * @returns {Promise} Statut (running, last_check, opportunities_count, etc.)
 */
export const getOptimizerStatus = async () => {
  try {
    const response = await apiClient.get("/company_dispatch/optimizer/status");
    return response.data;
  } catch (error) {
    console.error(
      "[DispatchMonitoring] Error fetching optimizer status:",
      error
    );
    throw error;
  }
};

/**
 * Récupère les opportunités d'optimisation détectées
 * @param {string} date - Date au format YYYY-MM-DD (optionnel)
 * @returns {Promise} Liste des opportunités avec suggestions
 */
export const getOptimizationOpportunities = async (date = null) => {
  try {
    const params = date ? { date } : {};
    const response = await apiClient.get(
      "/company_dispatch/optimizer/opportunities",
      {
        params,
      }
    );
    return response.data;
  } catch (error) {
    console.error("[DispatchMonitoring] Error fetching opportunities:", error);
    throw error;
  }
};

/**
 * Applique une suggestion de réassignation
 * @param {number} assignmentId - ID de l'assignation
 * @param {number} newDriverId - ID du nouveau chauffeur
 * @returns {Promise} Assignation mise à jour
 */
export const applySuggestion = async (assignmentId, newDriverId) => {
  try {
    const response = await apiClient.post(
      `/company_dispatch/assignments/${assignmentId}/reassign`,
      { new_driver_id: newDriverId }
    );
    return response.data;
  } catch (error) {
    console.error("[DispatchMonitoring] Error applying suggestion:", error);
    throw error;
  }
};

// Export d'un objet de service avec toutes les méthodes
const dispatchMonitoringService = {
  getDelays,
  getLiveDelays,
  startRealTimeOptimizer,
  stopRealTimeOptimizer,
  getOptimizerStatus,
  getOptimizationOpportunities,
  applySuggestion,
};

export default dispatchMonitoringService;
