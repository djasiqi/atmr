// frontend/src/services/dispatchMonitoringService.js
/**
 * Service pour le monitoring temps réel du dispatch
 * Interactions avec les endpoints de retards et d'optimisation
 */

import apiClient from '../utils/apiClient';

/**
 * Récupère les retards pour une date donnée
 * @param {string} date - Date au format YYYY-MM-DD
 * @returns {Promise} Liste des retards avec suggestions
 */
export const getDelays = async (date) => {
  try {
    const response = await apiClient.get('/company_dispatch/delays', {
      params: { date },
    });
    return response.data;
  } catch (error) {
    console.error('[DispatchMonitoring] Error fetching delays:', error);
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
    const response = await apiClient.get('/company_dispatch/delays/live', {
      params: { date },
    });
    return response.data;
  } catch (error) {
    // ⚡ Ignorer les erreurs 401 si le refresh est en cours ou réussi
    if (error?.response?.status === 401 && error?.config?._retryAfterRefresh) {
      // Refresh réussi, ne pas logger l'erreur
      return null;
    }

    // Ne logger que les vraies erreurs (pas les 401 en cours de refresh)
    if (error?.response?.status !== 401) {
      console.error('[DispatchMonitoring] Error fetching live delays:', error);
    } else {
      console.debug('[DispatchMonitoring] 401 error, refresh token will be attempted');
    }
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
    const response = await apiClient.post('/company_dispatch/optimizer/start', {
      check_interval_seconds: checkIntervalSeconds,
    });
    return response.data;
  } catch (error) {
    console.error('[DispatchMonitoring] Error starting optimizer:', error);
    throw error;
  }
};

/**
 * Arrête le monitoring automatique
 * @returns {Promise} Confirmation d'arrêt
 */
export const stopRealTimeOptimizer = async () => {
  try {
    const response = await apiClient.post('/company_dispatch/optimizer/stop');
    return response.data;
  } catch (error) {
    console.error('[DispatchMonitoring] Error stopping optimizer:', error);
    throw error;
  }
};

/**
 * Récupère le statut du monitoring automatique
 * @returns {Promise} Statut (running, last_check, opportunities_count, etc.)
 */
export const getOptimizerStatus = async () => {
  try {
    const response = await apiClient.get('/company_dispatch/optimizer/status');
    return response.data;
  } catch (error) {
    // ⚡ Ignorer les erreurs 401 si le refresh est en cours ou réussi
    if (error?.response?.status === 401 && error?.config?._retryAfterRefresh) {
      // Refresh réussi, ne pas logger l'erreur
      return null;
    }

    // Ne logger que les vraies erreurs (pas les 401 en cours de refresh)
    if (error?.response?.status !== 401) {
      console.error('[DispatchMonitoring] Error fetching optimizer status:', error);
    } else {
      console.debug('[DispatchMonitoring] 401 error, refresh token will be attempted');
    }
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
    const response = await apiClient.get('/company_dispatch/optimizer/opportunities', {
      params,
    });
    return response.data;
  } catch (error) {
    console.error('[DispatchMonitoring] Error fetching opportunities:', error);
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
    console.error('[DispatchMonitoring] Error applying suggestion:', error);
    throw error;
  }
};

/**
 * Rejette une suggestion d'assignation
 * @param {number} bookingId - ID de la réservation
 * @returns {Promise} Confirmation du rejet
 */
export const rejectSuggestion = async (bookingId) => {
  try {
    const response = await apiClient.post('/company_dispatch/reject_suggestion', {
      booking_id: bookingId,
    });
    return response.data;
  } catch (error) {
    console.error('[DispatchMonitoring] Error rejecting suggestion:', error);
    throw error;
  }
};

/**
 * Démarre l'agent dispatch intelligent
 * @returns {Promise} Statut de l'agent
 */
export const startAgent = async () => {
  try {
    const response = await apiClient.post('/company_dispatch/agent/start');
    return response.data;
  } catch (error) {
    console.error('[Agent] Error starting agent:', error);
    throw error;
  }
};

/**
 * Arrête l'agent dispatch intelligent
 * @returns {Promise} Confirmation d'arrêt
 */
export const stopAgent = async () => {
  try {
    const response = await apiClient.post('/company_dispatch/agent/stop');
    return response.data;
  } catch (error) {
    console.error('[Agent] Error stopping agent:', error);
    throw error;
  }
};

/**
 * Récupère le statut de l'agent dispatch intelligent
 * @returns {Promise} Statut (running, last_tick, actions_today, osrm_health, etc.)
 */
export const getAgentStatus = async () => {
  try {
    const response = await apiClient.get('/company_dispatch/agent/status');
    return response.data;
  } catch (error) {
    // ⚡ Ignorer les erreurs 401 si le refresh est en cours ou réussi
    if (error?.response?.status === 401 && error?.config?._retryAfterRefresh) {
      return null;
    }

    if (error?.response?.status !== 401) {
      console.error('[Agent] Error fetching agent status:', error);
    } else {
      console.debug('[Agent] 401 error, refresh token will be attempted');
    }
    throw error;
  }
};

/**
 * Réinitialise toutes les assignations et remet les courses au statut ACCEPTED
 * @param {string} date - Date au format YYYY-MM-DD (optionnel, défaut: toutes les dates)
 * @returns {Promise} Résultat de la réinitialisation
 */
export const resetAssignments = async (date = null) => {
  try {
    const body = date ? { date } : {};
    const response = await apiClient.post('/company_dispatch/reset', body);
    return response.data;
  } catch (error) {
    console.error('[DispatchMonitoring] Error resetting assignments:', error);
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
  rejectSuggestion,
  startAgent,
  stopAgent,
  getAgentStatus,
  resetAssignments,
};

export default dispatchMonitoringService;
