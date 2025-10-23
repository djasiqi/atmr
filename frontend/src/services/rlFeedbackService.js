// frontend/src/services/rlFeedbackService.js
/**
 * Service pour gérer les feedbacks utilisateurs sur les suggestions RL.
 * Permet d'améliorer le modèle DQN via apprentissage supervisé.
 */

import apiClient from '../utils/apiClient';

/**
 * Enregistre un feedback utilisateur sur une suggestion RL.
 *
 * @param {string} suggestionId - ID unique de la suggestion (metric_id)
 * @param {string} action - Action effectuée : "applied" | "rejected" | "ignored"
 * @param {string} [feedbackReason] - Raison du rejet (optionnel)
 * @param {object} [actualOutcome] - Résultat réel si appliqué
 * @returns {Promise<object>} Résultat de l'enregistrement
 */
export const provideFeedback = async ({
  suggestionId,
  action,
  feedbackReason = null,
  actualOutcome = null,
}) => {
  if (!suggestionId) {
    throw new Error('suggestionId requis');
  }

  if (!['applied', 'rejected', 'ignored'].includes(action)) {
    throw new Error('action doit être "applied", "rejected" ou "ignored"');
  }

  const payload = {
    suggestion_id: suggestionId,
    action: action,
  };

  if (feedbackReason) {
    payload.feedback_reason = feedbackReason;
  }

  if (actualOutcome) {
    payload.actual_outcome = actualOutcome;
  }

  try {
    const { data } = await apiClient.post('/company_dispatch/rl/feedback', payload);

    console.log('[RLFeedback] Feedback enregistré:', data);

    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[RLFeedback] Erreur enregistrement feedback:', error);

    // Gestion erreur 409 (déjà enregistré)
    if (error.response?.status === 409) {
      return {
        success: false,
        error: 'Feedback déjà enregistré pour cette suggestion',
        alreadyRecorded: true,
      };
    }

    return {
      success: false,
      error: error.response?.data?.error || error.message || 'Erreur inconnue',
    };
  }
};

/**
 * Enregistre feedback "Applied" avec résultat optionnel.
 *
 * @param {string} suggestionId - ID de la suggestion
 * @param {object} [outcome] - Résultat réel (gain_minutes, was_better, satisfaction)
 * @returns {Promise<object>}
 */
export const feedbackApplied = async (suggestionId, outcome = null) => {
  return provideFeedback({
    suggestionId,
    action: 'applied',
    actualOutcome: outcome,
  });
};

/**
 * Enregistre feedback "Rejected" avec raison.
 *
 * @param {string} suggestionId - ID de la suggestion
 * @param {string} [reason] - Raison du rejet
 * @returns {Promise<object>}
 */
export const feedbackRejected = async (suggestionId, reason = null) => {
  return provideFeedback({
    suggestionId,
    action: 'rejected',
    feedbackReason: reason,
  });
};

/**
 * Enregistre feedback "Ignored" (suggestion pas utilisée).
 *
 * @param {string} suggestionId - ID de la suggestion
 * @returns {Promise<object>}
 */
export const feedbackIgnored = async (suggestionId) => {
  return provideFeedback({
    suggestionId,
    action: 'ignored',
  });
};

/**
 * Récupère les statistiques des feedbacks pour l'entreprise.
 *
 * @param {number} [days=30] - Nombre de jours d'historique
 * @returns {Promise<object>} Statistiques feedbacks
 */
export const getFeedbackStats = async (days = 30) => {
  try {
    const { data } = await apiClient.get('/company_dispatch/rl/feedbacks/stats', {
      params: { days },
    });

    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[RLFeedback] Erreur récupération stats:', error);
    return {
      success: false,
      error: error.message,
    };
  }
};

const rlFeedbackService = {
  provideFeedback,
  feedbackApplied,
  feedbackRejected,
  feedbackIgnored,
  getFeedbackStats,
};

export default rlFeedbackService;
