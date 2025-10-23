import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { feedbackApplied, feedbackRejected } from '../../services/rlFeedbackService';
import { showSuccess, showError, showInfo } from '../../utils/toast';
import './RLSuggestionCard.css';

/**
 * Carte de suggestion RL avec score de confiance et métriques.
 *
 * Utilisée dans tous les modes:
 * - MANUAL: readonly (informatif seulement)
 * - SEMI-AUTO: cliquable (avec bouton "Appliquer")
 * - FULLY-AUTO: historique (actions déjà appliquées)
 *
 * @param {object} suggestion - Suggestion RL avec {confidence, suggested_driver_id, q_value, etc.}
 * @param {function} onApply - Callback quand l'utilisateur clique "Appliquer"
 * @param {boolean} readOnly - Mode lecture seule (Manual mode)
 * @param {boolean} applied - Déjà appliqué (Fully-Auto mode)
 */
const RLSuggestionCard = ({ suggestion, onApply, readOnly = false, applied = false }) => {
  const {
    booking_id,
    suggested_driver_id,
    suggested_driver_name,
    confidence,
    q_value: _q_value,
    expected_gain_minutes,
    distance_km,
    current_driver_id,
    current_driver_name,
    metric_id, // 🆕 ID pour tracking feedback
  } = suggestion;

  // 🆕 États pour feedback
  const [feedbackGiven, setFeedbackGiven] = useState(false);
  const [feedbackAction, setFeedbackAction] = useState(null);

  // Niveau de confiance avec couleurs et emojis
  const getConfidenceLevel = (conf) => {
    if (conf >= 0.9)
      return { label: 'Très élevée', class: 'very-high', emoji: '🟢', color: '#28a745' };
    if (conf >= 0.75) return { label: 'Élevée', class: 'high', emoji: '🟡', color: '#ffc107' };
    if (conf >= 0.5) return { label: 'Moyenne', class: 'medium', emoji: '🟠', color: '#ff9800' };
    return { label: 'Faible', class: 'low', emoji: '🔴', color: '#f44336' };
  };

  const confidenceInfo = getConfidenceLevel(confidence || 0);

  // Gestion du clic "Appliquer"
  const handleApply = async () => {
    if (confidence < 0.5) {
      const confirmed = window.confirm(
        `⚠️ Confiance faible (${(confidence * 100).toFixed(0)}%)\n\n` +
          `Voulez-vous vraiment appliquer cette suggestion?\n\n` +
          `Driver suggéré: ${suggested_driver_name || `#${suggested_driver_id}`}\n` +
          `Gain attendu: +${expected_gain_minutes || 0} min`
      );
      if (!confirmed) return;
    }

    // Appliquer la suggestion via le hook
    if (onApply) {
      await onApply(suggestion);

      // 🆕 Enregistrer feedback "applied" automatiquement
      if (metric_id) {
        try {
          await feedbackApplied(metric_id);
          setFeedbackGiven(true);
          setFeedbackAction('applied');
          showInfo('✅ Feedback enregistré pour amélioration du modèle');
        } catch (error) {
          console.error('[RLSuggestionCard] Error recording feedback:', error);
          // Non-bloquant : continuer même si feedback échoue
        }
      }
    }
  };

  // 🆕 Handler pour feedback positif (sans appliquer)
  const handlePositiveFeedback = async () => {
    if (!metric_id) {
      showError('❌ ID métrique manquant');
      return;
    }

    try {
      await feedbackApplied(metric_id, {
        was_better: true,
        satisfaction: 5,
      });

      setFeedbackGiven(true);
      setFeedbackAction('positive');
      showSuccess('👍 Merci ! Cette suggestion sera utilisée pour améliorer le modèle.');
    } catch (error) {
      showError(`❌ Erreur feedback: ${error.message}`);
    }
  };

  // 🆕 Handler pour feedback négatif (rejeter)
  const handleNegativeFeedback = async () => {
    if (!metric_id) {
      showError('❌ ID métrique manquant');
      return;
    }

    // Demander raison (optionnel)
    const reason = window.prompt(
      "👎 Pourquoi cette suggestion n'est pas bonne ?\n\n" +
        '(Optionnel - appuyez sur OK pour confirmer le rejet)'
    );

    // Si cancel, annuler
    if (reason === null) return;

    try {
      await feedbackRejected(metric_id, reason || undefined);

      setFeedbackGiven(true);
      setFeedbackAction('negative');
      showSuccess('👎 Merci ! Ce feedback aidera à améliorer le modèle.');
    } catch (error) {
      showError(`❌ Erreur feedback: ${error.message}`);
    }
  };

  return (
    <div
      className={`rl-suggestion-card confidence-${confidenceInfo.class} ${
        applied ? 'applied' : ''
      }`}
    >
      {/* Header compact */}
      <div className="suggestion-header">
        <div className="suggestion-icon">{applied ? '✅' : '🤖'}</div>
        <span className="booking-ref">Booking #{booking_id}</span>
        <div
          className={`confidence-badge ${confidenceInfo.class}`}
          title={`Confiance ${confidenceInfo.label}: ${(confidence * 100).toFixed(1)}%`}
        >
          {confidenceInfo.emoji} {((confidence || 0) * 100).toFixed(0)}%
        </div>
      </div>

      <div className="suggestion-body">
        {/* Driver actuel → Driver suggéré */}
        <div className="driver-change">
          {current_driver_id && (
            <>
              <div className="driver-item current">
                <div className="driver-avatar">👤</div>
                <div className="driver-info">
                  <span className="driver-label">Actuel</span>
                  <strong>{current_driver_name || `Driver #${current_driver_id}`}</strong>
                </div>
              </div>
              <div className="change-arrow">→</div>
            </>
          )}

          <div className="driver-item suggested">
            <div className="driver-avatar highlight">👤</div>
            <div className="driver-info">
              <span className="driver-label">{current_driver_id ? 'Suggéré' : 'Driver'}</span>
              <strong>{suggested_driver_name || `Driver #${suggested_driver_id}`}</strong>
              {distance_km && (
                <span className="driver-details">📍 {distance_km.toFixed(1)} km</span>
              )}
            </div>
          </div>
        </div>

        {/* Métriques simplifiées */}
        {expected_gain_minutes !== null &&
          expected_gain_minutes !== undefined &&
          expected_gain_minutes > 0 && (
            <div className="suggestion-metrics">
              <div className="metric">
                <span className="metric-label">Gain</span>
                <span className="metric-value positive">+{expected_gain_minutes} min</span>
              </div>
            </div>
          )}

        {/* Actions selon le mode */}
        {!readOnly && !applied && !feedbackGiven && (
          <div className="suggestion-actions">
            <button className="btn-apply" onClick={handleApply} disabled={!onApply}>
              ✅ Appliquer
            </button>

            {/* 🆕 Boutons feedback */}
            {metric_id && (
              <div className="feedback-buttons">
                <button
                  className="btn-feedback btn-thumbs-up"
                  onClick={handlePositiveFeedback}
                  title="Bonne suggestion (aide le modèle)"
                >
                  👍
                </button>
                <button
                  className="btn-feedback btn-thumbs-down"
                  onClick={handleNegativeFeedback}
                  title="Mauvaise suggestion (aide le modèle)"
                >
                  👎
                </button>
              </div>
            )}
          </div>
        )}

        {/* 🆕 Affichage feedback donné */}
        {feedbackGiven && (
          <div className={`feedback-confirmation ${feedbackAction}`}>
            {feedbackAction === 'positive' && '✅ Feedback positif enregistré'}
            {feedbackAction === 'negative' && '❌ Feedback négatif enregistré'}
            {feedbackAction === 'applied' && '✅ Application + Feedback enregistrés'}
          </div>
        )}

        {/* Warning si confiance faible */}
        {!readOnly && !applied && confidence < 0.5 && (
          <div className="suggestion-warning">⚠️ Confiance faible - Vérifier avant application</div>
        )}
      </div>
    </div>
  );
};

RLSuggestionCard.propTypes = {
  suggestion: PropTypes.shape({
    booking_id: PropTypes.number.isRequired,
    suggested_driver_id: PropTypes.number,
    suggested_driver_name: PropTypes.string,
    confidence: PropTypes.number,
    q_value: PropTypes.number,
    expected_gain_minutes: PropTypes.number,
    distance_km: PropTypes.number,
    current_driver_id: PropTypes.number,
    current_driver_name: PropTypes.string,
    assignment_id: PropTypes.number,
    metric_id: PropTypes.string, // 🆕 ID pour feedback
  }).isRequired,
  onApply: PropTypes.func,
  readOnly: PropTypes.bool,
  applied: PropTypes.bool,
};

export default RLSuggestionCard;
