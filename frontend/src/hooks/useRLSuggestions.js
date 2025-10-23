import { useState, useEffect, useCallback } from 'react';
import apiClient from '../utils/apiClient';

/**
 * Hook pour gérer les suggestions RL/MDI.
 *
 * Auto-refresh optionnel pour mode semi-auto/fully-auto.
 * Tri automatique par confiance décroissante.
 *
 * @param {string} date - Date du dispatch (YYYY-MM-DD)
 * @param {object} options - Options de configuration
 * @returns {object} - { suggestions, loading, error, reload, applySuggestion, ... }
 */
export const useRLSuggestions = (date, options = {}) => {
  const {
    autoRefresh = false,
    refreshInterval = 30000, // 30 secondes
    minConfidence = 0.0,
    limit = 20,
  } = options;

  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadSuggestions = useCallback(async () => {
    if (!date) return;

    setLoading(true);
    try {
      const { data } = await apiClient.get('/company_dispatch/rl/suggestions', {
        params: {
          for_date: date,
          min_confidence: minConfidence,
          limit: limit,
        },
      });

      // Trier par confiance décroissante
      const sortedSuggestions = (data.suggestions || []).sort(
        (a, b) => (b.confidence || 0) - (a.confidence || 0)
      );

      setSuggestions(sortedSuggestions);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('[useRLSuggestions] Error:', err);
      setSuggestions([]);
    } finally {
      setLoading(false);
    }
  }, [date, minConfidence, limit]);

  useEffect(() => {
    loadSuggestions();

    if (autoRefresh) {
      const interval = setInterval(loadSuggestions, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [loadSuggestions, autoRefresh, refreshInterval]);

  const applySuggestion = useCallback(
    async (suggestion) => {
      try {
        await apiClient.post(`/company_dispatch/assignments/${suggestion.assignment_id}/reassign`, {
          new_driver_id: suggestion.suggested_driver_id,
        });

        // Recharger suggestions après application
        await loadSuggestions();
        return { success: true };
      } catch (err) {
        return { success: false, error: err.message };
      }
    },
    [loadSuggestions]
  );

  // Métriques utiles dérivées
  const highConfidenceSuggestions = suggestions.filter((s) => (s.confidence || 0) > 0.8);
  const mediumConfidenceSuggestions = suggestions.filter((s) => {
    const conf = s.confidence || 0;
    return conf >= 0.5 && conf <= 0.8;
  });
  const lowConfidenceSuggestions = suggestions.filter((s) => (s.confidence || 0) < 0.5);

  const avgConfidence =
    suggestions.length > 0
      ? suggestions.reduce((sum, s) => sum + (s.confidence || 0), 0) / suggestions.length
      : 0;

  const totalExpectedGain = suggestions.reduce((sum, s) => sum + (s.expected_gain_minutes || 0), 0);

  return {
    suggestions,
    highConfidenceSuggestions,
    mediumConfidenceSuggestions,
    lowConfidenceSuggestions,
    avgConfidence,
    totalExpectedGain,
    loading,
    error,
    reload: loadSuggestions,
    applySuggestion,
  };
};

export default useRLSuggestions;
