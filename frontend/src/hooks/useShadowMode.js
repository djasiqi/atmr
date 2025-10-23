import { useState, useEffect, useCallback } from 'react';
import apiClient from '../utils/apiClient';

/**
 * Hook pour monitorer le Shadow Mode (Phase 1).
 *
 * Utilisé dans les dashboards admin pour suivre la validation du MDI.
 * Charge les stats, prédictions, et comparaisons MDI vs Système actuel.
 *
 * @param {object} options - Options de configuration
 * @returns {object} - { status, stats, predictions, comparisons, ... }
 */
export const useShadowMode = (options = {}) => {
  const {
    autoRefresh = true,
    refreshInterval = 30000, // 30 secondes
  } = options;

  const [status, setStatus] = useState(null);
  const [stats, setStats] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [comparisons, setComparisons] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadShadowData = useCallback(async () => {
    try {
      const [statusRes, statsRes, predsRes, compsRes] = await Promise.all([
        apiClient.get('/shadow-mode/status'),
        apiClient.get('/shadow-mode/stats'),
        apiClient.get('/shadow-mode/predictions', { params: { limit: 50 } }),
        apiClient.get('/shadow-mode/comparisons', { params: { limit: 50 } }),
      ]);

      setStatus(statusRes.data);
      setStats(statsRes.data.session_stats);
      setPredictions(predsRes.data.predictions || []);
      setComparisons(compsRes.data.comparisons || []);
      setError(null);
    } catch (err) {
      console.error('[useShadowMode] Error:', err);
      setError(err.message);

      // En cas d'erreur 404, Shadow Mode n'est pas actif (c'est OK)
      if (err.response?.status === 404 || err.response?.status === 403) {
        setStatus({ status: 'unavailable' });
        setStats(null);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadShadowData();

    if (autoRefresh) {
      const interval = setInterval(loadShadowData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [loadShadowData, autoRefresh, refreshInterval]);

  // Métriques dérivées
  const agreementRate = stats?.agreement_rate || 0;
  const totalComparisons = stats?.comparisons_count || 0;
  const totalPredictions = stats?.predictions_count || 0;
  const isReadyForPhase2 = agreementRate > 0.75 && totalComparisons >= 1000;

  // Analyse des désaccords
  const disagreements = comparisons.filter((c) => !c.agreement);
  const highConfidenceDisagreements = disagreements.filter((c) => (c.confidence || 0) > 0.8);

  return {
    status,
    stats,
    predictions,
    comparisons,
    disagreements,
    highConfidenceDisagreements,
    loading,
    error,
    reload: loadShadowData,
    isActive: status?.status === 'active',
    agreementRate,
    totalComparisons,
    totalPredictions,
    isReadyForPhase2,
  };
};

export default useShadowMode;
