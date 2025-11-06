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
  const [isBackendAvailable, setIsBackendAvailable] = useState(true);

  const loadShadowData = useCallback(async () => {
    if (!isBackendAvailable) {
      setLoading(false);
      return;
    }

    try {
      const statusRes = await apiClient.get('shadow-mode/status', {
        baseURL: '/api',
      });

      const [statsRes, predsRes, compsRes] = await Promise.all([
        apiClient.get('shadow-mode/stats', {
          baseURL: '/api',
        }),
        apiClient.get('shadow-mode/predictions', {
          baseURL: '/api',
          params: { limit: 50 },
        }),
        apiClient.get('shadow-mode/comparisons', {
          baseURL: '/api',
          params: { limit: 50 },
        }),
      ]);

      setStatus(statusRes.data);
      setStats(statsRes.data.session_stats);
      setPredictions(predsRes.data.predictions || []);
      setComparisons(compsRes.data.comparisons || []);
      setError(null);
    } catch (err) {
      const statusCode = err?.response?.status;

      if (statusCode === 404 || statusCode === 403) {
        setIsBackendAvailable(false);
        setStatus({ status: 'unavailable' });
        setStats(null);
        setPredictions([]);
        setComparisons([]);
        setError(null);
      } else {
        console.error('[useShadowMode] Error:', err);
        setError(err.message || 'Erreur inconnue lors du chargement du Shadow Mode');
      }
    } finally {
      setLoading(false);
    }
  }, [isBackendAvailable]);

  useEffect(() => {
    loadShadowData();

    if (autoRefresh && isBackendAvailable) {
      const interval = setInterval(loadShadowData, refreshInterval);
      return () => clearInterval(interval);
    }

    return undefined;
  }, [loadShadowData, autoRefresh, refreshInterval, isBackendAvailable]);

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
