// frontend/src/hooks/useRealtimeDashboard.js
/**
 * ✅ 3.4.2: Hook pour récupérer les données du dashboard temps réel dispatch
 * Utilise la nouvelle route /api/v1/dispatch/dashboard/realtime
 */

import { useState, useEffect, useCallback } from 'react';
import apiClient from '../utils/apiClient';

/**
 * Hook pour récupérer les données du dashboard temps réel dispatch
 *
 * @param {string} date - Date au format YYYY-MM-DD (optionnel, défaut: aujourd'hui)
 * @param {number} refreshInterval - Intervalle de refresh en ms (0 = pas d'auto-refresh)
 * @returns {object} { data, loading, error, refresh }
 */
export const useRealtimeDashboard = (date = null, refreshInterval = 0) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Obtenir la date (aujourd'hui par défaut)
  const getDate = useCallback(() => {
    return date || new Date().toISOString().split('T')[0];
  }, [date]);

  // Fonction de refresh
  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const currentDate = getDate();
      const response = await apiClient.get('/company_dispatch/dashboard/realtime', {
        params: { date: currentDate },
      });

      setData(response.data || null);
    } catch (err) {
      console.error('[useRealtimeDashboard] Error:', err);
      setError(err.response?.data?.error || err.message || 'Erreur lors du chargement du dashboard');
    } finally {
      setLoading(false);
    }
  }, [getDate]);

  // Charger les données initiales
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Auto-refresh si intervalle défini
  useEffect(() => {
    if (refreshInterval > 0) {
      const interval = setInterval(() => {
        refresh();
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [refreshInterval, refresh]);

  return {
    data,
    loading,
    error,
    refresh,
    // Helpers pour accéder facilement aux données
    qualityMetrics: data?.quality_metrics || null,
    currentDelays: data?.current_delays || [],
    opportunities: data?.opportunities || [],
    driverLoad: data?.driver_load || [],
    stats: data?.stats || null,
  };
};

export default useRealtimeDashboard;

