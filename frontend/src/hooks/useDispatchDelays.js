// frontend/src/hooks/useDispatchDelays.js
/**
 * Hook personnalisé pour gérer les retards de dispatch
 * Réutilisable dans n'importe quel composant
 */

import { useState, useEffect, useCallback } from 'react';
import { getLiveDelays, getOptimizerStatus } from '../services/dispatchMonitoringService';

/**
 * Hook pour récupérer et gérer les retards de dispatch
 *
 * @param {string} date - Date au format YYYY-MM-DD (optionnel, défaut: aujourd'hui)
 * @param {number} refreshInterval - Intervalle de refresh en ms (0 = pas d'auto-refresh)
 * @returns {object} { delays, summary, loading, error, refresh, optimizerStatus }
 */
export const useDispatchDelays = (date = null, refreshInterval = 0) => {
  const [delays, setDelays] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [optimizerStatus, setOptimizerStatus] = useState(null);

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
      const response = await getLiveDelays(currentDate);

      setDelays(response.delays || []);
      setSummary(response.summary || null);
    } catch (err) {
      console.error('[useDispatchDelays] Error:', err);
      setError(err.message || 'Erreur lors du chargement des retards');
    } finally {
      setLoading(false);
    }
  }, [getDate]);

  // Récupérer le statut de l'optimizer
  const fetchOptimizerStatus = useCallback(async () => {
    try {
      const status = await getOptimizerStatus();
      setOptimizerStatus(status);
    } catch (err) {
      console.error('[useDispatchDelays] Error fetching optimizer status:', err);
    }
  }, []);

  // Charger les données initiales
  useEffect(() => {
    refresh();
    fetchOptimizerStatus();
  }, [refresh, fetchOptimizerStatus]);

  // Auto-refresh si activé
  useEffect(() => {
    if (refreshInterval <= 0) return;

    const intervalId = setInterval(() => {
      refresh();
      fetchOptimizerStatus();
    }, refreshInterval);

    return () => clearInterval(intervalId);
  }, [refreshInterval, refresh, fetchOptimizerStatus]);

  // Compteurs utiles
  const delayCount = summary?.late || 0;
  const onTimeCount = summary?.on_time || 0;
  const earlyCount = summary?.early || 0;
  const totalCount = summary?.total || 0;
  const averageDelay = summary?.average_delay || 0;

  // Indicateurs de statut
  const hasDelays = delayCount > 0;
  const hasCriticalDelays = delays.some((d) =>
    d.suggestions?.some((s) => s.priority === 'critical')
  );

  return {
    // Données
    delays,
    summary,
    loading,
    error,
    optimizerStatus,

    // Fonctions
    refresh,

    // Compteurs
    delayCount,
    onTimeCount,
    earlyCount,
    totalCount,
    averageDelay,

    // Indicateurs
    hasDelays,
    hasCriticalDelays,
  };
};

export default useDispatchDelays;
