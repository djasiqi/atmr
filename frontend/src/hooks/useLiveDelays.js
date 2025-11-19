import { useState, useCallback } from 'react';
import { getLiveDelays } from '../services/dispatchMonitoringService';

/**
 * Hook personnalisé pour gérer le chargement des retards en temps réel
 */
export const useLiveDelays = (date) => {
  const [delays, setDelays] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadDelays = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await getLiveDelays(date);
      // ⚡ Si response est null (401 avec refresh réussi), ne pas mettre à jour les données
      if (response) {
        setDelays(response.delays || []);
        setSummary(response.summary || null);
      }
    } catch (err) {
      // ⚡ Ignorer les erreurs 401 si le refresh est en cours ou réussi
      if (err?.response?.status === 401 && err?.config?._retryAfterRefresh) {
        // Refresh réussi, ne pas logger l'erreur ni définir d'erreur
        return;
      }

      // Ne logger que les vraies erreurs (pas les 401 en cours de refresh)
      if (err?.response?.status !== 401) {
        console.error('[useLiveDelays] Error loading delays:', err);
        setError(err.message || 'Erreur lors du chargement des retards');
      } else {
        console.debug('[useLiveDelays] 401 error, refresh token will be attempted');
      }
    } finally {
      setLoading(false);
    }
  }, [date]);

  return {
    delays,
    summary,
    loading,
    error,
    loadDelays,
    setDelays,
    setSummary,
  };
};
