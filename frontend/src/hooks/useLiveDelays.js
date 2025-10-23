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
      setDelays(response.delays || []);
      setSummary(response.summary || null);
    } catch (err) {
      console.error('[useLiveDelays] Error loading delays:', err);
      setError(err.message || 'Erreur lors du chargement des retards');
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
