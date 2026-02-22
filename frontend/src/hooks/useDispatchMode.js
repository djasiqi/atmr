import { useState, useCallback, useEffect } from 'react';
import apiClient from '../utils/apiClient';

/**
 * Hook pour gérer le mode de dispatch.
 * Auto-charge le mode depuis l'API au montage.
 * dispatchMode vaut null tant que le chargement n'est pas terminé.
 */
export const useDispatchMode = () => {
  const [dispatchMode, setDispatchMode] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadDispatchMode = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const { data } = await apiClient.get('/company_dispatch/mode');
      setDispatchMode(data.dispatch_mode || 'manual');
    } catch (err) {
      console.error('[useDispatchMode] Error loading dispatch mode:', err);
      setError(err.message || 'Erreur lors du chargement du mode de dispatch');
      setDispatchMode('manual');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDispatchMode();
  }, [loadDispatchMode]);

  return {
    dispatchMode,
    loading,
    error,
    loadDispatchMode,
    setDispatchMode,
  };
};
