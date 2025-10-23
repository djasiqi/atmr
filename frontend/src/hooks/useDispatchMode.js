import { useState, useCallback } from 'react';
import apiClient from '../utils/apiClient';

/**
 * Hook personnalisé pour gérer le mode de dispatch
 */
export const useDispatchMode = () => {
  const [dispatchMode, setDispatchMode] = useState('semi_auto');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadDispatchMode = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const { data } = await apiClient.get('/company_dispatch/mode');
      setDispatchMode(data.dispatch_mode || 'semi_auto');
    } catch (err) {
      console.error('[useDispatchMode] Error loading dispatch mode:', err);
      setError(err.message || 'Erreur lors du chargement du mode de dispatch');
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    dispatchMode,
    loading,
    error,
    loadDispatchMode,
    setDispatchMode,
  };
};
