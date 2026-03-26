// frontend/src/hooks/useRealtimeDashboard.js
/**
 * Données dashboard dispatch temps réel (GET /company_dispatch/dashboard/realtime).
 * Source HTTP unique : TanStack Query + lirieKeys.dispatchRealtimeDashboard(day).
 */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import apiClient, { getCurrentAuthEnv } from '../utils/apiClient';
import { lirieKeys } from '../queryKeys/lirie';

const hasCompanyToken = () =>
  getCurrentAuthEnv() === 'demo'
    ? !!localStorage.getItem('demo_access_token')
    : !!(
        localStorage.getItem('company_access_token') ||
        localStorage.getItem('company_authToken') ||
        localStorage.getItem('app_access_token')
      );

/**
 * @param {string | null} date - YYYY-MM-DD (doit être le même que dispatchDay sur CompanyDashboard)
 * @param {number} refreshInterval - ms ; 0 = pas d’intervalle (pas de polling périodique via refetchInterval)
 */
export const useRealtimeDashboard = (date = null, refreshInterval = 0) => {
  const dayKey = useMemo(() => {
    if (date && String(date).trim()) {
      return String(date).trim().slice(0, 10);
    }
    return new Date().toISOString().split('T')[0];
  }, [date]);

  const tokenReady = hasCompanyToken();

  const {
    data = null,
    isLoading,
    isFetching,
    error: queryError,
    refetch,
  } = useQuery({
    queryKey: lirieKeys.dispatchRealtimeDashboard(dayKey),
    queryFn: async () => {
      const response = await apiClient.get('/company_dispatch/dashboard/realtime', {
        params: { date: dayKey },
      });
      return response.data ?? null;
    },
    enabled: tokenReady,
    staleTime: 20_000,
    refetchInterval: refreshInterval > 0 ? refreshInterval : false,
  });

  const errorMessage = useMemo(() => {
    if (!queryError) return null;
    return (
      queryError.response?.data?.error ||
      queryError.message ||
      'Erreur lors du chargement du dashboard'
    );
  }, [queryError]);

  const loading = !tokenReady ? false : isLoading || isFetching;

  return {
    data,
    loading,
    error: errorMessage,
    refresh: refetch,
    qualityMetrics: data?.quality_metrics || null,
    currentDelays: data?.current_delays || [],
    opportunities: data?.opportunities || [],
    driverLoad: data?.driver_load || [],
    stats: data?.stats || null,
  };
};

export default useRealtimeDashboard;
