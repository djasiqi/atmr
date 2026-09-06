import { useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import apiClient from '../utils/apiClient';
import { hasCompanyDispatchSession } from '../utils/webAuthSession';
import { recordDashboardApiCall } from '../utils/companyDashboardDuplicationReport';
import { isCompanyDashboardPerfEnabled } from '../utils/companyDashboardPerfInstrumentation';
import { lirieKeys } from '../queryKeys/lirie';

/**
 * Hook pour gérer le mode de dispatch.
 * Auto-charge le mode depuis l'API au montage (TanStack Query — déduplication partagée).
 * dispatchMode vaut null tant que le chargement n'est pas terminé.
 */
export const useDispatchMode = () => {
  const queryClient = useQueryClient();
  const canCallDispatch = hasCompanyDispatchSession();

  const query = useQuery({
    queryKey: lirieKeys.dispatchMode(),
    queryFn: async () => {
      if (isCompanyDashboardPerfEnabled()) {
        recordDashboardApiCall({
          key: 'dispatch_mode',
          url: '/company_dispatch/mode',
          componentId: 'useDispatchMode',
          callerStack: new Error().stack,
        });
      }
      const { data } = await apiClient.get('/company_dispatch/mode');
      return data.dispatch_mode || 'manual';
    },
    staleTime: 60_000,
    retry: 1,
    enabled: canCallDispatch,
  });

  const setDispatchMode = useCallback(
    (mode) => {
      queryClient.setQueryData(lirieKeys.dispatchMode(), mode);
    },
    [queryClient]
  );

  const loadDispatchMode = useCallback(() => query.refetch(), [query]);

  return {
    dispatchMode: !canCallDispatch
      ? 'manual'
      : query.isError
        ? 'manual'
        : (query.data ?? null),
    loading: canCallDispatch ? query.isLoading : false,
    error: query.isError
      ? (query.error?.message ?? 'Erreur lors du chargement du mode de dispatch')
      : null,
    loadDispatchMode,
    setDispatchMode,
  };
};
