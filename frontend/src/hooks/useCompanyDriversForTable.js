import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';
import { fetchCompanyDriversCanonical } from '../services/companyService';
import { lirieKeys } from '../queryKeys/lirie';
import { projectDriversForTable } from '../utils/companyDriverProjections';
import { useSocketConnected } from './useCompanySocket';

function useCompanyDriversLiveQueryPolicy(companyId) {
  const socketConnected = useSocketConnected();
  return Boolean(companyId != null && socketConnected);
}

/** Projection liste/table — même cache que useCompanyData, champs administratifs. */
export function useCompanyDriversForTable(companyId) {
  const driversLiveHealthy = useCompanyDriversLiveQueryPolicy(companyId);

  const companyDriversQueryOptions = useMemo(() => {
    if (driversLiveHealthy) {
      return {
        staleTime: Infinity,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
      };
    }
    return {
      staleTime: 45_000,
      refetchOnWindowFocus: true,
      refetchOnReconnect: true,
    };
  }, [driversLiveHealthy]);

  const { data: driversForTable = [], isLoading } = useQuery({
    queryKey: lirieKeys.companyDrivers(),
    queryFn: async () => {
      const data = await fetchCompanyDriversCanonical();
      return Array.isArray(data) ? data : data?.driver ?? [];
    },
    select: projectDriversForTable,
    enabled: Boolean(companyId),
    ...companyDriversQueryOptions,
  });

  return { driversForTable, loadingDriversForTable: isLoading };
}

export default useCompanyDriversForTable;
