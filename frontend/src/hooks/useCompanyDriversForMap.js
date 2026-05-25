import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';
import { fetchCompanyDriversCanonical } from '../services/companyService';
import { lirieKeys } from '../queryKeys/lirie';
import { projectDriversForMap } from '../utils/companyDriverProjections';
import { useSocketConnected } from './useCompanySocket';

function useCompanyDriversLiveQueryPolicy(companyId) {
  const socketConnected = useSocketConnected();
  return Boolean(companyId != null && socketConnected);
}

/**
 * Projection chauffeurs dédiée à DriverLiveMap — même cache TanStack que useCompanyData
 * (`lirieKeys.companyDrivers()`), sans second overlay socket.
 */
export function useCompanyDriversForMap(companyId) {
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

  const { data: driversForMap = [], isLoading } = useQuery({
    queryKey: lirieKeys.companyDrivers(),
    queryFn: async () => {
      const data = await fetchCompanyDriversCanonical();
      return Array.isArray(data) ? data : data?.driver ?? [];
    },
    select: projectDriversForMap,
    enabled: Boolean(companyId),
    ...companyDriversQueryOptions,
  });

  return { driversForMap, loadingDriversForMap: isLoading };
}

export default useCompanyDriversForMap;
