import { useQuery } from '@tanstack/react-query';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';

import { getInstitutionMe } from '@/services/institutionApi';
import {
  disconnectInstitutionRealtime,
  joinInstitutionRealtime,
  subscribeInstitutionEvents,
} from '@/services/institutionRealtimeBridge';
import { queryKeys } from '@/services/queryKeys';

export function useInstitutionRealtime(enabled = true): void {
  const queryClient = useQueryClient();
  const meQuery = useQuery({
    queryKey: queryKeys.institutionMe,
    queryFn: getInstitutionMe,
  });

  useEffect(() => {
    if (!enabled) return;
    const institutionId = meQuery.data?.id;
    if (!institutionId) return;

    joinInstitutionRealtime(institutionId);
    const unsubscribe = subscribeInstitutionEvents(() => {
      void queryClient.invalidateQueries({
        queryKey: ['institution', 'requests'],
      });
      void queryClient.invalidateQueries({
        queryKey: ['institution', 'request'],
      });
    });

    return () => {
      unsubscribe();
      disconnectInstitutionRealtime();
    };
  }, [enabled, meQuery.data?.id, queryClient]);
}
