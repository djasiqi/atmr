import { useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import DataFreshnessBadge from '../../../../components/common/DataFreshnessBadge';
import { LIRIE_QK_PREFIX } from '../../../../queryKeys/lirie';

/**
 * Badge de fraîcheur isolé : les mises à jour socket (GPS, overlay TanStack)
 * ne re-rendent pas le dashboard entier, seulement ce composant.
 */
export function CompanyDashboardFreshnessBadge({
  lastHttpSyncAt = null,
  isSyncing = false,
  realtimeConnected = true,
  className = '',
}) {
  const queryClient = useQueryClient();
  const [liveCacheUpdatedAt, setLiveCacheUpdatedAt] = useState(null);

  useEffect(() => {
    if (!realtimeConnected) {
      setLiveCacheUpdatedAt(null);
      return undefined;
    }

    const bumpFromCache = () => {
      let maxUpdatedAt = 0;
      queryClient.getQueryCache().findAll({
        predicate: (query) => Array.isArray(query.queryKey) && query.queryKey[0] === LIRIE_QK_PREFIX,
      }).forEach((query) => {
        const updatedAt = query.state.dataUpdatedAt ?? 0;
        if (updatedAt > maxUpdatedAt) maxUpdatedAt = updatedAt;
      });
      if (maxUpdatedAt > 0) {
        setLiveCacheUpdatedAt((prev) => Math.max(prev ?? 0, maxUpdatedAt));
      }
    };

    bumpFromCache();

    const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
      if (event?.type !== 'updated') return;
      const key = event.query?.queryKey;
      if (!Array.isArray(key) || key[0] !== LIRIE_QK_PREFIX) return;
      const updatedAt = event.query.state.dataUpdatedAt ?? 0;
      if (updatedAt > 0) {
        setLiveCacheUpdatedAt((prev) => Math.max(prev ?? 0, updatedAt));
      }
    });

    return unsubscribe;
  }, [realtimeConnected, queryClient]);

  const lastSyncAt = useMemo(() => {
    const httpMs = lastHttpSyncAt ?? 0;
    const liveMs = realtimeConnected && liveCacheUpdatedAt ? liveCacheUpdatedAt : 0;
    const merged = Math.max(httpMs, liveMs);
    return merged > 0 ? merged : null;
  }, [lastHttpSyncAt, liveCacheUpdatedAt, realtimeConnected]);

  return (
    <DataFreshnessBadge
      lastSyncAt={lastSyncAt}
      isSyncing={isSyncing}
      realtimeEnabled
      realtimeConnected={realtimeConnected}
      sourceLabel="Dispatch"
      className={className}
    />
  );
}
