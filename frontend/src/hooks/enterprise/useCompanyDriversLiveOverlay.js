import { useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { getCompanySocket } from '../../services/companySocket';
import { mergeOrUpdateDriverInList } from '../../utils/mergeDriverLiveUpdate';
import { lirieKeys } from '../../queryKeys/lirie';

/** Si le snapshot TanStack est encore frais, pas d’invalidate complet (réduit pics en flapping WS). */
const RECONNECT_MIN_FULL_REFETCH_MS = 60000;

/**
 * Fusionne les événements Socket chauffeurs dans le cache TanStack `lirieKeys.companyDrivers()`.
 */
export function useCompanyDriversLiveOverlay(companyId) {
  const queryClient = useQueryClient();

  useEffect(() => {
    const socket = getCompanySocket();
    if (!socket) return;

    const applyDelta = (payload, fromLiveState) => {
      queryClient.setQueryData(lirieKeys.companyDrivers(), (prev) =>
        mergeOrUpdateDriverInList(prev || [], payload, fromLiveState, companyId ?? null)
      );
    };

    const onLiveState = (payload) => applyDelta(payload, true);
    const onLocationUpdate = (payload) => applyDelta(payload, false);
    /** Reconnect : invalider seulement si pas de données récentes (overlay socket a déjà mis à jour le cache). */
    const onReconnected = () => {
      const state = queryClient.getQueryState(lirieKeys.companyDrivers());
      const updatedAt = state?.dataUpdatedAt ?? 0;
      if (
        updatedAt > 0 &&
        Date.now() - updatedAt < RECONNECT_MIN_FULL_REFETCH_MS
      ) {
        return;
      }
      queryClient.invalidateQueries({ queryKey: lirieKeys.companyDrivers() });
    };

    socket.on('driver_live_state_update', onLiveState);
    socket.on('driver_location_update', onLocationUpdate);
    if (typeof window !== 'undefined') {
      window.addEventListener('company_socket_reconnected', onReconnected);
    }

    return () => {
      socket.off('driver_live_state_update', onLiveState);
      socket.off('driver_location_update', onLocationUpdate);
      if (typeof window !== 'undefined') {
        window.removeEventListener('company_socket_reconnected', onReconnected);
      }
    };
  }, [queryClient, companyId]);
}
