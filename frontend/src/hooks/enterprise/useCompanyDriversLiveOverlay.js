import { useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { getCompanySocket } from '../../services/companySocket';
import { mergeOrUpdateDriverInList } from '../../utils/mergeDriverLiveUpdate';
import { lirieKeys } from '../../queryKeys/lirie';

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
    const onReconnected = () => {
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
