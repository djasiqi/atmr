// hooks/useInstitutionSocket.js
/**
 * ÉTAPE 6: Hook React pour Socket.IO Institution
 * 
 * Gère la connexion et l'invalidation automatique des queries React Query
 */

import { useEffect, useRef, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  ensureInstitutionSocket,
  joinInstitutionRoom,
  leaveInstitutionRoom,
  on,
  off,
  disconnectInstitutionSocket,
} from '../services/institutionSocket';
import { institutionQueryKeys } from './useInstitutionData';

/**
 * Hook pour gérer la connexion Socket.IO et les invalidations
 * @param {number|string} institutionId - ID de l'institution
 */
export function useInstitutionSocket(institutionId) {
  const queryClient = useQueryClient();
  const mountedRef = useRef(true);

  // Handler pour request_sent
  const handleRequestSent = useCallback((data) => {
    console.log('[InstitutionSocket] request_sent:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
  }, [queryClient]);

  // Handler pour offer_accepted
  const handleOfferAccepted = useCallback((data) => {
    console.log('[InstitutionSocket] offer_accepted:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(data.request_id) });
    }
  }, [queryClient]);

  // Handler pour request_converted
  const handleRequestConverted = useCallback((data) => {
    console.log('[InstitutionSocket] request_converted:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(data.request_id) });
    }
  }, [queryClient]);

  // Handler pour booking_status_updated
  const handleBookingStatusUpdated = useCallback((data) => {
    console.log('[InstitutionSocket] booking_status_updated:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(data.request_id) });
    }
  }, [queryClient]);

  // Handler pour request_cancelled
  const handleRequestCancelled = useCallback((data) => {
    console.log('[InstitutionSocket] request_cancelled:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(data.request_id) });
    }
  }, [queryClient]);

  // Handler pour booking_cancelled
  const handleBookingCancelled = useCallback((data) => {
    console.log('[InstitutionSocket] booking_cancelled:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(data.request_id) });
    }
  }, [queryClient]);

  // Handler pour new_notification (invalidation du cache notifications)
  const handleNewNotification = useCallback((data) => {
    console.log('[InstitutionSocket] new_notification:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.notifications() });
  }, [queryClient]);

  useEffect(() => {
    mountedRef.current = true;

    if (!institutionId) return;

    const setupSocket = async () => {
      try {
        await ensureInstitutionSocket();
        await joinInstitutionRoom(institutionId);

        // Écouter les événements
        await on('request_sent', handleRequestSent);
        await on('offer_accepted', handleOfferAccepted);
        await on('request_converted', handleRequestConverted);
        await on('booking_status_updated', handleBookingStatusUpdated);
        await on('request_cancelled', handleRequestCancelled);
        await on('booking_cancelled', handleBookingCancelled);
        await on('new_notification', handleNewNotification);

        console.log('[useInstitutionSocket] Socket configuré pour institution:', institutionId);
      } catch (err) {
        console.error('[useInstitutionSocket] Erreur setup:', err);
      }
    };

    setupSocket();

    return () => {
      mountedRef.current = false;
      
      // Cleanup listeners
      off('request_sent');
      off('offer_accepted');
      off('request_converted');
      off('booking_status_updated');
      off('request_cancelled');
      off('booking_cancelled');
      off('new_notification');
      
      // Quitter la room
      if (institutionId) {
        leaveInstitutionRoom(institutionId);
      }
    };
  }, [
    institutionId,
    handleRequestSent,
    handleOfferAccepted,
    handleRequestConverted,
    handleBookingStatusUpdated,
    handleRequestCancelled,
    handleBookingCancelled,
    handleNewNotification,
  ]);

  return {
    disconnect: disconnectInstitutionSocket,
  };
}

export default useInstitutionSocket;
