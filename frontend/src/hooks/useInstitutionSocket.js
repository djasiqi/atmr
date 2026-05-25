// hooks/useInstitutionSocket.js
/**
 * ÉTAPE 6: Hook React pour Socket.IO Institution
 *
 * Gère la connexion et l'invalidation automatique des queries React Query
 */

import { useEffect, useRef, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import {
  ensureInstitutionSocket,
  getInstitutionSocket,
  joinInstitutionRoom,
  leaveInstitutionRoom,
  on,
  off,
  disconnectInstitutionSocket,
} from '../services/institutionSocket';
import { institutionQueryKeys } from './useInstitutionData';

const CHAT_ACTIVE_STATUSES = new Set([
  'ASSIGNED',
  'EN_ROUTE',
  'IN_PROGRESS',
]);

/**
 * Hook pour gérer la connexion Socket.IO et les invalidations
 * @param {number|string} institutionId - ID de l'institution
 */
export function useInstitutionSocket(institutionId) {
  const queryClient = useQueryClient();
  const mountedRef = useRef(true);
  const lastNotifToastRef = useRef(null);

  const handleRequestSent = useCallback((data) => {
    console.log('[InstitutionSocket] request_sent:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
  }, [queryClient]);

  const handleOfferAccepted = useCallback((data) => {
    console.log('[InstitutionSocket] offer_accepted:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({
        queryKey: institutionQueryKeys.requestDetail(data.request_id),
      });
    }
    const companyName = data?.company_name || data?.payload?.company_name;
    toast.success(
      companyName
        ? `${companyName} a accepté votre demande de transport.`
        : 'Une entreprise a accepté votre demande de transport.'
    );
  }, [queryClient]);

  const handleRequestConverted = useCallback((data) => {
    console.log('[InstitutionSocket] request_converted:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({
        queryKey: institutionQueryKeys.requestDetail(data.request_id),
      });
    }
  }, [queryClient]);

  const handleBookingStatusUpdated = useCallback((data) => {
    console.log('[InstitutionSocket] booking_status_updated:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({
        queryKey: institutionQueryKeys.requestDetail(data.request_id),
      });
    }
  }, [queryClient]);

  const handleRequestCancelled = useCallback((data) => {
    console.log('[InstitutionSocket] request_cancelled:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({
        queryKey: institutionQueryKeys.requestDetail(data.request_id),
      });
    }
  }, [queryClient]);

  const handleBookingCancelled = useCallback((data) => {
    console.log('[InstitutionSocket] booking_cancelled:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    if (data?.request_id) {
      queryClient.invalidateQueries({
        queryKey: institutionQueryKeys.requestDetail(data.request_id),
      });
    }
  }, [queryClient]);

  const handleNewNotification = useCallback((data) => {
    console.log('[InstitutionSocket] new_notification:', data);
    if (!mountedRef.current) return;
    queryClient.invalidateQueries({ queryKey: institutionQueryKeys.notifications() });

    const eventType = data?.event_type || data?.payload?.event_type;
    if (eventType === 'offer_accepted') {
      return;
    }

    const dedupeKey = data?.id || data?.dedupe_key || data?.message;
    if (dedupeKey && lastNotifToastRef.current === dedupeKey) {
      return;
    }
    if (dedupeKey) {
      lastNotifToastRef.current = dedupeKey;
    }

    const title = data?.title || data?.payload?.title;
    const message = data?.message || data?.payload?.message;
    if (title || message) {
      toast.info(message || title);
    }
  }, [queryClient]);

  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;

    if (!institutionId) return undefined;

    const setupSocket = async () => {
      try {
        await ensureInstitutionSocket();
        if (cancelled || !mountedRef.current) return;

        await joinInstitutionRoom(institutionId);

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
      cancelled = true;
      mountedRef.current = false;

      const s = getInstitutionSocket();
      if (s) {
        s.off('request_sent', handleRequestSent);
        s.off('offer_accepted', handleOfferAccepted);
        s.off('request_converted', handleRequestConverted);
        s.off('booking_status_updated', handleBookingStatusUpdated);
        s.off('request_cancelled', handleRequestCancelled);
        s.off('booking_cancelled', handleBookingCancelled);
        s.off('new_notification', handleNewNotification);
      }

      off('request_sent');
      off('offer_accepted');
      off('request_converted');
      off('booking_status_updated');
      off('request_cancelled');
      off('booking_cancelled');
      off('new_notification');

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
    chatActiveStatuses: CHAT_ACTIVE_STATUSES,
  };
}

export default useInstitutionSocket;
