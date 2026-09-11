import { useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { acceptRequestOffer, fetchRequestOffer, rejectRequestOffer } from '../services/companyService';
import { getCurrentAuthEnv } from '../utils/apiClient';
import { canRespondToInstitutionOffer } from '../utils/institutionOfferResponse';
import {
  ACCEPT_OFFER_ERROR_TOAST,
  REJECT_OFFER_ERROR_TOAST,
  adjustBootstrapUnassignedKpi,
  bookingFromAcceptedOffer,
  clearLocalOfferDecision,
  isSupersededByLocalDecision,
  rememberLocalOfferDecision,
  removeInstitutionOfferFromCache,
  removeReservationFromOffer,
  restoreInstitutionOfferInCache,
  upsertPendingInstitutionOffer,
  upsertReservationInCache,
} from '../utils/institutionOffersCache';

function missionDayOf(offer) {
  const req = offer?.transport_request || {};
  return req.mission_date || req.scheduling?.mission_date || null;
}

/**
 * Accept / refus d'offres institution : mise à jour locale immédiate du cache canonique.
 * Le badge Institutions se recalcule depuis cette même liste.
 */
export function useInstitutionOfferMutations({
  dispatchDay,
  companyId,
  upsertReservation,
} = {}) {
  const queryClient = useQueryClient();
  const authEnv = getCurrentAuthEnv();

  const rejectOffer = useCallback(async (offerId, offerForGuard = null) => {
    if (offerForGuard && !canRespondToInstitutionOffer(offerForGuard)) {
      toast.error('Offre expirée, vous ne pouvez plus répondre.');
      return;
    }

    const revision = Date.now();
    rememberLocalOfferDecision(offerId, 'REJECTED', revision);
    const { removed, index } = removeInstitutionOfferFromCache(queryClient, offerId);

    try {
      await rejectRequestOffer(offerId);
    } catch (err) {
      clearLocalOfferDecision(offerId);
      restoreInstitutionOfferInCache(queryClient, removed || offerForGuard, index);
      toast.error(REJECT_OFFER_ERROR_TOAST);
      throw err;
    }
  }, [queryClient]);

  const acceptOffer = useCallback(async (offerId, proposedPickupTime, offerForGuard = null) => {
    if (offerForGuard && !canRespondToInstitutionOffer(offerForGuard)) {
      toast.error('Offre expirée, vous ne pouvez plus répondre.');
      return;
    }

    const revision = Date.now();
    rememberLocalOfferDecision(offerId, 'ACCEPTED', revision);
    const { removed, index } = removeInstitutionOfferFromCache(queryClient, offerId);
    const sourceOffer = removed || offerForGuard;
    const missionDay = missionDayOf(sourceOffer);
    const booking = sourceOffer
      ? bookingFromAcceptedOffer(sourceOffer, { proposedPickupTime })
      : null;
    const bookingDay = missionDay || dispatchDay;
    const bookingMatchesDay = !dispatchDay || !missionDay || missionDay === dispatchDay;

    if (booking && bookingMatchesDay) {
      if (typeof upsertReservation === 'function') {
        upsertReservation(booking);
      } else {
        upsertReservationInCache(queryClient, bookingDay, booking);
      }
      adjustBootstrapUnassignedKpi(
        queryClient,
        { authEnv, companyId, day: dispatchDay },
        1,
      );
    }

    try {
      const result = await acceptRequestOffer(offerId, proposedPickupTime);
      if (booking && bookingMatchesDay && result?.booking_id) {
        const confirmed = { ...booking, id: result.booking_id };
        removeReservationFromOffer(queryClient, bookingDay, offerId);
        if (typeof upsertReservation === 'function') {
          upsertReservation(confirmed);
        }
        upsertReservationInCache(queryClient, bookingDay, confirmed);
      }
      toast.success(
        proposedPickupTime
          ? 'Offre planifiée — réservation créée'
          : 'Offre validée — réservation créée',
      );
      return result;
    } catch (err) {
      clearLocalOfferDecision(offerId);
      restoreInstitutionOfferInCache(queryClient, sourceOffer, index);
      if (booking && bookingMatchesDay) {
        removeReservationFromOffer(queryClient, bookingDay, offerId);
        adjustBootstrapUnassignedKpi(
          queryClient,
          { authEnv, companyId, day: dispatchDay },
          -1,
        );
      }
      toast.error(ACCEPT_OFFER_ERROR_TOAST);
      throw err;
    }
  }, [authEnv, companyId, dispatchDay, queryClient, upsertReservation]);

  const applyRealtimeOfferEvent = useCallback(async (payload) => {
    const offerId = payload?.offer_id ?? payload?.metadata?.offer_id;
    if (!offerId) return;
    const incomingStatus = payload?.status || payload?.metadata?.status;
    if (isSupersededByLocalDecision(offerId, incomingStatus || 'PENDING', payload?.updated_at)) {
      return;
    }
    const status = String(incomingStatus || '').toUpperCase();
    if (status && ['ACCEPTED', 'REJECTED', 'EXPIRED', 'UNAVAILABLE'].includes(status)) {
      removeInstitutionOfferFromCache(queryClient, offerId);
      return;
    }
    try {
      const offer = await fetchRequestOffer(offerId);
      if (isSupersededByLocalDecision(offer?.id, offer?.status, offer?.updated_at)) {
        return;
      }
      upsertPendingInstitutionOffer(queryClient, offer);
    } catch (err) {
      console.warn('[institutionOffers] fetch offre temps réel impossible', err);
    }
  }, [queryClient]);

  const applyOfferUnavailable = useCallback((payload) => {
    const offerId = payload?.offer_id ?? payload?.metadata?.offer_id;
    if (!offerId) return;
    if (isSupersededByLocalDecision(offerId, 'UNAVAILABLE', payload?.updated_at)) {
      removeInstitutionOfferFromCache(queryClient, offerId);
      return;
    }
    removeInstitutionOfferFromCache(queryClient, offerId);
  }, [queryClient]);

  return {
    rejectOffer,
    acceptOffer,
    applyRealtimeOfferEvent,
    applyOfferUnavailable,
  };
}
