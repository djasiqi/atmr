/**
 * Cache canonique des offres institution (dashboard + page Réservations).
 * Le badge « Institutions » se dérive de cette même liste — jamais d'un compteur séparé.
 */

import { lirieKeys, listScopeHash } from '../queryKeys/lirie';
import { filterVisibleInstitutionOffers } from './institutionOfferResponse';

export const REJECT_OFFER_ERROR_TOAST =
  "Le refus n'a pas pu être enregistré. Réessayez.";
export const ACCEPT_OFFER_ERROR_TOAST =
  "L'acceptation n'a pas pu être enregistrée. Réessayez.";

const RESERVATIONS_SCOPE = listScopeHash({ flat: true, include_stats: false });
const DECISION_TTL_MS = 5 * 60 * 1000;
const TERMINAL_STATUSES = new Set(['ACCEPTED', 'REJECTED', 'EXPIRED', 'UNAVAILABLE']);

/** @type {Map<number, { action: string, at: number, revision: number }>} */
const localDecisions = new Map();

export function resetLocalInstitutionOfferDecisions() {
  localDecisions.clear();
}

export function rememberLocalOfferDecision(offerId, action, revision = Date.now()) {
  const id = Number(offerId);
  if (!Number.isFinite(id)) return;
  localDecisions.set(id, { action, at: Date.now(), revision: Number(revision) || Date.now() });
}

export function clearLocalOfferDecision(offerId) {
  localDecisions.set(Number(offerId), undefined);
  localDecisions.delete(Number(offerId));
}

function pruneDecisions(now = Date.now()) {
  localDecisions.forEach((entry, id) => {
    if (!entry || now - entry.at > DECISION_TTL_MS) {
      localDecisions.delete(id);
    }
  });
}

export function isSupersededByLocalDecision(
  offerId,
  incomingStatus,
  incomingRevision = null,
  now = Date.now(),
) {
  pruneDecisions(now);
  const entry = localDecisions.get(Number(offerId));
  if (!entry) return false;
  if (incomingRevision != null && Number(incomingRevision) > entry.revision) {
    return false;
  }
  const status = String(incomingStatus || 'PENDING').toUpperCase();
  if (entry.action === 'REJECTED' || entry.action === 'ACCEPTED') {
    return status === 'PENDING' || status === '';
  }
  return false;
}

export function countVisibleInstitutionOffers(offers, nowMs = Date.now()) {
  return filterVisibleInstitutionOffers(offers, nowMs).length;
}

export function reconcileInstitutionOffersResponse(data, now = Date.now()) {
  pruneDecisions(now);
  const offers = (data?.offers || []).filter(
    (offer) => !isSupersededByLocalDecision(offer?.id, offer?.status, offer?.updated_at, now),
  );
  return {
    ...data,
    offers,
    total: offers.length,
  };
}

export function findOfferIndex(offers, offerId) {
  return (offers || []).findIndex((offer) => Number(offer.id) === Number(offerId));
}

export function removeInstitutionOfferFromCache(queryClient, offerId) {
  let removed = null;
  let index = -1;
  queryClient.setQueryData(lirieKeys.institutionOffers(), (old) => {
    if (!old?.offers) return old;
    index = findOfferIndex(old.offers, offerId);
    if (index < 0) return old;
    removed = old.offers[index];
    const offers = old.offers.filter((_, i) => i !== index);
    return { ...old, offers, total: offers.length };
  });
  return { removed, index };
}

export function restoreInstitutionOfferInCache(queryClient, offer, index = null) {
  if (!offer) return;
  queryClient.setQueryData(lirieKeys.institutionOffers(), (old) => {
    const current = old?.offers || [];
    if (findOfferIndex(current, offer.id) >= 0) return old || { offers: current, total: current.length };
    const offers = [...current];
    const insertAt = Number.isInteger(index)
      ? Math.min(Math.max(index, 0), offers.length)
      : offers.length;
    offers.splice(insertAt, 0, offer);
    return { ...(old || {}), offers, total: offers.length };
  });
}

export function upsertPendingInstitutionOffer(queryClient, offer) {
  if (!offer?.id) return;
  if (isSupersededByLocalDecision(offer.id, offer.status, offer.updated_at)) return;
  const status = String(offer.status || 'PENDING').toUpperCase();
  if (TERMINAL_STATUSES.has(status)) {
    removeInstitutionOfferFromCache(queryClient, offer.id);
    return;
  }
  queryClient.setQueryData(lirieKeys.institutionOffers(), (old) => {
    const current = old?.offers || [];
    const index = findOfferIndex(current, offer.id);
    const offers = index >= 0
      ? current.map((item, i) => (i === index ? { ...item, ...offer } : item))
      : [offer, ...current];
    return { ...(old || {}), offers, total: offers.length };
  });
}

export function tempBookingIdFromOffer(offerId) {
  return -Math.abs(Number(offerId));
}

export function bookingFromAcceptedOffer(offer, { bookingId, proposedPickupTime } = {}) {
  const req = offer?.transport_request || {};
  const patient = req.patient || {};
  const name = req.patient_name
    || [patient.last_name, patient.first_name].filter(Boolean).join(' ')
    || 'Patient';
  return {
    id: bookingId ?? tempBookingIdFromOffer(offer.id),
    __fromInstitutionOfferId: Number(offer.id),
    customer_name: name,
    pickup_location: req.pickup_location,
    dropoff_location: req.dropoff_location,
    scheduled_time: proposedPickupTime || req.scheduled_time || req.next_confirmed_time,
    mission_date: req.mission_date || req.scheduling?.mission_date,
    status: 'accepted',
    driver_id: null,
    source: 'institution',
    client: { institution_name: req.institution_name },
  };
}

export function reservationsCacheKey(day) {
  return lirieKeys.companyReservations(day ?? '__all__', RESERVATIONS_SCOPE);
}

export function upsertReservationInCache(queryClient, day, booking) {
  if (!booking || booking.id == null) return;
  queryClient.setQueryData(reservationsCacheKey(day), (prev) => {
    const list = Array.isArray(prev) ? prev : [];
    const idx = list.findIndex((row) => (
      Number(row.id) === Number(booking.id)
      || Number(row.__fromInstitutionOfferId) === Number(booking.__fromInstitutionOfferId)
    ));
    if (idx >= 0) {
      const next = [...list];
      next[idx] = { ...next[idx], ...booking };
      return next;
    }
    return [booking, ...list];
  });
}

export function removeReservationFromOffer(queryClient, day, offerId) {
  queryClient.setQueryData(reservationsCacheKey(day), (prev) => {
    const list = Array.isArray(prev) ? prev : [];
    return list.filter((row) => (
      Number(row.__fromInstitutionOfferId) !== Number(offerId)
      && Number(row.id) !== tempBookingIdFromOffer(offerId)
    ));
  });
}

export function adjustBootstrapUnassignedKpi(queryClient, { authEnv, companyId, day }, delta) {
  if (companyId == null || !day) return;
  queryClient.setQueryData(
    lirieKeys.companyDashboardBootstrap(authEnv, companyId, day),
    (old) => {
      if (!old?.kpi) return old;
      const next = Math.max(0, (Number(old.kpi.unassigned) || 0) + delta);
      return { ...old, kpi: { ...old.kpi, unassigned: next } };
    },
  );
}
