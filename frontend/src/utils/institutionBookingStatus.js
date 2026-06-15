/**
 * Résolution du statut opérationnel booking (A/R, annulations, etc.)
 * et regroupement KPI dashboard institution.
 */

import {
  EXTERNAL_STATUSES,
  isConvertedLirie,
  isExternalRequest,
} from './requestStatus';

const TERMINAL_COMPLETED_KEYS = new Set(['COMPLETED', 'RETURN_COMPLETED']);

/**
 * @param {object|null|undefined} bookingSummary
 * @returns {string}
 */
export function resolveBookingStatusKey(bookingSummary) {
  if (!bookingSummary) return '';
  const raw = String(bookingSummary.status || '').toUpperCase();
  const normalized = raw === 'CANCELLED' ? 'CANCELED' : raw;
  const returnRaw = String(bookingSummary.return_booking?.status || '').toUpperCase();
  const returnStatus = returnRaw === 'CANCELLED' ? 'CANCELED' : returnRaw;
  const overall = String(bookingSummary.overall_status || '').toLowerCase();

  const hasReturn = Boolean(bookingSummary.return_booking);
  const returnCompleted = TERMINAL_COMPLETED_KEYS.has(returnStatus);
  const returnCancelled = returnStatus === 'CANCELED';
  const outboundCompleted = TERMINAL_COMPLETED_KEYS.has(normalized);

  if (hasReturn && overall) {
    if (overall === 'completed') return 'RETURN_COMPLETED';
    if (overall === 'cancelled') return 'CANCELED';
    if (overall === 'outbound_completed') return 'OUTBOUND_COMPLETED';
    if (overall === 'in_progress') return 'IN_PROGRESS';
    if (overall === 'planned') return 'ACCEPTED';
  }

  if (hasReturn) {
    if (returnCompleted) return 'RETURN_COMPLETED';
    if (returnCancelled) return 'CANCELED';
    if (outboundCompleted) return 'OUTBOUND_COMPLETED';
  }

  if (
    bookingSummary.completed_at &&
    !hasReturn &&
    normalized !== 'RETURN_COMPLETED' &&
    normalized !== 'CANCELED'
  ) {
    return 'COMPLETED';
  }

  if (
    bookingSummary.boarded_at &&
    normalized !== 'COMPLETED' &&
    normalized !== 'RETURN_COMPLETED' &&
    normalized !== 'CANCELED'
  ) {
    return 'IN_PROGRESS';
  }

  if (
    bookingSummary.en_route_at &&
    !bookingSummary.boarded_at &&
    (normalized === 'ACCEPTED' || normalized === 'ASSIGNED' || normalized === '')
  ) {
    return 'EN_ROUTE';
  }

  return normalized;
}

/**
 * @param {object|null|undefined} req
 * @returns {'draft'|'pending'|'active'|'completed'|'cancelled'|null}
 */
export function getInstitutionRequestKpiBucket(req) {
  if (!req) return null;

  if (req.status === 'DRAFT') return 'draft';
  if (req.status === 'CANCELLED') return 'cancelled';
  if (req.status === EXTERNAL_STATUSES.COMPLETED) return 'completed';
  if (req.status === 'SENT' || req.status === 'ACCEPTED') return 'pending';

  if (isExternalRequest(req)) {
    if (req.status === EXTERNAL_STATUSES.ASSIGNED) return 'active';
    return null;
  }

  if (isConvertedLirie(req) && req.booking_summary) {
    const bookingKey = resolveBookingStatusKey(req.booking_summary);
    if (bookingKey === 'CANCELED') return 'cancelled';
    if (TERMINAL_COMPLETED_KEYS.has(bookingKey)) return 'completed';
    if (bookingKey) return 'active';
  }

  if (req.status === 'CONVERTED') return 'active';

  return null;
}

/**
 * @param {object[]} items
 * @param {number} [total]
 * @returns {{ total: number, pending: number, active: number, completed: number, cancelled: number, needsAttention: number }}
 */
export function computeInstitutionRequestStats(items, total) {
  const stats = {
    total: total ?? items.length,
    pending: 0,
    active: 0,
    completed: 0,
    cancelled: 0,
    needsAttention: 0,
  };

  for (const req of items) {
    const bucket = getInstitutionRequestKpiBucket(req);
    if (bucket && stats[bucket] !== undefined) {
      stats[bucket] += 1;
    }

    if (req.status === 'SENT') {
      const sentTime = new Date(req.updated_at || req.created_at);
      const hoursAgo = (Date.now() - sentTime.getTime()) / 3600000;
      if (hoursAgo > 2) stats.needsAttention += 1;
    }
  }

  return stats;
}
