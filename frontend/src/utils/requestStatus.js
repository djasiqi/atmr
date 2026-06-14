/**
 * Helpers centralisés pour le statut et le mode d'exécution des demandes institution.
 */

export const CARRIER_SOURCE_EXTERNAL = 'external';
export const CARRIER_SOURCE_LIRIE = 'lirie';

export const EXTERNAL_STATUSES = {
  ASSIGNED: 'EXTERNAL_ASSIGNED',
  COMPLETED: 'EXTERNAL_DECLARED_COMPLETED',
};

const FALLBACK_STATUS_LABELS = {
  DRAFT: 'Brouillon',
  SENT: 'Envoyée',
  ACCEPTED: 'Acceptée',
  CONVERTED: 'Confirmée',
  CANCELLED: 'Annulée',
  EXPIRED: 'Expirée',
  [EXTERNAL_STATUSES.ASSIGNED]: 'Transporteur externe affecté',
  [EXTERNAL_STATUSES.COMPLETED]: 'Déclarée réalisée par l\'institution',
};

const TERMINAL_BOOKING_STATUSES = new Set(['COMPLETED', 'RETURN_COMPLETED']);

/**
 * @param {object|null|undefined} req
 * @returns {boolean}
 */
export function isExternalRequest(req) {
  return req?.carrier_source === CARRIER_SOURCE_EXTERNAL;
}

/**
 * @param {object|null|undefined} req
 * @returns {boolean}
 */
export function isAssignedRequest(req) {
  return req?.status === EXTERNAL_STATUSES.ASSIGNED;
}

/**
 * Mission terminée (externe déclarée ou booking LIRIE complété).
 * @param {object|null|undefined} req
 * @returns {boolean}
 */
export function isCompletedRequest(req) {
  if (!req) return false;
  if (req.status === EXTERNAL_STATUSES.COMPLETED) return true;
  const bs = req.booking_summary;
  if (!bs) return false;
  const raw = String(bs.status || '').toUpperCase();
  const normalized = raw === 'CANCELLED' ? 'CANCELED' : raw;
  if (normalized === 'CANCELED') return false;
  if (bs.completed_at && !bs.return_booking) return true;
  return TERMINAL_BOOKING_STATUSES.has(normalized);
}

/**
 * Demande convertie en booking LIRIE (pas externe).
 * @param {object|null|undefined} req
 * @returns {boolean}
 */
export function isConvertedLirie(req) {
  return req?.status === 'CONVERTED' && Boolean(req?.booking_id);
}

/**
 * Garde anti-crash : booking opérationnel disponible pour l'UI.
 * @param {object|null|undefined} req
 * @returns {boolean}
 */
export function hasBooking(req) {
  return Boolean(req?.booking_id && req?.booking_summary);
}

/**
 * Libellé de statut — privilégie status_label API.
 * @param {object|null|undefined} req
 * @returns {string}
 */
export function getRequestStatusLabel(req) {
  if (!req) return '';
  if (req.status_label) return req.status_label;
  return FALLBACK_STATUS_LABELS[req.status] || req.status || '';
}

/**
 * Libellé du mode d'exécution.
 * @param {object|null|undefined} req
 * @returns {string}
 */
export function getCarrierSourceLabel(req) {
  if (!req) return '';
  if (req.carrier_source_label) return req.carrier_source_label;
  if (isExternalRequest(req)) return 'Externe';
  return 'LIRIE';
}

/**
 * Peut-on affecter un transporteur externe depuis le détail ?
 * @param {object|null|undefined} req
 * @returns {boolean}
 */
export function canAssignExternalCarrier(req) {
  if (!req || isExternalRequest(req)) return false;
  return ['DRAFT', 'SENT'].includes(req.status);
}

/**
 * Peut-on déclarer une mission externe réalisée ?
 * @param {object|null|undefined} req
 * @returns {boolean}
 */
export function canCompleteExternalMission(req) {
  return isExternalRequest(req) && req?.status === EXTERNAL_STATUSES.ASSIGNED;
}
