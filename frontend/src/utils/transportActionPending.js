/** Détection d’une TransportAction en attente de réponse transporteur. */

const OPEN_ACTION_STATUSES = new Set(['pending', 'requested', 'counter_pending']);

/**
 * @param {object|null|undefined} changeRequest
 * @returns {boolean}
 */
export function isTransportActionPending(changeRequest) {
  if (!changeRequest) return false;
  const status = String(changeRequest.status || '').toLowerCase();
  if (
    [
      'completed',
      'rejected',
      'refused',
      'expired',
      'closed_replaced',
      'superseded',
      'conflicted',
      'accepted',
    ].includes(status)
  ) {
    return false;
  }
  if (OPEN_ACTION_STATUSES.has(status)) return true;
  // Compat legacy : pending=true uniquement si statut non terminal
  return changeRequest.pending === true;
}

/**
 * @param {object|null|undefined} changeRequest
 * @returns {boolean}
 */
export function isCancellationTransportAction(changeRequest) {
  const type = String(
    changeRequest?.action_type || changeRequest?.pending_action_type || '',
  ).toUpperCase();
  return type === 'CANCELLATION';
}

/**
 * Indexe l’action ouverte + booking propriétaire par route_group_id.
 * @param {Array<object>|null|undefined} reservations
 * @returns {Map<string|number, { action: object, booking: object }>}
 */
export function indexPendingActionsByRouteGroup(reservations) {
  const map = new Map();
  for (const booking of reservations || []) {
    const acr = booking?.active_change_request;
    if (!isTransportActionPending(acr)) continue;
    const groupId = booking.route_group_id;
    if (groupId == null || groupId === '') continue;
    if (!map.has(groupId)) {
      map.set(groupId, { action: acr, booking });
    }
  }
  return map;
}

/**
 * Action pendante propre au booking, ou héritée du parcours multi-trajets.
 * @param {object} booking
 * @param {Map<string|number, { action: object, booking: object }>|null|undefined} pendingByRouteGroup
 * @returns {object|null}
 */
export function resolvePendingTransportAction(booking, pendingByRouteGroup) {
  if (isTransportActionPending(booking?.active_change_request)) {
    return booking.active_change_request;
  }
  if (booking?.trip_flags?.change_request_pending && booking?.active_change_request) {
    return booking.active_change_request;
  }
  const groupId = booking?.route_group_id;
  if (groupId != null && groupId !== '' && pendingByRouteGroup?.has(groupId)) {
    return pendingByRouteGroup.get(groupId).action;
  }
  return null;
}

/**
 * Booking à ouvrir pour répondre (propriétaire de l’action, pas un sibling hérité).
 * @param {object} booking
 * @param {Map<string|number, { action: object, booking: object }>|null|undefined} pendingByRouteGroup
 * @param {Array<object>|null|undefined} reservations
 * @returns {object}
 */
export function resolveRespondTargetBooking(booking, pendingByRouteGroup, reservations = []) {
  if (isTransportActionPending(booking?.active_change_request)) {
    return booking;
  }

  const groupId = booking?.route_group_id;
  if (groupId != null && groupId !== '' && pendingByRouteGroup?.has(groupId)) {
    return pendingByRouteGroup.get(groupId).booking;
  }

  const inherited = resolvePendingTransportAction(booking, pendingByRouteGroup);
  const ownerId = inherited?.booking_id;
  if (ownerId != null) {
    const found = (reservations || []).find((b) => Number(b.id) === Number(ownerId));
    if (found) return found;
  }

  return booking;
}

/**
 * @param {object|null|undefined} changeRequest
 * @returns {{ label: string, title: string, isCancellation: boolean }|null}
 */
export function getPendingActionBadge(changeRequest) {
  if (!isTransportActionPending(changeRequest)) return null;
  const isCancellation = isCancellationTransportAction(changeRequest);
  return {
    isCancellation,
    label: isCancellation ? 'Annulation en attente' : 'Modif. en attente',
    title: isCancellation
      ? 'Annulation à confirmer'
      : 'Modification à confirmer',
  };
}
