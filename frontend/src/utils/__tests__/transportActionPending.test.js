import {
  getPendingActionBadge,
  indexPendingActionsByRouteGroup,
  isCancellationTransportAction,
  isTransportActionPending,
  resolvePendingTransportAction,
  resolveRespondTargetBooking,
} from '../transportActionPending';

describe('transportActionPending', () => {
  it('détecte une action ouverte', () => {
    expect(isTransportActionPending({ status: 'requested', pending: true })).toBe(true);
    expect(isTransportActionPending({ status: 'completed' })).toBe(false);
  });

  it('détecte une annulation', () => {
    expect(isCancellationTransportAction({ action_type: 'CANCELLATION' })).toBe(true);
    expect(isCancellationTransportAction({ action_type: 'CHANGE_TIME' })).toBe(false);
  });

  it('propage l’action ouverte aux trajets du même parcours', () => {
    const acr = { status: 'requested', pending: true, action_type: 'CANCELLATION', booking_id: 1 };
    const rows = [
      { id: 1, route_group_id: 'g1', active_change_request: acr },
      { id: 2, route_group_id: 'g1', active_change_request: null },
    ];
    const byGroup = indexPendingActionsByRouteGroup(rows);
    expect(resolvePendingTransportAction(rows[1], byGroup)).toEqual(acr);
    expect(getPendingActionBadge(resolvePendingTransportAction(rows[1], byGroup))).toEqual({
      isCancellation: true,
      label: 'Annulation en attente',
      title: 'Annulation à confirmer',
    });
  });

  it('ouvre le booking propriétaire pour Répondre, pas le sibling', () => {
    const acr = { status: 'requested', pending: true, action_type: 'CANCELLATION', booking_id: 1 };
    const owner = { id: 1, route_group_id: 'g1', active_change_request: acr };
    const sibling = { id: 2, route_group_id: 'g1', active_change_request: null };
    const byGroup = indexPendingActionsByRouteGroup([owner, sibling]);
    expect(resolveRespondTargetBooking(sibling, byGroup, [owner, sibling])).toBe(owner);
    expect(resolveRespondTargetBooking(owner, byGroup, [owner, sibling])).toBe(owner);
  });
});
