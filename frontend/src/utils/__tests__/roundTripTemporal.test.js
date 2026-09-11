import {
  RETURN_BEFORE_OUTBOUND_MESSAGE,
  isReturnPickupPossible,
  resolveReturnPickupConflict,
  wallClockKey,
} from '../roundTripTemporal';

describe('roundTripTemporal', () => {
  it('rejette un retour confirmé avant l’aller', () => {
    expect(isReturnPickupPossible({
      outboundPickup: '2026-09-12T14:15',
      returnPickup: '2026-09-12T12:15',
    })).toBe(false);
  });

  it('rejette un retour avant le RDV', () => {
    expect(isReturnPickupPossible({
      outboundPickup: '2026-09-12T14:15',
      returnPickup: '2026-09-12T14:45',
      appointment: '2026-09-12T15:00',
    })).toBe(false);
  });

  it('conserve un retour encore valide', () => {
    expect(isReturnPickupPossible({
      outboundPickup: '2026-09-12T12:15',
      returnPickup: '2026-09-12T16:00',
    })).toBe(true);
  });

  it('signale le conflit depuis le dossier retour', () => {
    const message = resolveReturnPickupConflict({
      reservation: { is_return: true, parent_booking_id: 10 },
      linkedBookings: [
        { id: 10, scheduled_time: '2026-09-12T14:15:00' },
      ],
      returnPickupIso: '2026-09-12T12:15:00',
    });
    expect(message).toBe(RETURN_BEFORE_OUTBOUND_MESSAGE);
  });

  it('détecte un retour route_group 2/2 avant l’aller', () => {
    const message = resolveReturnPickupConflict({
      reservation: {
        id: 45727,
        is_return: false,
        route_group_id: 'grp-cavadini',
        route_sequence_number: 2,
      },
      linkedBookings: [
        {
          id: 45726,
          route_group_id: 'grp-cavadini',
          route_sequence_number: 1,
          scheduled_time: '2026-09-12T14:15:00',
        },
      ],
      returnPickupIso: '2026-09-12T12:15:00',
    });
    expect(message).toBe(RETURN_BEFORE_OUTBOUND_MESSAGE);
  });

  it('normalise une clé murale', () => {
    expect(wallClockKey({ iso: '2026-09-12T14:15:00' })).toBe('2026-09-12T14:15');
  });
});
