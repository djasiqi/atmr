import {
  formatAppointmentTime,
  hasConfirmedPickupTime,
  hasScheduledPickupTime,
  isAppointmentTimeDefined,
  isPickupSentinel,
  isReturnLegNeedingTime,
  needsTimeBeforeDriverAssign,
} from '../bookingScheduling';

describe('bookingScheduling', () => {
  it('utilise scheduling.display_time', () => {
    const booking = {
      scheduling: {
        time_defined: true,
        time_scheduled: true,
        display_time: '14:30',
      },
    };
    expect(formatAppointmentTime(booking)).toBe('14:30');
  });

  it('retourne À définir si time_scheduled false', () => {
    const booking = {
      scheduling: {
        time_defined: false,
        time_scheduled: false,
        display_time: 'À définir',
      },
    };
    expect(formatAppointmentTime(booking)).toBe('À définir');
    expect(isAppointmentTimeDefined(booking)).toBe(false);
    expect(hasScheduledPickupTime(booking)).toBe(false);
  });

  it('13:30 non confirmé — heure présente mais pas confirmée', () => {
    const booking = {
      scheduled_time: '2026-06-12T13:30:00',
      time_confirmed: false,
    };
    expect(hasScheduledPickupTime(booking)).toBe(true);
    expect(hasConfirmedPickupTime(booking)).toBe(false);
    expect(isAppointmentTimeDefined(booking)).toBe(false);
  });

  it('minuit réel confirmé BK-01c', () => {
    const booking = {
      scheduled_time: '2026-06-12T00:00:00',
      time_confirmed: true,
    };
    expect(isPickupSentinel(booking.scheduled_time, true)).toBe(false);
    expect(hasScheduledPickupTime(booking)).toBe(true);
    expect(hasConfirmedPickupTime(booking)).toBe(true);
  });

  it('needsTimeBeforeDriverAssign — retour sans heure', () => {
    expect(
      needsTimeBeforeDriverAssign({
        is_return: true,
        time_confirmed: false,
        status: 'accepted',
      })
    ).toBe(true);
  });

  it('needsTimeBeforeDriverAssign — leg institution multi-étapes sans heure', () => {
    expect(
      needsTimeBeforeDriverAssign({
        route_group_id: 'grp-1',
        route_sequence_number: 2,
        time_confirmed: false,
        status: 'accepted',
        scheduling: { time_defined: false, time_scheduled: false },
      })
    ).toBe(true);
  });

  it('needsTimeBeforeDriverAssign — leg avec heure confirmée', () => {
    expect(
      needsTimeBeforeDriverAssign({
        route_group_id: 'grp-1',
        route_sequence_number: 1,
        status: 'accepted',
        scheduling: { time_defined: true, time_scheduled: true, display_time: '11:30' },
        scheduled_time: '2026-06-23T11:30:00',
      })
    ).toBe(false);
  });

  it('isReturnLegNeedingTime distingue retour et leg multi-étapes', () => {
    const retour = { is_return: true, time_confirmed: false };
    const legInstitution = {
      route_group_id: 'grp-1',
      time_confirmed: false,
      status: 'accepted',
    };
    expect(isReturnLegNeedingTime(retour)).toBe(true);
    expect(isReturnLegNeedingTime(legInstitution)).toBe(false);
  });
});
