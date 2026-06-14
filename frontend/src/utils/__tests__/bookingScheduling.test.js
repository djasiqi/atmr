import { formatAppointmentTime, isAppointmentTimeDefined } from '../bookingScheduling';

describe('bookingScheduling', () => {
  it('utilise scheduling.display_time', () => {
    const booking = {
      scheduling: { time_defined: true, display_time: '14:30' },
    };
    expect(formatAppointmentTime(booking)).toBe('14:30');
  });

  it('retourne À définir si time_defined false', () => {
    const booking = {
      scheduling: { time_defined: false, display_time: 'À définir' },
    };
    expect(formatAppointmentTime(booking)).toBe('À définir');
    expect(isAppointmentTimeDefined(booking)).toBe(false);
  });
});
