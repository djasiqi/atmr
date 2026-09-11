import {
  normalizeHhmm,
  resolveAppointmentShift,
  formatAppointmentShiftLead,
} from '../institutionAppointmentShift';

describe('normalizeHhmm', () => {
  it('normalise HH:MM et ISO naïf', () => {
    expect(normalizeHhmm('13:00')).toBe('13:00');
    expect(normalizeHhmm('9:15')).toBe('09:15');
    expect(normalizeHhmm('2026-09-12T15:00:00')).toBe('15:00');
    expect(normalizeHhmm('')).toBe('');
    expect(normalizeHhmm('25:00')).toBe('');
  });
});

describe('resolveAppointmentShift', () => {
  it('privilégie les query params de la notification', () => {
    const shift = resolveAppointmentShift({
      reservation: { institution_leg: { appointment_time: '2026-09-12T16:00:00' } },
      events: [{
        created_at: '2026-09-10T10:00:00',
        after_snapshot: { appointment_before: '12:00', appointment_after: '14:00' },
      }],
      searchParams: { appt_from: '13:00', appt_to: '15:00' },
    });
    expect(shift).toEqual({ before: '13:00', after: '15:00' });
  });

  it('lit le dernier change-event si la query est absente', () => {
    const shift = resolveAppointmentShift({
      reservation: { institution_leg: { appointment_time: '2026-09-12T15:00:00' } },
      events: [
        {
          created_at: '2026-09-10T08:00:00',
          after_snapshot: { appointment_before: '14:00', appointment_after: '13:00' },
        },
        {
          created_at: '2026-09-10T12:00:00',
          after_snapshot: { appointment_before: '13:00', appointment_after: '15:00' },
        },
      ],
    });
    expect(shift).toEqual({ before: '13:00', after: '15:00' });
  });

  it('retombe sur le RDV actuel du booking', () => {
    const shift = resolveAppointmentShift({
      reservation: { institution_leg: { appointment_time: '2026-09-12T15:00:00' } },
    });
    expect(shift).toEqual({ before: '', after: '15:00' });
  });
});

describe('formatAppointmentShiftLead', () => {
  it('énonce clairement le décalage before → after', () => {
    expect(formatAppointmentShiftLead({ before: '13:00', after: '15:00' }))
      .toBe('Le rendez-vous a été décalé de 13:00 à 15:00.');
  });

  it('retombe sur l’heure actuelle si l’ancien RDV est inconnu', () => {
    expect(formatAppointmentShiftLead({ before: '', after: '15:00' }))
      .toBe('Le rendez-vous a été décalé à 15:00.');
  });
});
