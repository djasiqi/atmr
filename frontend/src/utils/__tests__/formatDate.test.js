import { buildScheduleDisplay, renderBookingDateTime } from '../formatDate';

describe('renderBookingDateTime', () => {
  it('ignore un scheduling périmé après confirmation du départ', () => {
    const label = renderBookingDateTime({
      scheduled_time: '2026-09-12T14:15:00',
      time_confirmed: true,
      scheduling: {
        display_datetime: '12.09.2026 • 13:15 (non confirmé)',
        time_defined: false,
      },
    });
    expect(label).toBe('12.09.2026 • 14:15');
  });

  it('affiche (non confirmé) quand le départ n’est pas confirmé', () => {
    const label = renderBookingDateTime({
      scheduled_time: '2026-09-12T13:15:00',
      time_confirmed: false,
      scheduling: {
        display_datetime: '12.09.2026 • 13:15',
        time_defined: true,
      },
    });
    expect(label).toBe('12.09.2026 • 13:15 (non confirmé)');
  });

  it('n’expose pas une heure retour non confirmée', () => {
    const label = renderBookingDateTime({
      scheduled_time: '2026-09-12T12:15:00',
      time_confirmed: false,
      is_return: true,
      scheduling: {
        display_datetime: '12.09.2026 • 12:15',
        time_defined: true,
      },
    });
    expect(label).toBe('12.09.2026 • À confirmer');
    expect(label).not.toMatch(/12:15/);
  });
});

describe('buildScheduleDisplay', () => {
  it('construit le libellé confirmé', () => {
    expect(buildScheduleDisplay({
      scheduledTime: '2026-09-12T14:15:00',
      timeConfirmed: true,
    }).display_datetime).toBe('12.09.2026 • 14:15');
  });
});
