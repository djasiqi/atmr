import {
  formatLegTime,
  formatReturnTimeHint,
  formatReturnTimeLabel,
  formatDepartureTime,
  formatRouteStopTime,
  getNextConfirmedLegTime,
  getNextConfirmedScheduleInfo,
  formatLegScheduleSummary,
  formatMissionScheduleListLabel,
  getMissionScheduleCardDisplay,
} from '../formatLegTime';

describe('formatLegTime', () => {
  it('affiche l\'heure si time_confirmed et scheduled_time présents', () => {
    const leg = {
      scheduled_time: '2026-06-11T08:00:00',
      time_confirmed: true,
    };
    expect(formatLegTime(leg)).toMatch(/08:00/);
  });

  it('affiche À définir si scheduled_time absent', () => {
    expect(formatLegTime({ scheduled_time: null, time_confirmed: false })).toBe('À définir');
  });

  it('affiche heure indicative si time_confirmed=false', () => {
    const leg = {
      scheduled_time: '2026-06-11T14:00:00',
      time_confirmed: false,
    };
    expect(formatLegTime(leg)).toMatch(/14:00/);
    expect(formatLegTime(leg)).toMatch(/non confirmé/);
  });

  it('minuit réel confirmé affiche 00:00, pas À définir', () => {
    const leg = {
      scheduled_time: '2026-06-11T00:00:00',
      time_confirmed: true,
    };
    expect(formatLegTime(leg)).toMatch(/00:00/);
    expect(formatLegTime(leg)).not.toBe('À définir');
  });
});

describe('formatDepartureTime', () => {
  it('départ confirmé', () => {
    expect(formatDepartureTime({
      scheduled_time: '2026-06-11T13:15:00',
      pickup_time_confirmed: true,
    })).toMatch(/13:15/);
  });

  it('départ depuis booking_summary après conversion', () => {
    expect(formatDepartureTime({
      pickup_time_confirmed: false,
      booking_summary: { scheduled_time: '2026-06-11T19:00:00' },
    })).toMatch(/19:00/);
  });

  it('priorise booking_summary si request.scheduled_time diverge', () => {
    expect(formatDepartureTime({
      pickup_time_confirmed: true,
      scheduled_time: '2026-06-17T12:00:00',
      booking_summary: { scheduled_time: '2026-06-17T10:00:00' },
    })).toMatch(/10:00/);
  });

  it('départ indicatif', () => {
    const label = formatDepartureTime({
      scheduled_time: '2026-06-11T13:15:00',
      pickup_time_confirmed: false,
    });
    expect(label).toMatch(/13:15/);
    expect(label).toMatch(/non confirmé/);
  });
});

describe('getNextConfirmedLegTime', () => {
  it('retourne la plus proche heure confirmée', () => {
    const next = getNextConfirmedLegTime({
      mission_date: '2026-06-12',
      scheduled_time: '2026-06-12T13:00:00',
      pickup_time_confirmed: true,
      legs: [
        { sequence_index: 0, scheduled_time: '2026-06-12T14:00:00', time_confirmed: true },
      ],
    });
    expect(next).toContain('2026-06-12');
  });

  it('exclut les heures indicatives', () => {
    expect(getNextConfirmedLegTime({
      mission_date: '2026-06-12',
      legs: [
        { sequence_index: 0, scheduled_time: '2026-06-12T14:00:00', time_confirmed: false },
      ],
    })).toBeNull();
  });
});

describe('formatLegScheduleSummary', () => {
  it('résume départ et retour à définir', () => {
    const summary = formatLegScheduleSummary({
      mission_date: '2026-06-12',
      scheduled_time: '2026-06-12T13:15:00',
      pickup_time_confirmed: true,
      return_to_institution: true,
      legs: [
        { sequence_index: 0, dropoff_location: 'HUG', scheduled_time: '2026-06-12T14:00:00', time_confirmed: true },
        { sequence_index: 1, dropoff_location: 'Clinique', scheduled_time: null, time_confirmed: false },
      ],
    });
    expect(summary).toMatch(/13:15 Départ/);
    expect(summary).toMatch(/14:00 RDV/);
    expect(summary).toMatch(/Retour à définir/);
  });
});

describe('formatMissionScheduleListLabel', () => {
  it('affiche date + RDV seul si pas de départ confirmé', () => {
    const label = formatMissionScheduleListLabel({
      mission_date: '2026-06-15',
      pickup_time_confirmed: false,
      legs: [
        {
          sequence_index: 0,
          scheduled_time: '2026-06-15T20:00:00',
          time_confirmed: true,
          dropoff_establishment: 'HUG',
        },
      ],
      return_to_institution: true,
    });
    expect(label).toMatch(/15/);
    expect(label).toMatch(/RDV 20:00/);
    expect(label).not.toMatch(/Départ/);
  });

  it('affiche départ et RDV quand les deux sont confirmés', () => {
    const label = formatMissionScheduleListLabel({
      mission_date: '2026-06-15',
      scheduled_time: '2026-06-15T19:00:00',
      pickup_time_confirmed: true,
      legs: [
        {
          sequence_index: 0,
          scheduled_time: '2026-06-15T20:00:00',
          time_confirmed: true,
        },
      ],
    });
    expect(label).toMatch(/Départ 19:00/);
    expect(label).toMatch(/RDV 20:00/);
  });

  it('affiche le départ depuis booking_summary si la request n\'a pas pickup_time_confirmed', () => {
    const label = formatMissionScheduleListLabel({
      mission_date: '2026-06-15',
      status: 'CONVERTED',
      pickup_time_confirmed: false,
      booking_summary: { scheduled_time: '2026-06-15T19:00:00' },
      legs: [
        {
          sequence_index: 0,
          scheduled_time: '2026-06-15T20:00:00',
          time_confirmed: true,
        },
      ],
    });
    expect(label).toMatch(/Départ 19:00/);
    expect(label).toMatch(/RDV 20:00/);
  });
});

describe('getMissionScheduleCardDisplay', () => {
  it('met le départ en primary et le RDV en secondary', () => {
    const display = getMissionScheduleCardDisplay({
      mission_date: '2026-06-15',
      pickup_time_confirmed: true,
      scheduled_time: '2026-06-15T19:00:00',
      legs: [
        { sequence_index: 0, scheduled_time: '2026-06-15T20:00:00', time_confirmed: true },
      ],
    });
    expect(display.primary).toEqual({ label: 'Départ', time: '19:00' });
    expect(display.secondary).toEqual([{ label: 'RDV', time: '20:00' }]);
  });
});

describe('getNextConfirmedScheduleInfo', () => {
  it('distingue départ et RDV', () => {
    const info = getNextConfirmedScheduleInfo({
      mission_date: '2026-06-15',
      scheduled_time: '2026-06-15T18:00:00Z',
      pickup_time_confirmed: false,
      legs: [
        { sequence_index: 0, scheduled_time: '2026-06-15T18:00:00', time_confirmed: true },
      ],
    });
    expect(info?.label).toBe('RDV');
    expect(info?.time).toMatch(/18:00/);
  });
});

describe('formatReturnTimeHint', () => {
  it('return_date seul → À définir', () => {
    expect(
      formatReturnTimeHint({
        return_date: '2026-06-11',
        return_time_confirmed: false,
      }),
    ).toBe('À définir');
  });

  it('return_time confirmé → heure affichée', () => {
    const hint = formatReturnTimeHint({
      return_time: '2026-06-11T14:30:00',
      return_time_confirmed: true,
    });
    expect(hint).toMatch(/14:30/);
  });
});

describe('formatReturnTimeLabel', () => {
  it('préfixe retour · À définir', () => {
    expect(
      formatReturnTimeLabel({
        return_date: '2026-06-11',
        return_time_confirmed: false,
      }),
    ).toBe('retour · À définir');
  });
});

describe('formatRouteStopTime', () => {
  it('préfixe Départ sur l\'étape départ', () => {
    expect(
      formatRouteStopTime({
        kind: 'start',
        request: {
          pickup_time_confirmed: true,
          scheduled_time: '2026-06-15T19:00:00',
        },
      }),
    ).toBe('Départ 19:00');
  });

  it('préfixe RDV sur une destination sans heure', () => {
    expect(
      formatRouteStopTime({
        kind: 'destination',
        request: {},
        leg: { scheduled_time: null, time_confirmed: false },
      }),
    ).toBe('RDV · À définir');
  });

  it('préfixe RDV sur une destination confirmée', () => {
    expect(
      formatRouteStopTime({
        kind: 'destination',
        request: {},
        leg: { scheduled_time: '2026-06-15T20:00:00', time_confirmed: true },
      }),
    ).toBe('RDV 20:00');
  });

  it('place le départ sur l\'étape départ, pas sur destination 1', () => {
    const request = {
      return_to_institution: true,
      pickup_time_confirmed: true,
      scheduled_time: '2026-06-15T19:00:00',
      legs: [
        {
          sequence_index: 0,
          pickup_location: 'Anières',
          dropoff_location: 'HUG',
          scheduled_time: '2026-06-15T20:00:00',
          time_confirmed: true,
        },
        {
          sequence_index: 1,
          dropoff_location: 'Vésenaz',
          scheduled_time: null,
          time_confirmed: false,
        },
        {
          sequence_index: 2,
          dropoff_location: 'Anières',
          scheduled_time: null,
          time_confirmed: false,
        },
      ],
    };
    expect(formatRouteStopTime({ kind: 'start', request })).toBe('Départ 19:00');
    expect(formatRouteStopTime({ kind: 'destination', request, leg: request.legs[0] })).toBe('RDV 20:00');
    expect(formatRouteStopTime({ kind: 'destination', request, leg: request.legs[1] })).toBe('RDV · À définir');
    expect(formatRouteStopTime({ kind: 'return', request, leg: request.legs[2] })).toBe('Départ · À définir');
  });
});
