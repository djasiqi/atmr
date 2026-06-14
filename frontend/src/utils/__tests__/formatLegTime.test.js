import {
  formatLegTime,
  formatReturnTimeHint,
  formatReturnTimeLabel,
  formatDepartureTime,
  getNextConfirmedLegTime,
  formatLegScheduleSummary,
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
    expect(summary).toMatch(/Retour à définir/);
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
