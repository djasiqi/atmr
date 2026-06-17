import {
  DURATION_UNITS,
  displayValueToMinutes,
  formatDurationLabel,
  formatDurationRangeHint,
  minutesToDisplayValue,
  pickDefaultDurationUnit,
} from '../durationInput';

describe('durationInput', () => {
  it('choisit minutes jusqu’à 60 et heures au-delà', () => {
    expect(pickDefaultDurationUnit(5)).toBe(DURATION_UNITS.MINUTES);
    expect(pickDefaultDurationUnit(60)).toBe(DURATION_UNITS.MINUTES);
    expect(pickDefaultDurationUnit(61)).toBe(DURATION_UNITS.HOURS);
    expect(pickDefaultDurationUnit(10080)).toBe(DURATION_UNITS.HOURS);
  });

  it('convertit entre affichage et minutes', () => {
    expect(minutesToDisplayValue(10080, DURATION_UNITS.HOURS)).toBe(168);
    expect(displayValueToMinutes('168', DURATION_UNITS.HOURS)).toBe(10080);
    expect(displayValueToMinutes('5', DURATION_UNITS.MINUTES)).toBe(5);
    expect(displayValueToMinutes('1.5', DURATION_UNITS.HOURS)).toBe(90);
  });

  it('formate un libellé lisible', () => {
    expect(formatDurationLabel(5)).toBe('5 minutes');
    expect(formatDurationLabel(60)).toBe('1 heure');
    expect(formatDurationLabel(10080)).toBe('168 heures');
    expect(formatDurationLabel(90)).toBe('1 h 30 min');
  });

  it('décrit la plage selon l’unité', () => {
    expect(formatDurationRangeHint(1, 240, DURATION_UNITS.MINUTES)).toBe('1 à 240 min');
    expect(formatDurationRangeHint(1, 10080, DURATION_UNITS.HOURS)).toBe('1 minute à 168 h');
  });
});
