import {
  extractWallClockTime,
  extractWallClockDate,
  combineMissionDateTimeNaive,
  formatWallClockDateTime,
  isNaiveMissionIso,
  getGenevaTodayDateStr,
  minutesSinceMissionWallClock,
} from '../missionTimeDisplay';

describe('missionTimeDisplay', () => {
  it('conserve l\'heure murale sur ISO naïf', () => {
    expect(extractWallClockTime('2026-06-15T18:00:00')).toBe('18:00');
    expect(extractWallClockDate('2026-06-15T18:00:00')).toBe('2026-06-15');
    expect(isNaiveMissionIso('2026-06-15T18:00:00')).toBe(true);
  });

  it('convertit UTC Z vers heure Genève', () => {
    expect(extractWallClockTime('2026-06-15T16:00:00Z')).toBe('18:00');
  });

  it('combineMissionDateTimeNaive sans fuseau', () => {
    expect(combineMissionDateTimeNaive('2026-06-15', '18:00')).toBe('2026-06-15T18:00:00');
  });

  it('formatWallClockDateTime', () => {
    expect(formatWallClockDateTime('2026-06-15T18:00:00')).toEqual({
      date: '15.06.2026',
      time: '18:00',
    });
  });

  it('getGenevaTodayDateStr retourne YYYY-MM-DD', () => {
    const today = getGenevaTodayDateStr();
    expect(today).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it('minutesSinceMissionWallClock sur ISO naïf indépendant du fuseau navigateur', () => {
    const nowMs = Date.UTC(2026, 5, 16, 10, 45, 0);
    expect(minutesSinceMissionWallClock('2026-06-16T12:30:00', nowMs)).toBe(15);
    expect(minutesSinceMissionWallClock('2026-06-16T10:30:00', nowMs)).toBe(135);
  });

  it('minutesSinceMissionWallClock sur UTC Z normalise en Genève', () => {
    const nowMs = Date.UTC(2026, 5, 16, 10, 45, 0);
    expect(minutesSinceMissionWallClock('2026-06-16T08:30:00Z', nowMs)).toBe(135);
  });
});
