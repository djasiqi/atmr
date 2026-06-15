import {
  extractWallClockTime,
  extractWallClockDate,
  combineMissionDateTimeNaive,
  formatWallClockDateTime,
  isNaiveMissionIso,
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
});
