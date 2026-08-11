import {
  BUSINESS_TZ,
  formatBusinessAbsoluteDayTime,
  formatBusinessClockTime,
  getBusinessCalendarDate,
} from '../businessTime';
import { formatAbsolutePositionTime } from '../mapUtils';

describe('businessTime / TIME-2', () => {
  it('exporte Europe/Zurich', () => {
    expect(BUSINESS_TZ).toBe('Europe/Zurich');
  });

  it('18:00Z en août → 20:00 Genève', () => {
    expect(formatBusinessClockTime('2026-08-11T18:00:00.000Z')).toBe('20:00');
  });

  it('18:00Z en hiver → 19:00 Genève', () => {
    expect(formatBusinessClockTime('2026-01-11T18:00:00.000Z')).toBe('19:00');
  });

  it('23:30Z le 11 août → 01:30 le 12 août Genève', () => {
    const iso = '2026-08-11T23:30:00.000Z';
    expect(formatBusinessClockTime(iso)).toBe('01:30');
    expect(getBusinessCalendarDate(iso)).toBe('2026-08-12');
  });

  it("aujourd'hui / hier selon calendrier Zurich, pas navigateur", () => {
    // 11 août 22:00Z = 12 août 00:00 Genève ; « maintenant » = 12 août 10:00 Genève (08:00Z)
    const recorded = '2026-08-11T22:00:00.000Z';
    const now = new Date('2026-08-12T08:00:00.000Z');
    expect(formatBusinessAbsoluteDayTime(recorded, now)).toBe("aujourd'hui à 00:00");

    const yesterdayRecorded = '2026-08-11T10:00:00.000Z'; // 12:00 le 11 août Genève
    expect(formatBusinessAbsoluteDayTime(yesterdayRecorded, now)).toBe('hier à 12:00');
  });

  it('formatAbsolutePositionTime n’affiche pas de suffixe TZ', () => {
    const label = formatAbsolutePositionTime(
      '2026-08-11T18:00:00.000Z',
      new Date('2026-08-11T19:00:00.000Z'),
    );
    expect(label).toBe("aujourd'hui à 20:00");
    expect(label).not.toMatch(/UTC|CEST|GMT/i);
  });
});
