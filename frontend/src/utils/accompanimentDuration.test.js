import {
  parseAccompanimentDurationToHours,
  computeAccompanimentLineTotal,
} from './accompanimentDuration';

describe('parseAccompanimentDurationToHours', () => {
  it('décimales = heures', () => {
    expect(parseAccompanimentDurationToHours('2')).toBe(2);
    expect(parseAccompanimentDurationToHours('0.5')).toBe(0.5);
    expect(parseAccompanimentDurationToHours('1,5')).toBe(1.5);
  });
  it('1h, 1h30, 1:30', () => {
    expect(parseAccompanimentDurationToHours('1h')).toBe(1);
    expect(parseAccompanimentDurationToHours('1H30')).toBe(1.5);
    expect(parseAccompanimentDurationToHours('1:30')).toBe(1.5);
  });
  it('minutes', () => {
    expect(parseAccompanimentDurationToHours('30 min')).toBe(0.5);
    expect(parseAccompanimentDurationToHours('90min')).toBe(1.5);
  });
});

describe('computeAccompanimentLineTotal', () => {
  it('45 CHF/h × 0,5 h', () => {
    expect(computeAccompanimentLineTotal(45, '30min')).toBeCloseTo(22.5, 5);
  });
  it('45 CHF/h × 2 h', () => {
    expect(computeAccompanimentLineTotal(45, '2')).toBe(90);
  });
});
