import { periodRange, shiftPeriod } from './periodRange';

describe('periodRange', () => {
  const now = new Date('2026-09-30T22:30:00Z'); // 01.10 00:30 à Zurich (UTC+2)

  it('utilise le jour civil de Zurich, pas le jour UTC', () => {
    expect(periodRange('today', '', '', now)).toEqual({
      from: '2026-10-01',
      to: '2026-10-01',
    });
  });

  it('accepte un seul jour personnalisé', () => {
    expect(periodRange('custom', '2026-09-12', '2026-09-12', now)).toEqual({
      from: '2026-09-12',
      to: '2026-09-12',
    });
  });

  it('couvre le mois civil', () => {
    expect(periodRange('month', '', '', new Date('2026-09-15T10:00:00Z'))).toEqual({
      from: '2026-09-01',
      to: '2026-09-30',
    });
  });

  it('décale le mois civil', () => {
    expect(shiftPeriod('month', { from: '2026-09-01', to: '2026-09-30' }, 1)).toEqual({
      from: '2026-10-01',
      to: '2026-10-31',
      anchor: '2026-10-01',
    });
  });
});
