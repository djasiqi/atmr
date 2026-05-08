import {
  canonicalRealtimeTimeMs,
  shouldAcceptRealtimeEvent,
} from '../realtimeEventGuard';

describe('canonicalRealtimeTimeMs', () => {
  it('priorise recorded_at avant received_at', () => {
    const t = Date.parse('2026-01-15T12:00:00.000Z');
    const receivedOlder = Date.parse('2026-01-15T11:00:00.000Z');
    expect(
      canonicalRealtimeTimeMs({
        recorded_at: new Date(t).toISOString(),
        received_at: new Date(receivedOlder).toISOString(),
      })
    ).toBe(t);
  });

  it('retombe sur received_at si pas d’autre champ', () => {
    const r = Date.parse('2026-01-15T10:00:00.000Z');
    expect(
      canonicalRealtimeTimeMs({
        received_at: new Date(r).toISOString(),
      })
    ).toBe(r);
  });
});

describe('shouldAcceptRealtimeEvent', () => {
  it('accepte un événement plus récent que le précédent pour la même entité', () => {
    const base = Date.parse('2026-01-15T12:00:00.000Z');
    expect(
      shouldAcceptRealtimeEvent({
        entityKey: 'driver:1',
        canonicalTimeMs: base,
      })
    ).toBe(true);
    expect(
      shouldAcceptRealtimeEvent({
        entityKey: 'driver:1',
        canonicalTimeMs: base + 1000,
      })
    ).toBe(true);
  });

  it('rejette si canonicalTimeMs est strictement inférieur au précédent', () => {
    const t1 = Date.parse('2026-01-15T12:00:00.000Z');
    const t2 = Date.parse('2026-01-15T11:00:00.000Z');
    expect(
      shouldAcceptRealtimeEvent({
        entityKey: 'driver:42',
        canonicalTimeMs: t1,
      })
    ).toBe(true);
    expect(
      shouldAcceptRealtimeEvent({
        entityKey: 'driver:42',
        canonicalTimeMs: t2,
      })
    ).toBe(false);
  });
});
