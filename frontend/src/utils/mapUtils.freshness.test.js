import { getFreshnessStatus } from './mapUtils';

describe('getFreshnessStatus', () => {
  it('uses backend location_status when available', () => {
    expect(getFreshnessStatus({ location_status: 'live' })).toBe('live');
    expect(getFreshnessStatus({ location_status: 'recent' })).toBe('recent');
    expect(getFreshnessStatus({ location_status: 'stale' })).toBe('stale');
    expect(getFreshnessStatus({ location_status: 'offline' })).toBe('offline');
  });

  it('falls back to last_seen_seconds thresholds', () => {
    expect(getFreshnessStatus({ last_seen_seconds: 10 })).toBe('live');
    expect(getFreshnessStatus({ last_seen_seconds: 70 })).toBe('recent');
    expect(getFreshnessStatus({ last_seen_seconds: 250 })).toBe('stale');
    expect(getFreshnessStatus({ last_seen_seconds: 901 })).toBe('offline');
  });
});
