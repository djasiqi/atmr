import {
  ageToGpsFreshness,
  formatGpsFreshnessLabel,
  GPS_FRESHNESS_THRESHOLDS,
  normalizeGpsFreshnessMode,
  resolveGpsDisplayStatus,
  shouldDisplayGpsOffline,
} from '../gpsFreshnessContract';

describe('gpsFreshnessContract', () => {
  const nowMs = Date.parse('2026-08-23T22:00:00.000Z');

  it('PRESENCE : 150 s → recent (pas stale/offline)', () => {
    expect(ageToGpsFreshness(150, 'availability_presence')).toBe('recent');
    expect(ageToGpsFreshness(150, 'availability_presence')).not.toBe('stale');
  });

  it('PRESENCE : seuils 60 / 240 / 600', () => {
    expect(ageToGpsFreshness(45, 'availability_presence')).toBe('live');
    expect(ageToGpsFreshness(90, 'availability_presence')).toBe('recent');
    expect(ageToGpsFreshness(300, 'availability_presence')).toBe('stale');
    expect(ageToGpsFreshness(700, 'availability_presence')).toBe('verify');
  });

  it('mission_live : seuils plus stricts', () => {
    expect(ageToGpsFreshness(40, 'mission_live')).toBe('live');
    expect(ageToGpsFreshness(90, 'mission_live')).toBe('recent');
    expect(ageToGpsFreshness(150, 'mission_live')).toBe('stale');
    expect(ageToGpsFreshness(400, 'mission_live')).toBe('verify');
  });

  it('pipeline vivant → jamais offline par âge seul (PRESENCE)', () => {
    const driver = {
      location_mode: 'availability_presence',
      location_status: 'offline',
      device_health: {
        tracking_active: true,
        last_heartbeat_at: new Date(nowMs - 30_000).toISOString(),
      },
    };
    expect(shouldDisplayGpsOffline(driver, 150, nowMs)).toBe(false);
    expect(resolveGpsDisplayStatus(driver, 150, nowMs)).toBe('recent');
  });

  it('pipeline mort + position très vieille → offline_unknown', () => {
    const driver = {
      location_mode: 'availability_presence',
      device_health: {
        tracking_active: true,
        last_heartbeat_at: new Date(nowMs - 10 * 60_000).toISOString(),
      },
    };
    expect(shouldDisplayGpsOffline(driver, 700, nowMs)).toBe(true);
    expect(resolveGpsDisplayStatus(driver, 700, nowMs)).toBe('offline_unknown');
  });

  it('libellés GPS séparés du statut chauffeur', () => {
    expect(formatGpsFreshnessLabel('recent', 120)).toContain('Position mise à jour');
    expect(formatGpsFreshnessLabel('recent', 120)).toContain('il y a 2 min');
    expect(formatGpsFreshnessLabel('verify', 600)).toContain('GPS à vérifier');
  });

  it('normalizeGpsFreshnessMode', () => {
    expect(normalizeGpsFreshnessMode('availability_presence')).toBe('presence');
    expect(normalizeGpsFreshnessMode('mission_live')).toBe('mission_live');
    expect(GPS_FRESHNESS_THRESHOLDS.presence.live).toBe(60);
  });
});
