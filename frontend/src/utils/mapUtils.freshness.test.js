import {
  getFreshnessStatus,
  getDriverFreshnessLabel,
  isDeviceHealthSignalActive,
  isDevicePipelineAlive,
  isDeviceHeartbeatFresh,
} from './mapUtils';

describe('getFreshnessStatus', () => {
  const nowMs = Date.parse('2026-08-23T22:00:00.000Z');

  it('PRESENCE : 150 s + pipeline vivant → recent (pas offline)', () => {
    const driver = {
      location_mode: 'availability_presence',
      location_status: 'live',
      recorded_at: new Date(nowMs - 150_000).toISOString(),
      device_health: {
        tracking_active: true,
        last_heartbeat_at: new Date(nowMs - 30_000).toISOString(),
      },
    };
    expect(getFreshnessStatus(driver, nowMs)).toBe('recent');
  });

  it('uses backend location_status when pipeline offline and très vieux', () => {
    const driver = {
      location_status: 'offline',
      last_seen_seconds: 901,
      device_health: {
        last_heartbeat_at: new Date(nowMs - 10 * 60_000).toISOString(),
      },
    };
    expect(getFreshnessStatus(driver, nowMs)).toBe('offline_unknown');
  });

  it('mission_live : seuils plus stricts sans backend', () => {
    expect(getFreshnessStatus({ location_mode: 'mission_live', last_seen_seconds: 40 }, nowMs)).toBe(
      'live'
    );
    expect(getFreshnessStatus({ location_mode: 'mission_live', last_seen_seconds: 90 }, nowMs)).toBe(
      'recent'
    );
    expect(getFreshnessStatus({ location_mode: 'mission_live', last_seen_seconds: 150 }, nowMs)).toBe(
      'stale'
    );
  });
});

describe('C2 device-health / preuve de vie', () => {
  const nowMs = Date.parse('2026-08-17T10:00:00.000Z');

  it('last_fix_age seul ne prouve pas que le pipeline est vivant', () => {
    const driver = {
      device_health: {
        last_fix_age_seconds: 15,
        tracking_active: false,
      },
    };
    expect(isDeviceHealthSignalActive(driver, nowMs)).toBe(false);
    expect(isDevicePipelineAlive(driver, nowMs)).toBe(false);
    expect(isDeviceHeartbeatFresh(driver, nowMs)).toBe(false);
  });

  it('tracking_active ancien sans heartbeat frais ne ressuscite pas le pipeline', () => {
    const driver = {
      device_health: {
        tracking_active: true,
        last_heartbeat_at: new Date(nowMs - 10 * 60_000).toISOString(),
      },
    };
    expect(isDeviceHeartbeatFresh(driver, nowMs)).toBe(false);
    expect(isDevicePipelineAlive(driver, nowMs)).toBe(false);
    expect(isDeviceHealthSignalActive(driver, nowMs)).toBe(false);
  });

  it('heartbeat frais + tracking_active → pipeline vivant', () => {
    const driver = {
      device_health: {
        tracking_active: true,
        last_heartbeat_at: new Date(nowMs - 30_000).toISOString(),
      },
    };
    expect(isDeviceHeartbeatFresh(driver, nowMs)).toBe(true);
    expect(isDevicePipelineAlive(driver, nowMs)).toBe(true);
  });

  it('is_available=false → libellé Hors service, jamais GPS hors ligne', () => {
    expect(
      getDriverFreshnessLabel({
        is_available: false,
        status: 'off_duty',
        location_status: 'offline',
        last_seen_seconds: 900,
      })
    ).toBe('Hors service');
  });

  it('PRESENCE récente → Position mise à jour (pas Offline)', () => {
    const label = getDriverFreshnessLabel({
      is_available: true,
      status: 'available',
      location_mode: 'availability_presence',
      recorded_at: new Date(nowMs - 120_000).toISOString(),
      device_health: {
        tracking_active: true,
        last_heartbeat_at: new Date(nowMs - 30_000).toISOString(),
      },
    }, nowMs);
    expect(label).toContain('Position mise à jour');
    expect(label).not.toMatch(/Offline|hors ligne/i);
  });
});
