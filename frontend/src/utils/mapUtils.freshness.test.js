import {
  getFreshnessStatus,
  getDriverFreshnessLabel,
  isDeviceHealthSignalActive,
  isDevicePipelineAlive,
  isDeviceHeartbeatFresh,
} from './mapUtils';

describe('getFreshnessStatus', () => {
  it('uses backend location_status when available', () => {
    expect(getFreshnessStatus({ location_status: 'live' })).toBe('live');
    expect(getFreshnessStatus({ location_status: 'recent' })).toBe('recent');
    expect(getFreshnessStatus({ location_status: 'stale' })).toBe('stale');
    expect(getFreshnessStatus({ location_status: 'offline' })).toBe('offline');
    expect(getFreshnessStatus({ location_status: 'last_known' })).toBe('last_known');
  });

  it('backend live ignore last_seen_seconds élevé (stale autoritatif backend)', () => {
    expect(getFreshnessStatus({ location_status: 'live', last_seen_seconds: 150 })).toBe('live');
  });

  it('falls back to last_seen_seconds thresholds', () => {
    expect(getFreshnessStatus({ last_seen_seconds: 10 })).toBe('live');
    expect(getFreshnessStatus({ last_seen_seconds: 70 })).toBe('recent');
    expect(getFreshnessStatus({ last_seen_seconds: 250 })).toBe('stale');
    expect(getFreshnessStatus({ last_seen_seconds: 901 })).toBe('offline');
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
});
