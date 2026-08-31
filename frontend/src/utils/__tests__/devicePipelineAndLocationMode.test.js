import {
  isDevicePipelineAlive,
  isDeviceHeartbeatFresh,
} from '../devicePipelineUtils';
import {
  resolveDriverLocationMode,
  ageToGpsFreshness,
  normalizeGpsFreshnessMode,
} from '../gpsFreshnessContract';

describe('devicePipelineUtils', () => {
  const nowMs = Date.parse('2026-08-23T22:00:00.000Z');

  it('parse heartbeat epoch ms (Redis prod)', () => {
    const driver = {
      device_health: {
        platform: 'ios',
        tracking_active: true,
        last_heartbeat_at: String(nowMs - 30_000),
      },
    };
    expect(isDeviceHeartbeatFresh(driver, nowMs)).toBe(true);
    expect(isDevicePipelineAlive(driver, nowMs)).toBe(true);
  });

  it('iOS : fgs_running=0 ne tue pas le pipeline si tracking actif', () => {
    const driver = {
      device_health: {
        platform: 'ios',
        tracking_active: true,
        fgs_running: false,
        native_task_running: false,
        last_heartbeat_at: new Date(nowMs - 30_000).toISOString(),
      },
    };
    expect(isDevicePipelineAlive(driver, nowMs)).toBe(true);
  });

  it('iOS : fgs_running absent ne tue pas le pipeline si native_task_running', () => {
    const driver = {
      device_health: {
        platform: 'ios',
        native_task_running: true,
        last_heartbeat_at: new Date(nowMs - 30_000).toISOString(),
      },
    };
    expect(isDevicePipelineAlive(driver, nowMs)).toBe(true);
  });

  it('Android : fgs_running=true compte pour le pipeline', () => {
    const driver = {
      device_health: {
        platform: 'android',
        fgs_running: true,
        last_heartbeat_at: new Date(nowMs - 30_000).toISOString(),
      },
    };
    expect(isDevicePipelineAlive(driver, nowMs)).toBe(true);
  });
});

describe('resolveDriverLocationMode', () => {
  it('priorité SoT serveur : location_mode explicite', () => {
    expect(
      resolveDriverLocationMode({
        location_mode: 'availability_presence',
        status: 'busy',
        current_booking_id: 99,
      })
    ).toBe('availability_presence');
  });

  it('fallback métier mission_live si mode absent', () => {
    expect(
      resolveDriverLocationMode({ status: 'assigned', current_booking_id: 12 })
    ).toBe('mission_live');
  });

  it('disponible sans mission → PRESENCE', () => {
    expect(resolveDriverLocationMode({ status: 'available' })).toBe(
      'availability_presence'
    );
  });

  it('point ancien ne change pas le mode : seuils PRESENCE sur fix 150 s', () => {
    const mode = resolveDriverLocationMode({
      status: 'available',
      location_mode: 'availability_presence',
      last_seen_seconds: 150,
    });
    expect(normalizeGpsFreshnessMode(mode)).toBe('presence');
    expect(ageToGpsFreshness(150, mode)).toBe('recent');
  });

  it('transition mission → retour PRESENCE via SoT', () => {
    const inMission = resolveDriverLocationMode({
      location_mode: 'mission_live',
      status: 'busy',
    });
    expect(ageToGpsFreshness(90, inMission)).toBe('recent');

    const backPresence = resolveDriverLocationMode({
      location_mode: 'availability_presence',
      status: 'available',
    });
    expect(ageToGpsFreshness(150, backPresence)).toBe('recent');
    expect(ageToGpsFreshness(90, inMission)).toBe('recent');
    expect(ageToGpsFreshness(90, backPresence)).toBe('recent');
  });
});
