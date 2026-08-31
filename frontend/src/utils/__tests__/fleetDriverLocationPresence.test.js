import {
  formatDriverLocationPresenceLabel,
  resolveDriverLocationPresence,
} from '../fleetDriverLocationPresence';
import {
  applyLocalLocationFreshness,
  LOCAL_LIVE_MAX_SECONDS,
} from '../localDriverLocationFreshness';
import { isNonLiveGpsPosition } from '../companyDriverProjections';

function recordedAt(ageSeconds, nowMs) {
  return new Date(nowMs - ageSeconds * 1000).toISOString();
}

describe('fleetDriverLocationPresence', () => {
  const now = Date.parse('2026-08-11T12:00:00.000Z');

  it('PRESENCE : 150 s reste recent (cycle ~2–3 min)', () => {
    const view = resolveDriverLocationPresence(
      {
        latitude: 46,
        longitude: 6,
        location_mode: 'availability_presence',
        location_status: 'live',
        recorded_at: recordedAt(150, now),
        device_health: {
          tracking_active: true,
          last_heartbeat_at: new Date(now - 30_000).toISOString(),
        },
      },
      now
    );
    expect(view.presence).toBe('recent');
    expect(view.countedAsLocated).toBe(true);
  });

  it('matrice live / recent / stale / non-promotion (mission_live)', () => {
    expect(
      resolveDriverLocationPresence(
        {
          latitude: 46,
          longitude: 6,
          location_mode: 'mission_live',
          location_status: 'live',
          recorded_at: recordedAt(10, now),
        },
        now
      )
    ).toMatchObject({ presence: 'live', countedAsLocated: true });

    expect(
      resolveDriverLocationPresence(
        {
          latitude: 46,
          longitude: 6,
          location_mode: 'mission_live',
          location_status: 'live',
          recorded_at: recordedAt(70, now),
        },
        now
      ).presence
    ).toBe('recent');

    expect(
      resolveDriverLocationPresence(
        {
          latitude: 46,
          longitude: 6,
          location_mode: 'mission_live',
          location_status: 'live',
          recorded_at: recordedAt(400, now),
        },
        now
      ).countedAsLocated
    ).toBe(false);

    expect(
      resolveDriverLocationPresence(
        {
          latitude: 46,
          longitude: 6,
          location_mode: 'mission_live',
          location_status: 'stale',
          recorded_at: recordedAt(10, now),
        },
        now
      ).presence
    ).toBe('stale');
  });

  it('fallbacks → last_known ; sans coords → offline_unknown', () => {
    expect(
      resolveDriverLocationPresence(
        { latitude: 46, longitude: 6, position_source: 'db_fallback', location_status: 'live' },
        now
      ).presence
    ).toBe('last_known');
    expect(
      resolveDriverLocationPresence({ location_status: 'offline' }, now)
    ).toMatchObject({ presence: 'offline_unknown', showMarker: false });
  });

  it('isNonLiveGpsPosition aligné sur countedAsLocated', () => {
    const nowMs = Date.now();
    const live = {
      latitude: 46,
      longitude: 6,
      location_mode: 'availability_presence',
      location_status: 'live',
      recorded_at: recordedAt(LOCAL_LIVE_MAX_SECONDS - 1, nowMs),
    };
    expect(isNonLiveGpsPosition(live)).toBe(false);
    expect(
      isNonLiveGpsPosition({
        ...live,
        location_status: 'stale',
        recorded_at: recordedAt(400, nowMs),
      })
    ).toBe(true);
    expect(isNonLiveGpsPosition(live, { isFallback: true })).toBe(true);
  });

  it('applyLocalLocationFreshness préserve tracking_display', () => {
    const aged = applyLocalLocationFreshness(
      {
        recorded_at: recordedAt(90, now),
        location_mode: 'availability_presence',
        location_status: 'live',
        tracking_display_status: 'degraded_constrained',
        position_source: 'db_fallback',
      },
      now
    );
    expect(aged.location_status).toBe('recent');
    expect(aged.tracking_display_status).toBe('degraded_constrained');
    expect(aged.position_source).toBe('db_fallback');
  });

  it('formatDriverLocationPresenceLabel', () => {
    expect(formatDriverLocationPresenceLabel({ presence: 'live', ageSeconds: 8 })).toContain(
      'En direct'
    );
    expect(formatDriverLocationPresenceLabel({ presence: 'recent', ageSeconds: 120 })).toContain(
      'Position mise à jour'
    );
  });
});
