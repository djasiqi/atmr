import {
  buildDriverStructuralSetKey,
  isSameMarkerPosition,
  projectDriverForMap,
  projectDriversForMap,
  isDriverConstrained,
  resolveDriverMapVisualStatus,
  resolveDriverMapProjection,
  resolveDriverMapMarkerColor,
  isNonLiveGpsPosition,
  CONSTRAINED_MARKER_COLOR,
} from '../../utils/companyDriverProjections';
import { STATUS_COLORS } from '../../utils/mapUtils';

describe('companyDriverProjections', () => {
  it('buildDriverStructuralSetKey ignore lat/lng et reflète ids + filtre', () => {
    const driversA = [
      { id: 2, latitude: 46.2, longitude: 6.1 },
      { id: 1, latitude: 46.3, longitude: 6.2 },
    ];
    const driversB = [
      { id: 1, latitude: 46.9, longitude: 6.9 },
      { id: 2, latitude: 46.0, longitude: 6.0 },
    ];
    expect(buildDriverStructuralSetKey(driversA)).toBe(buildDriverStructuralSetKey(driversB));
    expect(buildDriverStructuralSetKey(driversA, 'bob')).not.toBe(buildDriverStructuralSetKey(driversA));
  });

  it('projectDriverForMap conserve id/coords/statut et champs GPS', () => {
    const projected = projectDriverForMap({
      id: 5,
      lat: 46.5,
      lng: 6.4,
      status: 'busy',
      notes: 'secret',
      location_status: 'last_known',
      position_source: 'db_fallback',
      recorded_at: '2026-07-29T09:47:47Z',
      accuracy: 80,
      speed: 4.5,
      heading: 120,
      location_mode: 'mission_live',
    });
    expect(projected).toMatchObject({
      id: 5,
      latitude: 46.5,
      longitude: 6.4,
      status: 'busy',
      location_status: 'last_known',
      position_source: 'db_fallback',
      recorded_at: '2026-07-29T09:47:47Z',
      businessStatus: 'busy',
      visualTreatment: 'gps_stale',
      accuracy: 80,
      speed: 4.5,
      heading: 120,
      location_mode: 'mission_live',
    });
    expect(projected.notes).toBeUndefined();
    expect(projected.lastPositionAt).toBe('2026-07-29T09:47:47Z');
  });

  it('isSameMarkerPosition tolère micro-variations', () => {
    expect(isSameMarkerPosition({ lat: 46.2, lng: 6.1 }, { lat: 46.200001, lng: 6.100001 })).toBe(true);
    expect(isSameMarkerPosition({ lat: 46.2, lng: 6.1 }, { lat: 46.21, lng: 6.1 })).toBe(false);
  });

  it('projectDriversForMap filtre les entrées invalides', () => {
    expect(projectDriversForMap([{ id: 1, latitude: 1, longitude: 2 }, null, {}])).toEqual([
      expect.objectContaining({ id: 1 }),
    ]);
  });
});

describe('DriverLiveMap anti-régression lat/lng only', () => {
  it('changement lat/lng seul ne modifie pas la clé structurelle', () => {
    const before = [{ id: 10, latitude: 46.1, longitude: 6.1 }];
    const after = [{ id: 10, latitude: 46.2, longitude: 6.2 }];
    expect(buildDriverStructuralSetKey(before)).toBe(buildDriverStructuralSetKey(after));
  });

  it('ajout chauffeur modifie la clé structurelle', () => {
    const before = [{ id: 10, latitude: 46.1, longitude: 6.1 }];
    const after = [
      { id: 10, latitude: 46.1, longitude: 6.1 },
      { id: 11, latitude: 46.2, longitude: 6.2 },
    ];
    expect(buildDriverStructuralSetKey(before)).not.toBe(buildDriverStructuralSetKey(after));
  });
});

describe('découplage métier / GPS', () => {
  it('busy + last_known + db_fallback → jamais couleur live', () => {
    const driver = {
      status: 'busy',
      location_status: 'last_known',
      position_source: 'db_fallback',
      recorded_at: '2026-07-29T09:47:47Z',
    };
    const projection = resolveDriverMapProjection(driver);
    expect(projection.businessStatus).toBe('busy');
    expect(projection.gpsFreshness).toBe('last_known');
    expect(projection.positionSource).toBe('db_fallback');
    expect(projection.visualTreatment).toBe('gps_stale');
    expect(projection.visualStatus).toBe('offline');
    expect(resolveDriverMapVisualStatus(driver)).toBe('offline');
    expect(isNonLiveGpsPosition(driver)).toBe(true);
  });

  it('busy + offline → GPS hors ligne', () => {
    const driver = {
      status: 'busy',
      location_status: 'offline',
      tracking_display_status: 'offline_unknown',
    };
    const projection = resolveDriverMapProjection(driver);
    expect(projection.businessStatus).toBe('busy');
    expect(projection.visualTreatment).toBe('gps_offline');
    expect(projection.visualStatus).toBe('offline');
  });

  it('assigned + stale → signal ancien', () => {
    const driver = {
      status: 'assigned',
      location_status: 'stale',
    };
    const projection = resolveDriverMapProjection(driver);
    expect(projection.businessStatus).toBe('assigned');
    expect(projection.visualTreatment).toBe('gps_stale');
    expect(projection.visualStatus).toBe('offline');
  });

  it('constrained + last_known → non-live dominant (pas orange actif)', () => {
    const driver = {
      location_status: 'last_known',
      presence_status: 'degraded_constrained',
      status: 'assigned',
    };
    expect(resolveDriverMapVisualStatus(driver)).toBe('offline');
    expect(resolveDriverMapProjection(driver).visualTreatment).toBe('gps_stale_constrained');
  });

  it('constrained + live → orange contrainte', () => {
    const driver = {
      location_status: 'live',
      presence_status: 'degraded_constrained',
      status: 'assigned',
    };
    expect(resolveDriverMapVisualStatus(driver)).toBe('constrained');
  });
});

describe('degraded_constrained / batterie restreinte', () => {
  it('isDriverConstrained détecte presence_status degraded_constrained', () => {
    expect(isDriverConstrained({ presence_status: 'degraded_constrained' })).toBe(true);
    expect(isDriverConstrained({ presence_status: 'online' })).toBe(false);
    expect(isDriverConstrained({})).toBe(false);
    expect(isDriverConstrained(null)).toBe(false);
  });

  it('isDriverConstrained détecte status assigned_constrained / available_constrained', () => {
    expect(isDriverConstrained({ status: 'assigned_constrained' })).toBe(true);
    expect(isDriverConstrained({ status: 'available_constrained' })).toBe(true);
    expect(isDriverConstrained({ status: 'assigned' })).toBe(false);
  });

  it('projectDriverForMap propage presence_status et device_health', () => {
    const projected = projectDriverForMap({
      id: 7,
      lat: 46.2,
      lng: 6.1,
      status: 'assigned_constrained',
      presence_status: 'degraded_constrained',
      location_status: 'live',
      device_health: { constraint_reason: 'battery_optimized', battery_optimized: true },
    });
    expect(projected).toMatchObject({
      id: 7,
      presence_status: 'degraded_constrained',
      device_health: { constraint_reason: 'battery_optimized', battery_optimized: true },
    });
  });

  it('resolveDriverMapVisualStatus renvoie constrained pour un chauffeur restreint frais', () => {
    const driver = {
      status: 'assigned',
      presence_status: 'degraded_constrained',
      location_status: 'live',
    };
    expect(resolveDriverMapVisualStatus(driver)).toBe('constrained');
    expect(resolveDriverMapVisualStatus(driver, { isFallback: true })).toBe('offline');
  });

  it('resolveDriverMapMarkerColor mappe constrained vers orange (#f97316)', () => {
    const colors = { ...STATUS_COLORS, available: '#4ade80' };
    expect(resolveDriverMapMarkerColor('constrained', colors)).toBe(CONSTRAINED_MARKER_COLOR);
    expect(resolveDriverMapMarkerColor('constrained', colors)).toBe('#f97316');
    expect(resolveDriverMapMarkerColor('assigned', colors)).toBe(STATUS_COLORS.assigned);
    expect(resolveDriverMapMarkerColor('offline', colors)).toBe(STATUS_COLORS.offline);
  });
});
