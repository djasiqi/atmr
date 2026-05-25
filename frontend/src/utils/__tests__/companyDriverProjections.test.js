import {
  buildDriverStructuralSetKey,
  isSameMarkerPosition,
  projectDriverForMap,
  projectDriversForMap,
} from '../../utils/companyDriverProjections';

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

  it('projectDriverForMap conserve id/coords/statut', () => {
    const projected = projectDriverForMap({
      id: 5,
      lat: 46.5,
      lng: 6.4,
      status: 'busy',
      notes: 'secret',
    });
    expect(projected).toMatchObject({
      id: 5,
      latitude: 46.5,
      longitude: 6.4,
      status: 'busy',
    });
    expect(projected.notes).toBeUndefined();
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
