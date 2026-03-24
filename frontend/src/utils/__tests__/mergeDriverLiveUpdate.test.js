import {
  hasExploitableCoords,
  mergeDriverLiveUpdate,
  mergeOrUpdateDriverInList,
} from '../mergeDriverLiveUpdate';

describe('mergeDriverLiveUpdate', () => {
  it('fusionne lng depuis live state', () => {
    const driver = { id: 1, latitude: 46.2, longitude: 6.1 };
    const out = mergeDriverLiveUpdate(driver, { lat: 46.3, lng: 6.2 }, true);
    expect(out.latitude).toBe(46.3);
    expect(out.longitude).toBe(6.2);
  });

  it('n’applique pas un point observabilité plus ancien que l’état courant', () => {
    const driver = {
      id: 1,
      latitude: 46.2,
      longitude: 6.1,
      received_at: '2026-03-24T14:00:00.000Z',
    };
    const out = mergeDriverLiveUpdate(
      driver,
      {
        latitude: 46.0,
        longitude: 6.0,
        accept_status: 'accepted_observability_only',
        received_at: '2026-03-24T13:00:00.000Z',
      },
      false
    );
    expect(out.latitude).toBe(46.2);
    expect(out.longitude).toBe(6.1);
  });
});

describe('hasExploitableCoords', () => {
  it('accepte lat/lng', () => {
    expect(hasExploitableCoords({ lat: 46, lng: 6 })).toBe(true);
  });
  it('rejette payload vide', () => {
    expect(hasExploitableCoords({ status: 'busy' })).toBe(false);
  });
});

describe('mergeOrUpdateDriverInList', () => {
  it('met à jour un chauffeur existant', () => {
    const prev = [{ id: 5, first_name: 'A', latitude: 1, longitude: 2 }];
    const next = mergeOrUpdateDriverInList(
      prev,
      { driver_id: 5, latitude: 46, longitude: 6 },
      false,
      10
    );
    expect(next).toHaveLength(1);
    expect(next[0].latitude).toBe(46);
    expect(next[0].longitude).toBe(6);
  });

  it('ajoute un chauffeur absent si coords exploitables', () => {
    const prev = [];
    const next = mergeOrUpdateDriverInList(
      prev,
      {
        driver_id: 99,
        first_name: 'X',
        company_id: 1,
        latitude: 46.2,
        longitude: 6.14,
      },
      false,
      1
    );
    expect(next).toHaveLength(1);
    expect(next[0].id).toBe(99);
    expect(next[0].latitude).toBe(46.2);
  });

  it('n’ajoute pas sans coords', () => {
    const prev = [];
    const next = mergeOrUpdateDriverInList(prev, { driver_id: 99, status: 'busy' }, true, 1);
    expect(next).toHaveLength(0);
  });
});
