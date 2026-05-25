import {
  canonicalTimeMs,
  hasExploitableCoords,
  mergeDriverLiveUpdate,
  mergeOrUpdateDriverInList,
} from '../mergeDriverLiveUpdate';

describe('mergeDriverLiveUpdate', () => {
  it('priorise lon canonique quand lon et lng sont présents', () => {
    const driver = { id: 1, latitude: 46.2, longitude: 6.1 };
    const out = mergeDriverLiveUpdate(driver, { lat: 46.3, lon: 6.4, lng: 6.2 }, true);
    expect(out.latitude).toBe(46.3);
    expect(out.longitude).toBe(6.4);
  });

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

  it('ignore observabilité arrivée après un canon plus récent déjà fusionné (race)', () => {
    const driver = {
      id: 1,
      latitude: 46.2,
      longitude: 6.1,
      received_at: '2026-03-24T15:00:00.000Z',
    };
    const out = mergeDriverLiveUpdate(
      driver,
      {
        latitude: 46.0,
        longitude: 6.0,
        accept_status: 'accepted_observability_only',
        received_at: '2026-03-24T14:30:00.000Z',
      },
      false
    );
    expect(out.latitude).toBe(46.2);
    expect(out.received_at).toBe('2026-03-24T15:00:00.000Z');
  });

  it('applique toujours les coords d’un point accepted_canonical (pas de garde temporelle observabilité)', () => {
    const driver = {
      id: 1,
      latitude: 46.2,
      longitude: 6.1,
      received_at: '2026-03-24T14:00:00.000Z',
    };
    const out = mergeDriverLiveUpdate(
      driver,
      {
        latitude: 46.5,
        longitude: 6.5,
        accept_status: 'accepted_canonical',
        received_at: '2026-03-24T13:00:00.000Z',
      },
      false
    );
    expect(out.latitude).toBe(46.5);
    expect(out.longitude).toBe(6.5);
  });

  it('accepte le premier point observabilité si le chauffeur n’a pas encore de canonical_time', () => {
    const driver = { id: 1, latitude: 46.2, longitude: 6.1 };
    const out = mergeDriverLiveUpdate(
      driver,
      {
        latitude: 46.25,
        longitude: 6.11,
        accept_status: 'accepted_observability_only',
        received_at: '2026-03-24T14:00:00.000Z',
      },
      false
    );
    expect(out.latitude).toBe(46.25);
    expect(out.longitude).toBe(6.11);
  });

  it('met à jour location_status/presence_status depuis driver_location_update même si le snapshot était offline', () => {
    const driver = {
      id: 1,
      latitude: 46.2,
      longitude: 6.1,
      location_status: 'offline',
      presence_status: 'offline',
      last_seen_seconds: 999,
    };
    const out = mergeDriverLiveUpdate(
      driver,
      {
        latitude: 46.21,
        longitude: 6.11,
        location_status: 'live',
        presence_status: 'online',
        last_seen_seconds: 3,
      },
      false
    );
    expect(out.location_status).toBe('live');
    expect(out.presence_status).toBe('online');
    expect(out.last_seen_seconds).toBe(3);
  });
});

describe('canonicalTimeMs', () => {
  it('priorise recorded_at / timestamp / ts avant received_at (temps événementiel)', () => {
    expect(
      canonicalTimeMs({
        received_at: '2026-01-01T12:00:00.000Z',
        recorded_at: '2026-01-01T11:00:00.000Z',
        timestamp: '2026-01-01T10:00:00.000Z',
      })
    ).toBe(Date.parse('2026-01-01T11:00:00.000Z'));
    expect(
      canonicalTimeMs({
        timestamp: '2026-01-01T10:00:00.000Z',
        received_at: '2026-01-01T12:00:00.000Z',
      })
    ).toBe(Date.parse('2026-01-01T10:00:00.000Z'));
    expect(
      canonicalTimeMs({
        ts: '2026-01-01T09:30:00.000Z',
        received_at: '2026-01-01T12:00:00.000Z',
      })
    ).toBe(Date.parse('2026-01-01T09:30:00.000Z'));
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
