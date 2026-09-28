import { clusterRouteGroups } from './ReservationTable';

describe('clusterRouteGroups', () => {
  it('ordonne les segments et garde une course simple sans groupe', () => {
    const simple = { id: 1, route_group_id: null, pickup_location: 'A', dropoff_location: 'B' };
    const returnLeg = {
      id: 4,
      route_group_id: 'grp',
      route_sequence_number: 3,
      is_return: true,
      dropoff_location: 'Rue du Test 1',
    };
    const second = {
      id: 3,
      route_group_id: 'grp',
      route_sequence_number: 2,
      dropoff_location: 'Clinique La Colline',
    };
    const first = {
      id: 2,
      route_group_id: 'grp',
      route_sequence_number: 1,
      dropoff_location: 'HUG',
    };

    const ordered = clusterRouteGroups([simple, returnLeg, second, first]);

    expect(ordered.map((row) => row.id)).toEqual([1, 2, 3, 4]);
    expect(ordered[3].dropoff_location).toBe('Rue du Test 1');
    expect(ordered[3].is_return).toBe(true);
  });
});
