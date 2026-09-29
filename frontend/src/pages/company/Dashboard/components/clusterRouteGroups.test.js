import { clusterRouteGroups, journeyPlaces, otherJourneyScheduleLines } from './ReservationTable';

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

  it('enchaîne prise en charge, étapes et retour d’une même demande', () => {
    const legs = [
      {
        id: 1,
        route_group_id: 'g',
        route_sequence_number: 1,
        pickup_location: 'Avenue Ernest-Pictet 9',
        dropoff_location: 'Hôpitaux Universitaires de Genève',
      },
      {
        id: 2,
        route_group_id: 'g',
        route_sequence_number: 2,
        pickup_location: 'Hôpitaux Universitaires de Genève',
        dropoff_location: 'Clinique de Joli-Mont',
      },
      {
        id: 3,
        route_group_id: 'g',
        route_sequence_number: 3,
        is_return: true,
        pickup_location: 'Clinique de Joli-Mont',
        dropoff_location: 'Avenue Ernest-Pictet 9',
      },
    ];
    expect(journeyPlaces(legs[0], legs)).toEqual([
      'Avenue Ernest-Pictet 9',
      'Hôpitaux Universitaires de Genève',
      'Clinique de Joli-Mont',
      'Avenue Ernest-Pictet 9',
    ]);
  });

  it('ajoute le départ suivant sans inventer un retour sans heure', () => {
    const legs = [
      {
        id: 1,
        route_group_id: 'g',
        route_sequence_number: 1,
        time_confirmed: false,
        scheduling: { time_scheduled: true, display_time: '23:00 (non confirmé)' },
      },
      {
        id: 2,
        route_group_id: 'g',
        route_sequence_number: 2,
        time_confirmed: true,
        scheduling: { time_scheduled: true, display_time: '23:45' },
      },
      {
        id: 3,
        route_group_id: 'g',
        route_sequence_number: 3,
        is_return: true,
        time_confirmed: false,
        scheduling: { time_scheduled: false, display_time: 'À définir' },
      },
    ];
    expect(otherJourneyScheduleLines(legs[0], legs).map((line) => line.text)).toEqual([
      'Départ 23:45',
    ]);
  });
});
