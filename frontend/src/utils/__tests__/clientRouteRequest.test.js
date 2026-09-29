import { foldClientRouteRequests } from '../clientRouteRequest';

describe('foldClientRouteRequests', () => {
  it('regroupe une demande à trois transports en une seule ligne', () => {
    const rows = foldClientRouteRequests([
      {
        id: 46797,
        route_group_id: 'grp',
        route_sequence_number: 1,
        is_return: false,
        is_round_trip: true,
        amount: 25,
        time_confirmed: false,
        scheduled_time: '2026-09-29T21:00:00.000Z',
        pickup_location: 'Avenue Ernest-Pictet 9',
        dropoff_location: 'HUG',
        hospital_service: 'Radiologie',
        doctor_name: 'Non spécifié',
      },
      {
        id: 46798,
        route_group_id: 'grp',
        route_sequence_number: 2,
        is_return: false,
        amount: 0.5,
        time_confirmed: true,
        scheduled_time: '2026-09-29T21:45:00.000Z',
        pickup_location: 'HUG',
        dropoff_location: 'Clinique de Joli-Mont',
        hospital_service: '',
        doctor_name: 'Docteur Rashiti',
      },
      {
        id: 46799,
        route_group_id: 'grp',
        route_sequence_number: 3,
        is_return: true,
        parent_booking_id: 46798,
        amount: 25,
        time_confirmed: false,
        scheduled_time: null,
        pickup_location: 'Clinique de Joli-Mont',
        dropoff_location: 'Avenue Ernest-Pictet 9',
      },
    ]);

    expect(rows).toHaveLength(1);
    expect(rows[0].id).toBe(46797);
    expect(rows[0].route_request_transport_count).toBe(3);
    expect(rows[0].route_request_amount).toBe(135);
    expect(rows[0].route_request_ids).toEqual([46797, 46798, 46799]);
    expect(rows[0].route_request_stops.map((stop) => stop.label)).toEqual([
      'Prise en charge',
      'Étape 1',
      'Étape 2',
      'Retour',
    ]);
    expect(rows[0].route_request_stops[1].detail).toBe('Radiologie');
    expect(rows[0].route_request_stops[2].detail).toBe('Docteur Rashiti');
    expect(rows[0].route_request_stops[0].when).toBe('Prise en charge à déterminer');
    expect(rows[0].route_request_stops[1].when).toMatch(/Rendez-vous/);
    expect(rows[0].route_request_stops[2].when).toMatch(/Heure de départ/);
    expect(rows[0].route_request_stops[3].when).toMatch(/Heure à définir/);
  });

  it('affiche le tarif entreprise une fois la demande acceptée', () => {
    const rows = foldClientRouteRequests([
      {
        id: 46797,
        company_id: 1,
        route_group_id: 'grp',
        route_sequence_number: 1,
        is_return: false,
        amount: 40,
      },
      {
        id: 46798,
        company_id: 1,
        route_group_id: 'grp',
        route_sequence_number: 2,
        is_return: false,
        amount: 40,
      },
      {
        id: 46799,
        company_id: 1,
        route_group_id: 'grp',
        route_sequence_number: 3,
        is_return: true,
        amount: 40,
      },
    ]);

    expect(rows[0].route_request_amount).toBe(120);
  });

  it('laisse un aller-retour simple sur une seule carte aller', () => {
    const rows = foldClientRouteRequests([
      {
        id: 1,
        is_return: false,
        is_round_trip: true,
        amount: 40,
        pickup_location: 'A',
        dropoff_location: 'B',
        return_booking: { id: 2, amount: 40 },
      },
      {
        id: 2,
        is_return: true,
        amount: 40,
        pickup_location: 'B',
        dropoff_location: 'A',
      },
    ]);

    expect(rows.map((row) => row.id)).toEqual([1]);
    expect(rows[0].route_request_stops).toBeUndefined();
  });
});
