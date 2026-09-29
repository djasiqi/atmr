import {
  journeyPlaces,
  pickupNeedsCompanyConfirmation,
  routeTimelinePoints,
} from '../routeGroupItinerary';

describe('routeGroupItinerary', () => {
  const booking = {
    id: 1,
    route_group_id: 'g',
    pickup_location: 'Avenue Ernest-Pictet 9',
    dropoff_location: 'HUG',
    route_group_legs: [
      {
        id: 1,
        sequence: 1,
        is_return: false,
        time_confirmed: false,
        time_scheduled: true,
        display_time: '23:00 (non confirmé)',
        pickup_location: 'Avenue Ernest-Pictet 9',
        dropoff_location: 'Hôpitaux Universitaires de Genève',
        hospital_service: 'Radiologie',
      },
      {
        id: 2,
        sequence: 2,
        is_return: false,
        time_confirmed: true,
        time_scheduled: true,
        display_time: '23:45',
        pickup_location: 'Hôpitaux Universitaires de Genève',
        dropoff_location: 'Clinique de Joli-Mont',
        doctor_name: 'Docteur Rashiti',
      },
      {
        id: 3,
        sequence: 3,
        is_return: true,
        time_confirmed: false,
        time_scheduled: false,
        display_time: 'À définir',
        pickup_location: 'Clinique de Joli-Mont',
        dropoff_location: 'Avenue Ernest-Pictet 9',
      },
    ],
  };

  it('retrouve tout le parcours depuis le résumé embarqué', () => {
    expect(journeyPlaces(booking, [booking])).toEqual([
      'Avenue Ernest-Pictet 9',
      'Hôpitaux Universitaires de Genève',
      'Clinique de Joli-Mont',
      'Avenue Ernest-Pictet 9',
    ]);
    const points = routeTimelinePoints(booking, [booking]);
    expect(points.map((point) => point.label)).toEqual([
      'Prise en charge',
      'Étape 1',
      'Étape 2',
      'Retour',
    ]);
    expect(points[0].timeLabel).toBe('À déterminer');
    expect(points[1].timeLabel).toBe('RDV 23:00');
    expect(points[1].details).toBe('Radiologie');
    expect(points[2].timeLabel).toBe('Départ 23:45');
    expect(points[2].details).toBe('Docteur Rashiti');
    expect(points[3].timeLabel).toBe('Heure à définir');
  });

  it('demande la confirmation de prise en charge tant que l heure n est pas confirmée', () => {
    expect(pickupNeedsCompanyConfirmation({
      ...booking,
      status: 'pending',
      time_confirmed: false,
      is_return: false,
    })).toBe(true);
    expect(pickupNeedsCompanyConfirmation({
      ...booking,
      status: 'pending',
      time_confirmed: true,
    })).toBe(false);
  });

  it('garde le rendez-vous après confirmation de la prise en charge', () => {
    const confirmed = {
      ...booking,
      time_confirmed: true,
      scheduling: { appointment_time: '23:00' },
      route_group_legs: booking.route_group_legs.map((leg) => (
        leg.id === 1
          ? {
            ...leg,
            time_confirmed: true,
            display_time: '22:56',
            appointment_time: '23:00',
          }
          : leg
      )),
    };
    const points = routeTimelinePoints(confirmed, [confirmed]);
    expect(points[0].timeLabel).toBe('Départ 22:56');
    expect(points[1].timeLabel).toBe('RDV 23:00');
  });
});