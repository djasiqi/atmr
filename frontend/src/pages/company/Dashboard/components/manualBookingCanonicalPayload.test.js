import { buildCanonicalReservationPayload } from './manualBookingCanonicalPayload';

const base = {
  clientId: 7,
  pickupLocation: 'Rue A',
  pickupCoords: { lat: 46.2, lon: 6.14 },
  dropoffLocation: 'Rue B',
  dropoffCoords: { lat: 46.21, lon: 6.15 },
  scheduledTime: '2026-10-02T09:00:00',
  destinationArrival: '2026-10-02T10:00:00',
  destinationDeparture: '2026-10-02T10:30:00',
  extraStops: [
    {
      location: 'Étape',
      coords: { lat: 46.22, lon: 6.16 },
      arrival: '2026-10-02T09:20:00',
      departure: '2026-10-02T09:40:00',
      destinationKind: 'medical',
    },
  ],
  isRoundTrip: true,
  returnArrival: '2026-10-02T11:30:00',
  pricingMode: 'manual',
  manualAmounts: ['40', '50', '45'],
  idempotencyKey: 'cle-formulaire',
  passengerName: 'Camille',
  isUrgent: true,
  isRecurring: true,
  recurrenceType: 'weekly',
  occurrences: 4,
};

describe('buildCanonicalReservationPayload', () => {
  it('envoie un parcours canonique avec deux heures sur l étape intermédiaire', () => {
    const { error, payload } = buildCanonicalReservationPayload(base);
    expect(error).toBeNull();
    expect(payload.pickup_location).toBeUndefined();
    expect(payload.is_round_trip).toBeUndefined();
    expect(payload.pricing_mode).toBe('manual');
    expect(payload.passenger_name).toBe('Camille');
    expect(payload.idempotency_key).toBe('cle-formulaire');
    expect(payload.route_steps.map((step) => step.kind)).toEqual([
      'pickup',
      'destination',
      'destination',
      'return',
    ]);
    expect(payload.route_steps[1].arrival_at).toBe('2026-10-02T09:20:00');
    expect(payload.route_steps[1].departure_at).toBe('2026-10-02T09:40:00');
    expect(payload.segment_amounts).toHaveLength(3);
    expect(payload.is_recurring).toBe(true);
  });

  it('refuse le mode automatique avec un montant de tronçon', () => {
    const { payload } = buildCanonicalReservationPayload({
      ...base,
      pricingMode: 'automatic',
      extraStops: [],
      isRoundTrip: false,
    });
    expect(payload.segment_amounts).toBeUndefined();
    expect(payload.preferential_amount).toBeUndefined();
    expect(payload.pricing_mode).toBe('automatic');
  });

  it('laisse le retour sans heure quand elle est à définir', () => {
    const { error, payload } = buildCanonicalReservationPayload({
      ...base,
      extraStops: [],
      isRoundTrip: true,
      destinationDeparture: '',
      returnArrival: '',
      destinationArrival: '2026-10-02T10:00:00',
      manualAmounts: ['40', '45'],
      pickupAccessNotes: 'code 1234',
    });
    expect(error).toBeNull();
    expect(payload.route_steps.map((step) => step.kind)).toEqual([
      'pickup',
      'destination',
      'return',
    ]);
    expect(payload.route_steps[0].departure_at).toBe('2026-10-02T09:00:00');
    expect(payload.route_steps[1].departure_at).toBeNull();
    expect(payload.route_steps[2].arrival_at).toBeNull();
    expect(payload.route_steps[2].access_notes).toBe('code 1234');
  });

  it('n envoie pas passenger_name quand le formulaire ne le saisit pas', () => {
    const { passengerName, ...withoutPassenger } = base;
    const { payload } = buildCanonicalReservationPayload(withoutPassenger);
    expect(passengerName).toBe('Camille');
    expect(payload.passenger_name).toBeUndefined();
  });
});
