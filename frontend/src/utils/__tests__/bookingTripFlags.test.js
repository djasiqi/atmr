import { buildTripBadgeDescriptors, resolveTripFlagsFromBooking } from '../bookingTripFlags';

describe('bookingTripFlags', () => {
  it('génère les badges depuis trip_flags API', () => {
    const flags = resolveTripFlagsFromBooking({
      trip_flags: {
        round_trip: true,
        multi_stop: true,
        leg_number: 2,
        leg_count: 3,
        transferred: true,
      },
    });
    const badges = buildTripBadgeDescriptors(flags);
    const keys = badges.map((b) => b.key);
    expect(keys).toContain('round_trip');
    expect(keys).toContain('multi_stop');
    expect(keys).toContain('transferred');
  });

  it('garde le badge Livraison même si trip_flags API est présent', () => {
    const flags = resolveTripFlagsFromBooking({
      display_model: 'booking',
      mission_type: 'material_delivery',
      trip_flags: {
        round_trip: false,
        multi_stop: false,
        transferred: false,
      },
    });
    expect(flags.materialDelivery).toBe(true);
    expect(buildTripBadgeDescriptors(flags).map((b) => b.key)).toContain('material_delivery');
  });

  it('ajoute le badge Livraison depuis mission_type', () => {
    const flags = resolveTripFlagsFromBooking({
      mission_type: 'material_delivery',
      delivery_description: 'Oxygène',
    });
    const badges = buildTripBadgeDescriptors(flags);
    expect(badges.map((b) => b.key)).toContain('material_delivery');
    expect(badges.find((b) => b.key === 'material_delivery')?.label).toBe('LIVRAISON');
  });

  it('n’ajoute pas le badge LIVRAISON si seule delivery_description est renseignée', () => {
    const flags = resolveTripFlagsFromBooking({
      mission_type: 'patient_transport',
      delivery_description: 'Livraison de documents',
    });
    const badges = buildTripBadgeDescriptors(flags);
    expect(flags.materialDelivery).toBe(false);
    expect(badges.map((b) => b.key)).not.toContain('material_delivery');
    expect(badges.map((b) => b.label)).not.toContain('LIVRAISON');
  });
});
