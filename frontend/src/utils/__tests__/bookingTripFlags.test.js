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
});
