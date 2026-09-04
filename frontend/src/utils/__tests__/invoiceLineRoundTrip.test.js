import { getRoundTripAuditLegs } from '../invoiceLineRoundTrip';

describe('getRoundTripAuditLegs', () => {
  it('expose les deux booking_id d’un A/R regroupé', () => {
    const legs = getRoundTripAuditLegs({
      reservation_id: 101,
      line_meta: {
        billing_unit: 'round_trip',
        primary_booking_id: 101,
        booking_ids: [101, 202],
        round_trip_merge_partner_reservation_id: 202,
        round_trip_primary_amount_ht: 40,
        round_trip_partner_amount_ht: 40,
      },
    });
    expect(legs).not.toBeNull();
    expect(legs.segmentsCount).toBe(2);
    expect(legs.outbound.bookingId).toBe(101);
    expect(legs.inbound.bookingId).toBe(202);
    expect(legs.outbound.amountHt).toBe(40);
    expect(legs.inbound.amountHt).toBe(40);
  });

  it('ne fusionne pas une ligne simple', () => {
    expect(
      getRoundTripAuditLegs({
        reservation_id: 7,
        line_meta: { billing_unit: 'single', booking_ids: [7] },
      })
    ).toBeNull();
  });
});
