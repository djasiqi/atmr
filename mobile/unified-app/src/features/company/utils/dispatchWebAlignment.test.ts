import { describe, expect, it } from "@jest/globals";
import {
  delaysByBookingLastWins,
  flattenCompanyDispatchDelays,
  mergeCompanyDispatchDelaySources,
  pickupDelaysByBookingLastWins,
  pickupEtaIsoByBookingId,
} from "./dispatchWebAlignment";

describe("dispatchWebAlignment", () => {
  it("flatten prend en compte pickup, dropoff et delay_minutes comme le delayMap web", () => {
    const rows = flattenCompanyDispatchDelays([
      { booking_id: 1, pickup_delay_minutes: 0, dropoff_delay_minutes: 56 },
    ]);
    expect(rows).toEqual([{ booking_id: 1, delay_minutes: 56, is_pickup: true }]);
    expect(pickupDelaysByBookingLastWins(rows).get(1)).toBe(56);
  });

  it("inclut les retards légers 1–4 min (aligné getDelayLevel web)", () => {
    const rows = flattenCompanyDispatchDelays([{ booking_id: 99, delay_minutes: 3 }]);
    expect(rows).toEqual([{ booking_id: 99, delay_minutes: 3, is_pickup: true }]);
  });

  it("exclut 0 minute de retard", () => {
    expect(flattenCompanyDispatchDelays([{ booking_id: 1, delay_minutes: 0 }])).toEqual([]);
  });

  it("priorise delay_minutes agrégé quand présent", () => {
    const rows = flattenCompanyDispatchDelays([
      {
        booking_id: 2,
        delay_minutes: 50,
        pickup_delay_minutes: 12,
        dropoff_delay_minutes: 40,
      },
    ]);
    expect(rows[0]?.delay_minutes).toBe(50);
  });

  it("retire booking.id imbriqué si booking_id absent (payload live)", () => {
    const rows = flattenCompanyDispatchDelays([
      { booking: { id: 42 }, delay_minutes: 7 },
    ]);
    expect(rows).toEqual([{ booking_id: 42, delay_minutes: 7, is_pickup: true }]);
  });

  it("prend le max pickup / dropoff / delay_minutes (marshal aligné agrégé)", () => {
    const rows = flattenCompanyDispatchDelays([
      { booking_id: 3, delay_minutes: 0, pickup_delay_minutes: 4, dropoff_delay_minutes: 52 },
    ]);
    expect(rows[0]?.delay_minutes).toBe(52);
  });

  it("delaysByBookingLastWins : dernier objet gagne pour un même booking", () => {
    const rows = [
      { booking_id: 1, delay_minutes: 10, is_pickup: true },
      { booking_id: 1, delay_minutes: 20, is_pickup: true },
    ];
    expect(delaysByBookingLastWins(rows).get(1)).toBe(20);
  });

  it("merge : live vide + snapshot conservé (retard + ETA)", () => {
    const merged = mergeCompanyDispatchDelaySources([], [
      {
        booking_id: 77,
        delay_minutes: 12,
        pickup_eta: "2026-01-02T09:40:00.000Z",
      },
    ]) as Record<string, unknown>[];
    expect(merged).toHaveLength(1);
    expect(merged[0]?.booking_id).toBe(77);
    expect(merged[0]?.delay_minutes).toBe(12);
    expect(merged[0]?.pickup_eta).toBe("2026-01-02T09:40:00.000Z");
    expect(pickupEtaIsoByBookingId(merged).get(77)).toBe("2026-01-02T09:40:00.000Z");
    expect(pickupDelaysByBookingLastWins(flattenCompanyDispatchDelays(merged)).get(77)).toBe(12);
  });

  it("merge : max des minutes live vs snapshot pour un même booking", () => {
    const merged = mergeCompanyDispatchDelaySources(
      [{ booking_id: 5, delay_minutes: 3 }],
      [{ booking_id: 5, delay_minutes: 15 }],
    ) as Record<string, unknown>[];
    expect(merged).toHaveLength(1);
    expect(merged[0]?.booking_id).toBe(5);
    expect(merged[0]?.delay_minutes).toBe(15);
  });
});
