import { canMarkRideUrgent, isPickupSentinel, isTimeUndefined } from "./pickupSentinel";

describe("pickupSentinel", () => {
  it("canMarkRideUrgent autorise l’urgence seulement sans horaire planifié", () => {
    expect(canMarkRideUrgent({ scheduled_at: null })).toBe(true);
    expect(canMarkRideUrgent({ scheduled_at: "2026-06-22T00:00:00" })).toBe(true);
    expect(canMarkRideUrgent({ scheduled_at: "2026-06-22T09:30:00" })).toBe(false);
    expect(
      canMarkRideUrgent({
        summary: {
          time: { pickup_at: "2026-06-22T07:30:00.000Z" },
          scheduling: { time_defined: true },
          time_confirmed: true,
        },
      })
    ).toBe(false);
  });

  it("isTimeUndefined reste distinct pour retours à confirmer avec heure affichée", () => {
    expect(
      isTimeUndefined({
        scheduled_at: "2026-06-22T09:30:00",
        time_confirmed: false,
      })
    ).toBe(true);
    expect(
      canMarkRideUrgent({
        scheduled_at: "2026-06-22T09:30:00",
        time_confirmed: false,
      })
    ).toBe(false);
  });

  it("isPickupSentinel détecte minuit sentinelle", () => {
    expect(isPickupSentinel("2026-06-22T00:00:00")).toBe(true);
    expect(isPickupSentinel("2026-06-22T09:45:00")).toBe(false);
  });
});
