import { getClientBookingStatusPillColors } from "./AppStatusBadge";

describe("getClientBookingStatusPillColors", () => {
  it("retourne un triplet cohérent pour chaque statut connu", () => {
    for (const s of [
      "pending",
      "requested",
      "confirmed",
      "assigned",
      "en_route",
      "in_progress",
      "completed",
      "cancelled",
    ]) {
      const c = getClientBookingStatusPillColors(s);
      expect(c.backgroundColor).toMatch(/^#|^rgba?\(/);
      expect(c.borderColor).toMatch(/^#|^rgba?\(/);
      expect(c.color).toMatch(/^#|^rgba?\(/);
    }
  });

  it("retombe sur le style neutre pour statut inconnu", () => {
    const a = getClientBookingStatusPillColors("totally_unknown");
    const b = getClientBookingStatusPillColors(null);
    expect(a).toEqual(b);
  });
});
