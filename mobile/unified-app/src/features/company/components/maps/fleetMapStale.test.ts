import { isFleetDriverMarkerStale } from "./fleetMapStale";
import { resolveMarkerVisual } from "./fleetMapStatusContract";

describe("fleetMapStale", () => {
  it("détecte stale via statut serveur si recorded_at absent", () => {
    expect(
      isFleetDriverMarkerStale({
        driver_id: 1,
        latitude: 46.2,
        longitude: 6.14,
        timestamp: null,
        recorded_at: null,
        location_status: "stale",
      } as never)
    ).toBe(true);
  });

  it("âge local écrase un live serveur figé", () => {
    const old = new Date(Date.now() - 180_000).toISOString();
    expect(
      isFleetDriverMarkerStale({
        driver_id: 2,
        latitude: 46.2,
        longitude: 6.14,
        timestamp: old,
        recorded_at: old,
        location_status: "live",
      })
    ).toBe(true);
  });

  it("serveur stale + recorded_at frais → reste stale (non-promotion)", () => {
    expect(
      isFleetDriverMarkerStale({
        driver_id: 3,
        latitude: 46.2,
        longitude: 6.14,
        timestamp: new Date().toISOString(),
        location_status: "stale",
      })
    ).toBe(true);
  });

  it("blend stale sur busy via présence", () => {
    const visual = resolveMarkerVisual("busy", "stale");
    expect(visual.opacity).toBe(0.88);
    expect(visual.fill).not.toBe("#00796B");
  });

  it("recent atténue la couleur métier", () => {
    const live = resolveMarkerVisual("busy", "live");
    const recent = resolveMarkerVisual("busy", "recent");
    expect(live.opacity).toBe(1);
    expect(recent.opacity).toBeLessThan(1);
    expect(recent.fill).toBe(live.fill);
  });

  it("last_known fantôme", () => {
    const visual = resolveMarkerVisual("available", "last_known");
    expect(visual.opacity).toBeLessThan(0.5);
  });
});
