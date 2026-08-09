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

  it("recorded_at frais → pas stale même si serveur dit stale", () => {
    expect(
      isFleetDriverMarkerStale({
        driver_id: 3,
        latitude: 46.2,
        longitude: 6.14,
        timestamp: new Date().toISOString(),
        location_status: "stale",
      })
    ).toBe(false);
  });

  it("blend stale sur busy sans changer le statut", () => {
    const visual = resolveMarkerVisual("busy", true);
    expect(visual.opacity).toBe(0.88);
    expect(visual.fill).not.toBe("#00796B");
  });

  it("constrained exempt du blend stale", () => {
    const visual = resolveMarkerVisual("constrained", true);
    expect(visual.fill).toBe("#f97316");
    expect(visual.opacity).toBe(1);
  });
});
