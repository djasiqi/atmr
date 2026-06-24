import { isFleetDriverMarkerStale } from "./fleetMapStale";
import { resolveMarkerVisual } from "./fleetMapStatusContract";

describe("fleetMapStale", () => {
  it("détecte stale via location_status", () => {
    expect(
      isFleetDriverMarkerStale({
        driver_id: 1,
        latitude: 46.2,
        longitude: 6.14,
        timestamp: new Date().toISOString(),
        location_status: "stale",
      })
    ).toBe(true);
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
