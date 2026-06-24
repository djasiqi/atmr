import { FLEET_WEB_STATUS_COLORS, resolveFleetMarkerFillColor } from "./fleetMapStatusContract";

describe("fleetMapStatusContract", () => {
  it("aligne les couleurs sur le portail web", () => {
    expect(FLEET_WEB_STATUS_COLORS).toMatchObject({
      available: "#4ade80",
      assigned: "#f59e0b",
      busy: "#00796B",
      offline: "#91A3A0",
      emergency: "#ef4444",
      constrained: "#f97316",
      brandDark: "#00695C",
    });
  });

  it("mappe les enrichissements locaux sur la palette web", () => {
    expect(resolveFleetMarkerFillColor("break")).toBe(FLEET_WEB_STATUS_COLORS.assigned);
    expect(resolveFleetMarkerFillColor("delayed")).toBe(FLEET_WEB_STATUS_COLORS.emergency);
    expect(resolveFleetMarkerFillColor("incident")).toBe(FLEET_WEB_STATUS_COLORS.emergency);
    expect(resolveFleetMarkerFillColor("busy")).toBe(FLEET_WEB_STATUS_COLORS.busy);
  });
});
