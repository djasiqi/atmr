import { Circle } from "react-native-maps";

import { buildImminentDepartureCircleElements } from "./ImminentDepartureMarkers";
import type { ImminentDeparture } from "../../dashboard/cockpit/imminentDepartures";

function departure(
  partial: Partial<ImminentDeparture> & Pick<ImminentDeparture, "missionId">
): ImminentDeparture {
  return {
    missionId: partial.missionId,
    scheduledAtMs: 0,
    minutesUntil: 10,
    risk: partial.risk ?? "normal",
    clusterKey: null,
    pickupLat: partial.pickupLat ?? null,
    pickupLon: partial.pickupLon ?? null,
  };
}

describe("enfants natifs carte entreprise", () => {
  it("n'insère pas de null quand une coordonnée de départ manque", () => {
    const nodes = buildImminentDepartureCircleElements([
      departure({ missionId: 1, pickupLat: null, pickupLon: 6.1 }),
      departure({ missionId: 2, pickupLat: 46.2, pickupLon: 6.14, risk: "critical" }),
    ]);

    expect(nodes).toHaveLength(1);
    expect(nodes.every((node) => node != null && node.type === Circle)).toBe(true);
  });
});
