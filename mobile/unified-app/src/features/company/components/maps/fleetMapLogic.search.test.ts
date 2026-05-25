import { matchFleetDriversByQuery } from "./fleetMapLogic";
import type { FleetDriverMapItem } from "./fleetMapTypes";

const driver = (id: number, name: string): FleetDriverMapItem =>
  ({
    driver_id: id,
    driver_name: name,
    latitude: 46.2,
    longitude: 6.14,
    timestamp: new Date().toISOString(),
    mission_id: null,
    location_status: "live",
    enrichment: {
      operationalStatus: "available",
      linkedMission: null,
      delayMinutes: null,
      vehicleType: null,
      licensePlate: null,
      currentAddress: null,
      destinationAddress: null,
      etaLabel: null,
      distanceLabel: null,
      phone: null,
    },
  }) as FleetDriverMapItem;

describe("matchFleetDriversByQuery", () => {
  const drivers = [driver(1, "Alice Martin"), driver(2, "Bob Dupont"), driver(12, "Claire Veux")];

  it("filtre par nom ou id", () => {
    expect(matchFleetDriversByQuery(drivers, "martin").map((d) => d.driver_id)).toEqual([1]);
    expect(matchFleetDriversByQuery(drivers, "12").map((d) => d.driver_id)).toEqual([12]);
  });

  it("retourne une liste triée quand la requête est vide", () => {
    const results = matchFleetDriversByQuery(drivers, "", 2);
    expect(results).toHaveLength(2);
    expect(results[0]!.driver_name).toBe("Alice Martin");
  });
});
