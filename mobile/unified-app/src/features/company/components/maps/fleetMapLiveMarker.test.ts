import { shouldFleetMarkerLivePulse } from "./fleetMapLiveMarker";
import type { FleetDriverMapItem } from "./fleetMapTypes";

function driver(partial: Partial<FleetDriverMapItem> & Pick<FleetDriverMapItem, "driver_id">): FleetDriverMapItem {
  return {
    driver_id: partial.driver_id,
    latitude: partial.latitude ?? 46.2,
    longitude: partial.longitude ?? 6.14,
    driver_name: "Test",
    full_name: "Test",
    location_status: partial.location_status ?? "live",
    mission_id: partial.mission_id ?? null,
    speed: partial.speed ?? 12,
    enrichment: partial.enrichment ?? {
      operationalStatus: "available",
      linkedMission: null,
      delayMinutes: null,
      vehicleType: null,
    },
  } as FleetDriverMapItem;
}

describe("shouldFleetMarkerLivePulse", () => {
  it("active pour disponible connecté", () => {
    expect(
      shouldFleetMarkerLivePulse(
        "available",
        driver({ driver_id: 1, location_status: "live" })
      )
    ).toBe(true);
  });

  it("active pour en mission connecté", () => {
    expect(
      shouldFleetMarkerLivePulse(
        "on_mission",
        driver({
          driver_id: 2,
          location_status: "live",
          mission_id: 9,
          enrichment: {
            operationalStatus: "on_mission",
            linkedMission: null,
            delayMinutes: null,
            vehicleType: null,
          },
        })
      )
    ).toBe(true);
  });

  it("inactif hors ligne", () => {
    expect(
      shouldFleetMarkerLivePulse(
        "offline",
        driver({ driver_id: 3, location_status: "offline" })
      )
    ).toBe(false);
  });
});
