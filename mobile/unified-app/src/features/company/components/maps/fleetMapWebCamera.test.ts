import {
  collectFleetMarkerPositions,
  collectValidFleetPositions,
  expandFleetBoundsSpan,
} from "./fleetMapWebCamera";
import type { FleetDriverMapItem, FleetMapMarker } from "./fleetMapTypes";

function mkDriver(id: number, lat: number, lng: number): FleetDriverMapItem {
  return {
    driver_id: id,
    latitude: lat,
    longitude: lng,
    enrichment: { operationalStatus: "available", displayName: `D${id}` },
  } as FleetDriverMapItem;
}

describe("fleetMapWebCamera", () => {
  it("inclut chaque chauffeur d’un cluster", () => {
    const markers: FleetMapMarker[] = [
      {
        kind: "cluster",
        clusterKey: "c1",
        latitude: 46.2,
        longitude: 6.14,
        count: 2,
        drivers: [mkDriver(1, 46.21, 6.15), mkDriver(2, 46.19, 6.13)],
      },
    ];
    expect(collectFleetMarkerPositions(markers)).toHaveLength(2);
  });

  it("étend les bornes si un seul point", () => {
    const expanded = expandFleetBoundsSpan([{ latitude: 46.2, longitude: 6.14 }]);
    expect(expanded.length).toBe(2);
    expect(expanded[0].latitude).not.toBe(expanded[1].latitude);
  });

  it("ignore les coordonnées invalides", () => {
    expect(
      collectValidFleetPositions([
        { latitude: Number.NaN, longitude: 6.14 },
        { latitude: 46.2, longitude: 6.14 },
      ])
    ).toHaveLength(1);
  });
});
