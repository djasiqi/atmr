import {
  pickClusterRepresentativeStatus,
  resolveFleetClusterMarkerHostLayout,
} from "./fleetLirieClusterMarker";
import { buildFleetClusterCountBadgeImageSource } from "./fleetNativeMarkerImage";
import type { FleetDriverMapItem } from "./fleetMapTypes";

function driverWithStatus(status: FleetDriverMapItem["enrichment"]["operationalStatus"]): FleetDriverMapItem {
  return {
    driver_id: 1,
    latitude: 46.2,
    longitude: 6.1,
    enrichment: {
      operationalStatus: status,
    },
  } as FleetDriverMapItem;
}

describe("fleetLirieClusterMarker", () => {
  it("choisit le statut le plus urgent du cluster", () => {
    expect(
      pickClusterRepresentativeStatus([
        driverWithStatus("available"),
        driverWithStatus("delayed"),
      ])
    ).toBe("delayed");
  });

  it("dimensionne le conteneur cluster avec pastille à droite", () => {
    const host = resolveFleetClusterMarkerHostLayout(2);
    expect(host.hostW).toBeGreaterThan(host.iconW);
    expect(host.fontSize).toBe(11);
    expect(host.width).toBe(24);
  });

  it("produit une pastille compteur raster agrandie", () => {
    const single = buildFleetClusterCountBadgeImageSource(2);
    expect(single.width).toBe(24);
    expect(single.height).toBe(24);

    const duo = buildFleetClusterCountBadgeImageSource(12);
    expect(duo.width).toBe(28);
    expect(duo.height).toBe(24);
    expect(duo.uri.length).toBeGreaterThan(0);
  });
});
