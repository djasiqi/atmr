import {
  pickClusterRepresentativeStatus,
  resolveClusterMarkerSizePx,
} from "./fleetLirieClusterMarker";
import { buildFleetClusterMarkerImageSource } from "./fleetNativeMarkerImage";
import { FLEET_WEB_STATUS_COLORS } from "./fleetMapStatusContract";
import type { FleetDriverMapItem } from "./fleetMapTypes";

function driverWithStatus(status: FleetDriverMapItem["enrichment"]["operationalStatus"]): FleetDriverMapItem {
  return {
    driver_id: 1,
    latitude: 46.2,
    longitude: 6.1,
    timestamp: new Date().toISOString(),
    enrichment: {
      operationalStatus: status,
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
    expect(
      pickClusterRepresentativeStatus([
        driverWithStatus("busy"),
        driverWithStatus("emergency"),
      ])
    ).toBe("emergency");
  });

  it("tailles cluster alignées web", () => {
    expect(resolveClusterMarkerSizePx(3)).toBe(40);
    expect(resolveClusterMarkerSizePx(12)).toBe(46);
    expect(resolveClusterMarkerSizePx(60)).toBe(52);
  });

  it("produit un cluster raster avec couleur dominante", () => {
    const src = buildFleetClusterMarkerImageSource(3, [driverWithStatus("incident")]);
    expect(src.width).toBe(40);
    expect(src.uri.length).toBeGreaterThan(0);
    expect(decodeURIComponent(src.uri)).toContain(FLEET_WEB_STATUS_COLORS.emergency);
  });
});
