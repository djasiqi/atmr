import { Platform } from "react-native";
import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import {
  buildFleetClusterMarkerImageSource,
  buildFleetDriverMarkerImageSource,
  FLEET_NATIVE_DRIVER_MARKER_SIZE_PX,
} from "./fleetNativeMarkerImage";
import { clearMetroAssetResolveCacheForTests } from "./resolveMetroAssetSource";
import type { FleetOperationalStatus } from "./mapStatusTheme";
import type { FleetDriverMapItem } from "./fleetMapTypes";

const baseDriverItem = (
  status: FleetOperationalStatus,
  overrides?: Partial<CompanyDriverLiveLocation>
): FleetDriverMapItem => ({
  driver_id: 1,
  driver_name: "Jean Dupont",
  latitude: 46.2,
  longitude: 6.14,
  timestamp: new Date().toISOString(),
  location_status: "live",
  ...overrides,
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
});

const ALL_STATUSES: FleetOperationalStatus[] = [
  "available",
  "busy",
  "assigned",
  "break",
  "delayed",
  "incident",
  "emergency",
  "constrained",
  "offline",
  "last_known",
];

describe("fleetNativeMarkerImage", () => {
  beforeEach(() => {
    clearMetroAssetResolveCacheForTests();
  });

  it("retourne PNG sur Android, SVG sur iOS", () => {
    const item = baseDriverItem("available");
    const src = buildFleetDriverMarkerImageSource("available", item);
    expect(src.uri.length).toBeGreaterThan(0);
    expect(src.width).toBe(FLEET_NATIVE_DRIVER_MARKER_SIZE_PX);
    expect(src.height).toBe(58);
    if (Platform.OS === "android") {
      expect(src.uri).toMatch(/^data:image\/png;base64,/);
    } else {
      expect(src.uri).toMatch(/^data:image\/svg\+xml/);
    }
  });

  it("retourne toujours uri non vide pour tous les statuts", () => {
    for (const status of ALL_STATUSES) {
      const src = buildFleetDriverMarkerImageSource(status, baseDriverItem(status));
      expect(src.uri.length).toBeGreaterThan(0);
      expect(src.width).toBeGreaterThan(0);
      expect(src.height).toBeGreaterThan(0);
    }
  });

  it("cluster coloré selon statut dominant", () => {
    const available = baseDriverItem("available");
    const incident = baseDriverItem("incident");
    const clusterAvailable = buildFleetClusterMarkerImageSource(2, [available, available]);
    const clusterIncident = buildFleetClusterMarkerImageSource(2, [available, incident]);
    expect(decodeURIComponent(clusterAvailable.uri)).toContain("#4ade80");
    expect(decodeURIComponent(clusterIncident.uri)).toContain("#ef4444");
  });
});
