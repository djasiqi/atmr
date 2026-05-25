import {
  connectFleetRouteToLiveDriver,
  dedupeFleetDirectionsPoints,
  fleetRoutePathKey,
  refineFleetDirectionsPath,
  simplifyFleetDirectionsPathForNative,
  resolveFleetMissionDirectionsPlan,
  resolveFleetMissionLegPlans,
  resolveFleetMissionRouteFocusLeg,
  resolveFleetOverlayRouteDrawPoints,
  type FleetDirectionsPlan,
  type FleetMapLatLng,
} from "./fleetMapDirections";
import { extractDetailedDirectionsPath } from "./fleetMapDirectionsWebApi";

describe("fleetMapDirections", () => {
  const pickup: FleetMapLatLng = { latitude: 46.2, longitude: 6.14 };
  const dropoff: FleetMapLatLng = { latitude: 46.19, longitude: 6.12 };
  const driver: FleetMapLatLng = { latitude: 46.21, longitude: 6.15 };

  it("focuses pickup leg before patient on board when selected", () => {
    expect(resolveFleetMissionRouteFocusLeg("en_route_pickup")).toBe("to_pickup");
    expect(resolveFleetMissionRouteFocusLeg("delayed")).toBe("to_pickup");
    expect(resolveFleetMissionRouteFocusLeg("patient_on_board")).toBe("to_dropoff");
  });

  it("splits directions into leg plans when mission is selected", () => {
    const legs = resolveFleetMissionLegPlans({
      driverPosition: driver,
      pickup,
      dropoff,
      lifecyclePhase: "en_route_pickup",
      isSelected: true,
    });
    expect(legs.focusLeg).toBe("to_pickup");
    expect(legs.legs.to_pickup).toEqual({ origin: driver, destination: pickup });
    expect(legs.legs.to_dropoff).toEqual({ origin: pickup, destination: dropoff });
  });

  it("routes patient on board from driver to dropoff", () => {
    const plan = resolveFleetMissionDirectionsPlan({
      driverPosition: driver,
      pickup,
      dropoff,
      lifecyclePhase: "patient_on_board",
    });
    expect(plan).toEqual({ origin: driver, destination: dropoff });
  });

  it("routes assigned mission through pickup waypoint", () => {
    const plan = resolveFleetMissionDirectionsPlan({
      driverPosition: driver,
      pickup,
      dropoff,
      lifecyclePhase: "assigned",
    });
    expect(plan).toEqual({
      origin: driver,
      destination: dropoff,
      waypoints: [pickup],
    });
  });

  it("falls back to pickup-dropoff without driver", () => {
    const plan = resolveFleetMissionDirectionsPlan({
      driverPosition: null,
      pickup,
      dropoff,
      lifecyclePhase: "assigned",
    });
    expect(plan).toEqual({ origin: pickup, destination: dropoff });
  });

  it("simplifies dense native paths for mobile rendering", () => {
    const dense: FleetMapLatLng[] = [];
    for (let index = 0; index < 120; index += 1) {
      dense.push({
        latitude: 46.2 + index * 0.00005,
        longitude: 6.14 + index * 0.00004,
      });
    }
    const simplified = simplifyFleetDirectionsPathForNative(dense);
    expect(simplified.length).toBeLessThan(dense.length);
    expect(simplified.length).toBeLessThanOrEqual(48);
    expect(simplified[0]).toEqual(dense[0]);
    expect(simplified[simplified.length - 1]).toEqual(dense[dense.length - 1]);
  });

  it("refines sparse paths into denser polylines", () => {
    const sparse: FleetMapLatLng[] = [
      { latitude: 46.2, longitude: 6.14 },
      { latitude: 46.205, longitude: 6.145 },
    ];
    const refined = refineFleetDirectionsPath(sparse);
    expect(refined.length).toBeGreaterThan(sparse.length);
  });

  it("connects live driver position to cached route start", () => {
    const routed: FleetMapLatLng[] = [
      { latitude: 46.2, longitude: 6.14 },
      { latitude: 46.201, longitude: 6.141 },
      { latitude: 46.202, longitude: 6.142 },
    ];
    const liveDriver = { latitude: 46.20001, longitude: 6.14001 };
    const connected = connectFleetRouteToLiveDriver(routed, liveDriver);
    expect(connected[0]).toEqual(liveDriver);
    expect(connected.length).toBe(routed.length);
  });

  it("extracts detailed step paths before overview fallback", () => {
    const points = extractDetailedDirectionsPath({
      legs: [
        {
          steps: [
            {
              path: [
                { lat: () => 46.2, lng: () => 6.14 },
                { lat: () => 46.201, lng: () => 6.141 },
                { lat: () => 46.202, lng: () => 6.142 },
              ],
            },
          ],
        },
      ],
      overview_path: [{ lat: () => 46.2, lng: () => 6.14 }],
    });
    expect(points.length).toBeGreaterThanOrEqual(3);
    expect(dedupeFleetDirectionsPoints(points).length).toBeGreaterThanOrEqual(3);
  });

  it("hides fallback straight line while routed path is loading", () => {
    const overlay = {
      missionId: 42,
      points: [
        { latitude: 46.2, longitude: 6.14 },
        { latitude: 46.19, longitude: 6.12 },
      ],
      directionsPlan: {
        origin: { latitude: 46.2, longitude: 6.14 },
        destination: { latitude: 46.19, longitude: 6.12 },
      },
      driverPosition: null,
    };
    const loading = resolveFleetOverlayRouteDrawPoints(
      overlay,
      new Map(),
      new Map([[fleetRoutePathKey(42), "loading"]])
    );
    expect(loading).toEqual([]);
    const failed = resolveFleetOverlayRouteDrawPoints(
      overlay,
      new Map(),
      new Map([[fleetRoutePathKey(42), "failed"]])
    );
    expect(failed).toEqual(overlay.points);
  });
});
