import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import { enrichFleetDriver } from "./fleetMapLogic";
import {
  buildFleetMissionOverlays,
  collectMapMissions,
  computeMissionOverlayFocusRegion,
  formatMapMissionEtaBadge,
  resolveMissionLifecyclePhase,
  resolveMissionRouteLegStyle,
  resolveMissionRouteStyle,
} from "./fleetMapMissionVisual";

const baseDriver = (id: number, lat: number, lng: number): CompanyDriverLiveLocation => ({
  driver_id: id,
  driver_name: `Driver ${id}`,
  latitude: lat,
  longitude: lng,
  timestamp: new Date().toISOString(),
  mission_id: null,
  location_status: "live",
});

describe("fleetMapMissionVisual", () => {
  it("resolve lifecycle phases", () => {
    expect(
      resolveMissionLifecyclePhase({ mission_id: 1, status: "assigned" })
    ).toBe("assigned");
    expect(
      resolveMissionLifecyclePhase({ mission_id: 1, status: "en_route" })
    ).toBe("en_route_pickup");
    expect(
      resolveMissionLifecyclePhase({ mission_id: 1, status: "in_progress" })
    ).toBe("patient_on_board");
  });

  it("builds overlays with route points and caps secondary routes", () => {
    const missions: CompanyDispatchMission[] = [
      {
        mission_id: 10,
        status: "en_route",
        driver_id: 1,
        pickup_lat: 46.21,
        pickup_lon: 6.15,
        dropoff_lat: 46.25,
        dropoff_lon: 6.19,
      },
      {
        mission_id: 11,
        status: "assigned",
        driver_id: 2,
        pickup_lat: 46.22,
        pickup_lon: 6.16,
        dropoff_lat: 46.26,
        dropoff_lon: 6.2,
      },
      {
        mission_id: 12,
        status: "assigned",
        driver_id: 3,
        pickup_lat: 46.23,
        pickup_lon: 6.17,
        dropoff_lat: 46.27,
        dropoff_lon: 6.21,
      },
      {
        mission_id: 13,
        status: "assigned",
        driver_id: 4,
        pickup_lat: 46.24,
        pickup_lon: 6.18,
        dropoff_lat: 46.28,
        dropoff_lon: 6.22,
      },
    ];
    const driversById = new Map(
      [1, 2, 3, 4].map((id, i) => [
        id,
        enrichFleetDriver(baseDriver(id, 46.2 + i * 0.01, 6.14 + i * 0.01), missions),
      ])
    );
    const overlays = buildFleetMissionOverlays({
      missions,
      driversById,
      selectedMissionId: 10,
      routeCap: 3,
    });
    expect(overlays.length).toBe(3);
    expect(overlays.some((o) => o.missionId === 10 && o.isSelected)).toBe(true);
    const selected = overlays.find((o) => o.isSelected);
    expect(selected?.points.length).toBeGreaterThanOrEqual(2);
    expect(selected?.displayOpacity).toBeGreaterThan(0);
    expect(selected?.showEtaBadge).toBe(false);
    expect(selected?.legDirectionsPlans?.to_pickup).toBeDefined();
  });

  it("computes mission focus region from driver pickup dropoff", () => {
    const mission: CompanyDispatchMission = {
      mission_id: 10,
      status: "en_route",
      driver_id: 1,
      pickup_lat: 46.21,
      pickup_lon: 6.15,
      dropoff_lat: 46.25,
      dropoff_lon: 6.19,
    };
    const driver = enrichFleetDriver(baseDriver(1, 46.2, 6.14), [mission]);
    const overlays = buildFleetMissionOverlays({
      missions: [mission],
      driversById: new Map([[1, driver]]),
      selectedMissionId: 10,
    });
    const region = computeMissionOverlayFocusRegion(overlays[0]!);
    expect(region).not.toBeNull();
    expect(region!.latitudeDelta).toBeGreaterThan(0);
  });

  it("collectMapMissions merges driver-linked missions", () => {
    const missions: CompanyDispatchMission[] = [];
    const driver = enrichFleetDriver(baseDriver(1, 46.2, 6.14), [
      {
        mission_id: 99,
        status: "en_route",
        driver_id: 1,
        pickup_lat: 46.21,
        pickup_lon: 6.15,
        dropoff_lat: 46.25,
        dropoff_lon: 6.19,
      },
    ]);
    const merged = collectMapMissions(missions, new Map([[1, driver]]));
    expect(merged).toHaveLength(1);
    expect(merged[0]?.mission_id).toBe(99);
  });

  it("does not show map ETA badge on selected mission", () => {
    const missions: CompanyDispatchMission[] = [
      {
        mission_id: 20,
        status: "en_route",
        driver_id: 1,
        pickup_lat: 46.21,
        pickup_lon: 6.15,
        dropoff_lat: 46.25,
        dropoff_lon: 6.19,
        scheduled_at: "2026-05-19T10:30:00+02:00",
        route_duration_min: 8,
      },
    ];
    const driversById = new Map([
      [1, enrichFleetDriver(baseDriver(1, 46.2, 6.14), missions)],
    ]);
    const overlays = buildFleetMissionOverlays({
      missions,
      driversById,
      selectedMissionId: 20,
      routeCap: 6,
    });
    const selected = overlays.find((o) => o.missionId === 20);
    expect(selected?.isSelected).toBe(true);
    expect(selected?.showEtaBadge).toBe(false);
    expect(selected?.legDirectionsPlans?.to_pickup).toBeDefined();
    expect(selected?.legDirectionsPlans?.to_dropoff).toBeDefined();
    expect(selected?.routeFocusLeg).toBe("to_pickup");
  });

  it("focuses dropoff leg when patient is on board and selected", () => {
    const missions: CompanyDispatchMission[] = [
      {
        mission_id: 21,
        status: "in_progress",
        driver_id: 1,
        pickup_lat: 46.21,
        pickup_lon: 6.15,
        dropoff_lat: 46.25,
        dropoff_lon: 6.19,
      },
    ];
    const driversById = new Map([
      [1, enrichFleetDriver(baseDriver(1, 46.22, 6.16), missions)],
    ]);
    const overlays = buildFleetMissionOverlays({
      missions,
      driversById,
      selectedMissionId: 21,
      routeCap: 6,
    });
    expect(overlays[0]?.routeFocusLeg).toBe("to_dropoff");
  });

  it("dims secondary route leg style when mission is selected", () => {
    const primary = resolveMissionRouteStyle("en_route_pickup", 1);
    const secondary = resolveMissionRouteLegStyle(primary, "to_dropoff", "to_pickup");
    expect(secondary.opacity).toBeLessThan(primary.opacity);
    expect(secondary.lineDashPattern).not.toBeNull();
  });

  it("uses red pickup anchor when mission lifecycle is delayed", () => {
    const missions: CompanyDispatchMission[] = [
      {
        mission_id: 99,
        status: "en_route",
        driver_id: 1,
        pickup_lat: 46.21,
        pickup_lon: 6.15,
        dropoff_lat: 46.25,
        dropoff_lon: 6.19,
        scheduled_at: "2026-05-19T08:00:00+02:00",
        assignment_pickup_delay_minutes: 15,
      },
    ];
    const driversById = new Map([
      [1, enrichFleetDriver(baseDriver(1, 46.2, 6.14), missions)],
    ]);
    const overlays = buildFleetMissionOverlays({
      missions,
      driversById,
      selectedMissionId: null,
      routeCap: 6,
    });
    expect(overlays[0]?.lifecyclePhase).toBe("delayed");
    expect(overlays[0]?.pickupAnchor?.fill).toBe("#EF4444");
    expect(overlays[0]?.routeStyle.color).toBe("#EF4444");
  });

  it("styles delayed routes with operational calm hierarchy", () => {
    const style = resolveMissionRouteStyle("delayed", 1);
    expect(style.color).toBe("#EF4444");
    expect(style.strokeWidth).toBeLessThanOrEqual(4);
    expect(style.glowColor).toBe("rgba(239, 68, 68, 0.10)");
    expect(style.opacity).toBeLessThanOrEqual(0.85);
  });

  it("formats compact map ETA badge without scheduled time suffix", () => {
    const label = formatMapMissionEtaBadge({
      mission_id: 1,
      status: "assigned",
      route_duration_min: 6,
      scheduled_at: "2026-05-18T08:00:00+02:00",
    });
    expect(label).toBe("~6 min");
    expect(label).not.toContain("·");
  });
});
