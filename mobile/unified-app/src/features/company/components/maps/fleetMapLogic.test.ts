import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import {
  buildFleetActiveRoute,
  buildFleetLiveOverlay,
  buildFleetLiveOverlayFromEnriched,
  clusterFleetMarkers,
  enrichFleetDriver,
  enrichFleetDrivers,
  patchEnrichedFleetDrivers,
  filterFleetDrivers,
  pickPrimaryFleetDriver,
  resolveFleetOperationalStatus,
  formatFleetConstraintReason,
} from "./fleetMapLogic";
import { DEFAULT_FLEET_MAP_FILTERS } from "./fleetMapTypes";

const baseDriver = (id: number, lat: number, lng: number, missionId?: number): CompanyDriverLiveLocation => ({
  driver_id: id,
  driver_name: `Driver ${id}`,
  latitude: lat,
  longitude: lng,
  timestamp: new Date().toISOString(),
  mission_id: missionId ?? null,
  location_status: "live",
});

describe("fleetMapLogic", () => {
  it("resolve incident status from large pickup delay", () => {
    const driver = baseDriver(1, 46.2, 6.14, 10);
    const mission: CompanyDispatchMission = {
      mission_id: 10,
      status: "en_route",
      driver_id: 1,
      assignment_pickup_delay_minutes: 22,
    };
    const enriched = enrichFleetDriver(driver, [mission]);
    expect(enriched.enrichment.operationalStatus).toBe("incident");
  });

  it("clusters nearby drivers", () => {
    const d1 = enrichFleetDriver(baseDriver(1, 46.2, 6.14), []);
    const d2 = enrichFleetDriver(baseDriver(2, 46.2001, 6.1401), []);
    const markers = clusterFleetMarkers([d1, d2], 0.05);
    expect(markers).toHaveLength(1);
    expect(markers[0].kind).toBe("cluster");
    if (markers[0].kind === "cluster") {
      expect(markers[0].count).toBe(2);
    }
  });

  it("filters by status available", () => {
    const d1 = enrichFleetDriver(baseDriver(1, 46.2, 6.14), []);
    const d2 = enrichFleetDriver(baseDriver(2, 46.3, 6.15, 5), []);
    const d3 = enrichFleetDriver(baseDriver(3, 46.31, 6.16, 8), [
      { mission_id: 8, status: "en_route", driver_id: 3 },
    ]);
    const filtered = filterFleetDrivers([d1, d2, d3], { ...DEFAULT_FLEET_MAP_FILTERS, status: "available" });
    expect(filtered.map((d) => d.driver_id).sort()).toEqual([1, 2]);
  });

  it("does not mark on_mission when mission is completed", () => {
    const driver = baseDriver(2, 46.3, 6.15, 5);
    const completed: CompanyDispatchMission = {
      mission_id: 5,
      status: "completed",
      driver_id: 2,
    };
    expect(resolveFleetOperationalStatus(driver, completed)).toBe("available");
    const enriched = enrichFleetDriver(driver, [completed]);
    expect(enriched.enrichment.operationalStatus).toBe("available");
  });

  it("offline when stale", () => {
    const driver: CompanyDriverLiveLocation = {
      ...baseDriver(1, 46.2, 6.14),
      last_seen_seconds: 500,
      location_status: "stale",
    };
    expect(resolveFleetOperationalStatus(driver, null)).toBe("offline");
  });

  it("last_known distinct de offline", () => {
    const driver: CompanyDriverLiveLocation = {
      ...baseDriver(1, 46.2, 6.14),
      location_status: "last_known",
      last_seen_seconds: 900,
    };
    expect(resolveFleetOperationalStatus(driver, null)).toBe("last_known");
  });

  it("constrained avant mission active", () => {
    const driver: CompanyDriverLiveLocation = {
      ...baseDriver(1, 46.2, 6.14),
      presence_status: "degraded_constrained",
      location_status: "live",
    };
    expect(resolveFleetOperationalStatus(driver, null)).toBe("constrained");
  });

  it("formatFleetConstraintReason fallback Raison inconnue", () => {
    const driver = baseDriver(1, 46.2, 6.14);
    expect(formatFleetConstraintReason(driver)).toBe("Raison inconnue");
  });

  it("buildFleetActiveRoute uses mission coordinates", () => {
    const driver = baseDriver(1, 46.2, 6.14, 10);
    const mission: CompanyDispatchMission = {
      mission_id: 10,
      status: "en_route",
      driver_id: 1,
      pickup_lat: 46.21,
      pickup_lon: 6.15,
      dropoff_lat: 46.25,
      dropoff_lon: 6.19,
    };
    const enriched = enrichFleetDriver(driver, [mission]);
    const route = buildFleetActiveRoute(enriched);
    expect(route).not.toBeNull();
    expect(route?.points.length).toBeGreaterThanOrEqual(2);
    expect(route?.points[1]).toEqual({ latitude: 46.21, longitude: 6.15 });
    expect(route?.points[2]).toEqual({ latitude: 46.25, longitude: 6.19 });
    expect(route?.color).toBe("#00796B");
  });

  it("pickPrimaryFleetDriver prioritise mission active", () => {
    const inProgressMission: CompanyDispatchMission = {
      mission_id: 20,
      status: "in_progress",
      driver_id: 2,
    };
    const available = enrichFleetDriver(baseDriver(1, 46.2, 6.14), []);
    const inProgress = enrichFleetDriver(baseDriver(2, 46.22, 6.17, 20), [inProgressMission]);
    expect(pickPrimaryFleetDriver([available, inProgress])?.driver_id).toBe(2);
  });

  it("buildFleetLiveOverlay counts actifs et retards", () => {
    const drivers = [baseDriver(1, 46.2, 6.14, 2), baseDriver(2, 46.3, 6.15)];
    const missions: CompanyDispatchMission[] = [
      { mission_id: 1, status: "in_progress", driver_id: 2 },
      {
        mission_id: 2,
        status: "en_route",
        driver_id: 1,
        assignment_pickup_delay_minutes: 22,
      },
    ];
    const overlay = buildFleetLiveOverlay(drivers, missions);
    expect(overlay.activeDrivers).toBe(2);
    expect(overlay.missionsInProgress).toBeGreaterThanOrEqual(1);
    expect(overlay.delayedMissions).toBeGreaterThanOrEqual(1);
  });

  it("patchEnrichedFleetDrivers updates a single driver in O(1)", () => {
    const missions: CompanyDispatchMission[] = [];
    const d1 = baseDriver(1, 46.2, 6.14);
    const d2 = baseDriver(2, 46.21, 6.15);
    const full = enrichFleetDrivers([d1, d2], missions);
    const moved = { ...d2, latitude: 46.22 };
    const patched = patchEnrichedFleetDrivers(full, [d1, moved], missions);
    expect(patched).not.toBeNull();
    expect(patched?.[1]?.latitude).toBe(46.22);
    expect(patched?.[0]).toBe(full[0]);
  });

  it("buildFleetLiveOverlayFromEnriched matches buildFleetLiveOverlay", () => {
    const drivers = [baseDriver(1, 46.2, 6.14, 10), baseDriver(2, 46.3, 6.15)];
    const missions: CompanyDispatchMission[] = [
      { mission_id: 10, status: "en_route", driver_id: 1, assignment_pickup_delay_minutes: 8 },
    ];
    const enriched = enrichFleetDrivers(drivers, missions);
    expect(buildFleetLiveOverlayFromEnriched(enriched, missions)).toEqual(
      buildFleetLiveOverlay(drivers, missions)
    );
  });
});
