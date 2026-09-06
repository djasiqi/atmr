import { describe, expect, it } from "@jest/globals";
import {
  DRIVER_DASHBOARD_HEADER_TO_STATUS_GAP,
  DRIVER_DASHBOARD_STATUS_AREA_HEIGHT,
  DRIVER_DASHBOARD_STATUS_TO_MISSION_GAP,
  resolveDriverDashboardPrimarySlot,
} from "./driverDashboardShell";
import {
  DRIVER_DASHBOARD_STATUS_AREA_HEIGHT as STATUS_AREA_HEIGHT,
  DRIVER_DASHBOARD_STATUS_LINE_HEIGHT,
} from "./driverHubStatusModel";

describe("resolveDriverDashboardPrimarySlot", () => {
  it("n’a plus de réserve StatusArea 48 px", () => {
    expect(DRIVER_DASHBOARD_STATUS_AREA_HEIGHT).toBe(STATUS_AREA_HEIGHT);
    expect(DRIVER_DASHBOARD_STATUS_LINE_HEIGHT).toBe(12);
    expect(DRIVER_DASHBOARD_STATUS_AREA_HEIGHT).toBe(DRIVER_DASHBOARD_STATUS_LINE_HEIGHT);
    expect(DRIVER_DASHBOARD_STATUS_AREA_HEIGHT).toBeLessThan(24);
  });

  it("ligne statut → mission = 24–32 px, sans bande vide", () => {
    expect(DRIVER_DASHBOARD_HEADER_TO_STATUS_GAP).toBe(0);
    expect(DRIVER_DASHBOARD_STATUS_TO_MISSION_GAP).toBeGreaterThanOrEqual(24);
    expect(DRIVER_DASHBOARD_STATUS_TO_MISSION_GAP).toBeLessThanOrEqual(32);
  });

  it("reste sur pending tant que les missions ne sont pas prêtes", () => {
    expect(resolveDriverDashboardPrimarySlot({ pending: true, hasActiveMission: false })).toBe(
      "pending"
    );
    expect(resolveDriverDashboardPrimarySlot({ pending: true, hasActiveMission: true })).toBe(
      "pending"
    );
  });

  it("remplit le même slot : mission ou idle", () => {
    expect(resolveDriverDashboardPrimarySlot({ pending: false, hasActiveMission: true })).toBe(
      "mission"
    );
    expect(resolveDriverDashboardPrimarySlot({ pending: false, hasActiveMission: false })).toBe(
      "idle"
    );
  });
});
