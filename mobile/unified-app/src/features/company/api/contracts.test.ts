import { describe, expect, it } from "@jest/globals";
import type {
  CompanyDispatchMission,
  CompanyDelayInvalidationEvent,
  CompanyDriverLiveLocation,
  CompanyOptimizerStatus,
} from "./contracts";
import {
  validateCompanyDispatchRealtimeDashboard,
  validateCompanyDriverLocationsResponse,
  validateCompanyMissionListResponse,
  validateCompanyOptimizerStatusResponse,
} from "./contracts";

describe("company contracts", () => {
  it("supports dispatch mission payload shape", () => {
    const mission: CompanyDispatchMission = {
      mission_id: 101,
      status: "assigned",
      company_id: 42,
      updated_at: new Date().toISOString(),
    };
    expect(mission.status).toBe("assigned");
  });

  it("supports live location and delay invalidation payloads", () => {
    const location: CompanyDriverLiveLocation = {
      driver_id: 7,
      latitude: 46.5,
      longitude: 6.6,
      timestamp: new Date().toISOString(),
    };
    const invalidation: CompanyDelayInvalidationEvent = {
      event_id: "evt_1",
      mission_id: 101,
      invalidated_reason: "delay_update",
      created_at: new Date().toISOString(),
    };
    const optimizer: CompanyOptimizerStatus = {
      optimizer_enabled: true,
      optimizer_state: "running",
    };

    expect(location.driver_id).toBe(7);
    expect(invalidation.invalidated_reason).toBe("delay_update");
    expect(optimizer.optimizer_state).toBe("running");
  });

  it("validates normalized runtime payloads for Gate E", () => {
    const now = new Date().toISOString();
    expect(
      validateCompanyMissionListResponse({
        context_id: "company:42",
        refreshed_at: now,
        missions: [{ mission_id: 1, status: "assigned" }],
      })
    ).toBe(true);
    expect(
      validateCompanyDriverLocationsResponse({
        context_id: "company:42",
        refreshed_at: now,
        locations: [{ driver_id: 5, latitude: 46.5, longitude: 6.6, timestamp: now }],
      })
    ).toBe(true);
    expect(
      validateCompanyOptimizerStatusResponse({
        context_id: "company:42",
        refreshed_at: now,
        status: { optimizer_enabled: true, optimizer_state: "running" },
      })
    ).toBe(true);
    expect(
      validateCompanyDispatchRealtimeDashboard({
        context_id: "company:42",
        refreshed_at: now,
        delayed_bookings_metrics_available: true,
        delayed_bookings: 2,
        opportunities_metrics_available: true,
        opportunities: 1,
        avg_delay_minutes: 4,
      })
    ).toBe(true);
  });

  it("rejects syntactically invalid payloads", () => {
    expect(
      validateCompanyMissionListResponse({
        context_id: "company:42",
        refreshed_at: new Date().toISOString(),
        missions: [{ mission_id: "101", status: "assigned" }],
      })
    ).toBe(false);
    expect(
      validateCompanyDriverLocationsResponse({
        context_id: "company:42",
        refreshed_at: new Date().toISOString(),
        locations: [{ driver_id: 5, latitude: null, longitude: 6.6, timestamp: "2026-01-01T10:00:00Z" }],
      })
    ).toBe(false);
    // Roster sans GPS (les deux coords absentes) — accepté.
    expect(
      validateCompanyDriverLocationsResponse({
        context_id: "company:42",
        refreshed_at: new Date().toISOString(),
        locations: [{ driver_id: 5 }],
      })
    ).toBe(true);
  });

  it("rejects semantically ambiguous payloads", () => {
    expect(
      validateCompanyMissionListResponse({
        context_id: "company:42",
        refreshed_at: new Date().toISOString(),
        missions: [{ mission_id: 101, status: "unknown_status" }],
      })
    ).toBe(false);
    expect(
      validateCompanyOptimizerStatusResponse({
        context_id: "company:42",
        refreshed_at: new Date().toISOString(),
        status: { optimizer_enabled: true, optimizer_state: "paused" },
      })
    ).toBe(false);
  });
});
