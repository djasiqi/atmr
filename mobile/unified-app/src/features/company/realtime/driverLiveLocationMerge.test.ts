import { describe, expect, it } from "@jest/globals";
import type { CompanyDriverLiveLocation } from "../api/contracts";
import {
  FLEET_GPS_MIN_MOVE_METERS,
  haversineMeters,
  hasMeaningfulDriverLocationChange,
  mergeDriverLocationPreserveReference,
} from "./driverLiveLocationMerge";
import { applyPendingDriverUpdates } from "./useCompanyDriverLiveTracking";

const baseDriver = (overrides: Partial<CompanyDriverLiveLocation> = {}): CompanyDriverLiveLocation => ({
  driver_id: 1,
  driver_name: "Driver 1",
  latitude: 46.2,
  longitude: 6.14,
  timestamp: "2026-01-01T10:00:00.000Z",
  recorded_at: "2026-01-01T10:00:00.000Z",
  received_at: "2026-01-01T10:00:01.000Z",
  mission_id: null,
  location_status: "live",
  ...overrides,
});

describe("driverLiveLocationMerge", () => {
  it("haversineMeters returns ~0 for identical points", () => {
    expect(haversineMeters(46.2, 6.14, 46.2, 6.14)).toBeLessThan(0.01);
  });

  it("ignores micro-movements below FLEET_GPS_MIN_MOVE_METERS", () => {
    const current = baseDriver();
    const incoming = baseDriver({
      latitude: current.latitude + 0.00001,
      longitude: current.longitude + 0.00001,
    });
    const moved = haversineMeters(
      current.latitude,
      current.longitude,
      incoming.latitude,
      incoming.longitude
    );
    expect(moved).toBeLessThan(FLEET_GPS_MIN_MOVE_METERS);
    expect(hasMeaningfulDriverLocationChange(current, incoming)).toBe(false);
    expect(mergeDriverLocationPreserveReference(current, incoming)).toBe(current);
  });

  it("detects meaningful movement above threshold", () => {
    const current = baseDriver();
    const incoming = baseDriver({ latitude: current.latitude + 0.001 });
    expect(hasMeaningfulDriverLocationChange(current, incoming)).toBe(true);
    expect(mergeDriverLocationPreserveReference(current, incoming)).toBe(incoming);
  });

  it("applyPendingDriverUpdates preserves references for unchanged drivers", () => {
    const d1 = baseDriver({ driver_id: 1 });
    const d2 = baseDriver({ driver_id: 2, latitude: 46.3 });
    const currentMap = { 1: d1, 2: d2 };

    const pending = new Map<number, CompanyDriverLiveLocation>();
    pending.set(1, d1);
    pending.set(2, { ...d2, latitude: d2.latitude + 0.00001, longitude: d2.longitude + 0.00001 });

    const next = applyPendingDriverUpdates(currentMap, pending);
    expect(next).toBe(currentMap);
    expect(next[1]).toBe(d1);
    expect(next[2]).toBe(d2);
  });
});
