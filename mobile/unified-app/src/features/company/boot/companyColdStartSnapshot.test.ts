import { beforeEach, describe, expect, it } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import { resolveDriverLocationPresence } from "../components/maps/driverLocationPresence";
import { contextScopedKey } from "../../../core/cache/contextCache";
import { companyQueryKeys } from "../companyQueryKeys";
import type {
  CompanyDispatchMissionListResponse,
  CompanyDispatchRealtimeDashboard,
  CompanyDriverLiveLocationResponse,
} from "../api/contracts";
import {
  applyCompanyColdStartSnapshot,
  boundMissionsForDisk,
  buildCompanyColdStartSnapshot,
  persistDriverLocationForDisk,
  resetCompanyColdStartSnapshotForTests,
} from "./companyColdStartSnapshot";

function isoMinutesAgo(minutes: number, nowMs = Date.now()): string {
  return new Date(nowMs - minutes * 60_000).toISOString();
}

describe("companyColdStartSnapshot", () => {
  beforeEach(() => {
    resetCompanyColdStartSnapshotForTests();
  });

  it("borne la journée persistée à 50 missions", () => {
    const missions = {
      context_id: "company:42",
      missions: Array.from({ length: 80 }, (_, i) => ({ mission_id: i + 1 })),
      refreshed_at: new Date().toISOString(),
      total: 80,
      page_size: 50,
      loaded: 80,
      is_complete: true,
      next_page: 3,
    } as CompanyDispatchMissionListResponse;
    const bounded = boundMissionsForDisk(missions);
    expect(bounded.missions).toHaveLength(50);
    expect(bounded.is_complete).toBe(false);
    expect(bounded.next_page).toBe(2);
  });

  it("retire last_seen_seconds pour ne pas rajeunir le GPS", () => {
    const persisted = persistDriverLocationForDisk({
      driver_id: 7,
      latitude: 46.2,
      longitude: 6.1,
      recorded_at: isoMinutesAgo(10),
      location_status: "live",
      last_seen_seconds: 3,
    });
    expect(persisted.last_seen_seconds).toBeUndefined();
    expect(persisted.recorded_at).toBeDefined();
    expect(persisted.location_status).toBe("live");
  });

  it("une position disque périmée ne redevient jamais LIVE", () => {
    const nowMs = Date.now();
    const recordedAt = isoMinutesAgo(10, nowMs);
    const snapshot = buildCompanyColdStartSnapshot({
      contextId: "company:42",
      date: "2026-09-06",
      nowIso: new Date(nowMs).toISOString(),
      roster: {
        context_id: "company:42",
        refreshed_at: recordedAt,
        locations: [
          {
            driver_id: 7,
            latitude: 46.2,
            longitude: 6.1,
            recorded_at: recordedAt,
            location_status: "live",
            last_seen_seconds: 2,
          },
        ],
      } satisfies CompanyDriverLiveLocationResponse,
    });
    const driver = snapshot.roster?.locations[0];
    expect(driver).toBeDefined();
    expect(driver?.last_seen_seconds).toBeUndefined();
    const presence = resolveDriverLocationPresence(driver!, nowMs);
    expect(presence.presence).not.toBe("live");
    expect(presence.presence).not.toBe("recent");
    expect(presence.countedAsLocated).toBe(false);
  });

  it("hydrate le QueryClient sans inventer de fraîcheur", () => {
    const queryClient = new QueryClient();
    const dashboard = {
      context_id: "company:42",
      refreshed_at: "2026-09-06T08:00:00.000Z",
      delayed_bookings_metrics_available: true,
      delayed_bookings: 2,
      opportunities_metrics_available: true,
      opportunities: 0,
      avg_delay_minutes: 4,
    } satisfies CompanyDispatchRealtimeDashboard;
    const snapshot = buildCompanyColdStartSnapshot({
      contextId: "company:42",
      date: "2026-09-06",
      dashboard,
    });
    applyCompanyColdStartSnapshot(queryClient, snapshot);
    const key = contextScopedKey("company:42", [
      ...companyQueryKeys.dashboard("company:42"),
      "2026-09-06",
    ] as unknown[]);
    expect(queryClient.getQueryData(key)).toEqual(dashboard);
    const query = queryClient.getQueryCache().find({ queryKey: key, exact: true });
    expect(query?.state.dataUpdatedAt).toBeLessThan(Date.now());
    queryClient.clear();
  });
});
