/**
 * Matrice P0-F UI — présence GPS flotte.
 */

import { describe, expect, it } from "@jest/globals";

import {
  applyLocalLocationFreshness,
  LOCAL_LIVE_MAX_SECONDS,
  LOCAL_RECENT_MAX_SECONDS,
} from "../../utils/localDriverLocationFreshness";
import {
  formatDriverLocationPresenceLabel,
  matchesFleetGpsFilter,
  resolveDriverLocationPresence,
  type FleetDriverPresenceInput,
} from "./driverLocationPresence";
import { resolveFleetOperationalStatus } from "./fleetMapLogic";

function driver(partial: Partial<FleetDriverPresenceInput> & { driver_id?: number }): FleetDriverPresenceInput {
  return {
    driver_id: partial.driver_id ?? 1,
    latitude: Object.prototype.hasOwnProperty.call(partial, "latitude") ? partial.latitude : 46.2,
    longitude: Object.prototype.hasOwnProperty.call(partial, "longitude") ? partial.longitude : 6.14,
    ...partial,
  };
}

function recordedAt(ageSeconds: number, nowMs: number): string {
  return new Date(nowMs - ageSeconds * 1000).toISOString();
}

describe("resolveDriverLocationPresence", () => {
  const now = Date.parse("2026-08-11T12:00:00.000Z");

  it("backend live + 10 s → live counted", () => {
    const view = resolveDriverLocationPresence(
      driver({ location_status: "live", recorded_at: recordedAt(10, now) }),
      now
    );
    expect(view.presence).toBe("live");
    expect(view.countedAsLocated).toBe(true);
    expect(view.showMarker).toBe(true);
  });

  it("backend live + 70 s → recent counted", () => {
    const view = resolveDriverLocationPresence(
      driver({ location_status: "live", recorded_at: recordedAt(70, now) }),
      now
    );
    expect(view.presence).toBe("recent");
    expect(view.countedAsLocated).toBe(true);
    expect(70).toBeGreaterThan(LOCAL_LIVE_MAX_SECONDS);
    expect(70).toBeLessThanOrEqual(LOCAL_RECENT_MAX_SECONDS);
  });

  it("backend live + 180 s → stale not counted", () => {
    const view = resolveDriverLocationPresence(
      driver({ location_status: "live", recorded_at: recordedAt(180, now) }),
      now
    );
    expect(view.presence).toBe("stale");
    expect(view.countedAsLocated).toBe(false);
  });

  it("backend stale + 10 s → stale (non-promotion)", () => {
    const view = resolveDriverLocationPresence(
      driver({ location_status: "stale", recorded_at: recordedAt(10, now) }),
      now
    );
    expect(view.presence).toBe("stale");
    expect(view.countedAsLocated).toBe(false);
  });

  it("db_fallback / company_fallback → last_known", () => {
    expect(
      resolveDriverLocationPresence(
        driver({ position_source: "db_fallback", location_status: "live", recorded_at: recordedAt(5, now) }),
        now
      ).presence
    ).toBe("last_known");
    expect(
      resolveDriverLocationPresence(
        driver({ position_source: "company_fallback", location_status: "live" }),
        now
      ).countedAsLocated
    ).toBe(false);
  });

  it("offline + coords → last_known ; sans coords → offline_unknown", () => {
    expect(
      resolveDriverLocationPresence(
        driver({ location_status: "offline", latitude: 46.2, longitude: 6.1 }),
        now
      ).presence
    ).toBe("last_known");
    const none = resolveDriverLocationPresence(
      driver({ location_status: "offline", latitude: null, longitude: null }),
      now
    );
    expect(none.presence).toBe("offline_unknown");
    expect(none.showMarker).toBe(false);
  });

  it("tracking_display degraded_constrained + offline + coords → last_known, contrainte séparée", () => {
    const view = resolveDriverLocationPresence(
      driver({
        location_status: "offline",
        tracking_display_status: "degraded_constrained",
        recorded_at: recordedAt(15, now),
      }),
      now
    );
    expect(view.presence).toBe("last_known");
  });

  it("tracking_display live + location_status stale + âge 10s → stale", () => {
    const view = resolveDriverLocationPresence(
      driver({
        location_status: "stale",
        tracking_display_status: "live",
        recorded_at: recordedAt(10, now),
      }),
      now
    );
    expect(view.presence).toBe("stale");
  });

  it("location_status absent + tracking live + âge 70s → recent", () => {
    const view = resolveDriverLocationPresence(
      driver({
        location_status: null,
        tracking_display_status: "live",
        recorded_at: recordedAt(70, now),
      }),
      now
    );
    expect(view.presence).toBe("recent");
  });

  it("tracking_display stale + âge 10s (location absent) → stale", () => {
    const view = resolveDriverLocationPresence(
      driver({
        location_status: null,
        tracking_display_status: "stale",
        recorded_at: recordedAt(10, now),
      }),
      now
    );
    expect(view.presence).toBe("stale");
  });

  it("degraded_constrained + âge inconnu + coords → last_known", () => {
    const view = resolveDriverLocationPresence(
      driver({
        location_status: null,
        tracking_display_status: "degraded_constrained",
        recorded_at: null,
        timestamp: null,
        last_seen_seconds: null,
      }),
      now
    );
    expect(view.presence).toBe("last_known");
  });

  it("busy + last_known → métier busy, presence last_known", () => {
    const d = {
      driver_id: 1,
      latitude: 46.2,
      longitude: 6.1,
      timestamp: recordedAt(600, now),
      location_status: "last_known" as const,
      status: "busy",
    };
    expect(resolveFleetOperationalStatus(d, null)).toBe("busy");
    expect(resolveDriverLocationPresence(d, now).presence).toBe("last_known");
  });

  it("prod 1/7 : live + stale + 5 offline_unknown", () => {
    const roster = [
      driver({ driver_id: 1, location_status: "live", recorded_at: recordedAt(10, now) }),
      driver({ driver_id: 2, location_status: "stale", recorded_at: recordedAt(300, now) }),
      driver({ driver_id: 3, latitude: null, longitude: null }),
      driver({ driver_id: 4, latitude: null, longitude: null }),
      driver({ driver_id: 5, latitude: null, longitude: null }),
      driver({ driver_id: 6, latitude: null, longitude: null }),
      driver({ driver_id: 7, latitude: null, longitude: null }),
    ];
    const liveCount = roster.filter((d) => resolveDriverLocationPresence(d, now).countedAsLocated).length;
    const spatial = roster.filter((d) => resolveDriverLocationPresence(d, now).showMarker);
    expect(roster.length).toBe(7);
    expect(liveCount).toBe(1);
    expect(spatial).toHaveLength(2);
    expect(matchesFleetGpsFilter("stale", "live")).toBe(false);
    expect(matchesFleetGpsFilter("stale", "not_recent")).toBe(true);
  });

  it("formatDriverLocationPresenceLabel", () => {
    expect(
      formatDriverLocationPresenceLabel({ presence: "live", ageSeconds: 8 })
    ).toBe("En direct · il y a 8 s");
    expect(
      formatDriverLocationPresenceLabel({ presence: "stale", ageSeconds: 280 })
    ).toMatch(/Position périmée/);
    expect(
      formatDriverLocationPresenceLabel({ presence: "offline_unknown", ageSeconds: null })
    ).toBe("Aucune position disponible");
  });
});

describe("applyLocalLocationFreshness — dégrade sans promotion", () => {
  const now = Date.parse("2026-08-11T12:00:00.000Z");

  it("ne promeut pas stale + 10s", () => {
    const aged = applyLocalLocationFreshness(
      {
        location_status: "stale",
        tracking_display_status: "degraded_constrained",
        position_source: "db_fallback",
        recorded_at: recordedAt(10, now),
      },
      now
    );
    expect(aged.location_status).toBe("stale");
    expect(aged.tracking_display_status).toBe("degraded_constrained");
    expect(aged.position_source).toBe("db_fallback");
  });

  it("dégrade live + 180s → stale", () => {
    const aged = applyLocalLocationFreshness(
      {
        location_status: "live",
        tracking_display_status: "live",
        recorded_at: recordedAt(180, now),
      },
      now
    );
    expect(aged.location_status).toBe("stale");
    expect(aged.tracking_display_status).toBe("live");
  });
});
