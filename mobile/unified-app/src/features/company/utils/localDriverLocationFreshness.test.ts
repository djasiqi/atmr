import {
  applyLocalLocationFreshness,
  LOCAL_LIVE_MAX_SECONDS,
  LOCAL_RECENT_MAX_SECONDS,
  resolveLocalLocationFreshnessStatus,
} from "./localDriverLocationFreshness";

describe("localDriverLocationFreshness", () => {
  const now = Date.parse("2026-08-09T12:00:00.000Z");

  it("résout live / recent / stale selon recorded_at", () => {
    expect(
      resolveLocalLocationFreshnessStatus(
        new Date(now - (LOCAL_LIVE_MAX_SECONDS - 1) * 1000).toISOString(),
        now
      )
    ).toBe("live");
    expect(
      resolveLocalLocationFreshnessStatus(
        new Date(now - (LOCAL_LIVE_MAX_SECONDS + 1) * 1000).toISOString(),
        now
      )
    ).toBe("recent");
    expect(
      resolveLocalLocationFreshnessStatus(
        new Date(now - (LOCAL_RECENT_MAX_SECONDS + 1) * 1000).toISOString(),
        now
      )
    ).toBe("stale");
  });

  it("timestamp invalide → offline_unknown", () => {
    expect(resolveLocalLocationFreshnessStatus(null, now)).toBe("offline_unknown");
    expect(resolveLocalLocationFreshnessStatus("not-a-date", now)).toBe("offline_unknown");
  });

  it("dégrade un statut serveur live figé sans écraser tracking_display", () => {
    const aged = applyLocalLocationFreshness(
      {
        recorded_at: new Date(now - 45_000).toISOString(),
        location_status: "live",
        tracking_display_status: "live",
        last_seen_seconds: 2,
      },
      now
    );
    expect(aged.location_status).toBe("recent");
    expect(aged.tracking_display_status).toBe("live");
    expect(aged.last_seen_seconds).toBe(45);
  });

  it("ne promeut pas stale", () => {
    const aged = applyLocalLocationFreshness(
      {
        recorded_at: new Date(now - 5_000).toISOString(),
        location_status: "stale",
        tracking_display_status: "degraded_constrained",
      },
      now
    );
    expect(aged.location_status).toBe("stale");
    expect(aged.tracking_display_status).toBe("degraded_constrained");
  });
});
