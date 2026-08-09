import { describe, expect, it, jest } from "@jest/globals";

jest.mock("../hooks", () => ({
  useActiveCompanyContextId: () => "company:42",
  useCompanyDriversLocationsSnapshotQuery: () => ({
    data: null,
    isLoading: false,
    error: null,
    refetch: async () => undefined,
  }),
}));

jest.mock("../../../core/realtime/contextRealtimeRouter", () => ({
  contextRealtimeRouter: {
    subscribe: () => () => undefined,
  },
}));

 
 
const {
  normalizeRealtimeLocation,
  shouldReplaceDriverLocation,
  applyPendingDriverUpdates,
  REALTIME_FLUSH_MS,
  MAX_BATCH_AGE_MS,
  MAP_SILENCE_RESYNC_MS,
  shouldTriggerMapSilenceResync,
} = require("./useCompanyDriverLiveTracking");

describe("company live drivers merge policy", () => {
  it("exposes bounded batch flush windows", () => {
    expect(REALTIME_FLUSH_MS).toBeGreaterThan(0);
    expect(MAX_BATCH_AGE_MS).toBeGreaterThanOrEqual(REALTIME_FLUSH_MS);
  });

  it("normalizes incoming realtime payload variants", () => {
    const normalized = normalizeRealtimeLocation({
      driver_id: "42",
      lat: 46.5,
      lng: 6.6,
      timestamp: "2026-01-01T10:00:00.000Z",
      recorded_at: "2026-01-01T10:00:00.000Z",
      received_at: "2026-01-01T10:00:02.000Z",
    });
    expect(normalized).toEqual(
      expect.objectContaining({
        driver_id: 42,
        latitude: 46.5,
        longitude: 6.6,
      })
    );
  });

  it("prioritizes lng over lon when both are present (extractLongitude order)", () => {
    const normalized = normalizeRealtimeLocation({
      driver_id: "42",
      lat: 46.5,
      lon: 6.8,
      lng: 6.6,
      timestamp: "2026-01-01T10:00:00.000Z",
    });
    expect(normalized).toEqual(
      expect.objectContaining({
        longitude: 6.6,
      })
    );
  });

  it("marks stale status when last_seen_seconds exceeds threshold", () => {
    const normalized = normalizeRealtimeLocation({
      driver_id: "42",
      lat: 46.5,
      lon: 6.6,
      timestamp: "2026-01-01T10:00:00.000Z",
      last_seen_seconds: 180,
    });
    expect(normalized).toEqual(
      expect.objectContaining({
        last_seen_seconds: 180,
        location_status: "stale",
      })
    );
  });

  it("accepts newer recorded_at and rejects stale optimistic overlay", () => {
    const current = {
      driver_id: 7,
      latitude: 46.5,
      longitude: 6.6,
      timestamp: "2026-01-01T10:00:01.000Z",
      recorded_at: "2026-01-01T10:00:01.000Z",
      received_at: "2026-01-01T10:00:02.000Z",
    };
    const staleSocket = {
      driver_id: 7,
      latitude: 46.6,
      longitude: 6.7,
      timestamp: "2026-01-01T09:59:58.000Z",
      recorded_at: "2026-01-01T09:59:58.000Z",
      received_at: "2026-01-01T10:00:03.000Z",
    };
    const freshSnapshot = {
      driver_id: 7,
      latitude: 46.8,
      longitude: 6.9,
      timestamp: "2026-01-01T10:00:05.000Z",
      recorded_at: "2026-01-01T10:00:05.000Z",
      received_at: "2026-01-01T10:00:05.000Z",
    };

    expect(shouldReplaceDriverLocation(current, staleSocket)).toBe(false);
    expect(shouldReplaceDriverLocation(current, freshSnapshot)).toBe(true);
  });

  it("falls back to timestamp then received_at when recorded_at ties", () => {
    const current = {
      driver_id: 7,
      latitude: 46.5,
      longitude: 6.6,
      timestamp: "2026-01-01T10:00:00.000Z",
      recorded_at: "2026-01-01T10:00:00.000Z",
      received_at: "2026-01-01T10:00:01.000Z",
    };
    const newerTimestamp = {
      ...current,
      timestamp: "2026-01-01T10:00:02.000Z",
      received_at: "2026-01-01T10:00:03.000Z",
    };
    const sameTimestampNewerReceived = {
      ...current,
      received_at: "2026-01-01T10:00:04.000Z",
    };

    expect(shouldReplaceDriverLocation(current, newerTimestamp)).toBe(true);
    expect(shouldReplaceDriverLocation(current, sameTimestampNewerReceived)).toBe(true);
  });

  // ─── observability_only anti-régression ───────────────────────────────────

  it("rejette une position observability_only plus ancienne que la position live courante", () => {
    const livePosition = {
      driver_id: 7,
      latitude: 46.5,
      longitude: 6.6,
      timestamp: "2026-01-01T10:00:05.000Z",
      recorded_at: "2026-01-01T10:00:05.000Z",
      received_at: "2026-01-01T10:00:06.000Z",
      accepted_observability_only: false,
    };
    const staleObsOnly = {
      driver_id: 7,
      latitude: 46.4,
      longitude: 6.5,
      timestamp: "2026-01-01T10:00:00.000Z",
      recorded_at: "2026-01-01T10:00:00.000Z",
      received_at: "2026-01-01T10:00:08.000Z", // reçu après, mais position plus ancienne
      accepted_observability_only: true,
    };

    expect(shouldReplaceDriverLocation(livePosition, staleObsOnly)).toBe(false);
  });

  it("accepte une position observability_only plus récente que la position live courante", () => {
    const livePosition = {
      driver_id: 7,
      latitude: 46.5,
      longitude: 6.6,
      timestamp: "2026-01-01T10:00:00.000Z",
      recorded_at: "2026-01-01T10:00:00.000Z",
      received_at: "2026-01-01T10:00:01.000Z",
      accepted_observability_only: false,
    };
    const freshObsOnly = {
      driver_id: 7,
      latitude: 46.6,
      longitude: 6.7,
      timestamp: "2026-01-01T10:00:10.000Z",
      recorded_at: "2026-01-01T10:00:10.000Z",
      received_at: "2026-01-01T10:00:11.000Z",
      accepted_observability_only: true,
    };

    expect(shouldReplaceDriverLocation(livePosition, freshObsOnly)).toBe(true);
  });

  it("ne bloque pas le remplacement si la position courante est aussi observability_only", () => {
    const obsOnlyCurrent = {
      driver_id: 7,
      latitude: 46.5,
      longitude: 6.6,
      timestamp: "2026-01-01T10:00:05.000Z",
      recorded_at: "2026-01-01T10:00:05.000Z",
      received_at: "2026-01-01T10:00:06.000Z",
      accepted_observability_only: true,
    };
    const olderObsOnly = {
      driver_id: 7,
      latitude: 46.4,
      longitude: 6.5,
      timestamp: "2026-01-01T10:00:01.000Z",
      recorded_at: "2026-01-01T10:00:01.000Z",
      received_at: "2026-01-01T10:00:08.000Z",
      accepted_observability_only: true,
    };

    // Cas obs_only vs obs_only : la comparaison timestamp normale s'applique
    expect(shouldReplaceDriverLocation(obsOnlyCurrent, olderObsOnly)).toBe(false);
  });

  it("accepte toujours si aucun état courant (première position)", () => {
    const incoming = {
      driver_id: 7,
      latitude: 46.5,
      longitude: 6.6,
      timestamp: "2026-01-01T10:00:00.000Z",
      accepted_observability_only: true,
    };
    expect(shouldReplaceDriverLocation(undefined, incoming)).toBe(true);
  });

  it("applyPendingDriverUpdates ne recrée pas la map si aucun changement significatif", () => {
    const current = {
      driver_id: 7,
      latitude: 46.5,
      longitude: 6.6,
      timestamp: "2026-01-01T10:00:05.000Z",
      recorded_at: "2026-01-01T10:00:05.000Z",
      received_at: "2026-01-01T10:00:06.000Z",
    };
    const currentMap = { 7: current };
    const pending = new Map();
    pending.set(7, {
      ...current,
      latitude: current.latitude + 0.00001,
      longitude: current.longitude + 0.00001,
    });
    const next = applyPendingDriverUpdates(currentMap, pending);
    expect(next).toBe(currentMap);
  });

  it("applyPendingDriverUpdates remplace uniquement les chauffeurs modifiés", () => {
    const d1 = {
      driver_id: 1,
      latitude: 46.5,
      longitude: 6.6,
      timestamp: "2026-01-01T10:00:00.000Z",
      recorded_at: "2026-01-01T10:00:00.000Z",
      received_at: "2026-01-01T10:00:01.000Z",
    };
    const d2 = {
      driver_id: 2,
      latitude: 46.6,
      longitude: 6.7,
      timestamp: "2026-01-01T10:00:00.000Z",
      recorded_at: "2026-01-01T10:00:00.000Z",
      received_at: "2026-01-01T10:00:01.000Z",
    };
    const currentMap = { 1: d1, 2: d2 };
    const pending = new Map();
    pending.set(2, {
      ...d2,
      latitude: 46.8,
      longitude: 6.9,
      timestamp: "2026-01-01T10:00:10.000Z",
      recorded_at: "2026-01-01T10:00:10.000Z",
    });
    const next = applyPendingDriverUpdates(currentMap, pending);
    expect(next).not.toBe(currentMap);
    expect(next[1]).toBe(d1);
    expect(next[2].latitude).toBe(46.8);
  });

  it("watchdog silence : refetch si aucun event socket ni succès snapshot depuis 25 s", () => {
    const now = 500_000;
    expect(
      shouldTriggerMapSilenceResync({
        nowMs: now,
        lastRealtimeEventAtMs: now - 10_000,
        lastWatchdogSuccessAtMs: now - 10_000,
      })
    ).toBe(false);
    expect(
      shouldTriggerMapSilenceResync({
        nowMs: now,
        lastRealtimeEventAtMs: now - MAP_SILENCE_RESYNC_MS - 1,
        lastWatchdogSuccessAtMs: now - 1_000,
      })
    ).toBe(false);
    expect(
      shouldTriggerMapSilenceResync({
        nowMs: now,
        lastRealtimeEventAtMs: now - MAP_SILENCE_RESYNC_MS - 1,
        lastWatchdogSuccessAtMs: now - MAP_SILENCE_RESYNC_MS - 1,
      })
    ).toBe(true);
  });
});
