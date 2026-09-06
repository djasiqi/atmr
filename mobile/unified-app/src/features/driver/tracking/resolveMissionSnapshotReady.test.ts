import { describe, expect, it } from "@jest/globals";
import {
  isMissionSourceSettledPostReady,
  resolveDriverMissionSnapshot,
  resolveMissionSnapshotReady,
} from "./resolveMissionSnapshotReady";

describe("resolveMissionSnapshotReady", () => {
  const readyAt = 1_000;

  it("reste HOLD tant que SESSION_READY n’a pas ouvert", () => {
    expect(
      resolveMissionSnapshotReady({
        networkReady: false,
        status: "success",
        fetchStatus: "idle",
        dataUpdatedAt: 2_000,
        networkReadyAtMs: readyAt,
      })
    ).toBe(false);
  });

  it("ne confond pas un cache pré-READY avec une réponse serveur", () => {
    expect(
      resolveMissionSnapshotReady({
        networkReady: true,
        status: "success",
        fetchStatus: "idle",
        dataUpdatedAt: 500,
        networkReadyAtMs: readyAt,
      })
    ).toBe(false);
  });

  it("reste HOLD pendant le fetch post-READY", () => {
    expect(
      resolveMissionSnapshotReady({
        networkReady: true,
        status: "success",
        fetchStatus: "fetching",
        dataUpdatedAt: 2_000,
        networkReadyAtMs: readyAt,
      })
    ).toBe(false);
  });

  it("résout seulement après un fetch terminé post-READY", () => {
    expect(
      resolveMissionSnapshotReady({
        networkReady: true,
        status: "success",
        fetchStatus: "idle",
        dataUpdatedAt: 2_000,
        networkReadyAtMs: readyAt,
      })
    ).toBe(true);
    expect(
      resolveMissionSnapshotReady({
        networkReady: true,
        status: "error",
        fetchStatus: "idle",
        dataUpdatedAt: 2_000,
        networkReadyAtMs: readyAt,
      })
    ).toBe(true);
  });
});

describe("resolveDriverMissionSnapshot", () => {
  it("reste pending avant SESSION_READY même si une mission est déjà en cache", () => {
    expect(
      resolveDriverMissionSnapshot({
        networkReady: false,
        networkReadyGeneration: 0,
        sources: [{ id: "bookings", settledPostReady: true, missionId: 45711 }],
      })
    ).toEqual({ status: "pending" });
  });

  it("une mission positive gagne immédiatement sans attendre le quorum", () => {
    expect(
      resolveDriverMissionSnapshot({
        networkReady: true,
        networkReadyGeneration: 1,
        sources: [
          { id: "bookings", settledPostReady: false, missionId: 45711 },
          { id: "company-bookings", settledPostReady: false, missionId: null },
        ],
      })
    ).toEqual({ status: "resolved_mission", missionId: 45711 });
  });

  it("un null isolé ne résout pas — quorum incomplet = pending", () => {
    expect(
      resolveDriverMissionSnapshot({
        networkReady: true,
        networkReadyGeneration: 1,
        sources: [
          { id: "bookings", settledPostReady: true, missionId: null },
          { id: "company-bookings", settledPostReady: false, missionId: null },
        ],
      })
    ).toEqual({ status: "pending" });
  });

  it("resolved_none seulement si toutes les sources sont settled sans mission", () => {
    expect(
      resolveDriverMissionSnapshot({
        networkReady: true,
        networkReadyGeneration: 1,
        sources: [
          { id: "bookings", settledPostReady: true, missionId: null },
          { id: "today", settledPostReady: true, missionId: null },
          { id: "company-bookings", settledPostReady: true, missionId: null },
        ],
      })
    ).toEqual({ status: "resolved_none" });
  });

  it("isMissionSourceSettledPostReady refuse le cache pré-READY", () => {
    expect(
      isMissionSourceSettledPostReady({
        networkReady: true,
        status: "success",
        fetchStatus: "idle",
        dataUpdatedAt: 500,
        networkReadyAtMs: 1_000,
      })
    ).toBe(false);
  });
});
