import { beforeEach, describe, expect, it } from "@jest/globals";
import { trackingQueueStore } from "./trackingQueueStore";

describe("trackingQueueStore (Annexe A.4)", () => {
  beforeEach(() => {
    trackingQueueStore._resetMemoryForTests();
  });

  it("conserve ingested_non_persisted après upsert (pas d'expiration silencieuse)", async () => {
    const queuedAt = Date.now() - 86_400_000 * 2;
    await trackingQueueStore.upsert({
      locationEventId: "e1",
      trackingSessionId: "s1",
      sessionGeneration: 1,
      sequenceId: 1,
      payloadJson: "{}",
      state: "ingested_non_persisted",
      queuedAt,
      lastAttemptAt: null,
      retryCount: 0,
      deliveryState: "backend_acked",
      missionId: null,
      locationMode: "mission_live",
      batchId: "b1",
      positionId: "p1",
      appState: "active",
      lastError: null,
      ackedAt: Date.now(),
    });
    const active = await trackingQueueStore.listActive();
    expect(active.some((r) => r.locationEventId === "e1")).toBe(true);
    expect(active.find((r) => r.locationEventId === "e1")?.state).toBe(
      "ingested_non_persisted"
    );
    await trackingQueueStore.upsert({
      ...(active.find((r) => r.locationEventId === "e1")!),
      queuedAt: queuedAt - 86_400_000,
    });
    const again = await trackingQueueStore.listActive();
    expect(again.some((r) => r.locationEventId === "e1")).toBe(true);
  });

  it("expose durable_unavailable après force test", () => {
    trackingQueueStore._forceDurableUnavailableForTests();
    expect(trackingQueueStore.isDurableUnavailable()).toBe(true);
    expect(trackingQueueStore.isDurableBackendAvailable()).toBe(false);
  });

  it("importLegacyOnce conserve les lignes déjà importées (mémoire Jest)", async () => {
    await trackingQueueStore.importLegacyOnce([
      {
        locationEventId: "m1",
        trackingSessionId: "s1",
        sessionGeneration: 1,
        sequenceId: 1,
        payloadJson: "{}",
        state: "non_ingested",
        queuedAt: 1,
        lastAttemptAt: null,
        retryCount: 0,
        deliveryState: "queued",
        missionId: null,
        locationMode: "mission_live",
        batchId: "b",
        positionId: "p",
        appState: "active",
        lastError: null,
        ackedAt: null,
      },
    ]);
    const active = await trackingQueueStore.listActive();
    expect(active.some((r) => r.locationEventId === "m1")).toBe(true);
  });
});
