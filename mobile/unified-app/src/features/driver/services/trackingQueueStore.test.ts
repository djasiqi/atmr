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
    // Re-upsert inchangé après 48h simulées — toujours présent
    await trackingQueueStore.upsert({
      ...(active.find((r) => r.locationEventId === "e1")!),
      queuedAt: queuedAt - 86_400_000,
    });
    const again = await trackingQueueStore.listActive();
    expect(again.some((r) => r.locationEventId === "e1")).toBe(true);
  });
});
