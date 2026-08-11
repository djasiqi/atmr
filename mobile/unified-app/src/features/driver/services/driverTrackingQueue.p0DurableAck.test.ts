import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { trackingQueueStore } from "./trackingQueueStore";

const mockSendDriverLocation = jest.fn();
const mockSendDriverLocationBatch = jest.fn();
const mockIsDriverSocketReady = jest.fn().mockReturnValue(false);

jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: {
    getItem: jest.fn(async () => null),
    setItem: jest.fn(async () => undefined),
  },
}));

jest.mock("../api/driverHttp", () => ({
  sendDriverLocation: (...args: unknown[]) => mockSendDriverLocation(...args),
}));

jest.mock("./trackingSessionsApi", () => ({
  registerTrackingSession: jest.fn(async () => ({
    session_generation: 1,
    first_sequence_id: 1,
  })),
  fetchTrackingWatermark: jest.fn(async () => ({
    last_acked_sequence_id: 0,
    tracking_session_id: null,
  })),
}));

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    sendDriverLocationBatch: (...args: unknown[]) => mockSendDriverLocationBatch(...args),
    isDriverSocketReady: () => mockIsDriverSocketReady(),
  },
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
}));

jest.mock("./trackingContextLease", () => ({
  readTrackingContextLease: async () => ({
    state: "driver_active",
    contextId: "driver:1",
    driverId: 1,
    sessionGenerationId: 1,
    trackingGenerationId: "trk-test",
    trackingIdentityId: "driver:1:company:1",
    updatedAt: Date.now(),
  }),
  leaseAllowsTransport: (lease: { state?: string } | null) =>
    Boolean(lease && lease.state === "driver_active"),
  leaseAllowsCapture: () => true,
}));

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: jest.fn((key: unknown) => key === "tracking_real_ack_semantics_enabled"),
}));

jest.mock("./socketBatchPacing", () => ({
  canEmitSocketBatchNow: () => true,
  getSocketBatchCooldownRemainingMs: () => 0,
  recordSocketBatchSent: jest.fn(),
  recordSocketBatchRateLimited: jest.fn(),
}));

 
const { driverTrackingQueue } = require("./driverTrackingQueue") as typeof import("./driverTrackingQueue");

describe("P0 file GPS — preuve durable + restart", () => {
  beforeEach(async () => {
    jest.useFakeTimers({ advanceTimers: true });
    (isFeatureEnabled as jest.Mock).mockImplementation(
      (key: unknown) => key === "tracking_real_ack_semantics_enabled"
    );
    mockSendDriverLocation.mockReset();
    mockSendDriverLocationBatch.mockReset();
    mockIsDriverSocketReady.mockReturnValue(false);
    trackingQueueStore._resetMemoryForTests();
    await driverTrackingQueue.resetForTests();
    await driverTrackingQueue.clearContextInactiveGate("test_setup");
  });

  afterEach(async () => {
    trackingQueueStore._resetMemoryForTests();
    await driverTrackingQueue.resetForTests();
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it("ACK HTTP persisted_sync retire durablement ; restart ne ressuscite pas", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 9,
      appState: "active",
      locationMode: "mission_live",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: 9,
        locationMode: "mission_live",
      },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.persistedEventIds).toContain(item.id);
    expect(flush.queueDepth).toBe(0);

    await driverTrackingQueue.resetForTests();
    // Simule redémarrage : listActive ne doit pas renvoyer l'item persisted
    const active = await trackingQueueStore.listActive();
    expect(active.find((r) => r.locationEventId === item.id)).toBeUndefined();
  });

  it("socket ACK ne retire pas de la file active", async () => {
    (isFeatureEnabled as jest.Mock).mockImplementation(
      (key: unknown) =>
        key === "tracking_real_ack_semantics_enabled" ||
        key === "tracking_socket_gps_ingest_enabled"
    );
    mockIsDriverSocketReady.mockReturnValue(true);
    mockSendDriverLocationBatch.mockReturnValue(true);
    const item = await driverTrackingQueue.enqueue({
      missionId: 3,
      appState: "active",
      locationMode: "mission_live",
      payload: {
        latitude: 46,
        longitude: 6,
        missionId: 3,
        locationMode: "mission_live",
      },
    });
    await driverTrackingQueue.flush({ networkProfile: "normal" });
    const acked = await driverTrackingQueue.markBackendAckedByIds([item.id]);
    expect(acked).toBe(1);
    expect((await driverTrackingQueue.getSnapshot()).queueDepth).toBe(1);
    const rows = await trackingQueueStore.listActive();
    expect(rows.some((r) => r.locationEventId === item.id)).toBe(true);
  });

  it("applyIngested : échec markState ne mute pas la mémoire et planifie un retry", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 11,
      appState: "active",
      locationMode: "mission_live",
      payload: {
        latitude: 46.1,
        longitude: 6.1,
        missionId: 11,
        locationMode: "mission_live",
      },
    });
    const spy = jest
      .spyOn(trackingQueueStore, "markState")
      .mockRejectedValueOnce(new Error("sqlite_busy"));
    await expect(driverTrackingQueue.applyIngestedEventIds([item.id])).rejects.toThrow(
      "sqlite_busy"
    );
    expect(item.persistState ?? "non_ingested").toBe("non_ingested");
    const row = (await trackingQueueStore.listActive()).find(
      (r) => r.locationEventId === item.id
    );
    expect(row?.state).toBe("non_ingested");
    spy.mockRestore();
  });

  it("tombstone + gap atomiques via transitionWithGaps (store)", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 12,
      appState: "active",
      locationMode: "mission_live",
      payload: {
        latitude: 46.2,
        longitude: 6.2,
        missionId: 12,
        locationMode: "mission_live",
      },
    });
    const n = await driverTrackingQueue.tombstoneByIds([item.id], "session_conflict");
    expect(n).toBe(1);
    expect((await trackingQueueStore.listActive()).length).toBe(0);
    const gaps = trackingQueueStore._listGapsForTests();
    expect(gaps.some((g) => g.reason === "session_conflict")).toBe(true);
  });

  it("kill→restart : store mémoire conserve ingested ; rehydrate sans faux persisted", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 13,
      appState: "active",
      locationMode: "mission_live",
      payload: {
        latitude: 46.3,
        longitude: 6.3,
        missionId: 13,
        locationMode: "mission_live",
      },
    });
    await driverTrackingQueue.applyIngestedEventIds([item.id]);
    const before = await trackingQueueStore.listActive();
    expect(before.find((r) => r.locationEventId === item.id)?.state).toBe(
      "ingested_non_persisted"
    );

    // Kill process : vide le miroir mémoire, conserve le store.
    await driverTrackingQueue.resetForTests({ keepStore: true });
    expect((await driverTrackingQueue.getSnapshot()).queueDepth).toBe(1);
    const after = await trackingQueueStore.listActive();
    const row = after.find((r) => r.locationEventId === item.id);
    expect(row?.state).toBe("ingested_non_persisted");
    expect(row?.state).not.toBe("persisted");
  });

  it.each([
    ["rejected", "rejected"],
    ["ignored", "tombstone"],
    ["stale", "tombstone"],
  ] as const)(
    "ACK HTTP %s → finalizeTerminal ; kill→restart ne ressuscite pas",
    async (ackStatus, expectedTerminalAbsentFromActive) => {
      void expectedTerminalAbsentFromActive;
      const item = await driverTrackingQueue.enqueue({
        missionId: 14,
        appState: "active",
        locationMode: "mission_live",
        payload: {
          latitude: 46.4,
          longitude: 6.4,
          missionId: 14,
          locationMode: "mission_live",
        },
      });
      mockSendDriverLocation.mockResolvedValue({
        ack_status: ackStatus,
        location_event_id: item.id,
        tracking_event_id: item.id,
      });
      const flush = await driverTrackingQueue.flush({
        forceHttpFallback: true,
        networkProfile: "normal",
      });
      expect(flush.dropped).toBe(1);
      expect(flush.queueDepth).toBe(0);
      expect((await trackingQueueStore.listActive()).length).toBe(0);
      if (ackStatus === "stale" || ackStatus === "ignored") {
        expect(
          trackingQueueStore._listGapsForTests().some((g) => g.reason === `ack_${ackStatus}`)
        ).toBe(true);
      }

      await driverTrackingQueue.resetForTests({ keepStore: true });
      const active = await trackingQueueStore.listActive();
      expect(active.find((r) => r.locationEventId === item.id)).toBeUndefined();
      expect((await driverTrackingQueue.getSnapshot()).queueDepth).toBe(0);
    }
  );

  it("ACK HTTP inattendu → retry_pending (pas de drop mémoire-only)", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 15,
      appState: "active",
      locationMode: "mission_live",
      payload: {
        latitude: 46.5,
        longitude: 6.5,
        missionId: 15,
        locationMode: "mission_live",
      },
    });
    mockSendDriverLocation.mockResolvedValue({
      // Statut hors contrat pour forcer la branche unexpected (pas de drop mémoire-only).
      ack_status: "totally_unknown" as unknown as "accepted",
      location_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.dropped).toBe(0);
    expect(flush.retried).toBe(1);
    expect(flush.queueDepth).toBe(1);
    const row = (await trackingQueueStore.listActive()).find(
      (r) => r.locationEventId === item.id
    );
    expect(row?.state).toBe("non_ingested");
    expect(row?.deliveryState).toBe("retry_pending");
  });
});
