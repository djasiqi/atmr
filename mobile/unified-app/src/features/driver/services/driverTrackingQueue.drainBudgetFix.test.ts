import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { trackingQueueStore } from "./trackingQueueStore";
import {
  MAX_AWAITING_DURABLE_ACK_ATTEMPTS,
  MAX_HEAD_SOFT_ACK_BLOCK_MS,
  softAckRetirementReason,
} from "./driverTrackingQueueSoftAckRetirement";

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
const { emitDriverTelemetry } = require("../../../core/observability/driverTelemetry") as {
  emitDriverTelemetry: jest.Mock;
};

describe("JZ-R1-DRAIN-BUDGET-FIX-23 soft-ACK head retirement", () => {
  beforeEach(async () => {
    jest.useFakeTimers({ advanceTimers: true });
    (isFeatureEnabled as jest.Mock).mockImplementation(
      (key: unknown) => key === "tracking_real_ack_semantics_enabled"
    );
    mockSendDriverLocation.mockReset();
    mockSendDriverLocationBatch.mockReset();
    mockIsDriverSocketReady.mockReturnValue(false);
    emitDriverTelemetry.mockClear();
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

  it("helper DOC bounds: attempts + âge soft-ACK", () => {
    expect(MAX_AWAITING_DURABLE_ACK_ATTEMPTS).toBe(20);
    expect(MAX_HEAD_SOFT_ACK_BLOCK_MS).toBe(15 * 60 * 1000);
    const now = 1_000_000;
    expect(
      softAckRetirementReason(
        {
          queuedAt: now - 60_000,
          retryCount: 1,
          persistState: "ingested_non_persisted",
          lastError: "awaiting_durable_ack",
        },
        now,
        86_400_000
      )
    ).toBeNull();
    expect(
      softAckRetirementReason(
        {
          queuedAt: now - 60_000,
          retryCount: 20,
          persistState: "ingested_non_persisted",
          lastError: "awaiting_durable_ack",
        },
        now,
        86_400_000
      )
    ).toBe("soft_ack_retry_exhausted");
    expect(
      softAckRetirementReason(
        {
          queuedAt: now - MAX_HEAD_SOFT_ACK_BLOCK_MS,
          retryCount: 1,
          persistState: "ingested_non_persisted",
          lastError: "awaiting_durable_ack",
        },
        now,
        86_400_000
      )
    ).toBe("ingested_non_persisted_expired");
    expect(
      softAckRetirementReason(
        {
          queuedAt: now - 60_000,
          retryCount: 5,
          persistState: "non_ingested",
          lastError: "send_failed",
          deliveryState: "retry_pending",
        },
        now,
        86_400_000
      )
    ).toBeNull();
  });

  it("A — ACK persisted_sync retire normalement", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 1,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 46.2, longitude: 6.1, missionId: 1, locationMode: "mission_live" },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.persistedEventIds).toContain(item.id);
    expect(flush.queueDepth).toBe(0);
    expect(flush.exitReason).toBe("HTTP_DISPATCHED");
  });

  it("B — premier ingested_non_persisted reste retryable (pas de retraite)", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 2,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 46.2, longitude: 6.1, missionId: 2, locationMode: "mission_live" },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "ingested_non_persisted",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.sent).toBe(1);
    expect(flush.queueDepth).toBe(1);
    const active = await trackingQueueStore.listActive();
    expect(active.find((r) => r.locationEventId === item.id)?.state).toBe(
      "ingested_non_persisted"
    );
    expect(
      emitDriverTelemetry.mock.calls.some((c) => c[0] === "tracking.queue.head_retired")
    ).toBe(false);
  });

  it("C — soft-ACK sous la borne : retries + budget compté", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 3,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 46.2, longitude: 6.1, missionId: 3, locationMode: "mission_live" },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "ingested_non_persisted",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(driverTrackingQueue.getDrainedInCurrentMinuteForTests()).toBe(1);
    await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(driverTrackingQueue.getDrainedInCurrentMinuteForTests()).toBe(2);
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.queueDepth).toBe(1);
  });

  it("D — borne âge atteinte → head expired + raison", async () => {
    const now = Date.now();
    const a = await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_old_a",
      sequenceId: 1,
      queuedAt: now - MAX_HEAD_SOFT_ACK_BLOCK_MS - 1000,
      persistState: "ingested_non_persisted",
      lastError: "awaiting_durable_ack",
      retryCount: 2,
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: "never",
      tracking_event_id: "never",
    });
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(
      emitDriverTelemetry.mock.calls.some(
        (c) =>
          c[0] === "tracking.queue.head_retired" &&
          (c[1] as { reason?: string }).reason === "ingested_non_persisted_expired"
      )
    ).toBe(true);
    expect((await trackingQueueStore.listActive()).find((r) => r.locationEventId === a.id)).toBeUndefined();
    expect(flush.exitReason).toBe("QUEUE_EMPTY");
  });

  it("E+clé — après retraite A, B devient head et SEND possible", async () => {
    const now = Date.now();
    const a = await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_sticky_a",
      sequenceId: 1,
      queuedAt: now - MAX_HEAD_SOFT_ACK_BLOCK_MS - 5000,
      persistState: "ingested_non_persisted",
      lastError: "awaiting_durable_ack",
    });
    const b = await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_fresh_b",
      sequenceId: 2,
      queuedAt: now - 1000,
      persistState: "non_ingested",
      deliveryState: "queued",
      lastError: null,
      retryCount: 0,
    });
    mockSendDriverLocation.mockImplementation(async (payload: { trackingEventId?: string }) => ({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: payload.trackingEventId,
      tracking_event_id: payload.trackingEventId,
    }));
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect((await trackingQueueStore.listActive()).find((r) => r.locationEventId === a.id)).toBeUndefined();
    expect(flush.persistedEventIds).toContain(b.id);
    expect(flush.sent).toBeGreaterThanOrEqual(1);
    expect(
      emitDriverTelemetry.mock.calls.filter((c) => c[0] === "tracking.queue.head_retired")
    ).toHaveLength(1);
  });

  it("F — budget minute = 60 → DRAIN_BUDGET_EXHAUSTED reste actif", async () => {
    await driverTrackingQueue.seedActiveItemForTests({
      sequenceId: 1,
      queuedAt: Date.now() - 1000,
      persistState: "non_ingested",
      deliveryState: "queued",
      lastError: null,
    });
    driverTrackingQueue.setDrainedInCurrentMinuteForTests(60);
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.exitReason).toBe("DRAIN_BUDGET_EXHAUSTED");
    expect(flush.httpDispatchStarted).toBe(false);
    expect(mockSendDriverLocation).not.toHaveBeenCalled();
  });

  it("G — retirement ne reset PAS le budget minute", async () => {
    const now = Date.now();
    await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_retire_budget",
      sequenceId: 1,
      queuedAt: now - MAX_HEAD_SOFT_ACK_BLOCK_MS - 1000,
      persistState: "ingested_non_persisted",
      lastError: "awaiting_durable_ack",
    });
    await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_next",
      sequenceId: 2,
      queuedAt: now - 500,
      persistState: "non_ingested",
      deliveryState: "queued",
      lastError: null,
    });
    driverTrackingQueue.setDrainedInCurrentMinuteForTests(60);
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(driverTrackingQueue.getDrainedInCurrentMinuteForTests()).toBe(60);
    expect(flush.exitReason).toBe("DRAIN_BUDGET_EXHAUSTED");
    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    const active = await trackingQueueStore.listActive();
    expect(active.find((r) => r.locationEventId === "trk_retire_budget")).toBeUndefined();
    expect(active.find((r) => r.locationEventId === "trk_next")).toBeDefined();
  });

  it("H — retry_pending normal non soft-ACK non purgé par âge 15 min", async () => {
    const now = Date.now();
    const item = await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_hard_retry",
      sequenceId: 1,
      queuedAt: now - MAX_HEAD_SOFT_ACK_BLOCK_MS - 1000,
      persistState: "non_ingested",
      deliveryState: "retry_pending",
      lastError: "send_failed",
      retryCount: 1,
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.persistedEventIds).toContain(item.id);
    expect(
      emitDriverTelemetry.mock.calls.some((c) => c[0] === "tracking.queue.head_retired")
    ).toBe(false);
  });

  it("I — item réellement persisted jamais classé expired/quarantined", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 9,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 46.2, longitude: 6.1, missionId: 9, locationMode: "mission_live" },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    await driverTrackingQueue.flush({ forceHttpFallback: true });
    const gaps = trackingQueueStore._listGapsForTests();
    expect(gaps.some((g) => g.reason === "ingested_non_persisted_expired")).toBe(false);
    expect(gaps.some((g) => g.reason === "soft_ack_retry_exhausted")).toBe(false);
  });

  it("J — pas de skip FIFO avant retraite légitime", async () => {
    const now = Date.now();
    const a = await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_fifo_a",
      sequenceId: 1,
      queuedAt: now - 30_000,
      persistState: "ingested_non_persisted",
      lastError: "awaiting_durable_ack",
      retryCount: 1,
    });
    const b = await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_fifo_b",
      sequenceId: 2,
      queuedAt: now - 1000,
      persistState: "non_ingested",
      deliveryState: "queued",
      lastError: null,
    });
    const sentIds: string[] = [];
    mockSendDriverLocation.mockImplementation(async (payload: { trackingEventId?: string }) => {
      sentIds.push(String(payload.trackingEventId));
      return {
        ack_status: "ingested_non_persisted",
        location_event_id: payload.trackingEventId,
        tracking_event_id: payload.trackingEventId,
      };
    });
    await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(sentIds[0]).toBe(a.id);
    expect(sentIds).toContain(b.id);
    expect(sentIds.indexOf(a.id)).toBeLessThan(sentIds.indexOf(b.id));
  });

  it("K — restart : head retiré ne ressuscite pas", async () => {
    const now = Date.now();
    const a = await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_no_resurrect",
      sequenceId: 1,
      queuedAt: now - MAX_HEAD_SOFT_ACK_BLOCK_MS - 1000,
      persistState: "ingested_non_persisted",
      lastError: "awaiting_durable_ack",
    });
    await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect((await trackingQueueStore.listActive()).find((r) => r.locationEventId === a.id)).toBeUndefined();
    await driverTrackingQueue.resetForTests({ keepStore: true });
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.queueDepth).toBe(0);
    expect((await trackingQueueStore.listActive()).find((r) => r.locationEventId === a.id)).toBeUndefined();
  });

  it("attempts bound — soft_ack_retry_exhausted", async () => {
    const now = Date.now();
    const a = await driverTrackingQueue.seedActiveItemForTests({
      id: "trk_attempts",
      sequenceId: 1,
      queuedAt: now - 60_000,
      persistState: "ingested_non_persisted",
      lastError: "awaiting_durable_ack",
      retryCount: MAX_AWAITING_DURABLE_ACK_ATTEMPTS,
    });
    await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(
      emitDriverTelemetry.mock.calls.some(
        (c) =>
          c[0] === "tracking.queue.head_retired" &&
          (c[1] as { reason?: string }).reason === "soft_ack_retry_exhausted"
      )
    ).toBe(true);
    expect((await trackingQueueStore.listActive()).find((r) => r.locationEventId === a.id)).toBeUndefined();
  });
});
