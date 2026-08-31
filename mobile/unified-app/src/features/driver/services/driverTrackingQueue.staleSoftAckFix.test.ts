import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { trackingQueueStore } from "./trackingQueueStore";
import {
  BACKEND_TOO_OLD_FOR_MODE_REASON,
  classifySoftAckForQueue,
} from "./driverTrackingQueueAckClassification";
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

describe("JZ-R1-STALE-SOFT-ACK-FIX-31 classifySoftAckForQueue", () => {
  it("too_old_for_mode + ingested_non_persisted → terminal", () => {
    expect(
      classifySoftAckForQueue({
        ack_status: "ingested_non_persisted",
        accept_reason: "too_old_for_mode",
      })
    ).toEqual({ kind: "terminal_backend_too_old" });
  });

  it.each([
    ["queued_kafka"],
    ["claim_in_flight"],
    ["duplicate_event_id_unproven"],
    ["db_persist_failed"],
    ["ledger_persist_failed"],
    ["awaiting_durable_ack"],
    ["older_than_canonical"],
    ["admission_not_canonical_eligible"],
    [null],
  ] as const)("soft-ACK accept_reason=%s → continue (UNCHANGED)", (reason) => {
    expect(
      classifySoftAckForQueue({
        ack_status: "ingested_non_persisted",
        accept_reason: reason,
      })
    ).toEqual({ kind: "continue" });
  });

  it("ack_status=stale n’est pas cette branche (géré ailleurs)", () => {
    expect(
      classifySoftAckForQueue({
        ack_status: "stale",
        accept_reason: "too_old_for_mode",
      })
    ).toEqual({ kind: "continue" });
  });
});

describe("JZ-R1-STALE-SOFT-ACK-FIX-31 queue terrain", () => {
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

  it("A/B/C — too_old_for_mode → expired terminal, jamais persisted, reason dédiée", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 1,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "ingested_non_persisted",
      accept_reason: "too_old_for_mode",
      durability: "queued_async",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.sent).toBe(1);
    expect(flush.dropped).toBe(1);
    expect(flush.retried).toBe(0);
    expect(flush.queueDepth).toBe(0);
    expect(flush.persistedEventIds).not.toContain(item.id);
    expect((await trackingQueueStore.listActive()).length).toBe(0);
    expect(
      trackingQueueStore
        ._listGapsForTests()
        .some((g) => g.reason === BACKEND_TOO_OLD_FOR_MODE_REASON)
    ).toBe(true);
    expect(
      emitDriverTelemetry.mock.calls.some(
        (c) =>
          c[0] === "tracking.queue.head_retired" &&
          (c[1] as { reason?: string }).reason === BACKEND_TOO_OLD_FOR_MODE_REASON
      )
    ).toBe(true);
    expect(mockSendDriverLocation).toHaveBeenCalledTimes(1);
  });

  it("D — item suivant immédiatement atteignable ; A jamais retenté", async () => {
    const stale = await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        locationMode: "availability_presence",
      },
    });
    const fresh = await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.21,
        longitude: 6.11,
        locationMode: "availability_presence",
      },
    });
    mockSendDriverLocation
      .mockResolvedValueOnce({
        ack_status: "ingested_non_persisted",
        accept_reason: "too_old_for_mode",
        location_event_id: stale.id,
        tracking_event_id: stale.id,
      })
      .mockResolvedValueOnce({
        ack_status: "persisted",
        durability: "persisted_sync",
        location_event_id: fresh.id,
        tracking_event_id: fresh.id,
      });
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(mockSendDriverLocation).toHaveBeenCalledTimes(2);
    expect(mockSendDriverLocation.mock.calls[0]?.[0]?.trackingEventId ?? stale.id).toBeTruthy();
    const firstPayload = mockSendDriverLocation.mock.calls[0]?.[0] as {
      trackingEventId?: string;
    };
    const secondPayload = mockSendDriverLocation.mock.calls[1]?.[0] as {
      trackingEventId?: string;
    };
    expect(firstPayload.trackingEventId ?? stale.id).toBe(stale.id);
    expect(secondPayload.trackingEventId ?? fresh.id).toBe(fresh.id);
    expect(flush.persistedEventIds).toContain(fresh.id);
    expect(flush.persistedEventIds).not.toContain(stale.id);
    expect(flush.queueDepth).toBe(0);
    await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(mockSendDriverLocation).toHaveBeenCalledTimes(2);
  });

  it.each([
    ["queued_kafka"],
    ["claim_in_flight"],
    ["db_persist_failed"],
    ["ledger_persist_failed"],
  ] as const)("E/F/G — accept_reason=%s reste retry_pending", async (reason) => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 2,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 46.2, longitude: 6.1, missionId: 2, locationMode: "mission_live" },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "ingested_non_persisted",
      accept_reason: reason,
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.queueDepth).toBe(1);
    expect(flush.dropped).toBe(0);
    expect(flush.retried).toBe(1);
    const row = (await trackingQueueStore.listActive()).find(
      (r) => r.locationEventId === item.id
    );
    expect(row?.state).toBe("ingested_non_persisted");
    expect(row?.deliveryState).toBe("retry_pending");
    expect(
      emitDriverTelemetry.mock.calls.some(
        (c) =>
          c[0] === "tracking.queue.head_retired" &&
          (c[1] as { reason?: string }).reason === BACKEND_TOO_OLD_FOR_MODE_REASON
      )
    ).toBe(false);
  });

  it("H — awaiting_durable_ack (sans too_old) inchangé", async () => {
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
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.queueDepth).toBe(1);
    expect(flush.retried).toBe(1);
  });

  it("I — ack_status=stale existant préservé", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 4,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 46.2, longitude: 6.1, missionId: 4, locationMode: "mission_live" },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "stale",
      accept_reason: "too_old_for_mode",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.dropped).toBe(1);
    expect(flush.queueDepth).toBe(0);
    expect(
      trackingQueueStore._listGapsForTests().some((g) => g.reason === "ack_stale")
    ).toBe(true);
  });

  it("J — persisted durable inchangé", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 5,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 46.2, longitude: 6.1, missionId: 5, locationMode: "mission_live" },
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
  });

  it("K — fallback âge/attempts soft-ACK non terminal toujours actif", () => {
    expect(MAX_AWAITING_DURABLE_ACK_ATTEMPTS).toBe(20);
    expect(MAX_HEAD_SOFT_ACK_BLOCK_MS).toBe(15 * 60 * 1000);
    const now = 2_000_000;
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
          retryCount: 20,
          persistState: "ingested_non_persisted",
          lastError: "awaiting_durable_ack",
        },
        now,
        86_400_000
      )
    ).toBe("soft_ack_retry_exhausted");
  });

  it("L — restart : terminal too_old ne ressuscite pas", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        locationMode: "availability_presence",
      },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "ingested_non_persisted",
      accept_reason: "too_old_for_mode",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    await driverTrackingQueue.flush({ forceHttpFallback: true });
    await driverTrackingQueue.resetForTests({ keepStore: true });
    const active = await trackingQueueStore.listActive();
    expect(active.find((r) => r.locationEventId === item.id)).toBeUndefined();
    expect((await driverTrackingQueue.getSnapshot()).queueDepth).toBe(0);
  });

  it("older_than_canonical UNCHANGED (pas regroupé)", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 6,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 46.2, longitude: 6.1, missionId: 6, locationMode: "mission_live" },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "ingested_non_persisted",
      accept_reason: "older_than_canonical",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.queueDepth).toBe(1);
    expect(flush.retried).toBe(1);
  });
});
