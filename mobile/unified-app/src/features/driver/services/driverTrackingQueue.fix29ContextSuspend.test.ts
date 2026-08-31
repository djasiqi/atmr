/**
 * FIX-29 — réconciliation context_inactive ↔ lease driver_active dans flush().
 * Reproduction RCA-28 + fail-closed + préservation des autres suspensions.
 */
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { trackingQueueStore } from "./trackingQueueStore";

const QUEUE_SUSPEND_STORAGE_KEY = "driver_tracking_queue_suspend_v1";

const mockSendDriverLocation = jest.fn<
  (payload: unknown) => Promise<{
    ack_status: string;
    durability?: string | null;
    tracking_event_id?: string | null;
  }>
>();
const mockAsyncStorageGetItem = jest.fn<(key: string) => Promise<string | null>>();
const mockAsyncStorageSetItem = jest.fn<(key: string, value: string) => Promise<void>>();
const mockEmitDriverTelemetry = jest.fn<(event: string, payload?: Record<string, unknown>) => void>();
const mockIsDriverSocketReady = jest.fn<() => boolean>().mockReturnValue(false);

type LeaseFixture = {
  state: "driver_active" | "inactive" | "switching";
  contextId: string;
  driverId: number;
  sessionGenerationId: number;
  trackingGenerationId: string;
  trackingIdentityId: string;
  updatedAt: number;
  fromDriver?: boolean;
  previousDriverActive?: unknown;
} | null;

let mockLeaseFixture: LeaseFixture = {
  state: "driver_active",
  contextId: "driver:1",
  driverId: 1,
  sessionGenerationId: 1,
  trackingGenerationId: "trk-test",
  trackingIdentityId: "driver:1:company:1",
  updatedAt: Date.now(),
};

const asyncStore = new Map<string, string>();

jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: {
    getItem: mockAsyncStorageGetItem,
    setItem: mockAsyncStorageSetItem,
  },
}));

jest.mock("../api/driverHttp", () => ({
  sendDriverLocation: mockSendDriverLocation,
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

jest.mock("../../../core/auth/authRefreshListeners", () => ({
  onAuthRefreshSuccess: () => undefined,
}));

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    sendDriverLocationBatch: () => false,
    isDriverSocketReady: () => mockIsDriverSocketReady(),
  },
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: mockEmitDriverTelemetry,
}));

jest.mock("./trackingContextLease", () => ({
  readTrackingContextLease: async () => mockLeaseFixture,
  leaseAllowsTransport: (lease: { state?: string } | null) =>
    Boolean(lease && lease.state === "driver_active"),
  leaseAllowsCapture: (lease: { state?: string; fromDriver?: boolean } | null) => {
    if (!lease) return false;
    if (lease.state === "driver_active") return true;
    if (lease.state === "switching" && lease.fromDriver) return true;
    return false;
  },
}));

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (key: unknown) =>
    key === "tracking_real_ack_semantics_enabled" ||
    key === "tracking_queue_compaction_enabled",
}));

jest.mock("./socketBatchPacing", () => ({
  canEmitSocketBatchNow: () => true,
  getSocketBatchCooldownRemainingMs: () => 0,
  recordSocketBatchSent: jest.fn(),
  recordSocketBatchRateLimited: jest.fn(),
}));

const { driverTrackingQueue } = require("./driverTrackingQueue") as typeof import("./driverTrackingQueue");

function setPersistedSuspend(reason: string, untilMs: number | null) {
  asyncStore.set(
    QUEUE_SUSPEND_STORAGE_KEY,
    JSON.stringify({ reason, untilMs })
  );
}

function readPersistedSuspend(): { reason?: string; untilMs?: number | null } | null {
  const raw = asyncStore.get(QUEUE_SUSPEND_STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as { reason?: string; untilMs?: number | null };
  } catch {
    return null;
  }
}

function telemetryCalls(event: string) {
  return mockEmitDriverTelemetry.mock.calls.filter((c) => c[0] === event);
}

describe("FIX-29 context_inactive reconcile with driver_active lease", () => {
  beforeEach(async () => {
    asyncStore.clear();
    mockAsyncStorageGetItem.mockReset();
    mockAsyncStorageSetItem.mockReset();
    mockSendDriverLocation.mockReset();
    mockEmitDriverTelemetry.mockReset();
    mockIsDriverSocketReady.mockReset();
    mockIsDriverSocketReady.mockReturnValue(false);

    mockAsyncStorageGetItem.mockImplementation(async (key: string) =>
      asyncStore.has(key) ? asyncStore.get(key)! : null
    );
    mockAsyncStorageSetItem.mockImplementation(async (key: string, value: string) => {
      if (value === "") {
        asyncStore.delete(key);
      } else {
        asyncStore.set(key, value);
      }
    });

    mockSendDriverLocation.mockImplementation(async (payload: unknown) => {
      const p = payload as { trackingEventId?: string };
      const id = p.trackingEventId ?? "unknown";
      return {
        ack_status: "persisted",
        durability: "persisted_sync",
        location_event_id: id,
        tracking_event_id: id,
      };
    });

    mockLeaseFixture = {
      state: "driver_active",
      contextId: "driver:1",
      driverId: 1,
      sessionGenerationId: 1,
      trackingGenerationId: "trk-test",
      trackingIdentityId: "driver:1:company:1",
      updatedAt: Date.now(),
    };

    trackingQueueStore._resetMemoryForTests();
    await driverTrackingQueue.resetForTests();
    await driverTrackingQueue.beginNewTrackingSession();
  });

  it("T29-A RCA-28: persisted context_inactive + driver_active → clear + HTTP same flush", async () => {
    await driverTrackingQueue.activateContextInactiveGate("rca28_seed");
    expect(readPersistedSuspend()?.reason).toBe("context_inactive");

    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.21156,
        longitude: 6.1263,
        missionId: null,
        locationMode: "availability_presence",
      },
    });

    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    const unblock = telemetryCalls("tracking.queue.transport_unblock");
    expect(
      unblock.some(
        (c) =>
          (c[1] as { reason?: string })?.reason === "flush_driver_active_reconcile" &&
          (c[1] as { previous_suspend_reason?: string })?.previous_suspend_reason ===
            "context_inactive" &&
          (c[1] as { lease_state?: string })?.lease_state === "driver_active"
      )
    ).toBe(true);

    expect(mockSendDriverLocation).toHaveBeenCalled();
    expect(flush.backendAcked).toBeGreaterThanOrEqual(1);
    expect(flush.sent).toBeGreaterThanOrEqual(1);
    expect(readPersistedSuspend()).toBeNull();
  });

  it("T29-B boot: cold memory + persisted gate + driver_active → HTTP", async () => {
    setPersistedSuspend("context_inactive", null);
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });

    // Kill process simulé : mémoire queueSuspend perdue, store + AsyncStorage conservés.
    await driverTrackingQueue.resetForTests({ keepStore: true });
    setPersistedSuspend("context_inactive", null);

    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(
      telemetryCalls("tracking.queue.transport_unblock").some(
        (c) =>
          (c[1] as { reason?: string })?.reason === "flush_driver_active_reconcile"
      )
    ).toBe(true);
    expect(mockSendDriverLocation).toHaveBeenCalled();
    expect(flush.backendAcked).toBeGreaterThanOrEqual(1);
    expect(readPersistedSuspend()).toBeNull();
  });

  it("T29-C inactive lease keeps gate / no HTTP", async () => {
    mockLeaseFixture = {
      state: "inactive",
      contextId: "company:1",
      driverId: 1,
      sessionGenerationId: 1,
      trackingGenerationId: "trk-test",
      trackingIdentityId: "driver:1:company:1",
      updatedAt: Date.now(),
    };
    await driverTrackingQueue.activateContextInactiveGate("seed");
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });

    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    expect(flush.backendAcked).toBe(0);
    expect(flush.sent).toBe(0);
    expect(readPersistedSuspend()?.reason).toBe("context_inactive");
    expect(
      telemetryCalls("tracking.queue.transport_blocked_context").some(
        (c) =>
          (c[1] as { reason?: string })?.reason === "lease_not_driver_active"
      )
    ).toBe(true);
    expect(
      telemetryCalls("tracking.queue.transport_unblock").some(
        (c) =>
          (c[1] as { reason?: string })?.reason === "flush_driver_active_reconcile"
      )
    ).toBe(false);
  });

  it("T29-D switching lease keeps gate / no HTTP", async () => {
    mockLeaseFixture = {
      state: "switching",
      contextId: "driver:1",
      driverId: 1,
      sessionGenerationId: 1,
      trackingGenerationId: "trk-test",
      trackingIdentityId: "driver:1:company:1",
      updatedAt: Date.now(),
      fromDriver: true,
    };
    await driverTrackingQueue.activateContextInactiveGate("seed");
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });

    await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    expect(readPersistedSuspend()?.reason).toBe("context_inactive");
  });

  it("T29-E null lease fail-closed", async () => {
    mockLeaseFixture = null;
    await driverTrackingQueue.activateContextInactiveGate("seed");
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });

    await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    expect(readPersistedSuspend()?.reason).toBe("context_inactive");
  });

  it("T29-F auth suspension not cleared by driver_active", async () => {
    await driverTrackingQueue.activateSuspensionForTests(60_000, "auth");
    expect(readPersistedSuspend()?.reason).toBe("auth");

    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });

    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    expect(flush.sent).toBe(0);
    expect(readPersistedSuspend()?.reason).toBe("auth");
    expect(
      telemetryCalls("tracking.queue.transport_unblock").some(
        (c) =>
          (c[1] as { reason?: string })?.reason === "flush_driver_active_reconcile"
      )
    ).toBe(false);
  });

  it("T29-G rate_limit backoff preserved", async () => {
    await driverTrackingQueue.activateSuspensionForTests(30_000, "rate_limit");
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });

    await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    expect(readPersistedSuspend()?.reason).toBe("rate_limit");
  });

  it("T29-G2 forbidden suspension preserved with driver_active", async () => {
    await driverTrackingQueue.activateSuspensionForTests(60_000, "forbidden");
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });

    await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    expect(readPersistedSuspend()?.reason).toBe("forbidden");
    expect(
      telemetryCalls("tracking.queue.transport_unblock").some(
        (c) =>
          (c[1] as { reason?: string })?.reason === "flush_driver_active_reconcile"
      )
    ).toBe(false);
  });

  it("T29-G3 circuit_breaker suspension preserved with driver_active", async () => {
    await driverTrackingQueue.activateSuspensionForTests(60_000, "circuit_breaker");
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });

    await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    expect(readPersistedSuspend()?.reason).toBe("circuit_breaker");
    expect(
      telemetryCalls("tracking.queue.transport_unblock").some(
        (c) =>
          (c[1] as { reason?: string })?.reason === "flush_driver_active_reconcile"
      )
    ).toBe(false);
  });

  it("T29-H context_entered_driver clear remains idempotent", async () => {
    await driverTrackingQueue.activateContextInactiveGate("seed");
    await driverTrackingQueue.clearContextInactiveGate("context_entered_driver");
    await driverTrackingQueue.clearContextInactiveGate("context_entered_driver");
    expect(readPersistedSuspend()).toBeNull();

    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.backendAcked).toBeGreaterThanOrEqual(1);
  });

  it("T29-I no gate + driver_active unchanged", async () => {
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: null,
        locationMode: "availability_presence",
      },
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(mockSendDriverLocation).toHaveBeenCalled();
    expect(flush.backendAcked).toBeGreaterThanOrEqual(1);
    expect(
      telemetryCalls("tracking.queue.transport_unblock").some(
        (c) =>
          (c[1] as { reason?: string })?.reason === "flush_driver_active_reconcile"
      )
    ).toBe(false);
  });
});
