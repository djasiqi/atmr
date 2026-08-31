/**
 * JZ-R1-FLUSH-OBS-11 — observabilité des sorties pré-HTTP du flush.
 * Comportement queue/send inchangé : seules les raisons / champs diag sont vérifiés.
 */
import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockSendDriverLocation = jest.fn<
  (payload: unknown) => Promise<{
    ack_status: string;
    tracking_event_id?: string | null;
  }>
>();
const mockAsyncStorageGetItem = jest.fn<(key: string) => Promise<string | null>>();
const mockAsyncStorageSetItem = jest.fn<(key: string, value: string) => Promise<void>>();
const mockAtmrJsDiag = jest.fn<(step: string, fields?: Record<string, unknown>) => void>();
const mockReadLease = jest.fn();

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

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    sendDriverLocationBatch: () => false,
    isDriverSocketReady: () => false,
  },
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
}));

jest.mock("./trackingContextLease", () => ({
  readTrackingContextLease: () => mockReadLease(),
  leaseAllowsTransport: (lease: { state?: string } | null) =>
    Boolean(lease && lease.state === "driver_active"),
  leaseAllowsCapture: () => true,
}));

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: () => false,
}));

jest.mock("./socketBatchPacing", () => ({
  canEmitSocketBatchNow: () => true,
  getSocketBatchCooldownRemainingMs: () => 0,
  recordSocketBatchSent: jest.fn(),
  recordSocketBatchRateLimited: jest.fn(),
}));

jest.mock("./atmrJsTaskDiag", () => ({
  atmrJsDiag: (step: string, fields?: Record<string, unknown>) => mockAtmrJsDiag(step, fields),
}));

jest.mock("../../../core/auth/authRefreshListeners", () => ({
  onAuthRefreshSuccess: () => undefined,
}));

const { driverTrackingQueue } = require("./driverTrackingQueue") as typeof import("./driverTrackingQueue");
const { trackingQueueStore } = require("./trackingQueueStore") as typeof import("./trackingQueueStore");

function activeLease() {
  return {
    state: "driver_active" as const,
    contextId: "driver:1",
    driverId: 1,
    sessionGenerationId: 1,
    trackingGenerationId: "trk-test",
    trackingIdentityId: "driver:1:company:1",
    updatedAt: Date.now(),
  };
}

describe("driverTrackingQueue flush observability (JZ-R1-FLUSH-OBS-11)", () => {
  beforeEach(async () => {
    mockAsyncStorageGetItem.mockReset();
    mockAsyncStorageSetItem.mockReset();
    mockSendDriverLocation.mockReset();
    mockAtmrJsDiag.mockReset();
    mockReadLease.mockReset();
    mockReadLease.mockResolvedValue(activeLease());
    mockAsyncStorageGetItem.mockResolvedValue(null);
    mockAsyncStorageSetItem.mockResolvedValue(undefined);
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      tracking_event_id: null,
    });
    trackingQueueStore._resetMemoryForTests();
    await driverTrackingQueue.resetForTests();
    await driverTrackingQueue.clearContextInactiveGate("test_setup");
  });

  it("QUEUE_SUSPEND_ACTIVE → exitReason + FLUSH_EXIT diag, sans HTTP", async () => {
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        locationMode: "availability_presence",
      },
    });
    // FIX-29 : context_inactive + driver_active est réconcilié (HTTP).
    // QUEUE_SUSPEND_ACTIVE doit être prouvé avec un suspend réellement bloquant.
    await driverTrackingQueue.activateSuspensionForTests(60_000, "auth");

    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(flush.exitReason).toBe("QUEUE_SUSPEND_ACTIVE");
    expect(flush.sent).toBe(0);
    expect(flush.httpDispatchStarted).toBe(false);
    expect(flush.suspended).toBe(true);
    expect(mockSendDriverLocation).not.toHaveBeenCalled();
    expect(mockAtmrJsDiag).toHaveBeenCalledWith(
      "FLUSH_EXIT",
      expect.objectContaining({ reason: "QUEUE_SUSPEND_ACTIVE" })
    );
  });

  it("LEASE_NOT_DRIVER_ACTIVE → exitReason, sans HTTP", async () => {
    mockReadLease.mockResolvedValue({
      state: "inactive",
      contextId: "x",
      driverId: 1,
      sessionGenerationId: 1,
      trackingGenerationId: "trk",
      trackingIdentityId: "x",
      updatedAt: Date.now(),
    });
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: { latitude: 1, longitude: 2, locationMode: "availability_presence" },
    });

    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.exitReason).toBe("LEASE_NOT_DRIVER_ACTIVE");
    expect(flush.httpDispatchStarted).toBe(false);
    expect(mockSendDriverLocation).not.toHaveBeenCalled();
  });

  it("FLUSH_ALREADY_RUNNING lorsque flush concurrent", async () => {
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: { latitude: 1, longitude: 2, locationMode: "availability_presence" },
    });
    let release!: (v: {
      ack_status: string;
      tracking_event_id?: string | null;
    }) => void;
    mockSendDriverLocation.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve;
        })
    );

    const first = driverTrackingQueue.flush({ forceHttpFallback: true });
    await new Promise((r) => setTimeout(r, 5));
    const second = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(second.exitReason).toBe("FLUSH_ALREADY_RUNNING");
    expect(second.httpDispatchStarted).toBe(false);
    expect(second.flushLockState).toBe("held");

    release({ ack_status: "persisted", tracking_event_id: null });
    const firstResult = await first;
    expect(firstResult.httpDispatchStarted).toBe(true);
    expect(firstResult.exitReason).toBe("HTTP_DISPATCHED");
  });

  it("item sélectionné → http_dispatch_started + sent>0 → HTTP_DISPATCHED", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "background",
      locationMode: "availability_presence",
      payload: { latitude: 46, longitude: 6, locationMode: "availability_presence" },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      tracking_event_id: item!.id,
    });

    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.httpDispatchStarted).toBe(true);
    expect(flush.selectedItem).toBe(true);
    expect(flush.sent).toBeGreaterThan(0);
    expect(flush.exitReason).toBe("HTTP_DISPATCHED");
    expect(mockSendDriverLocation).toHaveBeenCalled();
  });

  it("sent=0 → exitReason non vide (QUEUE_EMPTY)", async () => {
    const flush = await driverTrackingQueue.flush({ forceHttpFallback: true });
    expect(flush.sent).toBe(0);
    expect(flush.exitReason).toBeTruthy();
    expect(flush.exitReason).toBe("QUEUE_EMPTY");
  });
});
