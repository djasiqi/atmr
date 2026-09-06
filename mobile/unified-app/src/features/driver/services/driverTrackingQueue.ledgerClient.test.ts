import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockSendDriverLocation = jest.fn<
  (payload: unknown) => Promise<{
    ack_status: string;
    durability?: string | null;
    location_event_id?: string | null;
    tracking_event_id?: string | null;
  }>
>();
const mockRegisterTrackingSession = jest.fn();
const mockFetchTrackingWatermark = jest.fn();
const mockEmitDriverTelemetry = jest.fn<(event: string, payload?: Record<string, unknown>) => void>();
const mockAsyncStorageGetItem = jest.fn<(key: string) => Promise<string | null>>();
const mockAsyncStorageSetItem = jest.fn<(key: string, value: string) => Promise<void>>();

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
  registerTrackingSession: (...args: unknown[]) => mockRegisterTrackingSession(...args),
  fetchTrackingWatermark: (...args: unknown[]) => mockFetchTrackingWatermark(...args),
}));

jest.mock("../../../core/auth/authRefreshListeners", () => ({
  onAuthRefreshSuccess: () => undefined,
}));

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    sendDriverLocationBatch: () => false,
    isDriverSocketReady: () => false,
  },
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (...args: unknown[]) =>
    mockEmitDriverTelemetry(...(args as [string, Record<string, unknown>?])),
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
  isFeatureEnabled: () => false,
}));

jest.mock("./socketBatchPacing", () => ({
  canEmitSocketBatchNow: () => true,
  getSocketBatchCooldownRemainingMs: () => 0,
  recordSocketBatchSent: jest.fn(),
  recordSocketBatchRateLimited: jest.fn(),
}));

const { trackingQueueStore } = require("./trackingQueueStore") as typeof import("./trackingQueueStore");
const { driverTrackingQueue } = require("./driverTrackingQueue") as typeof import("./driverTrackingQueue");

function baseEnqueue() {
  return {
    missionId: null as number | null,
    appState: "active" as const,
    locationMode: "availability_presence" as const,
    payload: {
      latitude: 46.5,
      longitude: 6.6,
      locationMode: "availability_presence" as const,
    },
  };
}

async function waitForReadiness(
  expected: string | string[],
  maxTurns = 2000
): Promise<string> {
  const wanted = Array.isArray(expected) ? expected : [expected];
  for (let i = 0; i < maxTurns; i += 1) {
    const current = driverTrackingQueue.getSessionReadinessForTests();
    if (wanted.includes(current)) return current;
    await Promise.resolve();
  }
  throw new Error(
    `timeout readiness: wanted ${wanted.join("|")}, got ${driverTrackingQueue.getSessionReadinessForTests()}`
  );
}

describe("P0-C-LEDGER-CLIENT — readiness gate + anti-HOL", () => {
  beforeEach(async () => {
    mockAsyncStorageGetItem.mockReset();
    mockAsyncStorageSetItem.mockReset();
    mockSendDriverLocation.mockReset();
    mockRegisterTrackingSession.mockReset();
    mockFetchTrackingWatermark.mockReset();
    mockEmitDriverTelemetry.mockReset();
    mockAsyncStorageGetItem.mockResolvedValue(null);
    mockAsyncStorageSetItem.mockResolvedValue(undefined);
    mockRegisterTrackingSession.mockResolvedValue({
      tracking_session_id: "sess",
      session_generation: 7,
      first_sequence_id: 1,
      status: "active",
    });
    mockSendDriverLocation.mockImplementation(async (payload: unknown) => {
      const p = payload as { trackingEventId?: string };
      const id = p.trackingEventId ?? "evt";
      return {
        ack_status: "persisted",
        durability: "persisted_sync",
        location_event_id: id,
        tracking_event_id: id,
      };
    });
    trackingQueueStore._resetMemoryForTests();
    await driverTrackingQueue.resetForTests();
  });

  it("1 — beginNew + register OK → READY ; enqueue possible avec generation", async () => {
    await driverTrackingQueue.beginNewTrackingSession();
    expect(driverTrackingQueue.getSessionReadinessForTests()).toBe("READY");
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.sessionGeneration).toBe(7);
    expect(snap.sessionReadiness).toBe("READY");
    const item = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(item).not.toBeNull();
    expect(item!.sessionGeneration).toBe(7);
    expect(item!.trackingSessionId).toMatch(/^trk_sess_/);
  });

  it("2 — enqueue pendant REGISTERING → 0 row ; telemetry blocked", async () => {
    let resolveRegister: ((v: unknown) => void) | null = null;
    mockRegisterTrackingSession.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRegister = resolve;
        })
    );
    const beginPromise = driverTrackingQueue.beginNewTrackingSession();
    for (let i = 0; i < 2000 && !resolveRegister; i += 1) {
      await Promise.resolve();
    }
    expect(resolveRegister).not.toBeNull();
    await waitForReadiness(["CREATING", "REGISTERING"]);
    const blocked = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(blocked).toBeNull();
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.queueDepth).toBe(0);
    expect(
      mockEmitDriverTelemetry.mock.calls.some(
        (c) =>
          c[0] === "tracking.queue.enqueue_blocked" &&
          ((c[1] as { reason?: string })?.reason === "not_ready" ||
            (c[1] as { reason?: string })?.reason === "register_failed")
      )
    ).toBe(true);
    const sid = (
      driverTrackingQueue as unknown as { trackingSessionId: string }
    ).trackingSessionId;
    resolveRegister?.({
      tracking_session_id: sid,
      session_generation: 3,
      first_sequence_id: 1,
      status: "active",
    });
    await beginPromise;
  });

  it("3 — register fail → REGISTER_FAILED ; enqueue bloqué", async () => {
    mockRegisterTrackingSession.mockRejectedValue(new Error("network down"));
    await driverTrackingQueue.beginNewTrackingSession();
    expect(driverTrackingQueue.getSessionReadinessForTests()).toBe("REGISTER_FAILED");
    const blocked = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(blocked).toBeNull();
    expect(
      mockEmitDriverTelemetry.mock.calls.some(
        (c) => c[0] === "tracking.session.register_failed"
      )
    ).toBe(true);
    expect(
      mockEmitDriverTelemetry.mock.calls.some(
        (c) =>
          c[0] === "tracking.queue.enqueue_blocked" &&
          (c[1] as { reason?: string })?.reason === "register_failed"
      )
    ).toBe(true);
    // Pas de création de session à chaque fix
    const registerCalls = mockRegisterTrackingSession.mock.calls.length;
    await driverTrackingQueue.enqueue(baseEnqueue());
    await driverTrackingQueue.enqueue(baseEnqueue());
    expect(mockRegisterTrackingSession.mock.calls.length).toBe(registerCalls);
  });

  it("4 — register fail puis succès → READY ; enqueue avec generation", async () => {
    mockRegisterTrackingSession.mockRejectedValue(new Error("offline"));
    await driverTrackingQueue.beginNewTrackingSession();
    expect(driverTrackingQueue.getSessionReadinessForTests()).toBe("REGISTER_FAILED");
    expect(await driverTrackingQueue.enqueue(baseEnqueue())).toBeNull();

    mockRegisterTrackingSession.mockResolvedValue({
      tracking_session_id: "sess",
      session_generation: 11,
      first_sequence_id: 1,
      status: "active",
    });
    await driverTrackingQueue.retryRegisterForTests();
    expect(driverTrackingQueue.getSessionReadinessForTests()).toBe("READY");
    const item = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(item?.sessionGeneration).toBe(11);
  });

  it("5 — mid-rotate TTL : drop observé puis reprise après READY", async () => {
    await driverTrackingQueue.beginNewTrackingSession();
    const before = await driverTrackingQueue.getSnapshot();
    expect(before.sessionReadiness).toBe("READY");
    const first = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(first).not.toBeNull();

    let resolveRegister: ((v: unknown) => void) | null = null;
    mockRegisterTrackingSession.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRegister = resolve;
        })
    );
    driverTrackingQueue.forceExpireSessionForTests();

    const rotatePromise = driverTrackingQueue.enqueue(baseEnqueue());
    for (let i = 0; i < 2000 && !resolveRegister; i += 1) {
      await Promise.resolve();
    }
    expect(resolveRegister).not.toBeNull();
    await waitForReadiness(["CREATING", "REGISTERING"]);

    // Concurrent mid-rotate → drop, pas d'ancienne identité ni gen=null
    const concurrent = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(concurrent).toBeNull();

    const rotatingId = (
      driverTrackingQueue as unknown as { trackingSessionId: string }
    ).trackingSessionId;
    resolveRegister?.({
      tracking_session_id: rotatingId,
      session_generation: 22,
      first_sequence_id: 1,
      status: "active",
    });
    const afterRotate = await rotatePromise;
    expect(afterRotate).not.toBeNull();
    expect(afterRotate!.sessionGeneration).toBe(22);
    expect(afterRotate!.trackingSessionId).not.toBe(before.trackingSessionId);
    expect(afterRotate!.trackingSessionId).not.toBe(first!.trackingSessionId);
  });

  it("6 — tête historique gen=null quarantinée ; item READY derrière flushable", async () => {
    await driverTrackingQueue.beginNewTrackingSession();
    const valid = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(valid).not.toBeNull();

    const poisonId = `trk_poison_${Date.now()}`;
    await trackingQueueStore.upsert({
      locationEventId: poisonId,
      trackingSessionId: "trk_sess_orphan_poison",
      sessionGeneration: null,
      sequenceId: 1,
      payloadJson: JSON.stringify({ latitude: 1, longitude: 2 }),
      state: "non_ingested",
      queuedAt: Date.now() - 10_000,
      lastAttemptAt: null,
      retryCount: 0,
      deliveryState: "retry_pending",
      missionId: null,
      locationMode: "availability_presence",
      batchId: "b",
      positionId: "p",
      appState: "active",
      lastError: "ledger_ids_missing",
      ackedAt: null,
    });
    // Injecter en tête mémoire (simule HOL historique)
    const queue = driverTrackingQueue as unknown as {
      items: Record<string, unknown>[];
    };
    queue.items.unshift({
      id: poisonId,
      sequenceId: 1,
      trackingSessionId: "trk_sess_orphan_poison",
      sessionGeneration: null,
      batchId: "b",
      positionId: "p",
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: { latitude: 1, longitude: 2 },
      queuedAt: Date.now() - 10_000,
      retryCount: 3,
      deliveryState: "retry_pending",
      lastAttemptAt: Date.now(),
      ackedAt: null,
      lastError: "ledger_ids_missing",
      persistState: "non_ingested",
    });

    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(
      mockEmitDriverTelemetry.mock.calls.some(
        (c) => c[0] === "tracking.queue.ledger_invalid_quarantined"
      )
    ).toBe(true);
    expect(flush.backendAcked).toBeGreaterThanOrEqual(1);
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.queueDepth).toBe(0);
    expect(mockSendDriverLocation.mock.calls.length).toBeGreaterThanOrEqual(1);
    const sentIds = mockSendDriverLocation.mock.calls.map(
      (c) => (c[0] as { trackingEventId?: string }).trackingEventId
    );
    expect(sentIds).toContain(valid!.id);
    expect(sentIds).not.toContain(poisonId);
  });

  it("7 — pas de fire-and-forget ready : pas de seq sans generation", async () => {
    let resolveRegister: ((v: unknown) => void) | null = null;
    mockRegisterTrackingSession.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRegister = resolve;
        })
    );
    const beginPromise = driverTrackingQueue.beginNewTrackingSession();
    for (let i = 0; i < 2000 && !resolveRegister; i += 1) {
      await Promise.resolve();
    }
    expect(resolveRegister).not.toBeNull();
    await waitForReadiness(["CREATING", "REGISTERING"]);
    expect(await driverTrackingQueue.enqueue(baseEnqueue())).toBeNull();
    expect(driverTrackingQueue.getSessionReadinessForTests()).not.toBe("READY");
    const sid = (
      driverTrackingQueue as unknown as { trackingSessionId: string }
    ).trackingSessionId;

    resolveRegister?.({
      tracking_session_id: sid,
      session_generation: 9,
      first_sequence_id: 1,
      status: "active",
    });
    await beginPromise;
    const item = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(item?.sessionGeneration).toBe(9);
    expect(item?.sequenceId).toBe(1);
  });

  it("preuve — aucun chemin enqueue ne persiste generation=null", async () => {
    await driverTrackingQueue.beginNewTrackingSession();
    const item = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(item!.sessionGeneration).not.toBeNull();

    mockRegisterTrackingSession.mockRejectedValue(new Error("fail"));
    await driverTrackingQueue.beginNewTrackingSession();
    expect(await driverTrackingQueue.enqueue(baseEnqueue())).toBeNull();

    const rows = await trackingQueueStore.listActive();
    expect(rows.every((r) => r.sessionGeneration != null)).toBe(true);
  });
});
