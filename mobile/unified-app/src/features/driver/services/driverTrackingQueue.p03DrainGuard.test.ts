/**
 * P0.3 — stop drain au 1er 429, budget minute, coalescence flush.
 * Env forcé avant require : batch large pour tester le stop mid-flush ;
 * MAX_DRAIN=60 pour le budget serveur.
 */
process.env.EXPO_PUBLIC_DRIVER_TRACKING_DRAIN_BATCH_SIZE = "50";
process.env.EXPO_PUBLIC_DRIVER_TRACKING_MAX_DRAIN_POSITIONS_PER_MINUTE = "60";
process.env.EXPO_PUBLIC_DRIVER_TRACKING_DRAIN_INTERVAL_MS = "3000";

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
const mockAsyncStorageGetItem = jest.fn<(key: string) => Promise<string | null>>();
const mockAsyncStorageSetItem = jest.fn<(key: string, value: string) => Promise<void>>();
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
  emitDriverTelemetry: jest.fn(),
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

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { driverTrackingQueue } = require("./driverTrackingQueue") as typeof import("./driverTrackingQueue");

function rateLimitError(retryAfterSeconds = 30) {
  return {
    message: "rate_limit_exceeded",
    status: 429,
    retry_after_seconds: retryAfterSeconds,
  };
}

async function enqueueN(n: number): Promise<string[]> {
  const ids: string[] = [];
  for (let i = 0; i < n; i += 1) {
    const item = await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.5 + i * 0.0001,
        longitude: 6.6,
        locationMode: "availability_presence",
      },
    });
    ids.push(item.id);
  }
  return ids;
}

describe("driverTrackingQueue P0.3 drain guard", () => {
  beforeEach(async () => {
    asyncStore.clear();
    mockAsyncStorageGetItem.mockReset();
    mockAsyncStorageSetItem.mockReset();
    mockSendDriverLocation.mockReset();
    mockRegisterTrackingSession.mockReset();
    mockFetchTrackingWatermark.mockReset();
    mockAsyncStorageGetItem.mockImplementation(async (key) => asyncStore.get(key) ?? null);
    mockAsyncStorageSetItem.mockImplementation(async (key, value) => {
      if (!value) asyncStore.delete(key);
      else asyncStore.set(key, value);
    });
    mockRegisterTrackingSession.mockResolvedValue({
      tracking_session_id: "sess",
      session_generation: 1,
      first_sequence_id: 1,
      status: "active",
    });
    await driverTrackingQueue.resetForTests();
    await driverTrackingQueue.beginNewTrackingSession();
  });

  it("A — stop au premier 429 : PUT total = 11, 40 conservés", async () => {
    const ids = await enqueueN(50);
    let call = 0;
    mockSendDriverLocation.mockImplementation(async (payload: unknown) => {
      call += 1;
      const p = payload as { trackingEventId?: string };
      if (call <= 10) {
        return {
          ack_status: "persisted",
          durability: "persisted_sync",
          location_event_id: p.trackingEventId ?? null,
          tracking_event_id: p.trackingEventId ?? null,
        };
      }
      throw rateLimitError(30);
    });

    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    expect(mockSendDriverLocation).toHaveBeenCalledTimes(11);
    expect(flush.backendAcked).toBe(10);
    expect(flush.queueDepth).toBe(40);
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.suspendReason).toBe("rate_limit");
    expect(ids.length).toBe(50);
  });

  it("B — aucun HTTP pendant Retry-After / suspension", async () => {
    await enqueueN(5);
    mockSendDriverLocation.mockRejectedValueOnce(rateLimitError(30));

    await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(mockSendDriverLocation).toHaveBeenCalledTimes(1);

    mockSendDriverLocation.mockClear();
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: "x",
    });

    for (let i = 0; i < 4; i += 1) {
      const during = await driverTrackingQueue.flush({
        forceHttpFallback: true,
        networkProfile: "normal",
      });
      expect(during.sent).toBe(0);
      expect(mockSendDriverLocation).toHaveBeenCalledTimes(0);
    }

    await driverTrackingQueue.clearSuspension("test_expiry");
    // Ré-attacher location_event_id exact pour tombstone
    mockSendDriverLocation.mockImplementation(async (payload: unknown) => {
      const p = payload as { trackingEventId?: string };
      return {
        ack_status: "persisted",
        durability: "persisted_sync",
        location_event_id: p.trackingEventId ?? null,
        tracking_event_id: p.trackingEventId ?? null,
      };
    });
    const after = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(mockSendDriverLocation).toHaveBeenCalled();
    expect(after.backendAcked).toBeGreaterThan(0);
  });

  it("C — budget ≤ 60 PUT / minute", async () => {
    await enqueueN(80);
    mockSendDriverLocation.mockImplementation(async (payload: unknown) => {
      const p = payload as { trackingEventId?: string };
      return {
        ack_status: "persisted",
        durability: "persisted_sync",
        location_event_id: p.trackingEventId ?? null,
        tracking_event_id: p.trackingEventId ?? null,
      };
    });

    await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    const afterFirst = mockSendDriverLocation.mock.calls.length;
    expect(afterFirst).toBeLessThanOrEqual(50);

    await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    const afterSecond = mockSendDriverLocation.mock.calls.length;
    expect(afterSecond).toBeLessThanOrEqual(60);

    await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(mockSendDriverLocation.mock.calls.length).toBe(afterSecond);
    expect(mockSendDriverLocation.mock.calls.length).toBeLessThanOrEqual(60);
  });

  it("D — 5 succès + 1×429 → 95 conservés, rien perdu", async () => {
    await enqueueN(100);
    let call = 0;
    mockSendDriverLocation.mockImplementation(async (payload: unknown) => {
      call += 1;
      const p = payload as { trackingEventId?: string };
      if (call <= 5) {
        return {
          ack_status: "persisted",
          durability: "persisted_sync",
          location_event_id: p.trackingEventId ?? null,
          tracking_event_id: p.trackingEventId ?? null,
        };
      }
      throw rateLimitError(60);
    });

    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(mockSendDriverLocation).toHaveBeenCalledTimes(6);
    expect(flush.backendAcked).toBe(5);
    expect(flush.queueDepth).toBe(95);
  });

  it("E — persisted_sync reste strict (mauvais id conserve)", async () => {
    await enqueueN(1);
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: "other",
      tracking_event_id: "other",
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.backendAcked).toBe(0);
    expect(flush.queueDepth).toBe(1);
  });

  it("F — flush concurrents coalescés (une seule boucle)", async () => {
    await enqueueN(5);
    let resolveFirst!: (v: unknown) => void;
    const gate = new Promise((r) => {
      resolveFirst = r;
    });
    let calls = 0;
    mockSendDriverLocation.mockImplementation(async (payload: unknown) => {
      calls += 1;
      if (calls === 1) await gate;
      const p = payload as { trackingEventId?: string };
      return {
        ack_status: "persisted",
        durability: "persisted_sync",
        location_event_id: p.trackingEventId ?? null,
        tracking_event_id: p.trackingEventId ?? null,
      };
    });

    const pA = driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    const pB = driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    const pC = driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });

    const midB = await pB;
    const midC = await pC;
    expect(midB.sent).toBe(0);
    expect(midC.sent).toBe(0);

    resolveFirst(undefined);
    await pA;
    // Attendre le flush coalescé déclenché dans finally
    await new Promise((r) => setTimeout(r, 50));
    // Pas trois drains parallèles : le mock n'a jamais été appelé en parallèle >1
    expect(mockSendDriverLocation.mock.calls.length).toBeGreaterThanOrEqual(1);
    expect(mockSendDriverLocation.mock.calls.length).toBeLessThanOrEqual(5);
  });

  it("G — 429 pendant flush A → flush B pending n'envoie pas pendant suspension", async () => {
    await enqueueN(10);
    let resolve429!: (err: unknown) => void;
    const hold429 = new Promise((_, rej) => {
      resolve429 = rej;
    });
    let call = 0;
    mockSendDriverLocation.mockImplementation(async (payload: unknown) => {
      call += 1;
      const p = payload as { trackingEventId?: string };
      if (call === 1) {
        await hold429;
      }
      return {
        ack_status: "persisted",
        durability: "persisted_sync",
        location_event_id: p.trackingEventId ?? null,
        tracking_event_id: p.trackingEventId ?? null,
      };
    });

    const pA = driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    // Flush B pendant A actif → coalescé
    const pB = driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(await pB).toMatchObject({ sent: 0 });

    resolve429(rateLimitError(60));
    await pA;
    await new Promise((r) => setTimeout(r, 30));

    // Un seul PUT (celui qui a 429) — pas de reprise immédiate via pending
    expect(mockSendDriverLocation).toHaveBeenCalledTimes(1);
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.suspendReason).toBe("rate_limit");
    expect(snap.queueDepth).toBe(10);
  });
});
