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
let authRefreshCb: (() => void) | null = null;

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
  onAuthRefreshSuccess: (cb: () => void) => {
    authRefreshCb = cb;
  },
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

const { trackingQueueStore } = require("./trackingQueueStore") as typeof import("./trackingQueueStore");
const { driverTrackingQueue } = require("./driverTrackingQueue") as typeof import("./driverTrackingQueue");

describe("driverTrackingQueue http202AckSemantics + session", () => {
  beforeEach(async () => {
    mockAsyncStorageGetItem.mockReset();
    mockAsyncStorageSetItem.mockReset();
    mockSendDriverLocation.mockReset();
    mockRegisterTrackingSession.mockReset();
    mockFetchTrackingWatermark.mockReset();
    authRefreshCb = null;
    mockAsyncStorageGetItem.mockResolvedValue(null);
    mockAsyncStorageSetItem.mockResolvedValue(undefined);
    mockRegisterTrackingSession.mockResolvedValue({
      tracking_session_id: "sess",
      session_generation: 3,
      first_sequence_id: 1,
      status: "active",
    });
    trackingQueueStore._resetMemoryForTests();
    await driverTrackingQueue.resetForTests();
  });

  it("beginNewTrackingSession crée une session locale non nulle sans attendre le réseau", async () => {
    let resolveRegister: ((v: unknown) => void) | null = null;
    mockRegisterTrackingSession.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRegister = resolve;
        })
    );
    await driverTrackingQueue.beginNewTrackingSession();
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.trackingSessionId).toMatch(/^trk_sess_/);
    expect(snap.sequenceCounter).toBe(0);
    // débloquer pour ne pas laisser de promesse pendante
    resolveRegister?.({
      tracking_session_id: snap.trackingSessionId,
      session_generation: 1,
      first_sequence_id: 1,
      status: "active",
    });
  });

  it("202 queued_async ne retire pas l'item (pas d'ACK final)", async () => {
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "ingested_non_persisted",
      durability: "queued_async",
      location_event_id: "evt_q",
      tracking_event_id: "evt_q",
    });
    await driverTrackingQueue.beginNewTrackingSession();
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.5,
        longitude: 6.6,
        locationMode: "availability_presence",
        trackingEventId: "evt_q",
      },
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.backendAcked).toBe(0);
    expect(flush.queueDepth).toBeGreaterThanOrEqual(1);
  });

  it("200 persisted_sync avec location_event_id retire l'item", async () => {
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
    await driverTrackingQueue.beginNewTrackingSession();
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.5,
        longitude: 6.6,
        locationMode: "availability_presence",
      },
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.backendAcked).toBe(1);
  });

  it("200 accepted sans durability ne tombstone pas", async () => {
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "ingested",
      durability: null,
      location_event_id: "evt_legacy",
      tracking_event_id: "evt_legacy",
    });
    await driverTrackingQueue.beginNewTrackingSession();
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.5,
        longitude: 6.6,
        locationMode: "availability_presence",
        trackingEventId: "evt_legacy",
      },
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.backendAcked).toBe(0);
    expect(flush.queueDepth).toBeGreaterThanOrEqual(1);
  });

  it("auth refresh ne clear que la suspension auth", async () => {
    await driverTrackingQueue.beginNewTrackingSession();
    await driverTrackingQueue.activateSuspensionForTests(60_000, "rate_limit");
    authRefreshCb?.();
    await Promise.resolve();
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.suspendReason).toBe("rate_limit");
  });

  it("persisted + durability null → conserve la file", async () => {
    mockSendDriverLocation.mockImplementation(async (payload: unknown) => {
      const p = payload as { trackingEventId?: string };
      return {
        ack_status: "persisted",
        durability: null,
        location_event_id: p.trackingEventId ?? null,
        tracking_event_id: p.trackingEventId ?? null,
      };
    });
    await driverTrackingQueue.beginNewTrackingSession();
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: { latitude: 46.5, longitude: 6.6, locationMode: "availability_presence" },
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.backendAcked).toBe(0);
    expect(flush.queueDepth).toBeGreaterThanOrEqual(1);
  });

  it("persisted_sync + location_event_id null → conserve", async () => {
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: null,
      tracking_event_id: null,
    });
    await driverTrackingQueue.beginNewTrackingSession();
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: { latitude: 46.5, longitude: 6.6, locationMode: "availability_presence" },
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.backendAcked).toBe(0);
    expect(flush.queueDepth).toBeGreaterThanOrEqual(1);
  });

  it("persisted_sync + mauvais event id → conserve", async () => {
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: "other_event",
      tracking_event_id: "other_event",
    });
    await driverTrackingQueue.beginNewTrackingSession();
    await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "availability_presence",
      payload: { latitude: 46.5, longitude: 6.6, locationMode: "availability_presence" },
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.backendAcked).toBe(0);
    expect(flush.queueDepth).toBeGreaterThanOrEqual(1);
  });
});
