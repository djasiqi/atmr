/**
 * Q3 — reconnect ≠ conflict ; immutabilité file ; coalesce rotations.
 * Gates bloquants avant build >132.
 */
import { beforeEach, describe, expect, it, jest } from "@jest/globals";

const mockSendDriverLocation = jest.fn();
const mockRegisterTrackingSession = jest.fn();
const mockFetchTrackingWatermark = jest.fn();
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
  leaseAllowsTransport: () => true,
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
const { emitDriverTelemetry } = require("../../../core/observability/driverTelemetry") as {
  emitDriverTelemetry: jest.Mock;
};

function baseEnqueue() {
  return {
    missionId: 38243 as number | null,
    appState: "active" as const,
    locationMode: "mission" as const,
    payload: {
      latitude: 46.2,
      longitude: 6.1,
      locationMode: "mission" as const,
    },
  };
}

describe("driverTrackingQueue Q3 session ownership", () => {
  beforeEach(async () => {
    mockAsyncStorageGetItem.mockReset();
    mockAsyncStorageSetItem.mockReset();
    mockSendDriverLocation.mockReset();
    mockRegisterTrackingSession.mockReset();
    mockFetchTrackingWatermark.mockReset();
    emitDriverTelemetry.mockReset();
    mockAsyncStorageGetItem.mockResolvedValue(null);
    mockAsyncStorageSetItem.mockResolvedValue(undefined);
    mockRegisterTrackingSession.mockImplementation(async (body: unknown) => {
      const b = body as { tracking_session_id?: string };
      return {
        tracking_session_id: b.tracking_session_id ?? "sess",
        session_generation: Math.floor(Math.random() * 1000) + 1,
        first_sequence_id: 1,
        status: "active",
      };
    });
    trackingQueueStore._resetMemoryForTests();
    await driverTrackingQueue.resetForTests();
  });

  it("vrai session_conflict → exactement 1 rotate (register)", async () => {
    await driverTrackingQueue.beginNewTrackingSession();
    const before = await driverTrackingQueue.getSnapshot();
    expect(before.sessionReadiness).toBe("READY");
    const afterId = await driverTrackingQueue.reconcileAfterSessionConflict();
    expect(afterId).not.toBe(before.trackingSessionId);
    expect(mockRegisterTrackingSession).toHaveBeenCalledTimes(2); // begin + conflict
    const snap = await driverTrackingQueue.getSnapshot();
    expect(snap.trackingSessionId).toBe(afterId);
    expect(snap.sessionReadiness).toBe("READY");
  });

  it("2 session_conflict concurrents → exactement 1 nouvelle session", async () => {
    await driverTrackingQueue.beginNewTrackingSession();
    const before = await driverTrackingQueue.getSnapshot();

    mockRegisterTrackingSession.mockReset();
    const pendingResolvers: ((v: unknown) => void)[] = [];
    mockRegisterTrackingSession.mockImplementation(
      () =>
        new Promise((resolve) => {
          pendingResolvers.push(resolve);
        })
    );

    const p1 = driverTrackingQueue.reconcileAfterSessionConflict();
    for (let i = 0; i < 50 && pendingResolvers.length === 0; i += 1) {
      await Promise.resolve();
    }
    const p2 = driverTrackingQueue.reconcileAfterSessionConflict();
    for (let i = 0; i < 50; i += 1) {
      await Promise.resolve();
    }
    expect(pendingResolvers.length).toBe(1);
    const sid = (
      driverTrackingQueue as unknown as { trackingSessionId: string }
    ).trackingSessionId;
    pendingResolvers[0]?.({
      tracking_session_id: sid,
      session_generation: 99,
      first_sequence_id: 1,
      status: "active",
    });
    const [id1, id2] = await Promise.all([p1, p2]);
    expect(id1).toBe(id2);
    expect(id1).not.toBe(before.trackingSessionId);
    expect(mockRegisterTrackingSession).toHaveBeenCalledTimes(1);
    expect(emitDriverTelemetry).toHaveBeenCalledWith(
      "tracking.session.rotate_skipped",
      expect.objectContaining({ skip_cause: "rotate_in_flight_coalesced" })
    );
  });

  it("backlog A + rotate légitime vers B → items A immuables, nouveaux sur B", async () => {
    await driverTrackingQueue.beginNewTrackingSession();
    const sessionA = (await driverTrackingQueue.getSnapshot()).trackingSessionId;
    const itemA = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(itemA?.trackingSessionId).toBe(sessionA);

    const sessionB = await driverTrackingQueue.reconcileAfterSessionConflict();
    expect(sessionB).not.toBe(sessionA);
    expect(itemA?.trackingSessionId).toBe(sessionA);

    const itemB = await driverTrackingQueue.enqueue(baseEnqueue());
    expect(itemB?.trackingSessionId).toBe(sessionB);
    expect(itemB?.trackingSessionId).not.toBe(sessionA);
  });

  it("begin_new sur session READY reste autorisé (rotate explicite)", async () => {
    await driverTrackingQueue.beginNewTrackingSession();
    const a = (await driverTrackingQueue.getSnapshot()).trackingSessionId;
    await driverTrackingQueue.beginNewTrackingSession();
    const b = (await driverTrackingQueue.getSnapshot()).trackingSessionId;
    expect(b).not.toBe(a);
    expect(mockRegisterTrackingSession.mock.calls.length).toBeGreaterThanOrEqual(2);
  });
});
