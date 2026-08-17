/**
 * T1/T4 — retries HTTP du même eid : body wire deepEqual (payload figé à l'enqueue).
 */
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
const {
  buildDriverLocationHttpBody,
} = require("./freezeTrackingLocationPayload") as typeof import("./freezeTrackingLocationPayload");

describe("driverTrackingQueue payload immutability", () => {
  beforeEach(async () => {
    mockAsyncStorageGetItem.mockReset();
    mockAsyncStorageSetItem.mockReset();
    mockSendDriverLocation.mockReset();
    mockRegisterTrackingSession.mockReset();
    mockFetchTrackingWatermark.mockReset();
    mockAsyncStorageGetItem.mockResolvedValue(null);
    mockAsyncStorageSetItem.mockResolvedValue(undefined);
    mockRegisterTrackingSession.mockResolvedValue({
      tracking_session_id: "sess_immut",
      session_generation: 3,
      first_sequence_id: 1,
      status: "active",
    });
    trackingQueueStore._resetMemoryForTests();
    await driverTrackingQueue.resetForTests();
    await driverTrackingQueue.beginNewTrackingSession();
  });

  it("T1/T4 — deux flush HTTP du même eid → body deepEqual (recorded_at stable)", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 38243,
      appState: "background",
      locationMode: "mission_live",
      captureId: "cap_immut_1",
      payload: {
        latitude: 46.2116156,
        longitude: 6.1262053,
        accuracy: 7.8,
        timestamp: "2026-08-17T15:27:31.105Z",
        isBackground: true,
        missionId: 38243,
      },
    });
    expect(item).not.toBeNull();
    expect(item!.payload.recordedAt).toBe("2026-08-17T15:27:31.105Z");
    expect(item!.payload.trackingEventId).toBe(item!.id);

    mockSendDriverLocation.mockImplementation(async (payload) => {
      const p = payload as { trackingEventId?: string };
      return {
        ack_status: "ingested_non_persisted",
        durability: "queued_async",
        tracking_event_id: p.trackingEventId ?? item!.id,
        location_event_id: p.trackingEventId ?? item!.id,
      };
    });

    await driverTrackingQueue.flush({ forceHttpFallback: true });
    await driverTrackingQueue.flush({ forceHttpFallback: true });

    expect(mockSendDriverLocation.mock.calls.length).toBeGreaterThanOrEqual(2);
    const body0 = buildDriverLocationHttpBody(
      mockSendDriverLocation.mock.calls[0]![0] as Parameters<
        typeof buildDriverLocationHttpBody
      >[0]
    );
    const body1 = buildDriverLocationHttpBody(
      mockSendDriverLocation.mock.calls[1]![0] as Parameters<
        typeof buildDriverLocationHttpBody
      >[0]
    );
    expect(body1).toEqual(body0);
    expect(body0.recorded_at).toBe("2026-08-17T15:27:31.105Z");
    expect(body0.tracking_event_id).toBe(item!.id);
  });
});
