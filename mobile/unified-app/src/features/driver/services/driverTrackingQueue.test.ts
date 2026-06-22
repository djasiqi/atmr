import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";

const mockSendDriverLocation = jest.fn<
  (payload: unknown) => Promise<{ ack_status: "accepted" | "duplicate" | "stale" | "ignored" | "rejected" }>
>();
const mockSendDriverLocationBatch = jest.fn<(payload: unknown[]) => boolean>();
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

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    sendDriverLocationBatch: mockSendDriverLocationBatch,
    isDriverSocketReady: () => true,
  },
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: mockEmitDriverTelemetry,
}));

// Fabrique autonome (ne pas capter de const du fichier : ordre d’exécution Jest)
jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: jest.fn((key: unknown) =>
    key === "tracking_real_ack_semantics_enabled" || key === "tracking_queue_compaction_enabled"
  ),
}));

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { driverTrackingQueue } = require("./driverTrackingQueue") as typeof import("./driverTrackingQueue");

describe("driverTrackingQueue", () => {
  const mockFF = isFeatureEnabled as jest.MockedFunction<typeof isFeatureEnabled>;

  beforeEach(() => {
    mockAsyncStorageGetItem.mockReset();
    mockAsyncStorageSetItem.mockReset();
    mockSendDriverLocation.mockReset();
    mockSendDriverLocationBatch.mockReset();
    mockEmitDriverTelemetry.mockReset();
    mockFF.mockClear();
    mockFF.mockImplementation(
      (key) =>
        key === "tracking_real_ack_semantics_enabled" || key === "tracking_queue_compaction_enabled"
    );

    mockAsyncStorageGetItem.mockResolvedValue(null);
    mockAsyncStorageSetItem.mockResolvedValue(undefined);
    mockSendDriverLocation.mockResolvedValue({ ack_status: "accepted" });
    mockSendDriverLocationBatch.mockReturnValue(false);
  });

  it("keeps socket emitted item until backend ack", async () => {
    await driverTrackingQueue.enqueue({
      missionId: 42,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 46.5, longitude: 6.6, missionId: 42, locationMode: "mission_live" },
    });

    mockSendDriverLocationBatch.mockReturnValueOnce(true);
    const firstFlush = await driverTrackingQueue.flush({
      ackStaleMs: 60_000,
      networkProfile: "normal",
    });
    expect(firstFlush.socketEmitted).toBe(1);
    expect(firstFlush.backendAcked).toBe(0);
    expect(firstFlush.queueDepth).toBe(1);

    const queued = await driverTrackingQueue.getSnapshot();
    expect(queued.queueDepth).toBe(1);

    const itemRaw = mockAsyncStorageSetItem.mock.calls.at(-1)?.[1] as string;
    const itemId = (JSON.parse(itemRaw) as { id: string }[])[0].id;
    const ackedCount = await driverTrackingQueue.markBackendAckedByIds([itemId]);
    expect(ackedCount).toBe(1);
    expect((await driverTrackingQueue.getSnapshot()).queueDepth).toBe(0);
  });

  it("falls back to http when socket ack is stale", async () => {
    await driverTrackingQueue.enqueue({
      missionId: 7,
      appState: "background",
      locationMode: "mission_live",
      payload: { latitude: 45, longitude: 5, missionId: 7, locationMode: "mission_live" },
    });

    mockSendDriverLocationBatch.mockReturnValueOnce(true);
    await driverTrackingQueue.flush({ ackStaleMs: 1, networkProfile: "poor" });

    await new Promise((resolve) => setTimeout(resolve, 2));
    mockSendDriverLocation.mockResolvedValueOnce({ ack_status: "duplicate" });
    const secondFlush = await driverTrackingQueue.flush({ ackStaleMs: 1, networkProfile: "poor" });

    expect(secondFlush.backendAcked).toBe(1);
    expect(secondFlush.queueDepth).toBe(0);
    expect(mockSendDriverLocation).toHaveBeenCalled();
  });

  it("persists queue after enqueue for restart safety", async () => {
    await driverTrackingQueue.enqueue({
      missionId: 33,
      appState: "background",
      locationMode: "mission_live",
      payload: { latitude: 1, longitude: 2, missionId: 33, locationMode: "mission_live" },
    });
    expect(mockAsyncStorageSetItem).toHaveBeenCalled();
  });

  it("purges only items with sequence_id <= ack_last_sequence_id", async () => {
    await driverTrackingQueue.enqueue({
      missionId: 1,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 1, longitude: 1, missionId: 1, locationMode: "mission_live" },
    });
    await driverTrackingQueue.enqueue({
      missionId: 1,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 2, longitude: 2, missionId: 1, locationMode: "mission_live" },
    });
    const queueWrite = [...mockAsyncStorageSetItem.mock.calls]
      .reverse()
      .find((call) => String(call[0]).includes("driver_tracking_delivery_queue_v1"));
    const persistedRaw = (queueWrite?.[1] as string) ?? "[]";
    const persisted = JSON.parse(persistedRaw) as { sequenceId: number }[];
    const ackWatermark = Math.min(...persisted.map((item) => item.sequenceId));
    const before = await driverTrackingQueue.getSnapshot();
    const acked = await driverTrackingQueue.markBackendAckedByWatermark(ackWatermark);
    expect(acked).toBe(1);
    const snapshot = await driverTrackingQueue.getSnapshot();
    expect(snapshot.queueDepth).toBe(before.queueDepth - 1);
  });

  it("downgrade mission_live sans missionId vers availability_presence", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: null,
      appState: "active",
      locationMode: "mission_live",
      payload: { latitude: 1, longitude: 1, locationMode: "mission_live" },
    });
    expect(item.locationMode).toBe("availability_presence");
  });

  it("availability_presence n'utilise jamais le socket", async () => {
    mockSendDriverLocationBatch.mockReturnValue(true);
    mockSendDriverLocation.mockResolvedValue({ ack_status: "accepted" });
    while ((await driverTrackingQueue.getSnapshot()).queueDepth > 0) {
      await driverTrackingQueue.flush({
        ackStaleMs: 60_000,
        networkProfile: "normal",
        forceHttpFallback: true,
      });
    }
    mockSendDriverLocationBatch.mockClear();
    mockSendDriverLocation.mockClear();

    await driverTrackingQueue.enqueue({
      missionId: 99,
      appState: "active",
      locationMode: "availability_presence",
      payload: {
        latitude: 46.5,
        longitude: 6.6,
        missionId: 99,
        locationMode: "availability_presence",
      },
    });

    const flush = await driverTrackingQueue.flush({
      ackStaleMs: 60_000,
      networkProfile: "normal",
    });

    expect(mockSendDriverLocationBatch).not.toHaveBeenCalled();
    expect(mockSendDriverLocation).toHaveBeenCalled();
    expect(flush.backendAcked).toBeGreaterThanOrEqual(1);
  });
});
