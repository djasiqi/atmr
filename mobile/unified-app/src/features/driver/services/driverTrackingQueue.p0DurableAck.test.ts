import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { trackingQueueStore } from "./trackingQueueStore";

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

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    sendDriverLocationBatch: (...args: unknown[]) => mockSendDriverLocationBatch(...args),
    isDriverSocketReady: () => mockIsDriverSocketReady(),
  },
}));

jest.mock("../../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
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

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { driverTrackingQueue } = require("./driverTrackingQueue") as typeof import("./driverTrackingQueue");

describe("P0 file GPS — preuve durable + restart", () => {
  beforeEach(async () => {
    jest.useFakeTimers({ advanceTimers: true });
    (isFeatureEnabled as jest.Mock).mockImplementation(
      (key: unknown) => key === "tracking_real_ack_semantics_enabled"
    );
    mockSendDriverLocation.mockReset();
    mockSendDriverLocationBatch.mockReset();
    mockIsDriverSocketReady.mockReturnValue(false);
    trackingQueueStore._resetMemoryForTests();
    await driverTrackingQueue.resetForTests();
  });

  afterEach(async () => {
    await driverTrackingQueue.resetForTests();
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it("ACK HTTP persisted_sync retire durablement ; restart ne ressuscite pas", async () => {
    const item = await driverTrackingQueue.enqueue({
      missionId: 9,
      appState: "active",
      locationMode: "mission_live",
      payload: {
        latitude: 46.2,
        longitude: 6.1,
        missionId: 9,
        locationMode: "mission_live",
      },
    });
    mockSendDriverLocation.mockResolvedValue({
      ack_status: "persisted",
      durability: "persisted_sync",
      location_event_id: item.id,
      tracking_event_id: item.id,
    });
    const flush = await driverTrackingQueue.flush({
      forceHttpFallback: true,
      networkProfile: "normal",
    });
    expect(flush.persistedEventIds).toContain(item.id);
    expect(flush.queueDepth).toBe(0);

    await driverTrackingQueue.resetForTests();
    // Simule redémarrage : listActive ne doit pas renvoyer l'item persisted
    const active = await trackingQueueStore.listActive();
    expect(active.find((r) => r.locationEventId === item.id)).toBeUndefined();
  });

  it("socket ACK ne retire pas de la file active", async () => {
    (isFeatureEnabled as jest.Mock).mockImplementation(
      (key: unknown) =>
        key === "tracking_real_ack_semantics_enabled" ||
        key === "tracking_socket_gps_ingest_enabled"
    );
    mockIsDriverSocketReady.mockReturnValue(true);
    mockSendDriverLocationBatch.mockReturnValue(true);
    const item = await driverTrackingQueue.enqueue({
      missionId: 3,
      appState: "active",
      locationMode: "mission_live",
      payload: {
        latitude: 46,
        longitude: 6,
        missionId: 3,
        locationMode: "mission_live",
      },
    });
    await driverTrackingQueue.flush({ networkProfile: "normal" });
    const acked = await driverTrackingQueue.markBackendAckedByIds([item.id]);
    expect(acked).toBe(1);
    expect((await driverTrackingQueue.getSnapshot()).queueDepth).toBe(1);
    const rows = await trackingQueueStore.listActive();
    expect(rows.some((r) => r.locationEventId === item.id)).toBe(true);
  });
});
