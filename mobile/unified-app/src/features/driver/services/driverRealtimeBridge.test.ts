import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import { startDriverRealtimeBridge } from "./driverRealtimeBridge";

jest.mock(
  "expo-battery",
  () => ({
    getBatteryLevelAsync: async () => 0.5,
  }),
  { virtual: true }
);

const mockSubscribe = jest.fn();
const mockSubscribeDriverEvents = jest.fn();
const mockConnect = jest.fn();
const mockDisconnect = jest.fn();
const mockMarkByIds = jest.fn<() => Promise<number>>().mockResolvedValue(0);
const mockMarkByWatermark = jest.fn<() => Promise<number>>().mockResolvedValue(0);
const mockReleaseSocketForHttp = jest.fn<() => Promise<number>>().mockResolvedValue(0);
const mockReconcileSession = jest.fn<() => Promise<string>>().mockResolvedValue("trk_sess_new");
const mockSyncBridgeQueueDepth = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);

jest.mock("./driverTrackingBridge", () => {
  const actual = jest.requireActual("./driverTrackingBridge") as typeof import("./driverTrackingBridge");
  return {
    ...actual,
    syncBridgeQueueDepthFromPersistence: () => mockSyncBridgeQueueDepth(),
  };
});

jest.mock("./driverTrackingQueue", () => ({
  driverTrackingQueue: {
    markBackendAckedByIds: (...args: unknown[]) => mockMarkByIds(...args),
    markBackendAckedByWatermark: (...args: unknown[]) => mockMarkByWatermark(...args),
    releaseSocketEmittedForHttpRetry: (...args: unknown[]) => mockReleaseSocketForHttp(...args),
    reconcileAfterSessionConflict: (...args: unknown[]) => mockReconcileSession(...args),
  },
}));

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    connect: (...args: unknown[]) => mockConnect(...args),
    disconnect: () => mockDisconnect(),
    subscribe: (cb: unknown) => mockSubscribe(cb),
    subscribeDriverEvents: (cb: unknown) => mockSubscribeDriverEvents(cb),
  },
}));

jest.mock("../../../core/featureFlags/registry", () => ({
  isFeatureEnabled: () => false,
}));

jest.mock("../tracking", () => ({
  flushTrackingQueue: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
}));
const mockFlushTrackingQueue = jest.requireMock("../tracking").flushTrackingQueue as jest.Mock;

jest.mock("./socketBatchPacing", () => ({
  recordSocketBatchRateLimited: jest.fn(),
}));

describe("driverRealtimeBridge ack handling", () => {
  beforeEach(() => {
    mockSubscribe.mockReset();
    mockSubscribeDriverEvents.mockReset();
    mockConnect.mockReset();
    mockDisconnect.mockReset();
    mockMarkByIds.mockReset();
    mockMarkByWatermark.mockReset();
    mockReleaseSocketForHttp.mockReset();
    mockReconcileSession.mockReset();
    mockMarkByIds.mockResolvedValue(0);
    mockMarkByWatermark.mockResolvedValue(0);
    mockReleaseSocketForHttp.mockResolvedValue(0);
    mockReconcileSession.mockResolvedValue("trk_sess_new");
    mockFlushTrackingQueue.mockReset();
    mockFlushTrackingQueue.mockResolvedValue(undefined);
    mockSyncBridgeQueueDepth.mockReset();
    mockSyncBridgeQueueDepth.mockResolvedValue(undefined);
    mockSubscribe.mockImplementation((cb: (snapshot: any) => void) => {
      cb({ connected: false, activeContextId: "driver:1", mode: "polling" });
      return () => undefined;
    });
    mockSubscribeDriverEvents.mockImplementation((cb: (event: unknown) => void) => {
      cb({
        event_type: "driver_location_batch_ack",
        payload: {
          tracking_event_ids: ["a", "b"],
          ack_last_sequence_id: 15,
        },
      });
      return () => undefined;
    });
  });

  it("handles batch ack ids, watermark and resyncs bridge queue depth", async () => {
    const queryClient = new QueryClient();
    const dispose = startDriverRealtimeBridge(queryClient, "driver:1", {
      enableSocket: true,
    });
    expect(mockMarkByIds).toHaveBeenCalledWith(["a", "b"]);
    expect(mockMarkByWatermark).toHaveBeenCalledWith(15);
    await Promise.resolve();
    await Promise.resolve();
    expect(mockSyncBridgeQueueDepth).toHaveBeenCalled();
    dispose();
  });

  it("does not ack queue on rate_limited batch ack and schedules HTTP retry", async () => {
    jest.useFakeTimers();
    mockSubscribeDriverEvents.mockImplementation((cb: (event: unknown) => void) => {
      cb({
        event_type: "driver_location_batch_ack",
        payload: {
          rate_limited: true,
          positions_count: 0,
          tracking_event_ids: ["trk_lost"],
          ack_last_sequence_id: 99,
          retry_after_seconds: 4,
        },
      });
      return () => undefined;
    });
    const queryClient = new QueryClient();
    const dispose = startDriverRealtimeBridge(queryClient, "driver:1", {
      enableSocket: true,
    });
    expect(mockMarkByIds).not.toHaveBeenCalled();
    expect(mockMarkByWatermark).not.toHaveBeenCalled();
    expect(mockReleaseSocketForHttp).toHaveBeenCalled();
    dispose();
    jest.useRealTimers();
  });

  it("reconciles session and flushes after session_conflict batch ack", async () => {
    mockSubscribeDriverEvents.mockImplementation((cb: (event: unknown) => void) => {
      cb({
        event_type: "driver_location_batch_ack",
        payload: {
          session_conflict: true,
          positions_count: 0,
          tracking_event_ids: ["trk_stale"],
          ack_last_sequence_id: 42,
        },
      });
      return () => undefined;
    });
    const queryClient = new QueryClient();
    const dispose = startDriverRealtimeBridge(queryClient, "driver:1", {
      enableSocket: true,
    });
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(mockMarkByIds).toHaveBeenCalledWith(["trk_stale"]);
    expect(mockMarkByWatermark).toHaveBeenCalledWith(42);
    expect(mockReconcileSession).toHaveBeenCalled();
    await Promise.resolve();
    expect(mockFlushTrackingQueue).toHaveBeenCalled();
    dispose();
  });
});
