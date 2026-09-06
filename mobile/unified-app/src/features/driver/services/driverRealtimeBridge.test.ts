import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import { startDriverRealtimeBridge } from "./driverRealtimeBridge";
import { resetDriverForegroundResumeAuthorityForTests } from "../driverForegroundResumeAuthority";

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
const mockTombstoneByIds = jest.fn<() => Promise<number>>().mockResolvedValue(0);
const mockReleaseSocketForHttp = jest.fn<() => Promise<number>>().mockResolvedValue(0);
const mockReconcileSession = jest.fn<() => Promise<string>>().mockResolvedValue("trk_sess_new");
const mockGetSnapshot = jest.fn<() => Promise<{
  trackingSessionId: string;
  sessionGeneration: number | null;
  queueDepth: number;
}>>();
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
    tombstoneByIds: (...args: unknown[]) => mockTombstoneByIds(...args),
    releaseSocketEmittedForHttpRetry: (...args: unknown[]) => mockReleaseSocketForHttp(...args),
    reconcileAfterSessionConflict: (...args: unknown[]) => mockReconcileSession(...args),
    getSnapshot: (...args: unknown[]) => mockGetSnapshot(...args),
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
  isFeatureEnabled: jest.fn(() => false),
}));

jest.mock("../tracking", () => ({
  flushTrackingQueue: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
}));
const mockFlushTrackingQueue = jest.requireMock("../tracking").flushTrackingQueue as jest.Mock;
const mockIsFeatureEnabled = jest.requireMock("../../../core/featureFlags/registry")
  .isFeatureEnabled as jest.Mock;

jest.mock("./socketBatchPacing", () => ({
  recordSocketBatchRateLimited: jest.fn(),
}));

describe("driverRealtimeBridge ack handling", () => {
  beforeEach(() => {
    resetDriverForegroundResumeAuthorityForTests();
    mockSubscribe.mockReset();
    mockSubscribeDriverEvents.mockReset();
    mockConnect.mockReset();
    mockDisconnect.mockReset();
    mockMarkByIds.mockReset();
    mockMarkByWatermark.mockReset();
    mockTombstoneByIds.mockReset();
    mockReleaseSocketForHttp.mockReset();
    mockReconcileSession.mockReset();
    mockGetSnapshot.mockReset();
    mockIsFeatureEnabled.mockReset();
    mockIsFeatureEnabled.mockImplementation(() => false);
    mockMarkByIds.mockResolvedValue(0);
    mockMarkByWatermark.mockResolvedValue(0);
    mockTombstoneByIds.mockResolvedValue(0);
    mockReleaseSocketForHttp.mockResolvedValue(0);
    mockReconcileSession.mockResolvedValue("trk_sess_new");
    mockGetSnapshot.mockResolvedValue({
      trackingSessionId: "trk_sess_stable_a",
      sessionGeneration: 10,
      queueDepth: 2,
    });
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

  it("handles batch ack ids as socket_acked without durable watermark purge", async () => {
    const queryClient = new QueryClient();
    const dispose = startDriverRealtimeBridge(queryClient, "driver:1", {
      enableSocket: true,
    });
    expect(mockMarkByIds).toHaveBeenCalledWith(["a", "b"]);
    expect(mockMarkByWatermark).not.toHaveBeenCalled();
    await Promise.resolve();
    await Promise.resolve();
    expect(mockSyncBridgeQueueDepth).toHaveBeenCalled();
    dispose();
  });

  it("releases to HTTP on ingest_disabled without durable purge", async () => {
    mockSubscribeDriverEvents.mockImplementation((cb: (event: unknown) => void) => {
      cb({
        event_type: "driver_location_batch_ack",
        payload: {
          ingest_disabled: true,
          retry_event_ids: ["trk_1", "trk_2"],
          positions_count: 0,
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
    expect(mockMarkByIds).not.toHaveBeenCalled();
    expect(mockMarkByWatermark).not.toHaveBeenCalled();
    expect(mockReleaseSocketForHttp).toHaveBeenCalled();
    expect(mockFlushTrackingQueue).toHaveBeenCalled();
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
    expect(mockTombstoneByIds).toHaveBeenCalledWith(["trk_stale"], "session_conflict");
    expect(mockMarkByIds).not.toHaveBeenCalled();
    expect(mockMarkByWatermark).not.toHaveBeenCalled();
    expect(mockReconcileSession).toHaveBeenCalled();
    await Promise.resolve();
    expect(mockFlushTrackingQueue).toHaveBeenCalled();
    dispose();
  });

  it("Q3-A: socket reconnect flush/resync without session rotate", async () => {
    mockIsFeatureEnabled.mockImplementation(
      (key: string) => key === "tracking_resume_resync_enabled"
    );
    let lifecycleCb: ((snapshot: {
      connected: boolean;
      activeContextId: string;
      mode: string;
    }) => void) | null = null;
    mockSubscribe.mockImplementation((cb: (snapshot: any) => void) => {
      lifecycleCb = cb;
      cb({ connected: false, activeContextId: "driver:1", mode: "polling" });
      return () => undefined;
    });
    mockSubscribeDriverEvents.mockImplementation(() => () => undefined);

    const queryClient = new QueryClient();
    const dispose = startDriverRealtimeBridge(queryClient, "driver:1", {
      enableSocket: true,
    });
    expect(lifecycleCb).toBeTruthy();
    lifecycleCb!({
      connected: true,
      activeContextId: "driver:1",
      mode: "socket",
    });
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(mockReleaseSocketForHttp).toHaveBeenCalled();
    expect(mockFlushTrackingQueue).toHaveBeenCalled();
    expect(mockReconcileSession).not.toHaveBeenCalled();
    dispose();
  });

  it("Q3-A: two close reconnects still never rotate session", async () => {
    mockIsFeatureEnabled.mockImplementation(
      (key: string) =>
        key === "tracking_resume_resync_enabled" ||
        key === "realtime_resync_transition_gate_enabled"
    );
    let lifecycleCb: ((snapshot: {
      connected: boolean;
      activeContextId: string;
      mode: string;
    }) => void) | null = null;
    mockSubscribe.mockImplementation((cb: (snapshot: any) => void) => {
      lifecycleCb = cb;
      cb({ connected: false, activeContextId: "driver:1", mode: "polling" });
      return () => undefined;
    });
    mockSubscribeDriverEvents.mockImplementation(() => () => undefined);

    const queryClient = new QueryClient();
    const dispose = startDriverRealtimeBridge(queryClient, "driver:1", {
      enableSocket: true,
    });

    lifecycleCb!({ connected: true, activeContextId: "driver:1", mode: "socket" });
    await Promise.resolve();
    await Promise.resolve();
    lifecycleCb!({ connected: false, activeContextId: "driver:1", mode: "polling" });
    // Au-delà du throttle 3s défaut : 2e reconnect doit encore flusher sans rotate.
    const nowSpy = jest.spyOn(Date, "now").mockReturnValue(Date.now() + 10_000);
    lifecycleCb!({ connected: true, activeContextId: "driver:1", mode: "socket" });
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    nowSpy.mockRestore();

    expect(mockReconcileSession).not.toHaveBeenCalled();
    expect(mockReleaseSocketForHttp.mock.calls.length).toBeGreaterThanOrEqual(1);
    dispose();
  });
});
