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

describe("driverRealtimeBridge ack handling", () => {
  beforeEach(() => {
    mockSubscribe.mockReset();
    mockSubscribeDriverEvents.mockReset();
    mockConnect.mockReset();
    mockDisconnect.mockReset();
    mockMarkByIds.mockReset();
    mockMarkByWatermark.mockReset();
    mockMarkByIds.mockResolvedValue(0);
    mockMarkByWatermark.mockResolvedValue(0);
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
});
