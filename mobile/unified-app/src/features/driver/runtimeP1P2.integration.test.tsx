import React from "react";
import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { act, create } from "react-test-renderer";
import { AppState } from "react-native";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NotificationsProvider } from "../../core/providers/NotificationsProvider";
import { useDriverRealtimeSync } from "./hooks";
import {
  emitDriverProcessForegroundForTests,
  getDriverForegroundResumeListenerCountForTests,
  resetDriverForegroundResumeAuthorityForTests,
} from "./driverForegroundResumeAuthority";
import {
  resetDriverSessionNetworkGateForTests,
  setDriverSessionNetworkReady,
} from "../../core/network/driverSessionNetworkGate";

const mockRouterPush = jest.fn();
const mockRegisterDriverPushToken = jest.fn();
const mockHandleDriverPushQuickAction = jest.fn();
const mockStartDriverRealtimeBridge = jest.fn(() => jest.fn());
const mockRefreshAuthTokenNow = jest.fn();
const mockReconcileDriverMissions = jest.fn();
const mockOfflineFlush = jest.fn();
const mockFlushTrackingQueue = jest.fn();
const mockEmitDriverTelemetry = jest.fn();
const mockIsFeatureEnabled = jest.fn();

let mockAppStateHandlers: ((state: "active" | "inactive" | "background") => void)[] = [];
let mockNotificationResponseHandler: ((response: any) => void) | null = null;
let mockSessionState: {
  status: "ready" | "idle" | "error";
  activeContext:
    | { context_id: string; context_type: "driver" | "client"; permissions?: string[] }
    | null;
  bootstrap: { user: { id: number | null } } | null;
};

jest.mock("expo-router", () => ({
  useRouter: () => ({ push: (...args: unknown[]) => mockRouterPush(...args) }),
}));

jest.mock("expo-notifications", () => ({
  setNotificationHandler: jest.fn(),
  getPermissionsAsync: jest.fn().mockResolvedValue({ granted: true, status: "granted" }),
  requestPermissionsAsync: jest.fn().mockResolvedValue({ granted: true, status: "granted" }),
  addNotificationReceivedListener: jest.fn(() => ({ remove: jest.fn() })),
  addNotificationResponseReceivedListener: jest.fn((cb: (response: any) => void) => {
    mockNotificationResponseHandler = cb;
    return { remove: jest.fn() };
  }),
  addPushTokenListener: jest.fn(() => ({ remove: jest.fn() })),
  getExpoPushTokenAsync: jest.fn().mockResolvedValue({ data: "ExpoPushToken[integration]" }),
  getLastNotificationResponseAsync: jest.fn().mockResolvedValue({
    notification: {
      request: {
        identifier: "coldstart-1",
        content: {
          data: { mission_id: 501, type: "mission_assigned", event_id: "evt-cold-1" },
        },
      },
    },
  }),
}));

jest.mock("../../core/notifications/notificationDisclosurePersistence", () => ({
  readNotificationDisclosureAccepted: jest.fn(async () => true),
  ensureNotificationDisclosureSyncedWithOsPermission: jest.fn(async () => true),
  subscribeNotificationDisclosureAccepted: jest.fn(() => () => undefined),
}));

jest.mock("../../core/notifications/getStableDeviceId", () => ({
  getStableDeviceId: jest.fn(async () => "device-integration"),
  resetStableDeviceIdCacheForTests: jest.fn(),
}));

jest.mock("./services/backgroundLocationTask", () => ({
  getNativeTaskLifecycleStatus: jest.fn(async () => ({
    started: false,
    available: false,
  })),
  readNativeLocationUpdatesStarted: jest.fn(async () => false),
}));

jest.mock("./notificationActions", () => ({
  ensureDriverNotificationActions: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("./notificationGrouping", () => ({
  ensureDriverNotificationGrouping: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("./missionBarIOS", () => ({
  configureMissionBarIOS: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("./missionBarBackground", () => ({
  registerMissionBarBackgroundHandlers: jest.fn(),
}));

jest.mock("../../core/sessionProvider", () => ({
  useSession: () => ({
    status: mockSessionState.status,
    activeContext: mockSessionState.activeContext,
    bootstrap: mockSessionState.bootstrap,
    bootstrapSession: jest.fn().mockResolvedValue(undefined),
  }),
}));

jest.mock("../../core/featureFlags/registry", () => ({
  isFeatureEnabled: (flag: string) => mockIsFeatureEnabled(flag),
}));

jest.mock("../../core/api/client", () => ({
  refreshAuthTokenNow: () => mockRefreshAuthTokenNow(),
}));

jest.mock("../../core/auth/authTokenOrchestrator", () => ({
  refreshAuthTokenSingleflight: () => mockRefreshAuthTokenNow(),
}));

jest.mock("../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    connect: jest.fn(),
    getSnapshot: () => ({
      activeContextId: "driver:42",
      connected: true,
      mode: "socket",
      lastEventAt: null,
      lastError: null,
      reconnectAttempts: 0,
      reconnectBackoffMs: 0,
    }),
  },
}));

jest.mock("../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: (event: string, payload: unknown) => mockEmitDriverTelemetry(event, payload),
}));

jest.mock("./services/driverRealtimeBridge", () => ({
  startDriverRealtimeBridge: (...args: unknown[]) => mockStartDriverRealtimeBridge(...args),
}));

jest.mock("./sync", () => ({
  reconcileDriverMissions: (...args: unknown[]) => mockReconcileDriverMissions(...args),
}));

jest.mock("./offlineQueue", () => ({
  driverOfflineQueue: {
    flush: () => mockOfflineFlush(),
    setActiveContext: () => undefined,
  },
}));

jest.mock("./tracking", () => ({
  getTrackingSnapshot: () => ({
    missionId: 501,
    missionStatus: "ASSIGNED",
    appState: "active",
    isRunning: true,
    permission: "granted",
    lastSentAt: null,
    lastAckAt: null,
    queueDepth: 0,
    flushPathUsed: "http_fallback",
    lastAttemptAt: null,
    consecutiveFailures: 0,
    backoffUntilMs: 0,
  }),
  flushTrackingQueue: () => mockFlushTrackingQueue(),
  startDriverTracking: jest.fn(),
  stopDriverTracking: jest.fn(),
  updateDriverTrackingStatus: jest.fn(),
}));

jest.mock("./api/driverHttp", () => ({
  registerDriverPushToken: (payload: unknown) => mockRegisterDriverPushToken(payload),
  getDriverMissions: jest.fn().mockResolvedValue([]),
  getDriverMissionDetail: jest.fn().mockResolvedValue({}),
  updateDriverMissionStatus: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("./push", () => ({
  handleDriverPushQuickAction: (payload: unknown) => mockHandleDriverPushQuickAction(payload),
}));

jest.mock("./firebaseMessaging", () => ({
  initDriverFirebaseMessaging: jest.fn(),
  disposeDriverFirebaseMessaging: jest.fn(),
  driverFcmPlatform: () => "android",
}));

jest.mock("./notificationChannels", () => ({
  ensureBaseNotificationChannels: jest.fn().mockResolvedValue(undefined),
  ensureDriverNotificationChannels: jest.fn().mockResolvedValue(undefined),
  getRegisteredNotificationChannelCount: jest.fn().mockReturnValue(0),
  resolveDriverNotificationContract: jest.fn().mockReturnValue({}),
}));

jest.mock("./silentNotifications", () => ({
  handleSilentPushPayload: jest.fn().mockResolvedValue(undefined),
  isSilentPayload: () => false,
  shouldSuppressVisualPush: () => false,
}));

jest.mock("./driverRealtimeSync", () => ({
  configureDriverRealtimeSync: jest.fn(),
  requestMissionRefresh: jest.fn(),
  requestChatRefresh: jest.fn(),
}));

jest.mock("../../core/notifications/notificationDedupStore", () => ({
  buildNotificationDedupKey: () => "dedup-key",
  markNotificationHandled: () => false,
}));

function DriverRuntimeHarness() {
  useDriverRealtimeSync();
  return null;
}

describe("P1->P2 lightweight integration", () => {
  beforeEach(() => {
    mockAppStateHandlers = [];
    mockNotificationResponseHandler = null;
    mockSessionState = {
      status: "ready",
      activeContext: { context_id: "driver:42", context_type: "driver", permissions: [] },
      bootstrap: { user: { id: 42 } },
    };

    jest.spyOn(AppState, "addEventListener").mockImplementation((_event, callback) => {
      mockAppStateHandlers.push(callback as (state: "active" | "inactive" | "background") => void);
      return { remove: jest.fn() };
    });
    Object.defineProperty(AppState, "currentState", {
      configurable: true,
      get: () => "active",
    });
    resetDriverForegroundResumeAuthorityForTests();
    resetDriverSessionNetworkGateForTests();
    setDriverSessionNetworkReady(true);

    mockRouterPush.mockReset();
    mockRegisterDriverPushToken.mockReset();
    mockHandleDriverPushQuickAction.mockReset();
    mockStartDriverRealtimeBridge.mockReset();
    mockRefreshAuthTokenNow.mockReset();
    mockReconcileDriverMissions.mockReset();
    mockOfflineFlush.mockReset();
    mockFlushTrackingQueue.mockReset();
    mockRegisterDriverPushToken.mockReset();
    mockEmitDriverTelemetry.mockReset();
    mockIsFeatureEnabled.mockReset();

    mockHandleDriverPushQuickAction.mockResolvedValue(undefined);
    mockRegisterDriverPushToken.mockResolvedValue(undefined);
    mockRefreshAuthTokenNow.mockResolvedValue(true);
    mockReconcileDriverMissions.mockResolvedValue({
      missions: [],
      queue: { sent: 0, dropped: 0, failed: 0 },
    });
    mockOfflineFlush.mockResolvedValue({ sent: 0, dropped: 0, failed: 0 });
    mockFlushTrackingQueue.mockResolvedValue(undefined);
    mockIsFeatureEnabled.mockImplementation((flag: string) =>
      [
        "driver_push_enabled",
        "tracking_resume_resync_enabled",
        "realtime_auth_reconnect_enabled",
        "realtime_socket_enabled",
      ].includes(flag)
    );
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("routes cold-start notification and executes a single resume chain", async () => {
    const queryClient = new QueryClient();
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <NotificationsProvider>
            <DriverRuntimeHarness />
          </NotificationsProvider>
        </QueryClientProvider>
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRegisterDriverPushToken).toHaveBeenCalledWith(
      expect.objectContaining({ token: "ExpoPushToken[integration]", driverId: 42 })
    );
    expect(mockRouterPush).toHaveBeenCalledWith("/(app)/(driver)");
    expect(getDriverForegroundResumeListenerCountForTests()).toBeGreaterThan(0);

    let releaseReconcile: (() => void) | null = null;
    mockReconcileDriverMissions.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseReconcile = () => resolve({ missions: [], queue: { sent: 0, dropped: 0, failed: 0 } });
        })
    );

    await act(async () => {
      emitDriverProcessForegroundForTests(false);
      emitDriverProcessForegroundForTests(true);
      emitDriverProcessForegroundForTests(false);
      emitDriverProcessForegroundForTests(true);
      await Promise.resolve();
    });

    expect(mockReconcileDriverMissions).toHaveBeenCalledTimes(1);
    expect(mockFlushTrackingQueue).toHaveBeenCalledTimes(0);

    await act(async () => {
      releaseReconcile?.();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockFlushTrackingQueue).toHaveBeenCalledTimes(1);
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "driver.runtime.resume.success",
      expect.objectContaining({ context_id: "driver:42" })
    );

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("skips push routing after context switch away from driver", async () => {
    const queryClient = new QueryClient();
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <QueryClientProvider client={queryClient}>
          <NotificationsProvider>
            <DriverRuntimeHarness />
          </NotificationsProvider>
        </QueryClientProvider>
      );
      await Promise.resolve();
    });

    mockRouterPush.mockReset();
    mockSessionState = {
      status: "ready",
      activeContext: { context_id: "client:self", context_type: "client", permissions: [] },
      bootstrap: { user: { id: 42 } },
    };

    await act(async () => {
      renderer!.update(
        <QueryClientProvider client={queryClient}>
          <NotificationsProvider>
            <DriverRuntimeHarness />
          </NotificationsProvider>
        </QueryClientProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      mockNotificationResponseHandler?.({
        notification: {
          request: {
            identifier: "post-switch",
            content: { data: { mission_id: 777, type: "mission_updated" } },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockRouterPush).not.toHaveBeenCalled();

    await act(async () => {
      renderer!.unmount();
    });
  });
});
