import React from "react";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { act, create } from "react-test-renderer";
import { NotificationsProvider } from "./NotificationsProvider";

const mockRequestPermissionsAsync = jest.fn() as jest.Mock<any>;
const mockSetNotificationHandler = jest.fn() as jest.Mock<any>;
const mockAddNotificationReceivedListener = jest.fn() as jest.Mock<any>;
const mockAddNotificationResponseReceivedListener = jest.fn() as jest.Mock<any>;
const mockAddPushTokenListener = jest.fn() as jest.Mock<any>;
const mockGetExpoPushTokenAsync = jest.fn() as jest.Mock<any>;
const mockGetLastNotificationResponseAsync = jest.fn() as jest.Mock<any>;

const mockEmitDriverTelemetry = jest.fn() as jest.Mock<any>;
const mockRegisterDriverPushToken = jest.fn() as jest.Mock<any>;
const mockHandleDriverPushQuickAction = jest.fn() as jest.Mock<any>;
const mockHandleSilentPushPayload = jest.fn() as jest.Mock<any>;
const mockIsSilentPayload = jest.fn() as jest.Mock<any>;
const mockIsFeatureEnabled = jest.fn() as jest.Mock<any>;
const mockRouterPush = jest.fn() as jest.Mock<any>;
const mockUseSession = jest.fn() as jest.Mock<any>;
const mockEnsureDriverNotificationChannels = jest.fn() as jest.Mock<any>;
let currentSession: {
  status: string;
  activeContext: { context_type: string } | null;
  bootstrap: { user: { id: number | null } } | null;
};

let onReceived: ((notification: any) => void) | null = null;
let onResponse: ((response: any) => void) | null = null;
let notificationHandler: ((notification: any) => Promise<any>) | null = null;

jest.mock("expo-notifications", () => ({
  setNotificationHandler: (config: { handleNotification?: (notification: any) => Promise<any> }) => {
    notificationHandler = config.handleNotification ?? null;
    mockSetNotificationHandler(config);
  },
  requestPermissionsAsync: () => mockRequestPermissionsAsync(),
  addNotificationReceivedListener: (cb: (n: unknown) => void) => {
    onReceived = cb as any;
    mockAddNotificationReceivedListener(cb);
    return { remove: jest.fn() };
  },
  addNotificationResponseReceivedListener: (cb: (r: unknown) => void) => {
    onResponse = cb as any;
    mockAddNotificationResponseReceivedListener(cb);
    return { remove: jest.fn() };
  },
  addPushTokenListener: (cb: (e: unknown) => void) => {
    mockAddPushTokenListener(cb);
    return { remove: jest.fn() };
  },
  getExpoPushTokenAsync: () => mockGetExpoPushTokenAsync(),
  getLastNotificationResponseAsync: () => mockGetLastNotificationResponseAsync(),
}));

jest.mock("expo-router", () => ({
  useRouter: () => ({ push: (...args: unknown[]) => mockRouterPush(...args) }),
}));

jest.mock("../observability/driverTelemetry", () => ({
  emitDriverTelemetry: (event: string, payload: unknown) => mockEmitDriverTelemetry(event, payload),
}));

jest.mock("../featureFlags/registry", () => ({
  isFeatureEnabled: (flag: string) => mockIsFeatureEnabled(flag),
}));

jest.mock("../notifications/notificationDisclosurePersistence", () => ({
  readNotificationDisclosureAccepted: jest.fn(async () => true),
  subscribeNotificationDisclosureAccepted: jest.fn(() => () => undefined),
}));

jest.mock("../sessionProvider", () => ({
  useSession: () => mockUseSession(),
}));

jest.mock("../../features/driver/api/driverHttp", () => ({
  registerDriverPushToken: (payload: unknown) => mockRegisterDriverPushToken(payload),
}));

jest.mock("../../features/driver/push", () => ({
  handleDriverPushQuickAction: (payload: unknown) => mockHandleDriverPushQuickAction(payload),
}));

jest.mock("../../features/driver/silentNotifications", () => ({
  handleSilentPushPayload: (payload: unknown, callback: (missionId: number) => Promise<void>) =>
    mockHandleSilentPushPayload(payload, callback),
  isSilentPayload: (payload: unknown) => mockIsSilentPayload(payload),
}));

jest.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: jest.fn(),
  }),
}));

jest.mock("../../features/driver/driverRealtimeSync", () => ({
  configureDriverRealtimeSync: jest.fn(),
  requestMissionRefresh: jest.fn(),
  requestChatRefresh: jest.fn(),
}));

jest.mock("../../features/driver/notificationChannels", () => ({
  ensureDriverNotificationChannels: () => mockEnsureDriverNotificationChannels(),
}));

jest.mock("../../features/driver/notificationActions", () => ({
  ensureDriverNotificationActions: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("../../features/driver/notificationGrouping", () => ({
  ensureDriverNotificationGrouping: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("../../features/driver/missionBarIOS", () => ({
  configureMissionBarIOS: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("../../features/driver/missionBarBackground", () => ({
  registerMissionBarBackgroundHandlers: jest.fn(),
}));

jest.mock("../../features/driver/firebaseMessaging", () => ({
  initDriverFirebaseMessaging: jest.fn(),
  disposeDriverFirebaseMessaging: jest.fn(),
  driverFcmPlatform: () => "android",
}));

jest.mock("../notifications/notificationDedupStore", () => ({
  buildNotificationDedupKey: () => "dedup-key",
  markNotificationHandled: () => false,
}));

describe("NotificationsProvider", () => {
  beforeEach(() => {
    onReceived = null;
    onResponse = null;
    notificationHandler = null;
    mockRequestPermissionsAsync.mockReset();
    mockSetNotificationHandler.mockReset();
    mockAddNotificationReceivedListener.mockReset();
    mockAddNotificationResponseReceivedListener.mockReset();
    mockAddPushTokenListener.mockReset();
    mockGetExpoPushTokenAsync.mockReset();
    mockGetLastNotificationResponseAsync.mockReset();
    mockEmitDriverTelemetry.mockReset();
    mockRegisterDriverPushToken.mockReset();
    mockHandleDriverPushQuickAction.mockReset();
    mockHandleSilentPushPayload.mockReset();
    mockIsSilentPayload.mockReset();
    mockIsFeatureEnabled.mockReset();
    mockRouterPush.mockReset();
    mockUseSession.mockReset();
    mockEnsureDriverNotificationChannels.mockReset();

    mockIsFeatureEnabled.mockImplementation((flag: string) => flag === "driver_push_enabled");
    mockEnsureDriverNotificationChannels.mockResolvedValue(undefined);
    mockRequestPermissionsAsync.mockResolvedValue({ granted: true });
    mockGetExpoPushTokenAsync.mockResolvedValue({ data: "ExpoPushToken[test]" });
    mockGetLastNotificationResponseAsync.mockResolvedValue(null);
    mockRegisterDriverPushToken.mockResolvedValue(undefined);
    mockHandleDriverPushQuickAction.mockResolvedValue(undefined);
    mockHandleSilentPushPayload.mockResolvedValue(undefined);
    mockIsSilentPayload.mockReturnValue(false);
    currentSession = {
      status: "ready",
      activeContext: { context_type: "driver" },
      bootstrap: { user: { id: 42 } },
    };
    mockUseSession.mockImplementation(() => currentSession);
    jest.useRealTimers();
  });

  it("registers driver push token once session is ready", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRequestPermissionsAsync).toHaveBeenCalled();
    expect(mockRegisterDriverPushToken).toHaveBeenCalledWith(
      expect.objectContaining({
        token: "ExpoPushToken[test]",
        driverId: 42,
        provider: "expo",
      })
    );
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "push.token.registered",
      expect.objectContaining({ source: "driver.notifications.bridge", provider: "expo" })
    );

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("emits telemetry when notification channel setup fails", async () => {
    mockIsFeatureEnabled.mockImplementation(
      (flag: string) =>
        flag === "driver_push_enabled" || flag === "driver_notification_actions_enabled"
    );
    mockEnsureDriverNotificationChannels.mockRejectedValue(new Error("channel setup failed"));

    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "push.channels.setup_failed",
      expect.objectContaining({
        source: "core.notifications.provider",
        error: "channel setup failed",
      })
    );

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("routes mission payload when notification response is received", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        notification: {
          request: {
            identifier: "notif-1",
            content: {
              data: { mission_id: 321, type: "mission_assigned", event_id: "evt-1" },
            },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockHandleDriverPushQuickAction).toHaveBeenCalledWith(
      expect.objectContaining({ mission_id: 321, type: "mission_assigned" })
    );
    expect(mockRouterPush).toHaveBeenCalledWith("/(app)/(driver)");

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("maps DECLINE_MISSION response action to reject quick action", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        actionIdentifier: "DECLINE_MISSION",
        notification: {
          request: {
            identifier: "notif-decline-action",
            content: {
              data: { mission_id: 322, type: "mission_assigned", event_id: "evt-decline" },
            },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockHandleDriverPushQuickAction).toHaveBeenCalledWith(
      expect.objectContaining({ mission_id: 322, action: "reject" })
    );

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("filters non-driver recipient in foreground notification handler", async () => {
    await act(async () => {
      create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    const behavior = await notificationHandler?.({
      request: {
        content: {
          data: { recipient_role: "company", mission_id: 1, type: "mission_assigned" },
        },
      },
    });
    expect(behavior).toEqual(
      expect.objectContaining({
        shouldShowBanner: false,
        shouldShowList: false,
      })
    );
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "push.notification.ignored",
      expect.objectContaining({ stage: "foreground_handler", reason: "recipient_role_mismatch" })
    );
  });

  it("ignores self-actor notification in response listener", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        actionIdentifier: "ACCEPT_MISSION",
        notification: {
          request: {
            identifier: "notif-self-actor",
            content: {
              data: {
                mission_id: 900,
                type: "mission_assigned",
                recipient_role: "driver",
                actor_role: "driver",
                actor_id: 42,
              },
            },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockHandleDriverPushQuickAction).not.toHaveBeenCalled();
    expect(mockRouterPush).not.toHaveBeenCalled();
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "push.notification.ignored",
      expect.objectContaining({ stage: "response_listener", reason: "self_actor" })
    );

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("ignores company-targeted notification in received listener", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onReceived?.({
        request: {
          identifier: "notif-received-company",
          content: {
            data: {
              mission_id: 44,
              type: "mission_updated",
              recipient_role: "company",
            },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "push.notification.ignored",
      expect.objectContaining({ stage: "received_listener", reason: "recipient_role_mismatch" })
    );
    expect(mockEmitDriverTelemetry).not.toHaveBeenCalledWith(
      "push.notification.received",
      expect.anything()
    );

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("skips registration when context is not driver or user id invalid", async () => {
    currentSession = {
      status: "ready",
      activeContext: { context_type: "client" },
      bootstrap: { user: { id: null } },
    };

    await act(async () => {
      create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    expect(mockRegisterDriverPushToken).not.toHaveBeenCalled();
  });

  it("handles token fetch failure without registering", async () => {
    mockGetExpoPushTokenAsync.mockRejectedValueOnce(new Error("expo_token_error"));

    await act(async () => {
      create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    expect(mockRegisterDriverPushToken).not.toHaveBeenCalled();
  });

  it("ignores invalid payloads without routing", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        notification: {
          request: {
            identifier: "notif-invalid",
            content: {
              data: { type: "mission_assigned" },
            },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockRouterPush).not.toHaveBeenCalled();
    expect(mockHandleDriverPushQuickAction).not.toHaveBeenCalled();

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("emits route_failed telemetry when quick action handler throws", async () => {
    mockHandleDriverPushQuickAction.mockRejectedValueOnce(new Error("quick_action_failed"));
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        notification: {
          request: {
            identifier: "notif-failed-action",
            content: {
              data: { mission_id: 77, type: "mission_assigned", action: "accept" },
            },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "push.notification.route_failed",
      expect.objectContaining({ mission_id: 77, reason: "quick_action_failed" })
    );
    expect(mockRouterPush).not.toHaveBeenCalled();

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("does not route notification when session context changed away from driver", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    currentSession = {
      status: "ready",
      activeContext: { context_type: "client" },
      bootstrap: { user: { id: 42 } },
    };

    await act(async () => {
      renderer!.update(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        notification: {
          request: {
            identifier: "notif-context-changed",
            content: {
              data: { mission_id: 88, type: "mission_updated" },
            },
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

  it("prioritizes explicit deep link route over mission_id route", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        notification: {
          request: {
            identifier: "notif-deeplink-priority",
            content: {
              data: {
                mission_id: 10,
                type: "mission_updated",
                deep_link: "atmr://booking/99",
              },
            },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockRouterPush).toHaveBeenCalledWith("/(app)/(driver)/missions/99");

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("buffers notification while session is not ready and routes when driver context is ready", async () => {
    currentSession = {
      status: "idle",
      activeContext: null,
      bootstrap: { user: { id: 42 } },
    };
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        notification: {
          request: {
            identifier: "notif-buffered",
            content: { data: { mission_id: 602, type: "mission_assigned" } },
          },
        },
      });
      await Promise.resolve();
    });
    expect(mockRouterPush).not.toHaveBeenCalled();

    currentSession = {
      status: "ready",
      activeContext: { context_type: "driver" },
      bootstrap: { user: { id: 42 } },
    };

    await act(async () => {
      renderer!.update(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRouterPush).toHaveBeenCalledWith("/(app)/(driver)");

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("falls back to mission list when buffered payload times out before bootstrap ready", async () => {
    const previousTimeout = process.env.EXPO_PUBLIC_DRIVER_DEEPLINK_BOOTSTRAP_TIMEOUT_MS;
    process.env.EXPO_PUBLIC_DRIVER_DEEPLINK_BOOTSTRAP_TIMEOUT_MS = "5";
    jest.useFakeTimers();
    currentSession = {
      status: "idle",
      activeContext: null,
      bootstrap: { user: { id: 42 } },
    };

    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        notification: {
          request: {
            identifier: "notif-timeout",
            content: { data: { mission_id: 701, type: "mission_updated" } },
          },
        },
      });
      await Promise.resolve();
    });

    await act(async () => {
      jest.advanceTimersByTime(10);
      await Promise.resolve();
    });

    currentSession = {
      status: "ready",
      activeContext: { context_type: "driver" },
      bootstrap: { user: { id: 42 } },
    };

    await act(async () => {
      renderer!.update(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRouterPush).toHaveBeenCalledWith("/(app)/(driver)/missions");
    expect(mockEmitDriverTelemetry).toHaveBeenCalledWith(
      "push.notification.route_timeout",
      expect.objectContaining({ mission_id: 701, timeout_ms: 5 })
    );

    await act(async () => {
      renderer!.unmount();
    });

    process.env.EXPO_PUBLIC_DRIVER_DEEPLINK_BOOTSTRAP_TIMEOUT_MS = previousTimeout;
    jest.useRealTimers();
  });

  it("routes reassigned mission notifications to today trips page", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        notification: {
          request: {
            identifier: "notif-reassigned",
            content: {
              data: { mission_id: 730, type: "mission_reassigned", event_id: "evt-r1" },
            },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockRouterPush).toHaveBeenCalledWith("/(app)/(driver)/trips");

    await act(async () => {
      renderer!.unmount();
    });
  });

  it("routes chat notifications to the targeted thread", async () => {
    let renderer: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(
        <NotificationsProvider>
          <></>
        </NotificationsProvider>
      );
      await Promise.resolve();
    });

    await act(async () => {
      onResponse?.({
        notification: {
          request: {
            identifier: "notif-chat-route",
            content: {
              data: {
                type: "chat_message",
                thread_id: "dispatch",
                deep_link: "atmr://chat/thread/dispatch",
              },
            },
          },
        },
      });
      await Promise.resolve();
    });

    expect(mockRouterPush).toHaveBeenCalledWith("/(app)/(driver)/messages/dispatch");

    await act(async () => {
      renderer!.unmount();
    });
  });
});
