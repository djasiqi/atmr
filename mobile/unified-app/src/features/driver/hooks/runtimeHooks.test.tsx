import React from "react";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { act, create } from "react-test-renderer";
import { useNotificationActions } from "./useNotificationActions";
import { useNotifications } from "./useNotifications";
import { useSocketStatus } from "./useSocketStatus";
import { useTrackingState } from "./useTrackingState";

type TrackingSnapshot = {
  isRunning: boolean;
  missionStatus: string | null;
  lastSentAt: string | null;
};

type RealtimeSnapshot = {
  connected: boolean;
  actualTransport: "idle" | "polling" | "socket";
  mode: "polling" | "socket";
  desiredTransport: "polling" | "socket";
  authExhausted: boolean;
  authAttempts: number;
  transportAuthority: "socket" | "polling" | "reconcile" | "degraded";
  degradedMode: boolean;
  lastError: string | null;
};

type PresentedNotification = {
  request: {
    identifier: string;
    content: {
      title: string;
      body: string;
      data: { mission_id: number };
    };
  };
};

type ListenerSubscription = { remove: () => void };
type TrackingState = {
  isTracking: boolean;
  mode: "mission_live" | "availability_presence" | "idle";
  lastUpdate?: number;
  lastAckAt?: number;
  lastAckIsQueued?: boolean;
  lastAckStatus?: string | null;
  lastAckError?: string | null;
  currentAttemptSeq?: number;
  lastAckAttemptSeq?: number | null;
  currentAttemptEventId?: string | null;
  lastAckEventId?: string | null;
  queueDepth?: number;
  accuracy?: number;
};
type SocketStatus = {
  connected: boolean;
  reconnecting: boolean;
  degraded: boolean;
  authExhausted: boolean;
  authAttempts: number;
  transportMode: "idle" | "polling" | "socket";
  transportAuthority: "socket" | "polling" | "reconcile" | "degraded";
  lastError: string | null;
  lastConnectedAt?: number;
};
type NotificationsState = {
  notifications: { id: string }[];
  unreadCount: number;
  refresh: () => void;
};

const mockGetTrackingSnapshot = jest.fn<() => TrackingSnapshot>();
const mockSubscribeTrackingSnapshot = jest.fn<
  (listener: (snapshot: TrackingSnapshot) => void) => () => void
>();
const mockRealtimeSubscribe = jest.fn<
  (listener: (snapshot: RealtimeSnapshot) => void) => () => void
>();
const mockRealtimeGetSnapshot = jest.fn<() => RealtimeSnapshot>();

const mockGetPresentedNotificationsAsync = jest.fn<
  () => Promise<PresentedNotification[]>
>();
const mockAddNotificationReceivedListener = jest.fn<
  (listener: () => void) => ListenerSubscription
>();
const mockAddNotificationResponseReceivedListener = jest.fn<
  (listener: () => void) => ListenerSubscription
>();
const mockDismissNotificationAsync = jest.fn<(id: string) => Promise<void>>();
const mockDismissAllNotificationsAsync = jest.fn<() => Promise<void>>();

let trackingListener: ((snapshot: TrackingSnapshot) => void) | null = null;
let realtimeListener: ((snapshot: RealtimeSnapshot) => void) | null = null;

jest.mock("../tracking", () => ({
  getTrackingSnapshot: () => mockGetTrackingSnapshot(),
  subscribeTrackingSnapshot: (listener: (snapshot: unknown) => void) =>
    mockSubscribeTrackingSnapshot(listener),
}));

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    subscribe: (listener: (snapshot: unknown) => void) => mockRealtimeSubscribe(listener),
    getSnapshot: () => mockRealtimeGetSnapshot(),
  },
}));

jest.mock("expo-notifications", () => ({
  getPresentedNotificationsAsync: () => mockGetPresentedNotificationsAsync(),
  addNotificationReceivedListener: (listener: () => void) =>
    mockAddNotificationReceivedListener(listener),
  addNotificationResponseReceivedListener: (listener: () => void) =>
    mockAddNotificationResponseReceivedListener(listener),
  dismissNotificationAsync: (id: string) => mockDismissNotificationAsync(id),
  dismissAllNotificationsAsync: () => mockDismissAllNotificationsAsync(),
}));

jest.mock("react-native", () => ({
  Platform: { OS: "android" },
}));

function TrackingHarness(props: { onValue: (value: TrackingState) => void }) {
  const value = useTrackingState();
  props.onValue(value);
  return null;
}

function SocketHarness(props: { onValue: (value: SocketStatus) => void }) {
  const value = useSocketStatus();
  props.onValue(value);
  return null;
}

function NotificationsHarness(
  props: { onValue: (value: NotificationsState) => void }
) {
  const value = useNotifications();
  props.onValue(value);
  return null;
}

function NotificationActionsHarness(
  props: { onValue: (value: ReturnType<typeof useNotificationActions>) => void }
) {
  const value = useNotificationActions();
  props.onValue(value);
  return null;
}

describe("driver runtime hooks", () => {
  beforeEach(() => {
    trackingListener = null;
    realtimeListener = null;
    mockGetTrackingSnapshot.mockReset();
    mockSubscribeTrackingSnapshot.mockReset();
    mockRealtimeSubscribe.mockReset();
    mockRealtimeGetSnapshot.mockReset();
    mockGetPresentedNotificationsAsync.mockReset();
    mockAddNotificationReceivedListener.mockReset();
    mockAddNotificationResponseReceivedListener.mockReset();
    mockDismissNotificationAsync.mockReset();
    mockDismissAllNotificationsAsync.mockReset();

    mockGetTrackingSnapshot.mockReturnValue({
      isRunning: false,
      missionStatus: null,
      lastSentAt: null,
    });
    mockSubscribeTrackingSnapshot.mockImplementation((listener) => {
      trackingListener = listener;
      listener(mockGetTrackingSnapshot());
      return () => {
        trackingListener = null;
      };
    });
    mockRealtimeSubscribe.mockImplementation((listener) => {
      realtimeListener = listener;
      listener(mockRealtimeGetSnapshot());
      return () => {
        realtimeListener = null;
      };
    });
    mockRealtimeGetSnapshot.mockReturnValue({
      connected: false,
      actualTransport: "polling",
      mode: "polling",
      desiredTransport: "socket",
      authExhausted: false,
      authAttempts: 0,
      transportAuthority: "degraded",
      degradedMode: true,
      lastError: null,
    });
    mockGetPresentedNotificationsAsync.mockResolvedValue([
      {
        request: {
          identifier: "n-1",
          content: {
            title: "Mission",
            body: "Nouvelle mission",
            data: { mission_id: 12 },
          },
        },
      },
    ]);
    mockAddNotificationReceivedListener.mockImplementation((listener) => {
      return { remove: jest.fn() };
    });
    mockAddNotificationResponseReceivedListener.mockImplementation((listener) => {
      return { remove: jest.fn() };
    });
    mockDismissNotificationAsync.mockResolvedValue(undefined);
    mockDismissAllNotificationsAsync.mockResolvedValue(undefined);
  });

  it("maps tracking runtime snapshot and updates on subscription", async () => {
    let latest!: TrackingState;
    await act(async () => {
      create(<TrackingHarness onValue={(value) => (latest = value)} />);
    });

    expect(latest).toEqual({
      isTracking: false,
      mode: "idle",
      lastUpdate: undefined,
      lastAckAt: undefined,
      lastAckIsQueued: false,
      lastAckStatus: undefined,
      lastAckError: undefined,
      currentAttemptSeq: undefined,
      lastAckAttemptSeq: undefined,
      currentAttemptEventId: undefined,
      lastAckEventId: undefined,
      queueDepth: undefined,
      accuracy: undefined,
    });

    await act(async () => {
      trackingListener?.({
        isRunning: true,
        missionStatus: "ASSIGNED",
        lastSentAt: "2026-04-18T10:00:00.000Z",
      });
    });

    expect(latest.isTracking).toBe(true);
    expect(latest.mode).toBe("availability_presence");
    expect(latest.lastUpdate).toBe(Date.parse("2026-04-18T10:00:00.000Z"));
  });

  it("maps socket status from realtime manager lifecycle", async () => {
    let latest!: SocketStatus;
    await act(async () => {
      create(<SocketHarness onValue={(value) => (latest = value)} />);
    });

    expect(latest.connected).toBe(false);
    expect(latest.reconnecting).toBe(true);

    await act(async () => {
      realtimeListener?.({
        connected: true,
        actualTransport: "socket",
        mode: "polling",
        desiredTransport: "socket",
        authExhausted: false,
        authAttempts: 0,
        transportAuthority: "socket",
        degradedMode: false,
        lastError: null,
      });
    });

    expect(latest.connected).toBe(true);
    expect(latest.reconnecting).toBe(true);
    expect(typeof latest.lastConnectedAt).toBe("number");
  });

  it("exposes notifications facade and refreshes on listener events", async () => {
    let latest!: NotificationsState;
    await act(async () => {
      create(<NotificationsHarness onValue={(value) => (latest = value)} />);
      await Promise.resolve();
    });

    expect(latest.unreadCount).toBe(1);
    expect(latest.notifications[0]?.id).toBe("n-1");

    mockGetPresentedNotificationsAsync.mockResolvedValueOnce([]);
    await act(async () => {
      await latest.refresh();
    });

    expect(latest.unreadCount).toBe(0);
  });

  it("exposes notification actions facade", async () => {
    let latest: ReturnType<typeof useNotificationActions> | null = null;
    await act(async () => {
      create(<NotificationActionsHarness onValue={(value) => (latest = value)} />);
    });

    await act(async () => {
      await latest?.markAsRead("n-42");
      await latest?.dismiss("n-42");
      await latest?.markAllAsRead();
    });

    expect(mockDismissNotificationAsync).toHaveBeenCalledWith("n-42");
    expect(mockDismissAllNotificationsAsync).toHaveBeenCalledTimes(1);
  });
});

