import { useCallback, useEffect, useMemo, useState } from "react";
import { Platform } from "react-native";
import { getExpoNotificationsModule } from "../../../core/notifications/expoNotificationsCompat";

export type DriverNotification = {
  id: string;
  title: string;
  body: string;
  data: Record<string, unknown>;
};

export type NotificationsState = {
  notifications: DriverNotification[];
  unreadCount: number;
  refresh: () => void;
};

type NotificationShape = {
  request: {
    identifier: string;
    content: {
      title?: string | null;
      body?: string | null;
      data?: unknown;
    };
  };
};

function normalizeNotification(notification: NotificationShape): DriverNotification {
  return {
    id: notification.request.identifier,
    title: notification.request.content.title ?? "",
    body: notification.request.content.body ?? "",
    data: (notification.request.content.data ?? {}) as Record<string, unknown>,
  };
}

export function useNotifications(): NotificationsState {
  const [notifications, setNotifications] = useState<DriverNotification[]>([]);
  const Notifications = getExpoNotificationsModule();

  const refresh = useCallback(() => {
    if (Platform.OS === "web" || !Notifications) {
      setNotifications([]);
      return;
    }
    void Notifications.getPresentedNotificationsAsync()
      .then((items) => {
        setNotifications(items.map(normalizeNotification));
      })
      .catch(() => {
        setNotifications([]);
      });
  }, [Notifications]);

  useEffect(() => {
    if (Platform.OS === "web" || !Notifications) return;
    refresh();
    const receivedListener = Notifications.addNotificationReceivedListener(() => {
      refresh();
    });
    const responseListener =
      Notifications.addNotificationResponseReceivedListener(() => {
        refresh();
      });
    return () => {
      receivedListener.remove();
      responseListener.remove();
    };
  }, [Notifications, refresh]);

  return useMemo(
    () => ({
      notifications,
      unreadCount: notifications.length,
      refresh,
    }),
    [notifications, refresh]
  );
}

