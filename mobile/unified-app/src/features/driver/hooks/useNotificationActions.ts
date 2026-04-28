import { useCallback } from "react";
import { Platform } from "react-native";
import { getExpoNotificationsModule } from "../../../core/notifications/expoNotificationsCompat";

type NotificationActions = {
  markAsRead: (id: string) => Promise<void>;
  markAllAsRead: () => Promise<void>;
  dismiss: (id: string) => Promise<void>;
};

export function useNotificationActions(): NotificationActions {
  const Notifications = getExpoNotificationsModule();

  const dismiss = useCallback(async (id: string) => {
    if (Platform.OS === "web" || !Notifications) return;
    await Notifications.dismissNotificationAsync(id).catch(() => undefined);
  }, [Notifications]);

  const markAsRead = useCallback(async (id: string) => {
    if (Platform.OS === "web" || !Notifications) return;
    await Notifications.dismissNotificationAsync(id).catch(() => undefined);
  }, [Notifications]);

  const markAllAsRead = useCallback(async () => {
    if (Platform.OS === "web" || !Notifications) return;
    await Notifications.dismissAllNotificationsAsync().catch(() => undefined);
  }, [Notifications]);

  return {
    markAsRead,
    markAllAsRead,
    dismiss,
  };
}

