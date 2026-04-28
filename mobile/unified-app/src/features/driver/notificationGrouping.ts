import { getExpoNotificationsModule } from "../../core/notifications/expoNotificationsCompat";
import { DRIVER_NOTIFICATION_CHANNELS } from "./notificationChannels";

export async function ensureDriverNotificationGrouping() {
  const Notifications = getExpoNotificationsModule();
  if (!Notifications) return;

  await Notifications.setNotificationChannelAsync(DRIVER_NOTIFICATION_CHANNELS.missionUpdates, {
    name: "Driver missions grouped",
    importance: Notifications.AndroidImportance.HIGH,
    vibrationPattern: [0, 150, 150, 150],
    lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
    groupId: "missions",
  });
}
