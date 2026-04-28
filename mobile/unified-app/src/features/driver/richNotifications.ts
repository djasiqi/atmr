import { getExpoNotificationsModule } from "../../core/notifications/expoNotificationsCompat";
import { DRIVER_NOTIFICATION_CHANNELS } from "./notificationChannels";

export async function scheduleDriverRichNotification(params: {
  title: string;
  body: string;
  imageUrl?: string | null;
}) {
  const Notifications = getExpoNotificationsModule();
  if (!Notifications) return null;

  return Notifications.scheduleNotificationAsync({
    content: {
      title: params.title,
      body: params.body,
      sound: "default",
      categoryIdentifier: "driver-mission-actions",
      ...(params.imageUrl
        ? {
            attachments: [
              {
                identifier: "driver-mission-image",
                url: params.imageUrl,
                type: "public.jpeg",
              },
            ],
          }
        : null),
      data: { image_url: params.imageUrl ?? null },
    },
    trigger: {
      channelId: DRIVER_NOTIFICATION_CHANNELS.missionUpdates,
    },
  });
}
