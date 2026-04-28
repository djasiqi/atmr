import { getExpoNotificationsModule } from "../../core/notifications/expoNotificationsCompat";

export const DRIVER_NOTIFICATION_CATEGORY_ID = "MISSION_REQUEST";

export const DRIVER_NOTIFICATION_ACTION_IDS = {
  ACCEPT: "ACCEPT_MISSION",
  DECLINE: "DECLINE_MISSION",
  START: "START_MISSION",
  COMPLETE: "COMPLETE_MISSION",
} as const;

export async function ensureDriverNotificationActions(): Promise<void> {
  const Notifications = getExpoNotificationsModule();
  if (!Notifications) return;

  await Notifications.setNotificationCategoryAsync(DRIVER_NOTIFICATION_CATEGORY_ID, [
    {
      identifier: DRIVER_NOTIFICATION_ACTION_IDS.ACCEPT,
      buttonTitle: "Accept",
      options: { opensAppToForeground: false },
    },
    {
      identifier: DRIVER_NOTIFICATION_ACTION_IDS.DECLINE,
      buttonTitle: "Decline",
      options: { isDestructive: true, opensAppToForeground: false },
    },
    {
      identifier: DRIVER_NOTIFICATION_ACTION_IDS.START,
      buttonTitle: "Start",
      options: { opensAppToForeground: false },
    },
    {
      identifier: DRIVER_NOTIFICATION_ACTION_IDS.COMPLETE,
      buttonTitle: "Complete",
      options: { opensAppToForeground: false },
    },
  ]);
}
