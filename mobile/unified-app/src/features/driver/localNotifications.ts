import { getExpoNotificationsModule } from "../../core/notifications/expoNotificationsCompat";

export async function scheduleDriverMissionReminder(params: {
  missionId: number;
  title?: string;
  body?: string;
  secondsFromNow?: number;
}) {
  const Notifications = getExpoNotificationsModule();
  if (!Notifications) return null;

  return Notifications.scheduleNotificationAsync({
    content: {
      title: params.title ?? "Rappel mission",
      body: params.body ?? `Mission #${params.missionId} a verifier`,
      data: { mission_id: params.missionId, type: "reminder_action" },
    },
    trigger: {
      type: Notifications.SchedulableTriggerInputTypes.TIME_INTERVAL,
      seconds: Math.max(30, params.secondsFromNow ?? 300),
      repeats: false,
    },
  });
}
