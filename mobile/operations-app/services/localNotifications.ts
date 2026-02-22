import * as Notifications from "expo-notifications";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import { getLogger } from "@/utils/logger";
import { NotificationChannel } from "./notificationChannels";

const log = getLogger("LocalNotif");

const STORAGE_KEY = "scheduled_reminders";

const isNative = Platform.OS !== "web";

interface ReminderMapping {
  [bookingId: number]: string;
}

async function loadReminders(): Promise<ReminderMapping> {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : {};
  } catch {
    return {};
  }
}

async function saveReminders(reminders: ReminderMapping): Promise<void> {
  try {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(reminders));
  } catch { /* noop */ }
}

export async function scheduleMissionReminder(
  booking: {
    id: number;
    scheduled_time: string;
    pickup_location: string;
    dropoff_location?: string;
    passenger_name?: string;
  },
  minutesBefore: number = 30,
): Promise<string | null> {
  if (!isNative) return null;

  try {
    const scheduledTime = new Date(booking.scheduled_time);
    const reminderTime = new Date(
      scheduledTime.getTime() - minutesBefore * 60000,
    );

    if (reminderTime < new Date()) return null;

    const reminders = await loadReminders();
    if (reminders[booking.id]) return reminders[booking.id];

    const route =
      booking.pickup_location && booking.dropoff_location
        ? `${booking.pickup_location} → ${booking.dropoff_location}`
        : booking.pickup_location;

    const secondsFromNow = Math.max(
      1,
      Math.floor((reminderTime.getTime() - Date.now()) / 1000),
    );
    const trigger =
      secondsFromNow <= 60 * 60 * 24
        ? ({
            type: Notifications.SchedulableTriggerInputTypes.TIME_INTERVAL,
            seconds: secondsFromNow,
            repeats: false,
            ...(Platform.OS === "android" && {
              channelId: NotificationChannel.MISSIONS,
            }),
          } as Notifications.TimeIntervalTriggerInput)
        : ({
            type: Notifications.SchedulableTriggerInputTypes.DATE,
            date: reminderTime,
            ...(Platform.OS === "android" && {
              channelId: NotificationChannel.MISSIONS,
            }),
          } as Notifications.DateTriggerInput);

    const notificationId = await Notifications.scheduleNotificationAsync({
      content: {
        title:
          minutesBefore <= 10
            ? `Départ recommandé — course #${booking.id}`
            : `Course #${booking.id} dans ${minutesBefore} min`,
        body: booking.passenger_name
          ? `${booking.passenger_name} — ${route}`
          : route,
        data: {
          type: "local_reminder",
          booking_id: booking.id,
          deepLink: `atmr://booking/${booking.id}`,
        },
        sound: "default",
        priority: Notifications.AndroidNotificationPriority.HIGH,
        ...(Platform.OS === "android" && {
          channelId: NotificationChannel.MISSIONS,
        }),
      },
      trigger,
    });

    reminders[booking.id] = notificationId;
    await saveReminders(reminders);
    return notificationId;
  } catch (error) {
    log.error("schedule mission reminder failed", { bookingId: booking.id, error });
    return null;
  }
}

export async function cancelMissionReminder(bookingId: number): Promise<void> {
  if (!isNative) return;

  try {
    const reminders = await loadReminders();
    const notificationId = reminders[bookingId];
    if (!notificationId) return;

    await Notifications.cancelScheduledNotificationAsync(notificationId);
    delete reminders[bookingId];
    await saveReminders(reminders);
  } catch (error) {
    log.error("cancel mission reminder failed", { bookingId, error });
  }
}

export async function scheduleRemindersForActiveMissions(
  missions: Array<{
    id: number;
    scheduled_time: string;
    pickup_location: string;
    dropoff_location?: string;
    passenger_name?: string;
  }>,
): Promise<number> {
  if (!isNative) return 0;

  let scheduled = 0;
  for (const mission of missions) {
    const result = await scheduleMissionReminder(mission, 30);
    if (result) scheduled++;
  }
  return scheduled;
}

export async function cancelAllReminders(): Promise<void> {
  if (!isNative) return;

  try {
    await Notifications.cancelAllScheduledNotificationsAsync();
    await AsyncStorage.removeItem(STORAGE_KEY);
  } catch { /* noop */ }
}

export async function cleanupExpiredReminders(): Promise<void> {
  if (!isNative) return;

  try {
    const reminders = await loadReminders();
    const scheduled = await Notifications.getAllScheduledNotificationsAsync();
    const scheduledIds = new Set(scheduled.map((n) => n.identifier));

    const activeReminders: ReminderMapping = {};
    for (const [bookingId, notifId] of Object.entries(reminders)) {
      if (scheduledIds.has(notifId)) {
        activeReminders[Number(bookingId)] = notifId;
      }
    }
    await saveReminders(activeReminders);
  } catch { /* noop */ }
}
