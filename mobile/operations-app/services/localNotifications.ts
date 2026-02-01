// mobile/operations-app/services/localNotifications.ts
import * as Notifications from "expo-notifications";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import { NotificationChannel } from "./notificationChannels";

/**
 * Clé de stockage pour les rappels planifiés
 */
const STORAGE_KEY = "scheduled_reminders";

/**
 * Mapping booking_id → notification_id
 */
interface ReminderMapping {
  [bookingId: number]: string; // notification_id Expo
}

/**
 * Charge les rappels planifiés depuis le stockage local
 */
async function loadReminders(): Promise<ReminderMapping> {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : {};
  } catch (error) {
    console.error("❌ Erreur chargement rappels:", error);
    return {};
  }
}

/**
 * Sauvegarde les rappels planifiés
 */
async function saveReminders(reminders: ReminderMapping): Promise<void> {
  try {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(reminders));
  } catch (error) {
    console.error("❌ Erreur sauvegarde rappels:", error);
  }
}

/**
 * Planifie un rappel local X minutes avant une mission
 */
export async function scheduleMissionReminder(
  booking: {
    id: number;
    scheduled_time: string;
    pickup_location: string;
    dropoff_location?: string;
    passenger_name?: string;
  },
  minutesBefore: number = 30
): Promise<string | null> {
  try {
    const scheduledTime = new Date(booking.scheduled_time);
    const reminderTime = new Date(
      scheduledTime.getTime() - minutesBefore * 60000
    );

    // Ne planifier que si dans le futur
    if (reminderTime < new Date()) {
      console.log(`⏭️ Rappel pour mission ${booking.id} déjà passé, skip`);
      return null;
    }

    // Vérifier si un rappel existe déjà
    const reminders = await loadReminders();
    if (reminders[booking.id]) {
      console.log(`ℹ️ Rappel pour mission ${booking.id} déjà planifié`);
      return reminders[booking.id];
    }

    // Créer la notification locale
    const route =
      booking.pickup_location && booking.dropoff_location
        ? `${booking.pickup_location} → ${booking.dropoff_location}`
        : booking.pickup_location;

    // Format trigger valide pour expo-notifications (évite "trigger object invalid")
    const secondsFromNow = Math.max(
      1,
      Math.floor((reminderTime.getTime() - Date.now()) / 1000)
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
        // Android: utiliser le canal "missions"
        ...(Platform.OS === "android" && {
          channelId: NotificationChannel.MISSIONS,
        }),
      },
      trigger,
    });

    // Sauvegarder le mapping
    reminders[booking.id] = notificationId;
    await saveReminders(reminders);

    console.log(
      `✅ Rappel planifié pour mission ${
        booking.id
      } à ${reminderTime.toLocaleTimeString()}`
    );

    return notificationId;
  } catch (error) {
    console.error(
      `❌ Erreur planification rappel pour mission ${booking.id}:`,
      error
    );
    return null;
  }
}

/**
 * Annule un rappel local pour une mission
 */
export async function cancelMissionReminder(bookingId: number): Promise<void> {
  try {
    const reminders = await loadReminders();
    const notificationId = reminders[bookingId];

    if (!notificationId) {
      console.log(`ℹ️ Aucun rappel à annuler pour mission ${bookingId}`);
      return;
    }

    // Annuler la notification
    await Notifications.cancelScheduledNotificationAsync(notificationId);

    // Supprimer du mapping
    delete reminders[bookingId];
    await saveReminders(reminders);

    console.log(`✅ Rappel annulé pour mission ${bookingId}`);
  } catch (error) {
    console.error(
      `❌ Erreur annulation rappel pour mission ${bookingId}:`,
      error
    );
  }
}

/**
 * Planifie des rappels pour toutes les missions actives
 */
export async function scheduleRemindersForActiveMissions(
  missions: Array<{
    id: number;
    scheduled_time: string;
    pickup_location: string;
    dropoff_location?: string;
    passenger_name?: string;
  }>
): Promise<number> {
  let scheduled = 0;

  for (const mission of missions) {
    const result = await scheduleMissionReminder(mission, 30);
    if (result) scheduled++;
  }

  console.log(`📅 ${scheduled}/${missions.length} rappels planifiés`);
  return scheduled;
}

/**
 * Annule tous les rappels (ex: lors de la déconnexion)
 */
export async function cancelAllReminders(): Promise<void> {
  try {
    await Notifications.cancelAllScheduledNotificationsAsync();
    await AsyncStorage.removeItem(STORAGE_KEY);
    console.log("🗑️ Tous les rappels annulés");
  } catch (error) {
    console.error("❌ Erreur annulation tous rappels:", error);
  }
}

/**
 * Nettoie les rappels expirés (à appeler périodiquement)
 */
export async function cleanupExpiredReminders(): Promise<void> {
  try {
    const reminders = await loadReminders();
    const scheduled = await Notifications.getAllScheduledNotificationsAsync();

    // IDs des notifications encore planifiées
    const scheduledIds = new Set(scheduled.map((notif) => notif.identifier));

    // Filtrer les rappels qui n'existent plus
    const activeReminders: ReminderMapping = {};
    for (const [bookingId, notifId] of Object.entries(reminders)) {
      if (scheduledIds.has(notifId)) {
        activeReminders[Number(bookingId)] = notifId;
      }
    }

    await saveReminders(activeReminders);

    const cleaned =
      Object.keys(reminders).length - Object.keys(activeReminders).length;
    if (cleaned > 0) {
      console.log(`🧹 ${cleaned} rappels expirés nettoyés`);
    }
  } catch (error) {
    console.error("❌ Erreur nettoyage rappels expirés:", error);
  }
}
