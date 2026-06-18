import { getExpoNotificationsModule } from "./expoNotificationsCompat";
import { readNotificationDisclosureAccepted } from "./notificationDisclosurePersistence";

/** Vérifie disclosure + permission OS avant tout POST save-push-token (flush ou direct). */
export async function canRegisterPushTokenWithBackend(): Promise<boolean> {
  const disclosureAccepted = await readNotificationDisclosureAccepted();
  if (!disclosureAccepted) return false;

  const Notifications = getExpoNotificationsModule();
  if (!Notifications) return false;

  try {
    const perm = await Notifications.getPermissionsAsync();
    return Boolean(perm.granted || perm.status === "granted");
  } catch {
    return false;
  }
}
