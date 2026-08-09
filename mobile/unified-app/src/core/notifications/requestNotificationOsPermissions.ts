/**
 * Demande OS notifications — module léger (Expo uniquement, pas Firebase).
 * Seul site prod autorisé à appeler Notifications.requestPermissionsAsync
 * (gate Play `check:play-compliance`).
 */
import { getExpoNotificationsModule } from "./expoNotificationsCompat";

export type NotificationOsPermissionResult = {
  granted: boolean;
  status?: string;
  canAskAgain?: boolean;
};

export async function requestNotificationOsPermissionsAsync(): Promise<NotificationOsPermissionResult> {
  const Notifications = getExpoNotificationsModule();
  if (!Notifications?.requestPermissionsAsync) {
    return { granted: false, status: "unavailable" };
  }
  const perm = await Notifications.requestPermissionsAsync();
  const granted = Boolean(perm?.granted || perm?.status === "granted");
  return {
    granted,
    status: typeof perm?.status === "string" ? perm.status : undefined,
    canAskAgain: typeof perm?.canAskAgain === "boolean" ? perm.canAskAgain : undefined,
  };
}
