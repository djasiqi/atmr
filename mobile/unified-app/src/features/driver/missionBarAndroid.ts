import { canUseNotifee, loadNotifee } from "./notifeeCompat";

const MISSION_BAR_NOTIFICATION_ID = "driver-mission-bar";

export async function showMissionBarAndroid(_missionId: number, _status: string): Promise<void> {
  // Barre mission retirée du produit chauffeur (notif persistante Terminer/Annuler).
  return;
}

export async function hideMissionBarAndroid(): Promise<void> {
  if (!canUseNotifee()) return;
  try {
    const mod = await loadNotifee();
    if (!mod) return;
    const { default: notifee } = mod;
    await notifee.cancelNotification(MISSION_BAR_NOTIFICATION_ID);
    await notifee.stopForegroundService();
  } catch {
    // noop
  }
}
