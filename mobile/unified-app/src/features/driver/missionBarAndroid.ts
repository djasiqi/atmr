import { isFeatureEnabled } from "../../core/featureFlags/registry";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { canUseNotifee, loadNotifee } from "./notifeeCompat";
import { DRIVER_NOTIFICATION_CHANNELS } from "./notificationChannels";

const MISSION_BAR_NOTIFICATION_ID = "driver-mission-bar";

export async function showMissionBarAndroid(missionId: number, status: string): Promise<void> {
  if (!isFeatureEnabled("driver_mission_bar_enabled")) return;
  if (!canUseNotifee()) {
    emitDriverTelemetry("driver.mission_bar.android.unavailable", {
      source: "driver.mission_bar.android",
      mission_id: missionId,
      status,
    });
    return;
  }
  try {
    const mod = await loadNotifee();
    if (!mod) {
      emitDriverTelemetry("driver.mission_bar.android.unavailable", {
        source: "driver.mission_bar.android",
        mission_id: missionId,
        status,
      });
      return;
    }
    const { default: notifee, AndroidImportance } = mod;
    await notifee.createChannel({
      id: DRIVER_NOTIFICATION_CHANNELS.missionActive,
      name: "Driver Active Mission",
      importance: AndroidImportance.HIGH,
    });
    await notifee.displayNotification({
      id: MISSION_BAR_NOTIFICATION_ID,
      title: `Mission #${missionId}`,
      body: `Statut: ${status}`,
      android: {
        channelId: DRIVER_NOTIFICATION_CHANNELS.missionActive,
        // Évite un second FGS dismissable en parallèle du service expo-location.
        asForegroundService: false,
        ongoing: true,
        pressAction: {
          id: "open_mission",
          launchActivity: "default",
        },
        actions: [
          { title: "Terminer", pressAction: { id: "complete" } },
          { title: "Annuler", pressAction: { id: "cancel" } },
        ],
      },
    });
  } catch {
    emitDriverTelemetry("driver.mission_bar.android.unavailable", {
      source: "driver.mission_bar.android",
      mission_id: missionId,
      status,
    });
  }
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
