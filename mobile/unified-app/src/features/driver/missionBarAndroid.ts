import { isFeatureEnabled } from "../../core/featureFlags/registry";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { DRIVER_NOTIFICATION_CHANNELS } from "./notificationChannels";

const MISSION_BAR_NOTIFICATION_ID = "driver-mission-bar";

export async function showMissionBarAndroid(missionId: number, status: string): Promise<void> {
  if (!isFeatureEnabled("driver_mission_bar_enabled")) return;
  try {
    const { default: notifee, AndroidImportance } = await import("@notifee/react-native");
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
        asForegroundService: true,
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
  try {
    const { default: notifee } = await import("@notifee/react-native");
    await notifee.cancelNotification(MISSION_BAR_NOTIFICATION_ID);
    await notifee.stopForegroundService();
  } catch {
    // noop
  }
}
