import notifee, { AndroidImportance, AndroidVisibility } from "@notifee/react-native";
import * as Location from "expo-location";
import { Platform } from "react-native";
import type { MissionState, MissionBarStatus, BookingPreview } from "./missionState";
import { MissionStateManager } from "./missionState";

const BACKGROUND_LOCATION_TASK = "background-location-task";
const NOTIFICATION_ID = "mission-bar";
const CHANNEL_ID = "mission_active";

// ---------------------------------------------------------------------------
// Tracking guard: avoid 2 foreground services
// ---------------------------------------------------------------------------

async function isTrackingServiceActive(): Promise<boolean> {
  try {
    return await Location.hasStartedLocationUpdatesAsync(BACKGROUND_LOCATION_TASK);
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Notification content builders
// ---------------------------------------------------------------------------

function formatMissionTitle(state: MissionState): string {
  if (state.privacyMode) {
    switch (state.currentStatus) {
      case "ASSIGNED": return "Mission assignée";
      case "EN_ROUTE": return "En route vers la prise en charge";
      case "IN_PROGRESS": return "Mission en cours";
      case "COMPLETED": return "Mission terminée";
      default: return "Mission en cours";
    }
  }
  const dest = getCurrentDestination(state);
  switch (state.currentStatus) {
    case "ASSIGNED":
      return `ASSIGNÉE — ${dest}`;
    case "EN_ROUTE":
      return `EN ROUTE → ${dest}`;
    case "IN_PROGRESS":
      return `À BORD → ${dest}`;
    case "COMPLETED":
      return "Mission terminée";
    default:
      return "Mission en cours";
  }
}

function getCurrentDestination(state: MissionState): string {
  if (state.privacyMode) return "";
  const m = state.activeMission;
  if (!m) return "";
  if (state.currentStatus === "ASSIGNED" || state.currentStatus === "EN_ROUTE") {
    return shortenAddress(m.pickup_location);
  }
  return shortenAddress(m.dropoff_location);
}

function shortenAddress(addr: string | undefined): string {
  if (!addr) return "…";
  const parts = addr.split(",");
  return parts[0]?.trim().substring(0, 30) ?? addr.substring(0, 30);
}

function formatNextBooking(state: MissionState): string {
  const preview = state.nextBookingPreview;
  if (!preview) return "";
  const time = preview.pickup_at
    ? new Date(preview.pickup_at).toLocaleTimeString("fr-CH", { hour: "2-digit", minute: "2-digit" })
    : "";
  if (state.privacyMode || !preview.can_show_identity) {
    return time ? `Prochaine course à ${time}` : "Course suivante";
  }
  return `Prochaine ${time} · ${preview.client_display} · ${preview.pickup_short}`;
}

function buildActionsForStatus(status: MissionBarStatus): Array<{
  title: string;
  pressAction: { id: string };
}> {
  switch (status) {
    case "ASSIGNED":
      return [
        { title: "En route", pressAction: { id: "MISSION_STATUS_EN_ROUTE" } },
        { title: "Appeler", pressAction: { id: "CALL" } },
      ];
    case "EN_ROUTE":
      return [
        { title: "À bord", pressAction: { id: "MISSION_STATUS_IN_PROGRESS" } },
        { title: "Appeler", pressAction: { id: "CALL" } },
      ];
    case "IN_PROGRESS":
      return [
        { title: "Terminer", pressAction: { id: "MISSION_STATUS_COMPLETED" } },
        { title: "Appeler", pressAction: { id: "CALL" } },
      ];
    default:
      return [];
  }
}

// ---------------------------------------------------------------------------
// Presenter: show / update / dismiss
// ---------------------------------------------------------------------------

let channelCreated = false;

async function ensureChannel(): Promise<void> {
  if (channelCreated) return;
  await notifee.createChannel({
    id: CHANNEL_ID,
    name: "Mission en cours",
    importance: AndroidImportance.HIGH,
    visibility: AndroidVisibility.PUBLIC,
  });
  channelCreated = true;
}

export async function showMissionNotification(state: MissionState): Promise<void> {
  if (Platform.OS !== "android") return;
  await ensureChannel();

  const trackingActive = await isTrackingServiceActive();

  await notifee.displayNotification({
    id: NOTIFICATION_ID,
    title: formatMissionTitle(state),
    body: formatNextBooking(state),
    android: {
      channelId: CHANNEL_ID,
      asForegroundService: !trackingActive,
      ongoing: true,
      smallIcon: "notification_icon",
      pressAction: { id: "open-quick-actions", launchActivity: "default" },
      actions: buildActionsForStatus(state.currentStatus),
    },
  });
}

export async function updateMissionNotification(): Promise<void> {
  if (Platform.OS !== "android") return;
  const state = MissionStateManager.getState();
  if (!state.activeMission) {
    await dismissMissionNotification();
    return;
  }
  await showMissionNotification(state);
}

export async function dismissMissionNotification(): Promise<void> {
  if (Platform.OS !== "android") return;
  try {
    await notifee.cancelNotification(NOTIFICATION_ID);
    await notifee.stopForegroundService();
  } catch {
    // may not be running
  }
}
