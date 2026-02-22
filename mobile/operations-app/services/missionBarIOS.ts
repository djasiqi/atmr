import notifee, { IOSNotificationCategoryAction } from "@notifee/react-native";
import { Platform } from "react-native";
import type { MissionState } from "./missionState";
import { MissionStateManager } from "./missionState";

const NOTIFICATION_ID = "mission-bar-ios";
const CATEGORY_ID = "MISSION_ACTIVE";

// ---------------------------------------------------------------------------
// Category setup (call once at app boot)
// ---------------------------------------------------------------------------

let categoryCreated = false;

export async function setupIOSMissionCategory(): Promise<void> {
  if (Platform.OS !== "ios" || categoryCreated) return;
  categoryCreated = true;

  await notifee.setNotificationCategories([
    {
      id: CATEGORY_ID,
      actions: [
        {
          id: "MISSION_QUICK_ACTIONS",
          title: "Actions rapides",
          foreground: true,
        },
        {
          id: "MISSION_CALL",
          title: "Appeler",
          foreground: true,
        },
      ],
    },
  ]);
}

// ---------------------------------------------------------------------------
// Notification content builders (same logic as Android)
// ---------------------------------------------------------------------------

function formatMissionTitle(state: MissionState): string {
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
    ? new Date(preview.pickup_at).toLocaleTimeString("fr-CH", {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";
  const name = preview.can_show_identity ? preview.client_display : "Course suivante";
  return `Prochaine ${time} · ${name} · ${preview.pickup_short}`;
}

// ---------------------------------------------------------------------------
// Presenter: show / update / dismiss
// ---------------------------------------------------------------------------

export async function showMissionNotificationIOS(state: MissionState): Promise<void> {
  if (Platform.OS !== "ios") return;
  await setupIOSMissionCategory();

  await notifee.displayNotification({
    id: NOTIFICATION_ID,
    title: formatMissionTitle(state),
    body: formatNextBooking(state),
    ios: {
      categoryId: CATEGORY_ID,
      sound: undefined,
    },
  });
}

export async function updateMissionNotificationIOS(): Promise<void> {
  if (Platform.OS !== "ios") return;
  const state = MissionStateManager.getState();
  if (!state.activeMission) {
    await dismissMissionNotificationIOS();
    return;
  }
  await showMissionNotificationIOS(state);
}

export async function dismissMissionNotificationIOS(): Promise<void> {
  if (Platform.OS !== "ios") return;
  try {
    await notifee.cancelNotification(NOTIFICATION_ID);
  } catch {
    // may not exist
  }
}
