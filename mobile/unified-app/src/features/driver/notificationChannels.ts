import { getExpoNotificationsModule } from "../../core/notifications/expoNotificationsCompat";
import { normalizeDriverEventType } from "../../core/realtime/eventContracts";
import { PRODUCTION_LOCALE } from "../../i18n/productionLocale";

/**
 * Channel generique, cree pour TOUS les contextes (driver/company/client) des le
 * montage du provider, independamment des flags chauffeur. Sert de fallback
 * universel: sur Android targetSdk 36, une notification postee sans channel est
 * silencieusement supprimee.
 */
export const GENERIC_NOTIFICATION_CHANNEL_ID = "default";

export const DRIVER_NOTIFICATION_CHANNELS = {
  missionUpdates: "mission_updates",
  chat: "chat",
  urgent: "urgent",
  silent: "silent",
  lockscreen: "lock-screen",
  missionActive: "driver-missions-active",
} as const;

export type DriverNotificationEventType =
  | "mission_assigned"
  | "mission_updated"
  | "mission_cancelled"
  | "mission_reassigned"
  | "chat_message"
  | "mission_refresh"
  | "lockscreen_hint";

type DriverNotificationContract = {
  event: DriverNotificationEventType;
  channelId: string;
  priority: "MAX" | "HIGH" | "DEFAULT" | "LOW";
  silent: boolean;
  action: "open_mission" | "open_chat" | "sync_only";
};

const DRIVER_NOTIFICATION_CONTRACT: Record<DriverNotificationEventType, DriverNotificationContract> = {
  mission_assigned: {
    event: "mission_assigned",
    channelId: DRIVER_NOTIFICATION_CHANNELS.missionUpdates,
    priority: "HIGH",
    silent: false,
    action: "open_mission",
  },
  mission_updated: {
    event: "mission_updated",
    channelId: DRIVER_NOTIFICATION_CHANNELS.missionUpdates,
    priority: "HIGH",
    silent: false,
    action: "open_mission",
  },
  mission_cancelled: {
    event: "mission_cancelled",
    channelId: DRIVER_NOTIFICATION_CHANNELS.urgent,
    priority: "MAX",
    silent: false,
    action: "open_mission",
  },
  mission_reassigned: {
    event: "mission_reassigned",
    channelId: DRIVER_NOTIFICATION_CHANNELS.missionUpdates,
    priority: "HIGH",
    silent: false,
    action: "open_mission",
  },
  chat_message: {
    event: "chat_message",
    channelId: DRIVER_NOTIFICATION_CHANNELS.chat,
    priority: "DEFAULT",
    silent: false,
    action: "open_chat",
  },
  mission_refresh: {
    event: "mission_refresh",
    channelId: DRIVER_NOTIFICATION_CHANNELS.silent,
    priority: "LOW",
    silent: true,
    action: "sync_only",
  },
  lockscreen_hint: {
    event: "lockscreen_hint",
    channelId: DRIVER_NOTIFICATION_CHANNELS.lockscreen,
    priority: "HIGH",
    silent: false,
    action: "open_mission",
  },
};

export function resolveDriverNotificationContract(
  input: string | null | undefined
): DriverNotificationContract {
  const normalized = normalizeDriverEventType(input ?? "");
  const key = (normalized ?? input ?? "").toLowerCase() as DriverNotificationEventType;
  return DRIVER_NOTIFICATION_CONTRACT[key] ?? DRIVER_NOTIFICATION_CONTRACT.mission_updated;
}

/**
 * Cree le channel generique `default` (HIGH). A appeler au montage, sans aucun
 * gating (ni flag, ni role, ni contexte), pour garantir qu'au moins un channel
 * existe des le premier lancement quel que soit le compte.
 */
export async function ensureBaseNotificationChannels(): Promise<void> {
  const Notifications = getExpoNotificationsModule();
  if (!Notifications) return;

  await Notifications.setNotificationChannelAsync(GENERIC_NOTIFICATION_CHANNEL_ID, {
    name: PRODUCTION_LOCALE.notificationChannelGeneral,
    importance: Notifications.AndroidImportance.HIGH,
    lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
  });
}

/**
 * Nombre de channels Android actuellement enregistres (0 sur iOS/web : normal).
 * Permet d'instrumenter le cas anormal "0 channel" en production (Android).
 */
export async function getRegisteredNotificationChannelCount(): Promise<number> {
  const Notifications = getExpoNotificationsModule();
  if (!Notifications) return 0;
  if (typeof Notifications.getNotificationChannelsAsync !== "function") return 0;
  try {
    const channels = await Notifications.getNotificationChannelsAsync();
    return Array.isArray(channels) ? channels.length : 0;
  } catch {
    return 0;
  }
}

export async function ensureDriverNotificationChannels(): Promise<void> {
  const Notifications = getExpoNotificationsModule();
  if (!Notifications) return;

  await Notifications.setNotificationChannelAsync(DRIVER_NOTIFICATION_CHANNELS.urgent, {
    name: PRODUCTION_LOCALE.notificationChannelUrgent,
    importance: Notifications.AndroidImportance.MAX,
    vibrationPattern: [0, 300, 150, 300],
    lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
  });
  await Notifications.setNotificationChannelAsync(DRIVER_NOTIFICATION_CHANNELS.missionUpdates, {
    name: PRODUCTION_LOCALE.notificationChannelMissionUpdates,
    importance: Notifications.AndroidImportance.HIGH,
    vibrationPattern: [0, 180, 100, 180],
    lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
  });
  await Notifications.setNotificationChannelAsync(DRIVER_NOTIFICATION_CHANNELS.chat, {
    name: PRODUCTION_LOCALE.notificationChannelChat,
    importance: Notifications.AndroidImportance.DEFAULT,
    vibrationPattern: [0, 120],
    lockscreenVisibility: Notifications.AndroidNotificationVisibility.PRIVATE,
  });
  await Notifications.setNotificationChannelAsync(DRIVER_NOTIFICATION_CHANNELS.silent, {
    name: PRODUCTION_LOCALE.notificationChannelSilentSync,
    importance: Notifications.AndroidImportance.LOW,
    sound: null,
    vibrationPattern: [0],
    lockscreenVisibility: Notifications.AndroidNotificationVisibility.SECRET,
  });
  await Notifications.setNotificationChannelAsync(DRIVER_NOTIFICATION_CHANNELS.lockscreen, {
    name: PRODUCTION_LOCALE.notificationChannelLockScreen,
    importance: Notifications.AndroidImportance.HIGH,
    lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
  });
  await Notifications.setNotificationChannelAsync(DRIVER_NOTIFICATION_CHANNELS.missionActive, {
    name: PRODUCTION_LOCALE.notificationChannelMissionActive,
    importance: Notifications.AndroidImportance.HIGH,
    sound: null,
  });
}
