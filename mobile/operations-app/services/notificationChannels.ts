// mobile/operations-app/services/notificationChannels.ts
import * as Notifications from "expo-notifications";
import { Platform } from "react-native";
import { getLogger } from "@/utils/logger";

const log = getLogger("Channels");

/**
 * Types de canaux de notification Android
 */
export enum NotificationChannel {
  CRITICAL = "critical",
  MISSIONS = "missions",
  /** H2: Canal debug pour contourner channel legacy (missions créé en DEFAULT/LOW) */
  MISSIONS_V2 = "missions_v2",
  MISSION_ACTIVE = "mission_active",
  MESSAGES = "messages",
  INFO = "info",
}

/**
 * Configuration des canaux Android
 */
export interface ChannelConfig {
  id: NotificationChannel;
  name: string;
  description: string;
  importance: Notifications.AndroidImportance;
  sound: string | null;
  vibrationPattern: number[] | null;
  enableLights: boolean;
  lightColor?: string;
  bypassDnd?: boolean;
}

/**
 * Définitions des 4 canaux
 */
const CHANNEL_CONFIGS: ChannelConfig[] = [
  {
    id: NotificationChannel.CRITICAL,
    name: "🚨 Alertes Critiques",
    description:
      "Urgences médicales, accidents, alertes sécurité (ne peut pas être désactivé)",
    importance: Notifications.AndroidImportance.MAX,
    sound: "default", // TODO: Remplacer par "urgent_alert.mp3" si fichier custom
    vibrationPattern: [0, 500, 200, 500, 200, 500], // Triple vibration
    enableLights: true,
    lightColor: "#FF0000", // Rouge
    bypassDnd: true, // ✅ Ignorer mode silencieux
  },
  {
    id: NotificationChannel.MISSIONS,
    name: "🚗 Missions",
    description: "Nouvelles missions assignées, modifications et annulations",
    importance: Notifications.AndroidImportance.HIGH,
    sound: "default",
    vibrationPattern: [0, 250, 250, 250], // Double vibration
    enableLights: true,
    lightColor: "#2196F3", // Bleu
  },
  {
    id: NotificationChannel.MISSIONS_V2,
    name: "🚗 Missions (v2)",
    description: "Canal debug HIGH — contourne channel legacy missions",
    importance: Notifications.AndroidImportance.HIGH,
    sound: "default",
    vibrationPattern: [0, 250, 250, 250],
    enableLights: true,
    lightColor: "#2196F3",
  },
  {
    id: NotificationChannel.MISSION_ACTIVE,
    name: "📍 Mission en cours",
    description:
      "Notification persistante pendant la navigation avec actions rapides (En route, À bord, Terminer)",
    importance: Notifications.AndroidImportance.HIGH,
    sound: null,
    vibrationPattern: null,
    enableLights: false,
  },
  {
    id: NotificationChannel.MESSAGES,
    name: "💬 Messages",
    description: "Messages de l'équipe, du dispatcher et des clients",
    importance: Notifications.AndroidImportance.DEFAULT,
    sound: "default",
    vibrationPattern: [0, 100], // Simple vibration courte
    enableLights: true,
    lightColor: "#4CAF50", // Vert
  },
  {
    id: NotificationChannel.INFO,
    name: "📊 Informations",
    description: "Dispatch terminé, statistiques, informations générales",
    importance: Notifications.AndroidImportance.LOW,
    sound: null, // ❌ Pas de son
    vibrationPattern: null, // ❌ Pas de vibration
    enableLights: false,
  },
];

/**
 * Configure tous les canaux Android
 * À appeler au démarrage de l'app (App.tsx / _layout.tsx)
 */
export async function setupNotificationChannels(): Promise<void> {
  if (Platform.OS !== "android") {
    log.info("notification channels skipped (non-android)");
    return;
  }

  try {
    log.info("configuring notification channels");

    for (const config of CHANNEL_CONFIGS) {
      await Notifications.setNotificationChannelAsync(config.id, {
        name: config.name,
        description: config.description,
        importance: config.importance,
        sound: config.sound || undefined,
        vibrationPattern: config.vibrationPattern || undefined,
        enableLights: config.enableLights,
        lightColor: config.lightColor,
        bypassDnd: config.bypassDnd,
        lockscreenVisibility:
          Notifications.AndroidNotificationVisibility.PUBLIC,
      });

      // P2: Proof log — canal créé (diagnostic app killed)
      log.info("channel created", { id: config.id, name: config.name });
    }

    log.success("all channels configured");
  } catch (error) {
    log.error("channel configuration failed", { error });
  }
}

/** P0-A: Audit existence + importance + sound/vibration pour missions et missions_v2 */
export interface ChannelAuditResult {
  exists: boolean;
  importance: string;
  isHigh: boolean;
  hasSound: boolean;
  hasVibration: boolean;
}

async function auditChannelsForKillMode(): Promise<{
  missions: ChannelAuditResult;
  missions_v2: ChannelAuditResult;
}> {
  const empty: ChannelAuditResult = {
    exists: false,
    importance: "?",
    isHigh: false,
    hasSound: false,
    hasVibration: false,
  };
  const result = {
    missions: { ...empty },
    missions_v2: { ...empty },
  };

  if (Platform.OS !== "android") return result;

  const auditOne = async (
    channelId: string
  ): Promise<ChannelAuditResult> => {
    try {
      const ch = await Notifications.getNotificationChannelAsync(channelId);
      if (!ch) return empty;
      const raw = ch as unknown as Record<string, unknown>;
      const imp = raw.importance ?? (ch as any).importance ?? "?";
      const isHigh =
        imp === Notifications.AndroidImportance.HIGH ||
        imp === 4 ||
        imp === "high";
      const vib = raw.vibrationPattern ?? (ch as any).vibrationPattern;
      const hasVibration = Array.isArray(vib) ? vib.length > 0 : true;
      const hasSound = raw.sound != null || (ch as any).sound != null;
      return {
        exists: true,
        importance: String(imp),
        isHigh,
        hasSound: !!hasSound,
        hasVibration: !!hasVibration,
      };
    } catch {
      return empty;
    }
  };

  result.missions = await auditOne(NotificationChannel.MISSIONS);
  result.missions_v2 = await auditOne(NotificationChannel.MISSIONS_V2);

  // P0-A: Logs individuels pour diagnostic (existence + importance + sound/vibration)
  log.info("channel audit missions", {
    exists: result.missions.exists,
    importance: result.missions.importance,
    isHigh: result.missions.isHigh,
    hasSound: result.missions.hasSound,
    hasVibration: result.missions.hasVibration,
  });
  log.info("channel audit missions_v2", {
    exists: result.missions_v2.exists,
    importance: result.missions_v2.importance,
    isHigh: result.missions_v2.isHigh,
    hasSound: result.missions_v2.hasSound,
    hasVibration: result.missions_v2.hasVibration,
  });

  if (!result.missions_v2.exists) {
    log.warn("missions_v2 not created, open app once before kill test");
  }

  return result;
}

/** P0-B: Log unique "1 ligne" KILL-MODE readiness au boot */
export async function logKillModeReadiness(): Promise<void> {
  if (Platform.OS !== "android") return;

  try {
    const perm = await Notifications.getPermissionsAsync();
    const granted = perm.status === "granted";
    log.info("push proof permissions", {
      status: perm.status,
      granted,
      canAskAgain: perm.canAskAgain ?? "?",
    });
    const channels = await auditChannelsForKillMode();

    let manufacturer = "?";
    let model = "?";
    let androidVersion: number | string = "?";
    try {
      const Device = await import("expo-device");
      manufacturer = (Device as any).manufacturer ?? "?";
      model = (Device as any).modelName ?? (Device as any).modelId ?? "?";
      androidVersion = Platform.Version ?? "?";
    } catch {
      // ignore
    }

    const permOk = perm.status === "granted";
    const m2Exists = channels.missions_v2.exists;
    const m2High = channels.missions_v2.isHigh;
    const m1High = channels.missions.isHigh;

    let ready = "✓";
    if (!permOk) ready = "permission denied";
    else if (!m2Exists)
      ready = "missions_v2 non créé (ouvrir app une fois)";
    else if (!m2High) ready = "missions_v2 importance≠HIGH";
    else if (!m1High) ready = "missions legacy (missions_v2 OK)";

    log.info("kill-mode readiness", {
      ready,
      permissions: perm.status,
      missions: channels.missions.exists ? (channels.missions.isHigh ? "HIGH" : "legacy") : "absent",
      missions_v2: channels.missions_v2.exists ? (channels.missions_v2.isHigh ? "HIGH" : "low") : "absent",
      androidVersion,
      manufacturer,
      model,
    });
  } catch (e) {
    log.warn("kill-mode readiness check failed", { error: e });
  }
}

/** État complet pour Push Debug Card (dev-only) */
export interface KillModeState {
  platform: string;
  androidVersion: string | number;
  permissions: { status: string; granted: boolean };
  missions: ChannelAuditResult;
  missions_v2: ChannelAuditResult;
  manufacturer: string;
  model: string;
  ready: string;
  appOwnership: string;
}

/** P0-B: Récupère l'état complet pour la Push Debug Card */
export async function getKillModeState(): Promise<KillModeState | null> {
  if (Platform.OS !== "android") return null;

  try {
    const perm = await Notifications.getPermissionsAsync();
    const channels = await auditChannelsForKillMode();

    let manufacturer = "?";
    let model = "?";
    let androidVersion: string | number = "?";
    let appOwnership = "?";
    try {
      const Device = await import("expo-device");
      const Constants = await import("expo-constants").then((m) => m.default);
      manufacturer = (Device as any).manufacturer ?? "?";
      model = (Device as any).modelName ?? (Device as any).modelId ?? "?";
      androidVersion = Platform.Version ?? "?";
      appOwnership = Constants.appOwnership ?? "?";
    } catch {
      // ignore
    }

    const permOk = perm.status === "granted";
    const m2Exists = channels.missions_v2.exists;
    const m2High = channels.missions_v2.isHigh;
    const m1High = channels.missions.isHigh;

    let ready = "✓";
    if (!permOk) ready = "permission denied";
    else if (!m2Exists)
      ready = "missions_v2 non créé (ouvrir app une fois)";
    else if (!m2High) ready = "missions_v2 importance≠HIGH";
    else if (!m1High) ready = "missions legacy (missions_v2 OK)";

    return {
      platform: Platform.OS,
      androidVersion,
      permissions: { status: perm.status, granted: permOk },
      missions: channels.missions,
      missions_v2: channels.missions_v2,
      manufacturer,
      model,
      ready,
      appOwnership,
    };
  } catch (e) {
    log.warn("get kill mode state failed", { error: e });
    return null;
  }
}

/**
 * Récupère le canal approprié selon le type de notification
 */
export function getChannelForNotificationType(
  notificationType: string
): NotificationChannel {
  switch (notificationType) {
    case "urgent_alert":
    case "accident":
    case "emergency":
    case "security_zone":
    case "medical_emergency":
      return NotificationChannel.CRITICAL;

    case "booking":
    case "booking_updated":
    case "booking_cancelled":
    case "delay":
      return NotificationChannel.MISSIONS;

    case "message":
    case "chat_message":
    case "team_chat_message":
      return NotificationChannel.MESSAGES;

    case "mission_active":
      return NotificationChannel.MISSION_ACTIVE;

    case "dispatch_completed":
    case "stats":
    case "info":
      return NotificationChannel.INFO;

    default:
      return NotificationChannel.MISSIONS; // Par défaut
  }
}

/**
 * Récupère les informations d'un canal de notification
 * Utile pour vérifier la configuration d'un canal
 */
export async function getChannelSettings(
  channel: NotificationChannel
): Promise<Notifications.NotificationChannel | null> {
  if (Platform.OS !== "android") {
    log.warn("channels only available on android");
    return null;
  }

  try {
    const channelConfig = await Notifications.getNotificationChannelAsync(
      channel
    );
    return channelConfig;
  } catch (error) {
    log.error("get channel settings failed", { error });
    return null;
  }
}

// ========================================
// Phase 3.8 - Critical Alerts iOS
// ========================================

/**
 * Demande la permission pour Critical Alerts sur iOS
 *
 * ⚠️ IMPORTANT:
 * - Nécessite entitlement spécial d'Apple pour vraies Critical Alerts (bypass DnD)
 * - Sans entitlement: utilise interruptionLevel "critical" (iOS 15+)
 * - Avec entitlement: peut bypasser mode silencieux et DnD
 *
 * @returns true si permission accordée ou non nécessaire
 */
export async function requestCriticalAlertsPermission(): Promise<boolean> {
  if (Platform.OS !== "ios") {
    log.info("critical alerts only on ios");
    return true; // Pas nécessaire sur Android
  }

  try {
    log.info("requesting critical alerts permission ios");

    const { status } = await Notifications.requestPermissionsAsync({
      ios: {
        allowAlert: true,
        allowBadge: true,
        allowSound: true,
        // ⚠️ allowCriticalAlerts nécessite entitlement Apple
        // Décommentez après obtention entitlement:
        // allowCriticalAlerts: true,
      },
    });

    if (status === "granted") {
      log.success("notification permissions granted");
      return true;
    } else {
      log.warn("notification permissions denied");
      return false;
    }
  } catch (error) {
    log.error("critical alerts permission failed", { error });
    return false;
  }
}

/**
 * Vérifie si les Critical Alerts sont disponibles
 *
 * @returns true si Critical Alerts disponibles (entitlement approuvé)
 */
export async function areCriticalAlertsAvailable(): Promise<boolean> {
  if (Platform.OS !== "ios") {
    return false;
  }

  try {
    const permissions = await Notifications.getPermissionsAsync();

    // Vérifier si allowCriticalAlerts est accordé
    // Note: Cette propriété n'existe que si l'entitlement est configuré
    const ios = permissions.ios;
    if (ios && "criticalAlerts" in ios) {
      return ios.criticalAlerts === Notifications.IosAuthorizationStatus.AUTHORIZED;
    }

    return false;
  } catch (error) {
    log.info("critical alerts not available (missing entitlement)");
    return false;
  }
}
