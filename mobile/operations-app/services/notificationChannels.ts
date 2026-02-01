// mobile/operations-app/services/notificationChannels.ts
import * as Notifications from "expo-notifications";
import { Platform } from "react-native";

/**
 * Types de canaux de notification Android
 */
export enum NotificationChannel {
  CRITICAL = "critical",
  MISSIONS = "missions",
  /** H2: Canal debug pour contourner channel legacy (missions créé en DEFAULT/LOW) */
  MISSIONS_V2 = "missions_v2",
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
    console.log("📱 Canaux notifications skippés (non-Android)");
    return;
  }

  try {
    console.log("🔧 Configuration des canaux de notification...");

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
      console.log(`🔔 PUSH_PROOF channel created: ${config.id} (${config.name})`);
    }

    console.log("🎉 Tous les canaux configurés avec succès");
  } catch (error) {
    console.error("❌ Erreur lors de la configuration des canaux:", error);
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
  console.log(
    `🔔 PUSH_PROOF channel audit: missions exists=${result.missions.exists} importance=${result.missions.importance} isHigh=${result.missions.isHigh} sound=${result.missions.hasSound} vibration=${result.missions.hasVibration}`
  );
  console.log(
    `🔔 PUSH_PROOF channel audit: missions_v2 exists=${result.missions_v2.exists} importance=${result.missions_v2.importance} isHigh=${result.missions_v2.isHigh} sound=${result.missions_v2.hasSound} vibration=${result.missions_v2.hasVibration}`
  );

  if (!result.missions_v2.exists) {
    console.warn(
      "🔔 PUSH_PROOF missions_v2 NON CRÉÉ — ouvrir l'app une fois pour créer le canal avant test app kill"
    );
  }

  return result;
}

/** P0-B: Log unique "1 ligne" KILL-MODE readiness au boot */
export async function logKillModeReadiness(): Promise<void> {
  if (Platform.OS !== "android") return;

  try {
    const perm = await Notifications.getPermissionsAsync();
    const granted = perm.status === "granted";
    console.log(
      `🔔 PUSH_PROOF permissions: status=${perm.status} granted=${granted} canAskAgain=${perm.canAskAgain ?? "?"}`
    );
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

    console.log(
      `🔔 KILL-MODE readiness: ${ready} | permissions=${perm.status} ` +
        `missions=${channels.missions.exists ? (channels.missions.isHigh ? "HIGH" : "legacy") : "absent"} ` +
        `missions_v2=${channels.missions_v2.exists ? (channels.missions_v2.isHigh ? "HIGH" : "low") : "absent"} ` +
        `android=${androidVersion} device=${manufacturer}/${model}`
    );
  } catch (e) {
    console.warn("🔔 KILL-MODE readiness: impossible de vérifier", e);
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
    console.warn("🔔 getKillModeState: impossible de récupérer", e);
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
    console.warn("⚠️ Canaux disponibles uniquement sur Android");
    return null;
  }

  try {
    const channelConfig = await Notifications.getNotificationChannelAsync(
      channel
    );
    return channelConfig;
  } catch (error) {
    console.error("❌ Impossible de récupérer les infos du canal:", error);
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
    console.log("ℹ️ Critical Alerts uniquement sur iOS");
    return true; // Pas nécessaire sur Android
  }

  try {
    console.log("🔔 Demande permission Critical Alerts iOS...");

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
      console.log("✅ Permissions notifications accordées");
      return true;
    } else {
      console.warn("⚠️ Permissions notifications refusées");
      return false;
    }
  } catch (error) {
    console.error("❌ Erreur permission Critical Alerts:", error);
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
    console.log("ℹ️ Critical Alerts non disponibles (entitlement manquant)");
    return false;
  }
}
