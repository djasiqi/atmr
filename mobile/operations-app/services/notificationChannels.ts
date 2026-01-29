// mobile/operations-app/services/notificationChannels.ts
import * as Notifications from "expo-notifications";
import { Platform } from "react-native";

/**
 * Types de canaux de notification Android
 */
export enum NotificationChannel {
  CRITICAL = "critical",
  MISSIONS = "missions",
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

      console.log(`✅ Canal "${config.name}" créé`);
    }

    console.log("🎉 Tous les canaux configurés avec succès");
  } catch (error) {
    console.error("❌ Erreur lors de la configuration des canaux:", error);
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
