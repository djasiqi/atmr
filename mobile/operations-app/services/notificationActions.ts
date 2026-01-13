// mobile/operations-app/services/notificationActions.ts
import * as Notifications from "expo-notifications";
import { Platform } from "react-native";

/**
 * Catégories de notifications avec actions
 */
export enum NotificationCategory {
  MISSION_AVAILABLE = "mission_available",
  MISSION_URGENT = "mission_urgent",
  MESSAGE_RECEIVED = "message_received",
}

/**
 * Identifiants des actions
 */
export enum NotificationActionId {
  // Actions missions
  ACCEPT = "accept",
  DECLINE = "decline",
  VIEW = "view",

  // Actions urgentes
  CALL_DISPATCHER = "call_dispatcher",
  VIEW_DETAILS = "view_details",

  // Actions messages
  REPLY = "reply",
  MARK_READ = "mark_read",
}

/**
 * Configure toutes les catégories d'actions
 * À appeler au démarrage de l'app (après setupNotificationChannels)
 */
export async function setupNotificationActions(): Promise<void> {
  try {
    console.log("🔧 Configuration des actions notifications...");

    // ✅ Catégorie : Mission disponible
    await Notifications.setNotificationCategoryAsync(
      NotificationCategory.MISSION_AVAILABLE,
      [
        {
          identifier: NotificationActionId.ACCEPT,
          buttonTitle: "✅ Accepter",
          options: {
            opensAppToForeground: false, // API call en background
            isDestructive: false,
            isAuthenticationRequired: false,
          },
        },
        {
          identifier: NotificationActionId.DECLINE,
          buttonTitle: "❌ Refuser",
          options: {
            opensAppToForeground: false,
            isDestructive: true, // Rouge sur iOS
          },
        },
        {
          identifier: NotificationActionId.VIEW,
          buttonTitle: "👁️ Voir",
          options: {
            opensAppToForeground: true, // Ouvre l'app
            isDestructive: false,
          },
        },
      ],
      {
        previewPlaceholder: "Nouvelle mission disponible",
        intentIdentifiers: [],
        allowInCarPlay: true,
        showTitle: true,
        showSubtitle: true,
      }
    );

    console.log("✅ Catégorie 'mission_available' créée");

    // ✅ Catégorie : Mission urgente
    await Notifications.setNotificationCategoryAsync(
      NotificationCategory.MISSION_URGENT,
      [
        {
          identifier: NotificationActionId.CALL_DISPATCHER,
          buttonTitle: "📞 Appeler",
          options: {
            opensAppToForeground: true,
            isDestructive: false,
          },
        },
        {
          identifier: NotificationActionId.VIEW_DETAILS,
          buttonTitle: "🚨 Voir Détails",
          options: {
            opensAppToForeground: true,
            isDestructive: false,
          },
        },
      ],
      {
        previewPlaceholder: "Mission urgente",
        intentIdentifiers: [],
        allowInCarPlay: true,
        showTitle: true,
        showSubtitle: true,
      }
    );

    console.log("✅ Catégorie 'mission_urgent' créée");

    // ✅ Catégorie : Message reçu
    await Notifications.setNotificationCategoryAsync(
      NotificationCategory.MESSAGE_RECEIVED,
      [
        {
          identifier: NotificationActionId.REPLY,
          buttonTitle: "💬 Répondre",
          options: {
            opensAppToForeground: true,
            isDestructive: false,
          },
        },
        {
          identifier: NotificationActionId.MARK_READ,
          buttonTitle: "✓ Marquer lu",
          options: {
            opensAppToForeground: false,
            isDestructive: false,
          },
        },
      ],
      {
        previewPlaceholder: "Nouveau message",
        intentIdentifiers: [],
        allowInCarPlay: false, // Messages pas dans CarPlay
        showTitle: true,
        showSubtitle: true,
      }
    );

    console.log("✅ Catégorie 'message_received' créée");

    console.log("🎉 Toutes les catégories d'actions configurées");
  } catch (error) {
    console.error("❌ Erreur configuration actions:", error);
  }
}

/**
 * Récupère la catégorie appropriée selon le type de notification
 */
export function getCategoryForNotificationType(
  notificationType: string
): NotificationCategory | undefined {
  switch (notificationType) {
    case "booking":
    case "booking_assigned":
      return NotificationCategory.MISSION_AVAILABLE;

    case "urgent_alert":
    case "accident":
    case "emergency":
      return NotificationCategory.MISSION_URGENT;

    case "message":
    case "team_chat_message":
      return NotificationCategory.MESSAGE_RECEIVED;

    default:
      // Pas de catégorie = pas d'actions
      return undefined;
  }
}

/**
 * Vérifie si une action est disponible sur la plateforme
 */
export function isActionSupported(actionId: NotificationActionId): boolean {
  // Toutes les actions sont supportées sur iOS et Android modernes
  // Mais certaines peuvent être limitées selon les versions

  if (Platform.OS === "ios") {
    // iOS supporte toutes les actions depuis iOS 10+
    return true;
  }

  if (Platform.OS === "android") {
    // Android supporte les actions depuis API 19+
    // Les actions avec opensAppToForeground=false nécessitent API 24+
    return true;
  }

  return false;
}
