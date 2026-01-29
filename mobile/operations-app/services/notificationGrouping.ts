// mobile/operations-app/services/notificationGrouping.ts
import * as Notifications from "expo-notifications";
import { Platform } from "react-native";

/**
 * Service de groupement intelligent des notifications
 * Phase 2 - Enrichissement
 *
 * Évite le spam en regroupant les notifications similaires
 */

/**
 * Types de groupes de notifications
 */
export enum NotificationGroup {
  MISSIONS = "missions",
  MESSAGES = "messages",
  INFOS = "infos",
  ALERTS = "alerts",
}

/**
 * Récupère le groupe approprié selon le type de notification
 */
export function getGroupForNotificationType(
  notificationType: string
): NotificationGroup {
  switch (notificationType) {
    case "booking":
    case "booking_assigned":
    case "booking_updated":
    case "booking_cancelled":
      return NotificationGroup.MISSIONS;

    case "message":
    case "chat_message":
    case "team_chat_message":
      return NotificationGroup.MESSAGES;

    case "urgent_alert":
    case "accident":
    case "emergency":
      return NotificationGroup.ALERTS;

    case "dispatch_completed":
    case "stats":
    case "info":
      return NotificationGroup.INFOS;

    default:
      return NotificationGroup.INFOS;
  }
}

/**
 * Compte le nombre de notifications actives dans un groupe
 *
 * @param groupId Identifiant du groupe
 * @returns Nombre de notifications dans ce groupe
 */
export async function countNotificationsInGroup(
  groupId: NotificationGroup
): Promise<number> {
  try {
    // Récupérer toutes les notifications présentes
    const notifications =
      await Notifications.getPresentedNotificationsAsync();

    // Compter celles du groupe
    const count = notifications.filter((notif) => {
      const data = notif.request.content.data;
      return data?.group === groupId || data?.threadId === groupId;
    }).length;

    return count;
  } catch (error) {
    console.error("❌ Erreur comptage notifications groupe:", error);
    return 0;
  }
}

/**
 * Crée une notification groupée pour plusieurs missions
 *
 * @param missions Liste des missions à regrouper
 * @param channelId Canal Android à utiliser
 */
export async function createGroupedMissionsNotification(
  missions: Array<{
    id: number;
    pickup_location: string;
    scheduled_time?: string;
  }>,
  channelId?: string
): Promise<string | null> {
  if (missions.length === 0) return null;

  // Si une seule mission, pas besoin de grouper
  if (missions.length === 1) {
    return await Notifications.scheduleNotificationAsync({
      content: {
        title: "🚗 Nouvelle mission",
        body: missions[0].pickup_location,
        data: {
          type: "booking",
          booking_id: missions[0].id,
          group: NotificationGroup.MISSIONS,
        },
        ...(Platform.OS === "android" && channelId && { channelId }),
      },
      trigger: null,
    });
  }

  // Plusieurs missions : notification groupée
  const title = `🚗 ${missions.length} nouvelles missions`;
  const body = `${missions.length} missions vous attendent`;

  try {
    if (Platform.OS === "android") {
      // Android : InboxStyle pour liste
      const lines = missions
        .slice(0, 7)
        .map((m) => `#${m.id} - ${m.pickup_location}`);

      return await Notifications.scheduleNotificationAsync({
        content: {
          title,
          body,
          data: {
            type: "booking_group",
            mission_count: missions.length,
            group: NotificationGroup.MISSIONS,
            threadId: NotificationGroup.MISSIONS,
          },
          ...(channelId && { channelId }),

          // @ts-ignore - Android-specific
          android: {
            style: "inbox",
            lines,
            summaryText:
              missions.length > 7
                ? `+${missions.length - 7} autres missions`
                : undefined,
            groupSummary: true,
            group: NotificationGroup.MISSIONS,
          },
        },
        trigger: null,
      });
    } else {
      // iOS : notification simple avec compteur
      return await Notifications.scheduleNotificationAsync({
        content: {
        title,
        body: missions.map((m) => m.pickup_location).join(", "),
        data: {
          type: "booking_group",
          mission_count: missions.length,
          group: NotificationGroup.MISSIONS,
          threadIdentifier: NotificationGroup.MISSIONS,
          summaryArgument: `${missions.length} missions`,
          summaryArgumentCount: missions.length,
        },
      },
        trigger: null,
      });
    }
  } catch (error) {
    console.error("❌ Erreur création notification groupée:", error);
    return null;
  }
}

/**
 * Crée une notification groupée pour plusieurs messages
 *
 * @param messages Liste des messages à regrouper
 * @param channelId Canal Android à utiliser
 */
export async function createGroupedMessagesNotification(
  messages: Array<{
    id: number;
    sender: string;
    text: string;
    timestamp?: string;
  }>,
  channelId?: string
): Promise<string | null> {
  if (messages.length === 0) return null;

  // Si un seul message, pas besoin de grouper
  if (messages.length === 1) {
    return await Notifications.scheduleNotificationAsync({
      content: {
        title: messages[0].sender,
        body: messages[0].text,
        data: {
          type: "message",
          message_id: messages[0].id,
          group: NotificationGroup.MESSAGES,
        },
        ...(Platform.OS === "android" && channelId && { channelId }),
      },
      trigger: null,
    });
  }

  // Plusieurs messages : notification groupée
  const title = `💬 ${messages.length} nouveaux messages`;

  try {
    if (Platform.OS === "android") {
      // Android : MessagingStyle pour conversation
      const lines = messages
        .slice(0, 7)
        .map((m) => `${m.sender}: ${m.text.substring(0, 50)}`);

      return await Notifications.scheduleNotificationAsync({
        content: {
          title,
          body: `${messages.length} nouveaux messages`,
          data: {
            type: "message_group",
            message_count: messages.length,
            group: NotificationGroup.MESSAGES,
            threadId: NotificationGroup.MESSAGES,
          },
          ...(channelId && { channelId }),

          // @ts-ignore - Android-specific
          android: {
            style: "inbox", // Ou "messaging" si disponible
            lines,
            summaryText:
              messages.length > 7
                ? `+${messages.length - 7} autres messages`
                : undefined,
            groupSummary: true,
            group: NotificationGroup.MESSAGES,
          },
        },
        trigger: null,
      });
    } else {
      // iOS : notification simple avec compteur
      return await Notifications.scheduleNotificationAsync({
        content: {
          title,
        body: messages.map((m) => `${m.sender}: ${m.text}`).join("\n"),
        data: {
          type: "message_group",
          message_count: messages.length,
          group: NotificationGroup.MESSAGES,
          threadIdentifier: NotificationGroup.MESSAGES,
          summaryArgument: `${messages.length} messages`,
          summaryArgumentCount: messages.length,
        },
      },
        trigger: null,
      });
    }
  } catch (error) {
    console.error("❌ Erreur création notification messages groupés:", error);
    return null;
  }
}

/**
 * Supprime toutes les notifications d'un groupe
 *
 * @param groupId Identifiant du groupe
 */
export async function clearNotificationGroup(
  groupId: NotificationGroup
): Promise<void> {
  try {
    const notifications =
      await Notifications.getPresentedNotificationsAsync();

    for (const notif of notifications) {
      const data = notif.request.content.data;
      if (data?.group === groupId || data?.threadId === groupId) {
        await Notifications.dismissNotificationAsync(
          notif.request.identifier
        );
      }
    }

    console.log(`🧹 Groupe "${groupId}" nettoyé`);
  } catch (error) {
    console.error("❌ Erreur nettoyage groupe:", error);
  }
}

/**
 * Met à jour une notification groupée avec un nouveau compteur
 *
 * @param groupId Identifiant du groupe
 * @param count Nouveau compteur
 * @param title Titre personnalisé (optionnel)
 * @param body Corps personnalisé (optionnel)
 */
export async function updateGroupedNotificationCount(
  groupId: NotificationGroup,
  count: number,
  title?: string,
  body?: string
): Promise<void> {
  if (count <= 0) {
    await clearNotificationGroup(groupId);
    return;
  }

  try {
    // Supprimer l'ancienne notification groupée
    await clearNotificationGroup(groupId);

    // Créer une nouvelle avec le bon compteur
    const defaultTitle = getGroupTitle(groupId, count);
    const defaultBody = getGroupBody(groupId, count);

    await Notifications.scheduleNotificationAsync({
      content: {
        title: title || defaultTitle,
        body: body || defaultBody,
        data: {
          type: `${groupId}_group`,
          count,
          group: groupId,
          threadId: groupId,
        },
        badge: count,
      },
      trigger: null,
    });

    console.log(
      `✅ Notification groupée mise à jour: ${groupId} (${count})`
    );
  } catch (error) {
    console.error("❌ Erreur mise à jour notification groupée:", error);
  }
}

/**
 * Génère un titre pour une notification groupée
 */
function getGroupTitle(groupId: NotificationGroup, count: number): string {
  switch (groupId) {
    case NotificationGroup.MISSIONS:
      return `🚗 ${count} ${count > 1 ? "nouvelles missions" : "nouvelle mission"}`;
    case NotificationGroup.MESSAGES:
      return `💬 ${count} ${count > 1 ? "nouveaux messages" : "nouveau message"}`;
    case NotificationGroup.ALERTS:
      return `🚨 ${count} ${count > 1 ? "alertes" : "alerte"}`;
    case NotificationGroup.INFOS:
      return `📊 ${count} ${count > 1 ? "nouvelles infos" : "nouvelle info"}`;
    default:
      return `${count} notifications`;
  }
}

/**
 * Génère un corps pour une notification groupée
 */
function getGroupBody(groupId: NotificationGroup, count: number): string {
  switch (groupId) {
    case NotificationGroup.MISSIONS:
      return `${count} missions vous attendent`;
    case NotificationGroup.MESSAGES:
      return `${count} messages non lus`;
    case NotificationGroup.ALERTS:
      return `${count} alertes nécessitent votre attention`;
    case NotificationGroup.INFOS:
      return `${count} nouvelles informations`;
    default:
      return `${count} notifications en attente`;
  }
}

/**
 * Vérifie si le groupement est activé sur la plateforme
 */
export function isGroupingSupported(): boolean {
  // Le groupement est supporté sur Android et iOS (avec variations)
  return true;
}
