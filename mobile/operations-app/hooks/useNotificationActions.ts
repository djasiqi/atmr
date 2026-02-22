// mobile/operations-app/hooks/useNotificationActions.ts
import { useEffect, useRef } from "react";
import * as Notifications from "expo-notifications";
import { router } from "expo-router";
import { Alert } from "react-native";

import { getLogger } from "@/utils/logger";
import { api } from "@/services/api";
import { enterpriseStandardApi } from "@/services/enterpriseStandardApi";
import { NotificationActionId } from "@/services/notificationActions";
import { useAuth } from "@/hooks/useAuth";
import {
  trackNotificationOpened,
  trackNotificationActionClicked,
} from "@/services/notificationAnalytics";

const log = getLogger("NotifAction");

/**
 * Hook pour gérer les réponses aux actions des notifications
 */
export function useNotificationActions() {
  const responseListener = useRef<Notifications.Subscription | null>(null);
  const { mode, enterpriseSession } = useAuth();
  const modeRef = useRef({ mode, enterpriseSession });
  modeRef.current = { mode, enterpriseSession };

  useEffect(() => {
    // Écouter les réponses aux actions
    responseListener.current =
      Notifications.addNotificationResponseReceivedListener(handleActionResponse);

    return () => {
      if (responseListener.current) {
        responseListener.current.remove();
      }
    };
  }, []);

  /**
   * Gère la réponse à une action de notification
   */
  async function handleActionResponse(
    response: Notifications.NotificationResponse
  ): Promise<void> {
    try {
      const { actionIdentifier, notification } = response;
      const { data } = notification.request.content;

      log.info("notification action", { actionIdentifier, data });

      const notificationType = (data?.type as string) || "unknown";
      const notificationId = notification.request.identifier;

      // ✅ Phase 2 - Analytics: Tracker ouverture notification
      trackNotificationOpened(notificationType, notificationId, data || {});

      // Si pas d'action spécifique (juste ouverture notification)
      if (actionIdentifier === Notifications.DEFAULT_ACTION_IDENTIFIER) {
        handleDefaultAction(data);
        return;
      }

      // ✅ Phase 2 - Analytics: Tracker action spécifique
      trackNotificationActionClicked(
        notificationType,
        notificationId,
        actionIdentifier as string,
        data || {}
      );

      // Gérer les actions spécifiques
      switch (actionIdentifier) {
        case NotificationActionId.ACCEPT:
          await handleAcceptMission(data);
          break;

        case NotificationActionId.DECLINE:
          await handleDeclineMission(data);
          break;

        case NotificationActionId.VIEW:
        case NotificationActionId.VIEW_DETAILS:
          handleViewDetails(data);
          break;

        case NotificationActionId.CALL_DISPATCHER:
          handleCallDispatcher(data);
          break;

        case NotificationActionId.MARK_READ:
          await handleMarkMessageRead(data);
          break;

        case NotificationActionId.REPLY:
          handleReplyMessage(data);
          break;

        case NotificationActionId.MISSION_QUICK_ACTIONS:
          router.push("/quick-action?fromNavigation=true");
          break;

        case NotificationActionId.MISSION_CALL:
          handleCallDispatcher(data);
          break;

        default:
          log.warn("unknown action", { actionIdentifier });
      }
    } catch (error) {
      log.error("notification action handling failed", { error });
      Alert.alert(
        "Erreur",
        "Impossible de traiter l'action. Veuillez réessayer."
      );
    }
  }

  /**
   * Action par défaut : Ouvrir l'app et naviguer vers l'écran approprié
   */
  function handleDefaultAction(data: any): void {
    const deepLink = data?.deepLink;
    const bookingId = data?.booking_id;

    if (deepLink) {
      // Utiliser le deep link si disponible
      router.push(deepLink);
    } else if (bookingId) {
      // Naviguer vers la mission
      router.push(`/(tabs)/mission`);
    } else {
      // Par défaut, aller à l'écran d'accueil
      router.push(`/(tabs)/mission`);
    }
  }

  /**
   * Accepter une mission rapidement (sans ouvrir l'app)
   */
  async function handleAcceptMission(data: any): Promise<void> {
    const bookingId = data?.booking_id;

    if (!bookingId) {
      log.error("missing booking_id in data", { data });
      Alert.alert("Erreur", "Impossible d'accepter la mission.");
      return;
    }

    try {
      log.info("accepting mission", { bookingId });

      // Appel API pour accepter la mission
      const response = await api.post(
        `/driver/me/bookings/${bookingId}/quick-accept`,
        {}
      );

      if (response.status === 200 && response.data.ok) {
        log.success("mission accepted", { bookingId });

        // Afficher notification de confirmation locale
        await Notifications.scheduleNotificationAsync({
          content: {
            title: "✅ Mission acceptée",
            body: "La mission a été acceptée avec succès",
            sound: true,
          },
          trigger: null, // Immédiat
        });

        // Optionnel : Ouvrir l'app sur les détails de la mission
        router.push(`/(tabs)/mission`);
      } else {
        throw new Error("Échec acceptation mission");
      }
    } catch (error) {
      log.error("accept mission failed", { error });
      Alert.alert(
        "Erreur",
        "Impossible d'accepter la mission. Veuillez réessayer manuellement."
      );
    }
  }

  /**
   * Refuser une mission rapidement
   */
  async function handleDeclineMission(data: any): Promise<void> {
    const bookingId = data?.booking_id;

    if (!bookingId) {
      log.error("missing booking_id in data", { data });
      Alert.alert("Erreur", "Impossible de refuser la mission.");
      return;
    }

    try {
      log.info("declining mission", { bookingId });

      // Appel API pour refuser la mission
      const response = await api.post(
        `/driver/me/bookings/${bookingId}/quick-reject`,
        {}
      );

      if (response.status === 200 && response.data.ok) {
        log.success("mission declined", { bookingId });

        // Notification de confirmation
        await Notifications.scheduleNotificationAsync({
          content: {
            title: "❌ Mission refusée",
            body: "La mission a été refusée",
            sound: false, // Pas de son
          },
          trigger: null,
        });
      } else {
        throw new Error("Échec refus mission");
      }
    } catch (error) {
      log.error("decline mission failed", { error });
      Alert.alert(
        "Erreur",
        "Impossible de refuser la mission. Veuillez réessayer manuellement."
      );
    }
  }

  /**
   * Voir les détails d'une notification
   */
  function handleViewDetails(data: any): void {
    const deepLink = data?.deepLink;
    const bookingId = data?.booking_id;

    if (deepLink) {
      router.push(deepLink);
    } else if (bookingId) {
      router.push(`/(tabs)/mission`);
    } else {
      router.push(`/(tabs)/mission`);
    }
  }

  /**
   * Appeler le dispatcher
   */
  function handleCallDispatcher(data: any): void {
    const dispatcherPhone = data?.dispatcher_phone;

    if (dispatcherPhone) {
      // Ouvrir le dialer avec le numéro
      router.push(`tel:${dispatcherPhone}`);
    } else {
      Alert.alert(
        "Appeler le dispatcher",
        "Voulez-vous appeler le dispatcher ?",
        [
          { text: "Annuler", style: "cancel" },
          {
            text: "Appeler",
            onPress: () => {
              // Numéro par défaut ou récupéré depuis les settings
              router.push("tel:+33123456789");
            },
          },
        ]
      );
    }
  }

  /**
   * Marquer un message comme lu (mode-aware : enterprise vs driver)
   */
  async function handleMarkMessageRead(data: any): Promise<void> {
    const messageId = data?.message_id;

    if (!messageId) {
      log.error("missing message_id in data", { data });
      return;
    }

    try {
      log.info("marking message as read", { messageId });

      const { mode, enterpriseSession } = modeRef.current;
      // Politique : mode=enterprise + session valide → enterpriseStandardApi ; sinon → api (driver).
      // Si mode=enterprise mais enterpriseSession=null (état transitoire), fallback driver explicite.
      const isEnterprise = mode === "enterprise" && !!enterpriseSession;
      const client = isEnterprise ? enterpriseStandardApi : api;

      await client.post(`/messages/${messageId}/read`, {
        quick_action: true,
      });

      log.success("message marked as read", { messageId });
    } catch (error) {
      log.error("mark message read failed", { error });
    }
  }

  /**
   * Répondre à un message
   */
  function handleReplyMessage(data: any): void {
    // Ouvrir l'app sur l'écran de mission (pas d'écran messages pour l'instant)
    router.push(`/(tabs)/mission`);
  }

  return {
    // Exposer des méthodes si nécessaire pour tests
    handleAcceptMission,
    handleDeclineMission,
  };
}
