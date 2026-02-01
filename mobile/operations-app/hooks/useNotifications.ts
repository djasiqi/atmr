// operations-app/hooks/useNotifications.ts — utilisé en mode driver (mission, trips, dashboard)
import { useEffect } from "react";
import * as Notifications from "expo-notifications";
import type { NotificationBehavior } from "expo-notifications";
import { Platform } from "react-native";
import * as Linking from "expo-linking";
import { useAuth } from "@/hooks/useAuth";
import { logError } from "@/utils/errorHandler";
import { validateDeepLink } from "@/services/deepLinkHandler";

// 🔔 Configuration du comportement des notifications en mode foreground
const foregroundBehavior: NotificationBehavior = {
  shouldShowAlert: true,
  shouldPlaySound: true,
  shouldSetBadge: true,
  shouldShowBanner: true,
  shouldShowList: true,
};

// P0.2: Driver ID pour filtrer self-notifications (actor=driver)
let _currentDriverIdForFilter: number | undefined;

export function setCurrentDriverIdForNotificationFilter(id: number | undefined) {
  _currentDriverIdForFilter = id;
}

function _shouldIgnoreNotificationForDriver(data: Record<string, unknown>): {
  ignore: boolean;
  reason?: string;
} {
  const recipientRole = data.recipient_role as string | undefined;
  const actorRole = data.actor_role as string | undefined;
  const actorId = data.actor_id as number | undefined;
  if (recipientRole === "company") {
    return { ignore: true, reason: "recipient_role=company (not for driver)" };
  }
  if (
    actorRole === "driver" &&
    actorId !== undefined &&
    _currentDriverIdForFilter !== undefined &&
    actorId === _currentDriverIdForFilter
  ) {
    return { ignore: true, reason: "actor=driver self-notification (exclude_actor)" };
  }
  return { ignore: false };
}

export const useNotifications = () => {
  const { driver, loading } = useAuth();

  // P0.5: Vérification bundle — si ce log n'apparaît pas, mauvais code chargé
  useEffect(() => {
    if (__DEV__) {
      console.log("🚨 useNotifications OPERATIONS-APP (driver mode) LOADED");
    }
  }, []);

  // P0.2: Handler dynamique pour filtrer affichage (recipient_role, actor self)
  useEffect(() => {
    setCurrentDriverIdForNotificationFilter(driver?.id);
    if (Platform.OS === "web") return;
    Notifications.setNotificationHandler({
      handleNotification: async (notification) => {
        const data = (notification.request.content.data || {}) as Record<string, unknown>;
        const { ignore } = _shouldIgnoreNotificationForDriver(data);
        if (ignore) {
          return {
            shouldShowAlert: false,
            shouldPlaySound: false,
            shouldSetBadge: false,
            shouldShowBanner: false,
            shouldShowList: false,
          };
        }
        return foregroundBehavior;
      },
    });
    return () => {
      setCurrentDriverIdForNotificationFilter(undefined);
      Notifications.setNotificationHandler({ handleNotification: async () => foregroundBehavior });
    };
  }, [driver?.id]);

  useEffect(() => {
    // Ne rien faire tant que l'utilisateur n'est pas chargé et identifié
    // Et ne rien faire sur web (expo-notifications non supporté)
    if (Platform.OS === "web") {
      console.warn("🔔 [useNotifications] Platform.OS === 'web' - notifications push désactivées sur web");
      return;
    }
    if (loading || !driver) {
      console.log(`🔔 [useNotifications] Attente: loading=${loading}, driver=${!!driver}`);
      return;
    }

    // ✅ IMPORTANT: l'enregistrement du push token driver est centralisé dans `app/_layout.tsx`
    // pour éviter les doubles appels `/driver/save-push-token` (races + bruit réseau).

    // Étape 2 : Mettre en place les écouteurs d'événements
    const notificationListener = Notifications.addNotificationReceivedListener(
      async (notification) => {
        const data = notification.request.content.data || {};
        const notificationType = data.type || "";

        // P0.2: Filtrer affichage si notification pas pour le chauffeur
        const filterResult = _shouldIgnoreNotificationForDriver(data);
        if (filterResult.ignore) {
          console.log("📩 Notification ignorée (driver filter):", {
            ignored_reason: filterResult.reason,
            trace_id: data.trace_id,
            recipient_role: data.recipient_role,
            routing_decision: data.routing_decision,
            actor_role: data.actor_role,
            actor_id: data.actor_id,
            data,
          });
          return;
        }

        // P0.4: Log payload proof (trace_id, recipient_role, etc.)
        console.log("📩 Notification reçue pendant que l'app est ouverte:", {
          trace_id: data.trace_id,
          type: notificationType,
          event_type: data.event_type ?? data.event,
          booking_id: data.booking_id ?? data.bookingId,
          status: data.status,
          recipient_role: data.recipient_role,
          routing_decision: data.routing_decision,
          actor_role: data.actor_role,
          actor_id: data.actor_id,
          data: notification.request.content.data,
        });

        // ✅ Phase 2.6: Router les notifications silencieuses vers handleSilentNotification
        if (notificationType === "silent_update" || data["content-available"] === 1) {
          try {
            const { handleSilentNotification } = await import(
              "@/services/silentNotifications"
            );
            await handleSilentNotification(data);
            console.log("✅ Notification silencieuse traitée:", data.sync_type);
          } catch (error) {
            console.error("❌ Erreur traitement notification silencieuse:", error);
          }
          // Ne pas afficher la notification si c'est silencieuse
          return;
        }

        // Pour les autres notifications, afficher normalement
        console.log("📩 Notification normale:", notificationType);
      }
    );

    const responseListener =
      Notifications.addNotificationResponseReceivedListener((response) => {
        console.log(
          "📲 L'utilisateur a interagi avec une notification:",
          response
        );
        
        // ✅ Extract deep link from notification data
        const notificationData = response.notification.request.content.data;
        const deepLink = notificationData?.deepLink;
        
        if (deepLink) {
          console.log("🔗 Deep link détecté:", deepLink);
          
          // ✅ Valider le deep link (sécurité: anti-injection, anti-open-redirect)
          if (typeof deepLink === "string") {
            const validation = validateDeepLink(deepLink);
            
            if (validation.valid) {
              // Use expo-linking to handle the deep link
              // This will trigger the router to navigate
              Linking.openURL(deepLink).catch((error) => {
                console.warn("❌ Failed to open deep link:", error);
                logError("Deep link navigation failed", error);
              });
            } else {
              console.warn("⚠️ Deep link invalide:", validation.error, deepLink);
            }
          } else {
            console.warn("⚠️ Deep link n'est pas une chaîne:", typeof deepLink);
          }
        } else {
          console.log("ℹ️ Aucun deep link dans la notification");
        }
      });

    // Étape 3 : Nettoyer les écouteurs quand le composant est démonté
    return () => {
      notificationListener.remove();
      responseListener.remove();
    };
  }, [driver, loading]); // Ce `useEffect` se relancera si le chauffeur ou l'état de chargement change
};
