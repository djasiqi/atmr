// operations-app/hooks/useNotifications.ts — utilisé en mode driver (mission, trips, dashboard)
import { useEffect } from "react";
import * as Notifications from "expo-notifications";
import type { NotificationBehavior } from "expo-notifications";
import { Platform } from "react-native";
import * as Linking from "expo-linking";
import { useAuth } from "@/hooks/useAuth";
import { logError } from "@/utils/errorHandler";
import { validateDeepLink } from "@/services/deepLinkHandler";
import { getLogger } from "@/utils/logger";
import { emitInAppNotification } from "@/components/common/InAppNotificationToast";

const log = getLogger("Notifs");

// 🔔 Foreground: pas de notification système, uniquement in-app (toast/banner)
const foregroundBehavior: NotificationBehavior = {
  shouldShowAlert: false,
  shouldPlaySound: false,
  shouldSetBadge: false,
  shouldShowBanner: false,
  shouldShowList: false,
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
      log.info("operations-app driver notifications loaded", {});
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
      log.warn("push disabled on web", {});
      return;
    }
    if (loading || !driver) {
      log.info("waiting for auth", { loading, hasDriver: !!driver });
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
          log.info("notification ignored by driver filter", {
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
        log.info("notification received while app open", {
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
            log.success("silent notification handled", { sync_type: data.sync_type });
          } catch (error) {
            log.error("silent notification handling failed", { error });
          }
          // Ne pas afficher la notification si c'est silencieuse
          return;
        }

        // Afficher en toast in-app (pas de notification système)
        const title = notification.request.content.title;
        const body = notification.request.content.body;
        if (title || body) {
          emitInAppNotification({
            title: title || "",
            body: body || "",
            data: data as Record<string, unknown>,
          });
        }
        log.info("in-app toast shown", { notificationType });
      }
    );

    const responseListener =
      Notifications.addNotificationResponseReceivedListener((response) => {
        log.info("user interacted with notification", { response });
        
        // ✅ Extract deep link from notification data
        const notificationData = response.notification.request.content.data;
        const deepLink = notificationData?.deepLink;
        
        if (deepLink) {
          log.info("deep link detected", { deepLink });
          
          // ✅ Valider le deep link (sécurité: anti-injection, anti-open-redirect)
          if (typeof deepLink === "string") {
            const validation = validateDeepLink(deepLink);
            
            if (validation.valid) {
              // Use expo-linking to handle the deep link
              // This will trigger the router to navigate
              Linking.openURL(deepLink).catch((error) => {
                log.warn("failed to open deep link", { error });
                logError("Deep link navigation failed", error);
              });
            } else {
              log.warn("invalid deep link", { error: validation.error, deepLink });
            }
          } else {
            log.warn("deep link not a string", { type: typeof deepLink });
          }
        } else {
          log.info("no deep link in notification", {});
        }
      });

    // Étape 3 : Nettoyer les écouteurs quand le composant est démonté
    return () => {
      notificationListener.remove();
      responseListener.remove();
    };
  }, [driver, loading]); // Ce `useEffect` se relancera si le chauffeur ou l'état de chargement change
};
