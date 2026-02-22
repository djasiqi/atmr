// hooks/useEnterpriseNotifications.ts
import { useEffect, useRef } from "react";
import * as Notifications from "expo-notifications";
import type { NotificationBehavior } from "expo-notifications";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Device from "expo-device";
import { Platform, AppState } from "react-native";
import Constants from "expo-constants";
import { useRouter } from "expo-router";
import { useAuth } from "@/hooks/useAuth";
import { useEnterpriseSocket } from "@/hooks/useEnterpriseSocket";
import api from "@/services/api";
import { secureStorage } from "@/services/storage";
import { getErrorMessage, logError } from "@/utils/errorHandler";
import { getLogger } from "@/utils/logger";

const log = getLogger("EntNotif");

// 🔔 Configuration du comportement des notifications en mode foreground
const foregroundBehavior: NotificationBehavior = {
  shouldShowAlert: true,
  shouldPlaySound: true,
  shouldSetBadge: true,
  shouldShowBanner: true,
  shouldShowList: true,
};

Notifications.setNotificationHandler({
  handleNotification: async () => foregroundBehavior,
});

export const useEnterpriseNotifications = () => {
  const { enterpriseSession, loading } = useAuth();
  const socket = useEnterpriseSocket();
  const router = useRouter();
  const appState = useRef(AppState.currentState);
  const notificationHandlersRef = useRef<{
    newBooking?: (data: any) => void;
    bookingUpdated?: (data: any) => void;
    bookingCancelled?: (data: any) => void;
    chatMessage?: (data: any) => void;
  }>({});

  // Enregistrer le token push pour l'entreprise
  useEffect(() => {
    if (Platform.OS === "web" || loading || !enterpriseSession?.company?.id) {
      return;
    }

    const isDevEnv = __DEV__ === true;
    const isBare = Constants.executionEnvironment === "bare";
    const forceDevPush =
      String(process.env.EXPO_PUBLIC_ENABLE_PUSH_DEV || "").trim() === "1";
    if ((isDevEnv || isBare) && !forceDevPush) {
      log.info("notifications disabled in dev/local, skip registration", {});
      return;
    }

    const setupAndRegister = async () => {
      try {
        const token = await registerForPushNotificationsAsync();
        const companyId = Number(enterpriseSession.company.id);

        if (!token || !Number.isInteger(companyId)) {
          const isDevLocal = __DEV__ === true || Constants.executionEnvironment === "bare";
          if (isDevLocal && !token) {
            log.info("no push token in dev/local, normal without firebase", {});
          } else {
            log.warn("invalid token or company id, registration cancelled", {});
          }
          return;
        }

        // Sauvegarder company_id dans AsyncStorage
        try {
          await AsyncStorage.setItem("company_id", String(companyId));
          log.info("company_id saved to AsyncStorage", { companyId });
        } catch (e) {
          log.warn("failed to save company_id", { error: e });
        }

        // Ne contacter le serveur que si le token est nouveau
        const storageKey = `push_token_enterprise_${companyId}`;
        const lastSentToken = await AsyncStorage.getItem(storageKey);

        if (lastSentToken === token) {
          log.info("push token unchanged, no new registration", {});
        } else {
          log.info("new token detected, registering on server", { companyId });

          try {
            // ✅ Envoyer avec Authorization Enterprise (sinon l'intercepteur driver ne met pas le bon token)
            const enterpriseJwt = await secureStorage.getEnterpriseToken();
            await api.post(
              "/companies/save-push-token",
              {
                companyId: Number(companyId),
                token,
              },
              {
                headers: enterpriseJwt ? { Authorization: `Bearer ${enterpriseJwt}` } : {},
              }
            );
            log.success("push token registered on server", {});
          } catch (e: any) {
            log.error("push token send failed", {
              companyId,
              status: e?.response?.status,
              data: e?.response?.data,
              message: e?.message,
            });
            throw e;
          }

          await AsyncStorage.setItem(storageKey, token);
          log.success("token saved on server and locally", {});
        }
      } catch (error: unknown) {
        const errorMessage = getErrorMessage(error);
        
        if (errorMessage.includes("FIS_AUTH_ERROR")) {
          log.warn("firebase error during notification setup, normal in dev", {});
        } else {
          logError("Erreur durant la configuration des notifications", error);
        }
      }
    };

    setupAndRegister();
  }, [enterpriseSession, loading]);

  // Écouter les événements Socket.IO et envoyer des notifications
  useEffect(() => {
    if (!socket || !enterpriseSession?.company?.id) {
      return;
    }

    // Fonction pour envoyer une notification
    const sendNotification = async (
      title: string,
      body: string,
      data?: any,
      sound: boolean = true
    ) => {
      try {
        // Vérifier si l'app est en arrière-plan ou inactive
        const currentAppState = AppState.currentState;
        const isAppInactive = currentAppState !== "active";

        // Envoyer la notification même si l'app est inactive
        await Notifications.scheduleNotificationAsync({
          content: {
            title,
            body,
            data: data || {},
            sound: sound,
            priority: Notifications.AndroidNotificationPriority.HIGH,
            categoryIdentifier: "enterprise",
          },
          trigger: null, // Envoyer immédiatement
        });

        log.info("notification sent", { title, body, appInactive: isAppInactive });
      } catch (error) {
        log.error("send notification failed", { error });
      }
    };

    // Handler pour nouvelle course (jamais d'ID visible)
    const handleNewBooking = (data: any) => {
      log.info("new booking received", { data });
      const clientName = data.client_name || data.client_display_name || "Client";
      const timeShort = data.time_formatted || (data.scheduled_time ? String(data.scheduled_time).slice(11, 16) : "");
      const body = timeShort ? `${clientName} • ${timeShort}` : clientName;
      sendNotification(
        "Nouvelle course • Assignée",
        body,
        { type: "new_booking", booking_id: data.booking_id ?? data.id, ...data },
        true
      );
    };

    // Handler pour course mise à jour (Driver→Company: chauffeur change statut)
    // Format: "Course • {STATUT}" + "Chauffeur → Client" + trajet (jamais d'ID visible)
    const handleBookingUpdated = (data: any) => {
      log.info("booking updated", { data });
      const statusLabels: Record<string, string> = {
        assigned: "Assignée",
        en_route: "En route",
        in_progress: "À bord",
        completed: "Terminée",
        return_completed: "Retour terminé",
        cancelled: "Annulée",
        canceled: "Annulée",
      };
      const status = String(data.status || "").toLowerCase();
      const statusLabel = statusLabels[status] || "Mise à jour";

      const driverName =
        data.driver?.full_name ||
        [data.driver?.first_name, data.driver?.last_name].filter(Boolean).join(" ") ||
        "Chauffeur";
      const clientName = data.client_name || data.client_display_name || "Client";
      const shorten = (v: string, max = 32) => (v.length > max ? v.slice(0, max - 1) + "…" : v);
      const pickupShort =
        data.pickup_short || (data.pickup_location ? shorten(String(data.pickup_location).split(",")[0].trim(), 32) : null);
      const dropoffShort =
        data.dropoff_short || (data.dropoff_location ? shorten(String(data.dropoff_location).split(",")[0].trim(), 32) : null);

      const formatRouteLine = (p: string, d: string, maxTotal = 65) => {
        const combined = `${p} → ${d}`;
        if (combined.length <= maxTotal) return combined;
        const maxEach = Math.floor((maxTotal - 3) / 2);
        return `${p.length > maxEach ? p.slice(0, maxEach - 1) + "…" : p} → ${d.length > maxEach ? d.slice(0, maxEach - 1) + "…" : d}`;
      };

      const title = `Course • ${statusLabel}`;
      const bodyLines = [`${driverName} → ${clientName}`];
      if (pickupShort && dropoffShort) {
        bodyLines.push(formatRouteLine(pickupShort, dropoffShort));
      } else if (pickupShort) bodyLines.push(pickupShort);
      else if (dropoffShort) bodyLines.push(dropoffShort);
      const body = bodyLines.join("\n");

      sendNotification(
        title,
        body,
        { type: "booking_updated", booking_id: data.booking_id ?? data.id, status: data.status, ...data },
        true
      );
    };

    // Handler pour course annulée (jamais d'ID visible)
    const handleBookingCancelled = (data: any) => {
      log.info("booking cancelled", { data });
      const clientName = data.client_name || data.client_display_name || "Client";
      const timeShort = data.time_formatted || (data.scheduled_time ? String(data.scheduled_time).slice(11, 16) : "");
      const body = timeShort ? `${clientName} • ${timeShort}` : clientName;
      sendNotification(
        "Course • Annulée",
        body,
        { type: "booking_cancelled", booking_id: data.booking_id ?? data.id, ...data },
        true
      );
    };

    // Handler pour message de chat (Équipe • Sender, preview 90 chars, collapse/dedupe)
    const handleChatMessage = (message: any) => {
      if (message.sender_id === enterpriseSession?.user?.id) {
        return;
      }

      log.info("new chat message", { message });
      const role = String(message.sender_role || "").toLowerCase();
      const roleLabel = role === "company" ? "Entreprise" : role === "driver" ? "Chauffeur" : "Équipe";
      const senderName = message.sender_name || message.sender_first_name || roleLabel || "Message";

      const normalizePreview = (content: string | null | undefined, maxLen = 90): string => {
        if (!content || typeof content !== "string") return "Nouveau message";
        let s = content.trim().replace(/\r?\n/g, " ").replace(/\s+/g, " ");
        if (!s) return "Nouveau message";
        return s.length > maxLen ? s.slice(0, maxLen - 1) + "…" : s;
      };
      const messagePreview = normalizePreview(message.content);

      const title = `Équipe • ${senderName}`;
      const threadId = String(message.company_id ?? "team");

      sendNotification(
        title,
        messagePreview,
        {
          type: "chat_message",
          message_id: message.id,
          messageId: message.id,
          thread_id: threadId,
          company_id: message.company_id,
          sender_name: senderName,
          deepLink: "atmr://enterprise/chat",
          ...message,
        },
        true
      );
    };

    // Stocker les handlers pour le cleanup
    notificationHandlersRef.current = {
      newBooking: handleNewBooking,
      bookingUpdated: handleBookingUpdated,
      bookingCancelled: handleBookingCancelled,
      chatMessage: handleChatMessage,
    };

    // Attacher les listeners Socket.IO
    socket.on("new_booking", handleNewBooking);
    socket.on("booking_updated", handleBookingUpdated);
    socket.on("booking_cancelled", handleBookingCancelled);
    socket.on("team_chat_message", handleChatMessage);

    // Cleanup
    return () => {
      const handlers = notificationHandlersRef.current;
      if (handlers.newBooking) socket.off("new_booking", handlers.newBooking);
      if (handlers.bookingUpdated) socket.off("booking_updated", handlers.bookingUpdated);
      if (handlers.bookingCancelled) socket.off("booking_cancelled", handlers.bookingCancelled);
      if (handlers.chatMessage) socket.off("team_chat_message", handlers.chatMessage);
    };
  }, [socket, enterpriseSession]);

  // Écouter les notifications reçues
  useEffect(() => {
    const notificationListener = Notifications.addNotificationReceivedListener(
      (notification) => {
        log.info("notification received", { notification });
      }
    );

    const responseListener = Notifications.addNotificationResponseReceivedListener(
      (response) => {
        log.info("user interacted with notification", { response });
        const data = response.notification.request.content.data as Record<string, unknown> | undefined;
        if (!data) return;

        // Extraire booking_id : backend envoie booking_id (snake_case) ou legacy bookingId
        const rawId = data.booking_id ?? data.bookingId;
        const deepLink = typeof data.deep_link === "string" ? data.deep_link : typeof data.deepLink === "string" ? data.deepLink : null;

        let rideId: string | null = null;
        if (rawId != null) {
          const n = typeof rawId === "number" ? rawId : parseInt(String(rawId), 10);
          if (!Number.isNaN(n) && n > 0) rideId = String(n);
        }
        if (!rideId && deepLink) {
          const match = /atmr:\/\/enterprise\/rides\/(\d+)/i.exec(deepLink);
          if (match?.[1]) rideId = match[1];
        }

        const notifType = typeof data.type === "string" ? data.type : "";

        if (rideId) {
          router.push({
            pathname: "/(enterprise)/ride-details",
            params: { rideId },
          } as any);
        } else if (notifType === "chat_message" || notifType === "message") {
          router.push("/(enterprise)/chat" as any);
        } else if (notifType && (notifType.includes("booking") || notifType.includes("booking_assigned"))) {
          log.warn("booking notification without booking_id, cannot open", { data });
        }
      }
    );

    return () => {
      notificationListener.remove();
      responseListener.remove();
    };
  }, [router]);
};

async function registerForPushNotificationsAsync(): Promise<string | null> {
  try {
    if (!Device.isDevice) {
      log.warn("emulator detected, notifications may be limited", {});
    }
    if (Platform.OS === "web") {
      return null;
    }

    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== "granted") {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== "granted") {
      log.warn("notification permissions denied", {});
      return null;
    }

    // Configurer le canal Android avec son
    if (Platform.OS === "android") {
      await Notifications.setNotificationChannelAsync("enterprise", {
        name: "Notifications Entreprise",
        importance: Notifications.AndroidImportance.MAX,
        vibrationPattern: [0, 250, 250, 250],
        lightColor: "#0A7F59",
        sound: "default", // Son par défaut
        enableVibrate: true,
        showBadge: true,
      });
    }

    const projectId =
      (Constants as any)?.expoConfig?.extra?.eas?.projectId ||
      (Constants as any)?.easConfig?.projectId;
    const token = await Notifications.getExpoPushTokenAsync(
      projectId ? { projectId } : undefined
    );

    log.info("expo push token obtained", { prefix: token.data.substring(0, 50) + "..." });

    return token.data;
  } catch (error: unknown) {
    const errorMessage = getErrorMessage(error);
    
    const isFirebaseError =
      errorMessage.includes("FIS_AUTH_ERROR") ||
      errorMessage.includes("Missing FIS auth token") ||
      errorMessage.includes("FIS_AUTH") ||
      (errorMessage.includes("Firebase") && errorMessage.includes("auth"));
    
    if (isFirebaseError) {
      const isDevLocal = __DEV__ || process.env.NODE_ENV === "development";
      if (isDevLocal) {
        log.info("firebase error in dev/local, normal if firebase unavailable", {});
      } else {
        log.warn("firebase error, expo token should still work", {});
      }
    } else {
      logError("Error registering for notifications", error);
    }

    return null;
  }
}

