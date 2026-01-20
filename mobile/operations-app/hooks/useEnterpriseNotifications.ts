// hooks/useEnterpriseNotifications.ts
import { useEffect, useRef } from "react";
import * as Notifications from "expo-notifications";
import type { NotificationBehavior } from "expo-notifications";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Device from "expo-device";
import { Platform, AppState } from "react-native";
import Constants from "expo-constants";
import { useAuth } from "@/hooks/useAuth";
import { useEnterpriseSocket } from "@/hooks/useEnterpriseSocket";
import api from "@/services/api";
import { secureStorage } from "@/services/storage";
import { getErrorMessage, logError } from "@/utils/errorHandler";

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
      console.log("🔔 Notifications désactivées en développement/local - skip registration");
      return;
    }

    const setupAndRegister = async () => {
      try {
        const token = await registerForPushNotificationsAsync();
        const companyId = Number(enterpriseSession.company.id);

        if (!token || !Number.isInteger(companyId)) {
          const isDevLocal = __DEV__ === true || Constants.executionEnvironment === "bare";
          if (isDevLocal && !token) {
            console.log("ℹ️ Pas de token push en dev/local - normal sans Firebase");
          } else {
            console.warn("⛔ Token ou ID d'entreprise invalide, enregistrement annulé.");
          }
          return;
        }

        // Sauvegarder company_id dans AsyncStorage
        try {
          await AsyncStorage.setItem("company_id", String(companyId));
          console.log(`💾 company_id sauvegardé dans AsyncStorage: ${companyId}`);
        } catch (e) {
          console.warn("⚠️ Impossible de sauvegarder company_id:", e);
        }

        // Ne contacter le serveur que si le token est nouveau
        const storageKey = `push_token_enterprise_${companyId}`;
        const lastSentToken = await AsyncStorage.getItem(storageKey);

        if (lastSentToken === token) {
          console.log("✅ Token de notification inchangé, pas de nouvel enregistrement.");
        } else {
          console.log("🔔 Nouveau token détecté, enregistrement sur le serveur pour l'entreprise:", companyId);

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
            console.log("✅ Token push enregistré avec succès sur le serveur");
          } catch (e: any) {
            console.error("❌ Envoi push token échoué:", {
              companyId,
              status: e?.response?.status,
              data: e?.response?.data,
              message: e?.message,
            });
            throw e;
          }

          await AsyncStorage.setItem(storageKey, token);
          console.log("✅ Token enregistré sur le serveur et sauvegardé localement.");
        }
      } catch (error: unknown) {
        const errorMessage = getErrorMessage(error);
        
        if (errorMessage.includes("FIS_AUTH_ERROR")) {
          console.warn("⚠️ Firebase Error lors de la configuration des notifications - normal en dev/local");
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

        console.log(`📩 Notification envoyée: ${title} - ${body} (app inactive: ${isAppInactive})`);
      } catch (error) {
        console.error("❌ Erreur lors de l'envoi de la notification:", error);
      }
    };

    // Handler pour nouvelle course
    const handleNewBooking = (data: any) => {
      console.log("📦 Nouvelle course reçue:", data);
      sendNotification(
        "Nouvelle course",
        `Une nouvelle course a été créée${data.booking_id ? ` (#${data.booking_id})` : ""}`,
        { type: "new_booking", bookingId: data.booking_id, ...data },
        true
      );
    };

    // Handler pour course mise à jour
    const handleBookingUpdated = (data: any) => {
      console.log("🔄 Course mise à jour:", data);
      const statusMessages: Record<string, string> = {
        assigned: "a été assignée",
        in_progress: "est en cours",
        completed: "est terminée",
        cancelled: "a été annulée",
      };
      const statusMessage = statusMessages[data.status] || "a été mise à jour";
      
      sendNotification(
        "Course mise à jour",
        `La course${data.booking_id ? ` #${data.booking_id}` : ""} ${statusMessage}`,
        { type: "booking_updated", bookingId: data.booking_id, status: data.status, ...data },
        true
      );
    };

    // Handler pour course annulée
    const handleBookingCancelled = (data: any) => {
      console.log("❌ Course annulée:", data);
      sendNotification(
        "Course annulée",
        `La course${data.booking_id ? ` #${data.booking_id}` : ""} a été annulée`,
        { type: "booking_cancelled", bookingId: data.booking_id, ...data },
        true
      );
    };

    // Handler pour message de chat
    const handleChatMessage = (message: any) => {
      // Ne pas envoyer de notification si c'est notre propre message
      if (message.sender_id === enterpriseSession?.user?.id) {
        return;
      }

      console.log("💬 Nouveau message de chat:", message);
      const senderName = message.sender_name || "Un membre de l'équipe";
      const messagePreview = message.content
        ? message.content.substring(0, 50) + (message.content.length > 50 ? "..." : "")
        : "Nouveau message";

      sendNotification(
        senderName,
        messagePreview,
        { type: "chat_message", messageId: message.id, ...message },
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
        console.log("📩 Notification reçue:", notification);
      }
    );

    const responseListener = Notifications.addNotificationResponseReceivedListener(
      (response) => {
        console.log("📲 L'utilisateur a interagi avec une notification:", response);
        // Ici on pourrait gérer la navigation vers la page appropriée
        const data = response.notification.request.content.data;
        if (data?.type === "new_booking" || data?.type === "booking_updated") {
          // Naviguer vers la page de détails de la course si nécessaire
          // router.push(`/ride-details/${data.bookingId}`);
        } else if (data?.type === "chat_message") {
          // Naviguer vers le chat si nécessaire
          // router.push("/chat");
        }
      }
    );

    return () => {
      notificationListener.remove();
      responseListener.remove();
    };
  }, []);
};

async function registerForPushNotificationsAsync(): Promise<string | null> {
  try {
    if (!Device.isDevice) {
      console.warn("⚠️ Emulator detected - notifications may be limited");
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
      console.warn("⚠️ Notification permissions denied");
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

    console.log("📱 Expo push token:", token.data.substring(0, 50) + "...");

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
        console.log("ℹ️ Erreur Firebase en dev/local - Firebase non accessible, c'est normal");
      } else {
        console.warn("⚠️ Firebase Error - Expo token should still work");
      }
    } else {
      logError("Error registering for notifications", error);
    }

    return null;
  }
}

