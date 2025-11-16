// C:\Users\jasiq\atmr\mobile\driver-app\hooks\useNotifications.ts
import { useEffect } from "react";
import * as Notifications from "expo-notifications";
import type { NotificationBehavior } from "expo-notifications";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Device from "expo-device";
import { Platform } from "react-native";
import Constants from "expo-constants";
import { useAuth } from "@/hooks/useAuth";
import api from "@/services/api";
import { getErrorMessage, logError } from "@/utils/errorHandler";

// 🔔 Configuration du comportement des notifications en mode foreground
const foregroundBehavior: NotificationBehavior = {
  shouldShowAlert: true,
  shouldPlaySound: true,
  shouldSetBadge: true,
  // requis par les types Expo récents
  shouldShowBanner: true,
  shouldShowList: true,
};
Notifications.setNotificationHandler({
  handleNotification: async () => foregroundBehavior,
});

export const useNotifications = () => {
  const { driver, loading } = useAuth();

  useEffect(() => {
    // Ne rien faire tant que l'utilisateur n'est pas chargé et identifié
    // Et ne rien faire sur web (expo-notifications non supporté)
    if (Platform.OS === "web" || loading || !driver) {
      return;
    }

    // Désactiver proprement les notifications en dev/local (évite FIS_AUTH_ERROR)
    // Dev Client/Bare en local sans config Firebase ne peut pas obtenir de token fiable
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
        // Étape 1 : Obtenir le token de l'appareil
        const token = await registerForPushNotificationsAsync(); // (Cette fonction doit exister ailleurs dans votre code)

        // Cast fort en entier pour correspondre au backend (évite "ID du chauffeur invalide ou manquant.")
        const driverId = Number((driver as any)?.id);
        if (!token || !Number.isInteger(driverId)) {
          console.warn(
            "⛔ Token ou ID de chauffeur invalide, enregistrement annulé."
          );
          return;
        }
        // ✅ FIX: Sauvegarder driver_id dans AsyncStorage pour Socket.IO
        try {
          await AsyncStorage.setItem("driver_id", String(driverId));
          console.log(`💾 driver_id sauvegardé dans AsyncStorage: ${driverId}`);
        } catch (e) {
          console.warn("⚠️ Impossible de sauvegarder driver_id:", e);
        }

        // ✅ **OPTIMISATION : Ne contacter le serveur que si le token est nouveau**
        const storageKey = `push_token_driver_${driverId}`;
        const lastSentToken = await AsyncStorage.getItem(storageKey);

        if (lastSentToken === token) {
          console.log(
            "✅ Token de notification inchangé, pas de nouvel enregistrement."
          );
        } else {
          console.log(
            "🔔 Nouveau token détecté, enregistrement sur le serveur pour le chauffeur:",
            driverId
          );

          try {
            // ✅ FIX: S'assurer que driverId est bien un nombre
            await api.post("/driver/save-push-token", {
              driverId: Number(driverId),
              token,
            });
            console.log("✅ Token push enregistré avec succès sur le serveur");
          } catch (e: any) {
            // Log détaillé côté client pour diagnostiquer un 400 éventuel
            console.error("❌ Envoi push token échoué:", {
              driverId,
              status: e?.response?.status,
              data: e?.response?.data,
              message: e?.message,
            });
            throw e;
          }

          await AsyncStorage.setItem(storageKey, token);
          console.log(
            "✅ Token enregistré sur le serveur et sauvegardé localement."
          );
        }
      } catch (error: unknown) {
        logError("Erreur durant la configuration des notifications", error);
      }
    };

    setupAndRegister();

    // Étape 2 : Mettre en place les écouteurs d'événements
    const notificationListener = Notifications.addNotificationReceivedListener(
      (notification) => {
        console.log(
          "📩 Notification reçue pendant que l'app est ouverte:",
          notification
        );
      }
    );

    const responseListener =
      Notifications.addNotificationResponseReceivedListener((response) => {
        console.log(
          "📲 L'utilisateur a interagi avec une notification:",
          response
        );
        // TODO: Ajouter la logique de navigation ici (ex: rediriger vers un écran)
      });

    // Étape 3 : Nettoyer les écouteurs quand le composant est démonté
    return () => {
      notificationListener.remove();
      responseListener.remove();
    };
  }, [driver, loading]); // Ce `useEffect` se relancera si le chauffeur ou l'état de chargement change
};

async function registerForPushNotificationsAsync(): Promise<string | null> {
  try {
    if (!Device.isDevice) {
      console.warn("⚠️ Emulator detected - notifications may be limited");
    }
    if (Platform.OS === "web") {
      // Pas de notifications push sur web via expo-notifications
      return null;
    }
    const { status: existingStatus } =
      await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== "granted") {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== "granted") {
      console.warn("⚠️ Notification permissions denied");
      return null;
    }

    // ✅ **IMPROVEMENT 2: Set up Android channel before getting token**
    if (Platform.OS === "android") {
      await Notifications.setNotificationChannelAsync("default", {
        name: "Notifications Driver",
        importance: Notifications.AndroidImportance.MAX,
        vibrationPattern: [0, 250, 250, 250],
        lightColor: "#FF231F7C",
        sound: "default",
      });
    }

    // ✅ Fournir projectId si disponible (SDK récents + Dev Client)
    const projectId =
      (Constants as any)?.expoConfig?.extra?.eas?.projectId ||
      (Constants as any)?.easConfig?.projectId;
    const token = await Notifications.getExpoPushTokenAsync(
      projectId ? { projectId } : undefined
    );

    console.log("📱 Expo push token:", token.data.substring(0, 50) + "...");

    return token.data;
  } catch (error: unknown) {
    logError("Error registering for notifications", error);

    const errorMessage = getErrorMessage(error);
    if (errorMessage.includes("FIS_AUTH_ERROR")) {
      console.warn("⚠️ Firebase Error - Expo token should still work");
    }

    return null;
  }
}
