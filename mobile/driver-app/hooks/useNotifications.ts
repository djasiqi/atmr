// C:\Users\jasiq\atmr\mobile\driver-app\hooks\useNotifications.ts
import { useEffect } from "react";
import * as Notifications from "expo-notifications";
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Device from "expo-device";
import { Platform } from "react-native";
import { useAuth } from "@/hooks/useAuth";
import api from "@/services/api";
import { getErrorMessage, logError } from "@/utils/errorHandler";

// 🔔 Configuration du comportement des notifications en mode foreground
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
    // Add these two lines
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

export const useNotifications = () => {
  const { driver, loading } = useAuth();

  useEffect(() => {
    // Ne rien faire tant que l'utilisateur n'est pas chargé et identifié
    if (loading || !driver) {
      return;
    }

    const setupAndRegister = async () => {
      try {
        // Étape 1 : Obtenir le token de l'appareil
        const token = await registerForPushNotificationsAsync(); // (Cette fonction doit exister ailleurs dans votre code)

        if (!token || !driver.id || typeof driver.id !== 'number') {
          console.warn("⛔ Token ou ID de chauffeur invalide, enregistrement annulé.");
          return;
        }

        // ✅ **OPTIMISATION : Ne contacter le serveur que si le token est nouveau**
        const storageKey = `push_token_driver_${driver.id}`;
        const lastSentToken = await AsyncStorage.getItem(storageKey);

        if (lastSentToken === token) {
          console.log("✅ Token de notification inchangé, pas de nouvel enregistrement.");
        } else {
          console.log("🔔 Nouveau token détecté, enregistrement sur le serveur pour le chauffeur:", driver.id);
          
          await api.post("/driver/save-push-token", {
            driverId: driver.id,
            token,
          });

          await AsyncStorage.setItem(storageKey, token);
          console.log("✅ Token enregistré sur le serveur et sauvegardé localement.");
        }
      } catch (error: unknown) {
        logError("Erreur durant la configuration des notifications", error);
      }
    };

    setupAndRegister();

    // Étape 2 : Mettre en place les écouteurs d'événements
    const notificationListener = Notifications.addNotificationReceivedListener(notification => {
      console.log("📩 Notification reçue pendant que l'app est ouverte:", notification);
    });

    const responseListener = Notifications.addNotificationResponseReceivedListener(response => {
      console.log("📲 L'utilisateur a interagi avec une notification:", response);
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

    // ✅ **IMPROVEMENT 3: Removed hardcoded projectId**
    const token = await Notifications.getExpoPushTokenAsync();
    
    console.log("📱 Expo push token:", token.data.substring(0, 50) + "...");

    return token.data;
  } catch (error: unknown) {
    logError("Error registering for notifications", error);
    
    const errorMessage = getErrorMessage(error);
    if (errorMessage.includes('FIS_AUTH_ERROR')) {
      console.warn("⚠️ Firebase Error - Expo token should still work");
    }
    
    return null;
  }
}