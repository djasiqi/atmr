// C:\Users\jasiq\atmr\mobile\driver-app\hooks\useNotifications.ts
import { useEffect } from "react";
import * as Notifications from "expo-notifications";
import type { NotificationBehavior } from "expo-notifications";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Device from "expo-device";
import { Platform } from "react-native";
import Constants from "expo-constants";
import * as Linking from "expo-linking";
import { useAuth } from "@/hooks/useAuth";
import api from "@/services/api";
import { getErrorMessage, logError } from "@/utils/errorHandler";
import { validateDeepLink } from "@/services/deepLinkHandler";

// Détecter si on est en dev/local
const isDevLocal = __DEV__ === true || Constants.executionEnvironment === "bare";

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
    if (Platform.OS === "web") {
      console.warn("🔔 [useNotifications] Platform.OS === 'web' - notifications push désactivées sur web");
      return;
    }
    if (loading || !driver) {
      console.log(`🔔 [useNotifications] Attente: loading=${loading}, driver=${!!driver}`);
      return;
    }

    // ✅ LOGGING AMÉLIORÉ: Log des conditions d'environnement
    const isDevEnv = __DEV__ === true;
    const isBare = Constants.executionEnvironment === "bare";
    const forceDevPush =
      String(process.env.EXPO_PUBLIC_ENABLE_PUSH_DEV || "").trim() === "1";
    const appOwnership = Constants.appOwnership;
    
    console.log("🔔 [useNotifications] Conditions d'environnement:", {
      Platform: Platform.OS,
      isDevEnv,
      isBare,
      forceDevPush,
      appOwnership,
      executionEnvironment: Constants.executionEnvironment,
    });

    // ✅ CORRECTIF: Ne pas désactiver en production, même si détecté comme dev
    // Seulement désactiver si vraiment en dev ET forceDevPush n'est pas activé
    // En production (appOwnership === "standalone" ou "production"), toujours activer
    const isProduction = appOwnership === "standalone" || appOwnership === "production";
    if (!isProduction && (isDevEnv || isBare) && !forceDevPush) {
      console.warn("🔔 [useNotifications] Notifications désactivées en développement/local - skip registration");
      return;
    }

    const setupAndRegister = async () => {
      try {
        console.log("🔔 [useNotifications] Début enregistrement token push...");
        
        // Étape 1 : Obtenir le token de l'appareil
        const token = await registerForPushNotificationsAsync();

        // Cast fort en entier pour correspondre au backend (évite "ID du chauffeur invalide ou manquant.")
        const driverId = Number((driver as any)?.id);
        
        console.log("🔔 [useNotifications] Résultat obtention token:", {
          hasToken: !!token,
          tokenLength: token?.length || 0,
          driverId,
          isInteger: Number.isInteger(driverId),
          driverIdType: typeof driverId,
          isProduction,
        });
        
        // ✅ CORRECTIF: En production, forcer l'enregistrement même si token est null
        // (le backend peut réactiver un token existant)
        if (!Number.isInteger(driverId)) {
          console.error("⛔ [useNotifications] ID de chauffeur invalide, enregistrement annulé.", {
            driverId,
            isInteger: Number.isInteger(driverId),
          });
          return;
        }
        
        if (!token) {
          // En dev/local sans Firebase, c'est normal de ne pas avoir de token
          const isDevLocal = __DEV__ === true || Constants.executionEnvironment === "bare";
          if (isDevLocal && !isProduction) {
            console.log("ℹ️ [useNotifications] Pas de token push en dev/local - normal sans Firebase");
            return;
          } else if (isProduction) {
            // En production, essayer de réactiver le token existant même sans nouveau token
            console.warn("⚠️ [useNotifications] Pas de token push obtenu en production - tentative réactivation token existant");
            // On continue quand même pour voir si on peut réactiver un token existant
          } else {
            console.error("⛔ [useNotifications] Token invalide, enregistrement annulé.", {
              hasToken: !!token,
              driverId,
            });
            return;
          }
        }
        // ✅ FIX: Sauvegarder driver_id dans AsyncStorage pour Socket.IO
        try {
          await AsyncStorage.setItem("driver_id", String(driverId));
          console.log(`💾 driver_id sauvegardé dans AsyncStorage: ${driverId}`);
        } catch (e) {
          console.warn("⚠️ Impossible de sauvegarder driver_id:", e);
        }

        // ✅ CORRECTIF: Toujours enregistrer le token lors de la connexion
        // même s'il n'a pas changé, pour réactiver les tokens inactifs
        // (les tokens peuvent être invalidés lors du logout et doivent être réactivés)
        const storageKey = `push_token_driver_${driverId}`;
        const lastSentToken = await AsyncStorage.getItem(storageKey);
        
        // ✅ FORCER l'enregistrement à chaque connexion pour réactiver les tokens inactifs
        // Ne pas vérifier si le token a changé, toujours enregistrer
        if (token) {
          console.log(
            "🔔 [useNotifications] Enregistrement token pour réactivation (connexion):",
            driverId,
            lastSentToken === token ? "(token identique - réactivation)" : "(token changé)"
          );

          try {
            console.log("🔔 [useNotifications] Envoi token au backend...", {
              driverId: Number(driverId),
              tokenPreview: token.substring(0, 30) + "...",
            });
            
            // ✅ FIX: S'assurer que driverId est bien un nombre
            const response = await api.post("/driver/save-push-token", {
              driverId: Number(driverId),
              token,
            });
            
            console.log("✅ [useNotifications] Token push enregistré/réactivé avec succès sur le serveur", {
              response: response.data,
            });
            
            await AsyncStorage.setItem(storageKey, token);
            console.log(
              "✅ [useNotifications] Token enregistré sur le serveur et sauvegardé localement."
            );
          } catch (e: any) {
            // Log détaillé côté client pour diagnostiquer un 400 éventuel
            console.error("❌ [useNotifications] Envoi push token échoué:", {
              driverId,
              status: e?.response?.status,
              statusText: e?.response?.statusText,
              data: e?.response?.data,
              message: e?.message,
              code: e?.code,
              stack: e?.stack,
            });
            
            // ✅ ENVOI AU BACKEND: Logger l'erreur pour diagnostic
            try {
              await fetch("http://127.0.0.1:7242/ingest/push-token-error", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  driverId,
                  error: e?.message || "Unknown error",
                  status: e?.response?.status,
                  data: e?.response?.data,
                  platform: Platform.OS,
                  timestamp: new Date().toISOString(),
                }),
              }).catch(() => {}); // Ignorer les erreurs de connexion
            } catch {}
            
            throw e;
          }
        } else {
          console.warn("⚠️ [useNotifications] Pas de token push disponible - impossible d'enregistrer/réactiver");
        }
      } catch (error: unknown) {
        const errorMessage = getErrorMessage(error);
        
        console.error("❌ [useNotifications] Erreur durant la configuration des notifications:", {
          error: errorMessage,
          errorType: error instanceof Error ? error.constructor.name : typeof error,
          driverId: (driver as any)?.id,
          platform: Platform.OS,
          stack: error instanceof Error ? error.stack : undefined,
        });
        
        // Ne pas logger comme erreur si c'est FIS_AUTH_ERROR (normal en dev/local)
        if (errorMessage.includes("FIS_AUTH_ERROR")) {
          console.warn("⚠️ [useNotifications] Firebase Error lors de la configuration des notifications - normal en dev/local");
        } else {
          logError("Erreur durant la configuration des notifications", error);
        }
      }
    };

    setupAndRegister();

    // Étape 2 : Mettre en place les écouteurs d'événements
    const notificationListener = Notifications.addNotificationReceivedListener(
      async (notification) => {
        const data = notification.request.content.data || {};
        const notificationType = data.type || "";

        console.log(
          "📩 Notification reçue pendant que l'app est ouverte:",
          notificationType
        );

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

async function registerForPushNotificationsAsync(): Promise<string | null> {
  try {
    console.log("🔔 [registerForPushNotificationsAsync] Début obtention token...");
    
    if (!Device.isDevice) {
      console.warn("⚠️ [registerForPushNotificationsAsync] Emulator detected - notifications may be limited");
    }
    if (Platform.OS === "web") {
      // Pas de notifications push sur web via expo-notifications
      console.warn("⚠️ [registerForPushNotificationsAsync] Platform.OS === 'web' - retour null");
      return null;
    }
    
    console.log("🔔 [registerForPushNotificationsAsync] Vérification permissions...");
    const { status: existingStatus } =
      await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;
    
    console.log("🔔 [registerForPushNotificationsAsync] Permission existante:", existingStatus);

    if (existingStatus !== "granted") {
      console.log("🔔 [registerForPushNotificationsAsync] Demande de permission...");
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
      console.log("🔔 [registerForPushNotificationsAsync] Permission après demande:", status);
    }

    if (finalStatus !== "granted") {
      console.error("❌ [registerForPushNotificationsAsync] Notification permissions denied ou refusées:", {
        existingStatus,
        finalStatus,
        platform: Platform.OS,
      });
      return null;
    }
    
    console.log("✅ [registerForPushNotificationsAsync] Permissions accordées");

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
    console.log("🔔 [registerForPushNotificationsAsync] Obtention token Expo...", {
      hasProjectId: !!projectId,
      projectId: projectId ? projectId.substring(0, 20) + "..." : null,
    });
    
    const token = await Notifications.getExpoPushTokenAsync(
      projectId ? { projectId } : undefined
    );

    console.log("✅ [registerForPushNotificationsAsync] Expo push token obtenu:", {
      tokenPreview: token.data.substring(0, 50) + "...",
      tokenLength: token.data.length,
    });

    return token.data;
  } catch (error: unknown) {
    const errorMessage = getErrorMessage(error);
    
    console.error("❌ [registerForPushNotificationsAsync] Erreur lors de l'obtention du token:", {
      error: errorMessage,
      errorType: error instanceof Error ? error.constructor.name : typeof error,
      platform: Platform.OS,
      stack: error instanceof Error ? error.stack : undefined,
    });
    
    // Ne pas logger comme erreur si c'est une erreur Firebase (normal en dev/local)
    // Détecter plusieurs variantes d'erreurs Firebase
    const isFirebaseError = 
      errorMessage.includes("FIS_AUTH_ERROR") ||
      errorMessage.includes("Missing FIS auth token") ||
      errorMessage.includes("FIS_AUTH") ||
      errorMessage.includes("Firebase") && errorMessage.includes("auth");
    
    if (isFirebaseError) {
      // En dev/local, juste un log informatif, pas d'erreur
      const isDevLocal = __DEV__ || process.env.NODE_ENV === "development";
      if (isDevLocal) {
        console.log("ℹ️ Erreur Firebase en dev/local - Firebase non accessible, c'est normal");
      } else {
        console.warn("⚠️ Firebase Error - Expo token should still work");
      }
    } else {
      // Logger uniquement les vraies erreurs
      logError("Error registering for notifications", error);
    }

    return null;
  }
}
