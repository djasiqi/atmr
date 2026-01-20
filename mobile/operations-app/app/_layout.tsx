import * as Sentry from "@sentry/react-native";
import React, { useEffect, useRef } from "react";
import { Slot, useSegments, useRouter } from "expo-router";
import * as Linking from "expo-linking";
import { AuthProvider, useAuth } from "@/hooks/useAuth";
import {
  configureNotifications,
  initNotifications,
} from "@/services/notification";
import { setupNotificationChannels } from "@/services/notificationChannels";
import { setupNotificationActions } from "@/services/notificationActions";
import { useNotificationActions } from "@/hooks/useNotificationActions";
import { Platform, View, Text, ActivityIndicator, StyleSheet } from "react-native";
import Constants from "expo-constants";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { registerPushToken, initializeCSRFToken } from "@/services/api"; // si l'alias '@' n'est pas configuré: ../services/api
import {
  startAdaptiveLocationTracking,
  stopAdaptiveLocationTracking,
} from "@/services/locationTracker";
import { validateDeepLink } from "@/services/deepLinkHandler";
// Version check - gestion des mises à jour obligatoires/recommandées
import { VersionProvider, useVersion } from "@/contexts/VersionContext";
import { UpdateRequiredScreen } from "@/components/version/UpdateRequiredScreen";
import { UpdateRecommendedModal } from "@/components/version/UpdateRecommendedModal";

// ✅ Enregistrer la tâche de localisation en arrière-plan (uniquement si le module natif est disponible)
// Note: expo-task-manager nécessite un rebuild natif. En développement avec Expo Go, on skip.
if (Platform.OS !== "web") {
  try {
    require("@/tasks/locationTask");
  } catch (error) {
    // Module natif non disponible (Expo Go ou build non mis à jour)
    // C'est normal en développement, le tracking arrière-plan nécessite un development build
    console.log("ℹ️ TaskManager non disponible (normal en Expo Go, nécessite un development build)");
  }
}

// ✅ Phase 2.6: Définir la tâche de synchronisation silencieuse en arrière-plan
if (Platform.OS !== "web") {
  try {
    const { defineBackgroundSyncTask } = require("@/services/silentNotifications");
    defineBackgroundSyncTask();
    console.log("✅ Tâche background sync définie");
  } catch (error) {
    console.log("ℹ️ defineBackgroundSyncTask non disponible:", error);
  }
}

Sentry.init({
  dsn: "https://500ea836dce2e802b27109d857cb3534@o4509736814772224.ingest.de.sentry.io/4509736867201104",
  sendDefaultPii: true,
  tracesSampleRate: 1.0,
  profilesSampleRate: 1.0,
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,
  integrations: [Sentry.mobileReplayIntegration()],
});

export default function RootLayout() {
  return (
    <VersionProvider>
      <AuthProvider>
        <RootNav />
      </AuthProvider>
    </VersionProvider>
  );
}

function RootNav() {
  const {
    mode,
    isAuthenticated,
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
    loading,
    driver,
  } = useAuth();
  const userId = driver?.id ?? null;

  // Version check - récupération du statut de mise à jour
  const { status: updateStatus, isLoading: versionLoading } = useVersion();

  const segments = useSegments();
  const router = useRouter();
  const registeringRef = useRef(false);

  // ✅ Phase 2 - Gestionnaire des actions de notifications
  useNotificationActions();

  // ✅ Deep link handler: parse atmr:// URLs and navigate to appropriate routes
  const handleDeepLink = React.useCallback((url: string) => {
    try {
      console.log("🔗 Handling deep link:", url);

      // ✅ Valider le deep link (sécurité: anti-injection, anti-open-redirect)
      const validation = validateDeepLink(url);
      if (!validation.valid) {
        console.warn("⚠️ Deep link invalide:", validation.error, url);
        return;
      }

      // Utiliser les valeurs validées
      const route = validation.type; // "booking", "bookings", "chat", "dispatch"
      const id = validation.id; // ID validé (entier positif) ou undefined

      // Only navigate if driver is authenticated
      if (!isDriverAuthenticated || loading) {
        console.log("⏳ Driver not authenticated yet, deferring deep link navigation");
        return;
      }

      // Map deep link paths to app routes (utiliser les valeurs validées)
      if (route === "booking" && id) {
        // Navigate to trip details
        console.log("📍 Navigating to trip details for booking:", id);
        router.push(`/(dashboard)/trip-details?id=${id}` as any);
      } else if (route === "bookings") {
        // Navigate to trips list
        console.log("📍 Navigating to trips list");
        router.push("/(tabs)/trips" as any);
      } else if (route === "chat" && id) {
        // Navigate to chat (le format chat/message/{id} est validé comme chat/{id})
        console.log("📍 Navigating to chat with message:", id);
        router.push(`/(tabs)/chat?messageId=${id}` as any);
      } else if (route === "dispatch" && id) {
        // Navigate to schedule/dispatch (le format dispatch/run/{id} est validé comme dispatch/{id})
        console.log("📍 Navigating to schedule with dispatch run:", id);
        router.push(`/(dashboard)/schedule?dispatchRunId=${id}` as any);
      } else {
        console.warn("⚠️ Deep link route non gérée:", route, id);
      }
    } catch (error) {
      console.error("❌ Error handling deep link:", error);
    }
  }, [isDriverAuthenticated, loading, router]);

  // Si une mise à jour est REQUIRED, afficher l'écran bloquant
  // (avant même l'authentification)
  if (updateStatus === "UPDATE_REQUIRED") {
    return <UpdateRequiredScreen />;
  }

  // 🔐 Redirections selon l’état d’auth
  useEffect(() => {
    if (loading) return;
    const firstSegment = (segments[0] as string | undefined) ?? "";
    const isDriverAuthGroup = firstSegment === "(auth)";
    const isEnterpriseAuthGroup = firstSegment === "(enterprise-auth)";
    const isEnterpriseGroup = firstSegment === "(enterprise)";

    if (mode === "enterprise") {
      if (!isEnterpriseAuthenticated) {
        if (
          !isEnterpriseAuthGroup &&
          firstSegment !== "" &&
          firstSegment !== "index"
        ) {
          router.replace("/(enterprise-auth)/login" as any);
        }
      } else {
        if (
          isEnterpriseAuthGroup ||
          firstSegment === "(auth)" ||
          firstSegment === "" ||
          firstSegment === "index"
        ) {
          router.replace("/(enterprise)/dashboard" as any);
        }
      }
      if (firstSegment === "(tabs)" || firstSegment === "(dashboard)") {
        router.replace("/(enterprise)/dashboard" as any);
      }
    } else {
      // En mode driver, attendre que le chargement soit terminé avant de naviguer
      // pour éviter de naviguer vers login pendant un switch de compte
      if (!isDriverAuthenticated && !loading) {
        // Ne pas naviguer vers login si on est dans un contexte d'entreprise
        // (cela peut arriver lors d'un switch de compte)
        if (
          !isDriverAuthGroup &&
          !isEnterpriseGroup &&
          !isEnterpriseAuthGroup &&
          firstSegment !== "" &&
          firstSegment !== "index"
        ) {
          router.replace("/(auth)/login" as any);
        }
      } else if (
        isDriverAuthenticated &&
        (isDriverAuthGroup ||
          firstSegment === "" ||
          firstSegment === "index" ||
          isEnterpriseAuthGroup ||
          isEnterpriseGroup)
      ) {
        router.replace("/(tabs)/mission" as any);
      }
    }
  }, [
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
    loading,
    mode,
    router,
    segments,
  ]);

  // 🔔 Config + enregistrement push (quand prêt)
  useEffect(() => {
    if (loading || !isDriverAuthenticated || !driver) return;
    const currentUserId = driver.id;

    // ✅ Initialiser le token CSRF au démarrage (pour les requêtes POST/PUT/DELETE/PATCH)
    initializeCSRFToken();

    // Expo Go n'embarque pas google-services.json → skip pour éviter l'erreur Firebase
    if (Constants.appOwnership === "expo") {
      console.warn(
        "Skip FCM in Expo Go. Use a Development Build to test push."
      );
      return;
    }

    // éviter les doubles exécutions
    if (registeringRef.current) return;
    registeringRef.current = true;

    let cancelled = false;

    (async () => {
      try {
        console.log("🔔 [_layout] Initialisation des notifications…", {
          driverId: currentUserId,
          platform: Platform.OS,
          appOwnership: Constants.appOwnership,
          executionEnvironment: Constants.executionEnvironment,
        });
        await configureNotifications();

        // ✅ Configurer les canaux Android (Phase 1 - Quick Wins)
        await setupNotificationChannels();

        // ✅ Configurer les actions directes (Phase 2 - Enrichissement)
        await setupNotificationActions();

        // ✅ Configurer la synchronisation silencieuse en arrière-plan (Phase 2.6)
        const { setupBackgroundSync } = await import("@/services/silentNotifications");
        await setupBackgroundSync();

        // Récupération token (device ou expo) avec peu de retries pour éviter le spam
        const tokens = await initNotifications({
          withExpoToken: true,
          maxRetries: 2,
        });

        if (cancelled) return;

        const device = (tokens as any)?.device ?? null;
        const expo = (tokens as any)?.expo ?? null;
        const tokenToUse = device || expo;

        if (!tokenToUse) {
          // En dev/local, c'est normal de ne pas avoir de token
          const isDevLocal = __DEV__ === true || Constants.executionEnvironment === "bare";
          console.error("❌ [_layout] Aucun token push disponible", {
            driverId: currentUserId,
            platform: Platform.OS,
            isDevLocal,
            hasDevice: !!device,
            hasExpo: !!expo,
          });
          if (isDevLocal) {
            console.log("ℹ️ [_layout] Pas de token push en dev/local - normal sans Firebase");
          } else {
            console.warn("⚠️ [_layout] Aucun token push disponible (APK sans Firebase ?)");
          }
          return;
        }

        // ✅ INSTRUMENTATION: Device ID
        let deviceId = "unknown";
        try {
          const Device = await import("expo-device");
          // expo-device exporte modelId directement
          deviceId = Device.modelId || Device.deviceName || "unknown";
        } catch (error) {
          console.warn("⚠️ Impossible de récupérer Device ID:", error);
        }

        // ✅ INSTRUMENTATION: Logs avec Device ID et User ID
        console.log("🔔 Device ID:", deviceId);
        console.log("🔔 Token enregistré:", tokenToUse.substring(0, 20) + "...");
        console.log("🔔 Enregistrement token pour driver:", currentUserId);

        // ✅ CORRECTIF: Toujours enregistrer le token lors de la connexion
        // même s'il n'a pas changé, pour réactiver les tokens inactifs
        // (les tokens peuvent être invalidés lors du logout et doivent être réactivés)
        const key = currentUserId
          ? `push_token_${currentUserId}`
          : "push_token_default";
        const last = await AsyncStorage.getItem(key);

        // ✅ FORCER l'enregistrement à chaque connexion pour réactiver les tokens inactifs
        // Ne pas vérifier si le token a changé, toujours enregistrer
        console.log(
          "🔔 [_layout] Enregistrement token pour réactivation (connexion):",
          currentUserId,
          last === tokenToUse ? "(token identique - réactivation)" : "(token changé)"
        );

        try {
          console.log("🔔 [_layout] Envoi token au backend...", {
            driverId: currentUserId,
            tokenPreview: tokenToUse.substring(0, 30) + "...",
          });
          const response = await registerPushToken({
            token: tokenToUse,
            driverId: currentUserId,
          });
          await AsyncStorage.setItem(key, tokenToUse);
          console.log("✅ [_layout] Push token enregistré/réactivé côté backend", {
            response,
          });
        } catch (e: any) {
          console.error("❌ [_layout] Enregistrement token échoué:", {
            driverId: currentUserId,
            status: e?.response?.status,
            statusText: e?.response?.statusText,
            data: e?.response?.data,
            message: e?.message,
          });
          // Ne pas throw pour ne pas bloquer l'app, mais logger l'erreur
        }
      } catch (e: any) {
        console.error(
          "❌ [_layout] Enregistrement des notifications échoué:",
          {
            driverId: currentUserId,
            error: e?.message || String(e),
            status: e?.response?.status,
            data: e?.response?.data,
            platform: Platform.OS,
          }
        );
      } finally {
        registeringRef.current = false;
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [driver, isDriverAuthenticated, loading]);

  // ✅ Initialiser le token CSRF pour les entreprises
  useEffect(() => {
    if (loading || !isEnterpriseAuthenticated) return;

    // ✅ Initialiser le token CSRF au démarrage (pour les requêtes POST/PUT/DELETE/PATCH)
    initializeCSRFToken();
  }, [isEnterpriseAuthenticated, loading]);

  // ✅ Deep link handling: listen for deep links when app is running
  useEffect(() => {
    if (Platform.OS === "web") return;

    // Handle initial deep link when app opens from notification
    Linking.getInitialURL().then((url) => {
      if (url) {
        console.log("🔗 Initial deep link detected:", url);
        // Wait for auth to complete before handling
        setTimeout(() => {
          if (isDriverAuthenticated && !loading) {
            handleDeepLink(url);
          }
        }, 1000);
      }
    });

    // Listen for deep links when app is already running
    const subscription = Linking.addEventListener("url", (event) => {
      console.log("🔗 Deep link received while app running:", event.url);
      if (isDriverAuthenticated && !loading) {
        handleDeepLink(event.url);
      }
    });

    return () => {
      subscription.remove();
    };
  }, [isDriverAuthenticated, loading, handleDeepLink]);

  // ✅ 4. Fréquence GPS Adaptative Mobile : Démarrer tracking adaptatif pour les drivers
  useEffect(() => {
    if (loading || !isDriverAuthenticated || !driver) {
      // Arrêter le tracking si le driver se déconnecte
      if (!isDriverAuthenticated) {
        stopAdaptiveLocationTracking();
      }
      return;
    }

    // Démarrer le tracking adaptatif pour le driver authentifié
    let cancelled = false;

    (async () => {
      try {
        console.log("📍 Démarrage du tracking GPS adaptatif...");
        await startAdaptiveLocationTracking();
        if (!cancelled) {
          console.log("✅ Tracking GPS adaptatif démarré");
        }
      } catch (e: any) {
        console.warn(
          "❌ Erreur démarrage tracking GPS adaptatif:",
          e?.message || String(e)
        );
      }
    })();

    return () => {
      cancelled = true;
      // Arrêter le tracking lors du démontage ou déconnexion
      stopAdaptiveLocationTracking();
    };
  }, [driver, isDriverAuthenticated, loading]);

  // ✅ UX : Afficher un écran de chargement pendant l'auto-login
  console.log("🔴 [RootNav] État de chargement:", {
    loading,
    versionLoading,
    isAuthenticated,
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
    mode,
    timestamp: new Date().toISOString()
  });

  if (loading || versionLoading) {
    console.log("⏳ [RootNav] Affichage écran de chargement...");
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Reconnexion en cours…</Text>
      </View>
    );
  }

  console.log("✅ [RootNav] Rendu <Slot /> (tabs vont être affichés)");

  return (
    <>
      <Slot />
      {/* Modal de mise à jour recommandée (non bloquante) */}
      {updateStatus === "UPDATE_RECOMMENDED" && <UpdateRecommendedModal />}
    </>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: "#666666",
  },
});
