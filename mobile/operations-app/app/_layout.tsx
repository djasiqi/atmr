import * as Sentry from "@sentry/react-native";
import React, { useEffect, useRef, useState } from "react";
import { Slot, useSegments, useRouter } from "expo-router";
import * as Linking from "expo-linking";
import * as SplashScreen from "expo-splash-screen";
import { AuthProvider, useAuth } from "@/hooks/useAuth";
import * as Notifications from "expo-notifications";
import {
  configureNotifications,
  initNotifications,
  setNotificationAppMode,
} from "@/services/notification";
import {
  logKillModeReadiness,
  setupNotificationChannels,
} from "@/services/notificationChannels";
import { setupNotificationActions } from "@/services/notificationActions";
import { useNotificationActions } from "@/hooks/useNotificationActions";
import {
  getFCMToken,
  onFCMTokenRefresh,
  requestFCMPermission,
  registerForegroundHandler,
  registerNotificationOpenedHandler,
  registerNotifeeForegroundHandler,
  handleInitialNotification,
  setFCMNavigationHandler,
} from "@/services/firebaseMessaging";
import { Platform, View, Text, Image, Animated, ActivityIndicator, StyleSheet, LogBox } from "react-native";

// ✅ Supprimer les avertissements GPS bruyants en mode dev
// Ces messages sont des retries normaux quand le backend est lent (non bloquants).
LogBox.ignoreLogs([
  "Timeout waiting for ACK",
  "Erreur resync queue GPS",
  "Erreur resync pour driver",
  "Retry #",
  "[expo-notifications] Listening to push token changes",
  '"shadow*" style props are deprecated',
  "refresh token rejected",
  "refresh token expired",
  "refresh token failed",
]);
import Constants from "expo-constants";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { registerPushToken, initializeCSRFToken } from "@/services/api"; // si l'alias '@' n'est pas configuré: ../services/api
import {
  startAdaptiveLocationTracking,
  stopAdaptiveLocationTracking,
} from "@/services/locationTracker";
import { validateDeepLink } from "@/services/deepLinkHandler";
import { initNetworkStateCache } from "@/services/networkState";
import { initLogContext } from "@/services/logContext";
import { OfflineBanner } from "@/components/common/OfflineBanner";
import { PushFailureBanner } from "@/components/common/PushFailureBanner";
import { GpsDisabledBanner } from "@/components/common/GpsDisabledBanner";
import { InAppNotificationToast } from "@/components/common/InAppNotificationToast";
import { BatteryOptimizationGuide } from "@/components/common/BatteryOptimizationGuide";
import { checkBatteryOptimization } from "@/services/batteryOptimization";
import { getLogger } from "@/utils/logger";

const log = getLogger("App");

SplashScreen.preventAutoHideAsync().catch(() => {});

// P0.2.C — Init cache réseau pour logs (corrélation logout ↔ offline)
initNetworkStateCache();
// P2.2 — Init log context (device_id hash pour corrélation multi-tenant)
initLogContext();
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
    log.info("task manager unavailable (expo go, needs dev build)", { error });
  }
}

// ✅ Phase 2.6: Définir la tâche de synchronisation silencieuse en arrière-plan
if (Platform.OS !== "web") {
  try {
    const { defineBackgroundSyncTask } = require("@/services/silentNotifications");
    defineBackgroundSyncTask();
    log.success("background sync task defined");
  } catch (error) {
    log.info("background sync task not available", { error });
  }
}

// ✅ Mission Bar: enregistrer le handler Notifee headless (actions notif app tuée)
if (Platform.OS !== "web") {
  try {
    const { registerNotifeeBackgroundHandler } = require("@/services/missionBarBackground");
    registerNotifeeBackgroundHandler();
  } catch (error) {
    log.info("mission bar background handler not available", { error });
  }
}

Sentry.init({
  dsn: "https://500ea836dce2e802b27109d857cb3534@o4509736814772224.ingest.de.sentry.io/4509736867201104",
  sendDefaultPii: true,
  tracesSampleRate: 1.0,
  profilesSampleRate: 1.0,
  // Session Replay desactive : ReplayIntegration provoque des Background ANR
  // sur Android via un ReentrantLock dans AndroidConnectionStatusProvider.updateCache()
  // appelé de maniere synchrone sur le main thread pendant onStart.
  // Bug SDK Sentry @sentry/react-native ~7.2.x — reactiver apres upgrade.
  // replaysSessionSampleRate: 0.1,
  // replaysOnErrorSampleRate: 1.0,
  // integrations: [Sentry.mobileReplayIntegration()],
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
  const [pushFailed, setPushFailed] = useState(false);
  const [showBatteryGuide, setShowBatteryGuide] = useState(false);

  // Version check - récupération du statut de mise à jour
  const { status: updateStatus, isLoading: versionLoading } = useVersion();

  const splashHiddenRef = useRef(false);
  useEffect(() => {
    if (splashHiddenRef.current) return;
    if (loading || versionLoading) return;
    splashHiddenRef.current = true;
    SplashScreen.hideAsync().catch(() => {});
  }, [loading, versionLoading]);

  // P0.5: Mettre à jour le mode notif dès que l'auth est connue (handler boot-level)
  useEffect(() => {
    if (Platform.OS === "web") return;
    const mode =
      isEnterpriseAuthenticated
        ? "enterprise"
        : isDriverAuthenticated
          ? "driver"
          : null;
    setNotificationAppMode(mode);
    if (__DEV__ && mode) {
      log.info("notification app mode set", { mode });
    }
  }, [isDriverAuthenticated, isEnterpriseAuthenticated]);

  const segments = useSegments();
  const router = useRouter();
  const registeringRef = useRef(false);

  // ✅ Phase 2 - Gestionnaire des actions de notifications
  useNotificationActions();

  // ✅ Deep link handler: parse atmr:// URLs and navigate to appropriate routes
  const handleDeepLink = React.useCallback((url: string) => {
    try {
      log.info("handling deep link", { url });

      // ✅ Valider le deep link (sécurité: anti-injection, anti-open-redirect)
      const validation = validateDeepLink(url);
      if (!validation.valid) {
        log.warn("invalid deep link", { error: validation.error, url });
        return;
      }

      // Utiliser les valeurs validées
      const route = validation.type; // "booking", "bookings", "chat", "dispatch"
      const id = validation.id; // ID validé (entier positif) ou undefined

      // Only navigate if driver is authenticated
      if (!isDriverAuthenticated || loading) {
        log.info("driver not authenticated yet, deferring deep link");
        return;
      }

      // Map deep link paths to app routes (utiliser les valeurs validées)
      if (route === "booking" && id) {
        // Navigate to trip details
        log.info("navigating to trip details", { bookingId: id });
        router.push(`/(dashboard)/trip-details?id=${id}` as any);
      } else if (route === "bookings") {
        // Navigate to trips list
        log.info("navigating to trips list");
        router.push("/(tabs)/trips" as any);
      } else if (route === "chat") {
        // Navigate to chat: chat/message/{id} | chat/thread/{id} | chat
        const subType = validation.subType;
        if (id && subType === "message") {
          log.info("navigating to chat with message", { messageId: id });
          router.push(`/(tabs)/chat?messageId=${id}` as any);
        } else if (id && subType === "thread") {
          log.info("navigating to chat with thread", { threadId: id });
          router.push(`/(tabs)/chat?threadId=${id}` as any);
        } else if (id) {
          // Fallback rétrocompatible: chat/123 → messageId
          router.push(`/(tabs)/chat?messageId=${id}` as any);
        } else {
          log.info("navigating to chat");
          router.push("/(tabs)/chat" as any);
        }
      } else if (route === "dispatch" && id) {
        // Navigate to schedule/dispatch (le format dispatch/run/{id} est validé comme dispatch/{id})
        log.info("navigating to schedule with dispatch run", { dispatchRunId: id });
        router.push(`/(dashboard)/schedule?dispatchRunId=${id}` as any);
      } else {
        log.warn("unhandled deep link route", { route, id });
      }
    } catch (error) {
      log.error("error handling deep link", { error });
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
      log.warn("skip fcm in expo go (use dev build for push)");
      return;
    }

    // éviter les doubles exécutions
    if (registeringRef.current) return;
    registeringRef.current = true;

    let cancelled = false;

    (async () => {
      try {
        log.info("notifications init", {
          driverId: currentUserId,
          platform: Platform.OS,
          appOwnership: Constants.appOwnership,
          executionEnvironment: Constants.executionEnvironment,
        });
        await configureNotifications();

        // ✅ Configurer les canaux Android (Phase 1 - Quick Wins)
        await setupNotificationChannels();

        // P0.6: Log unique KILL-MODE readiness (permissions + channel + device)
        await logKillModeReadiness();

        // ✅ Configurer les actions directes (Phase 2 - Enrichissement)
        await setupNotificationActions();

        // ✅ Configurer la synchronisation silencieuse en arrière-plan (Phase 2.6)
        const { setupBackgroundSync } = await import("@/services/silentNotifications");
        await setupBackgroundSync();

        // FCM: request permission (mainly iOS) + get native FCM token
        await requestFCMPermission();
        const fcmToken = await getFCMToken();

        // Fallback: try legacy Expo token if FCM unavailable (Expo Go)
        let tokenToUse: string | null = fcmToken;
        let provider: "fcm" | "expo" = "fcm";

        if (!fcmToken) {
          log.info("FCM token unavailable, trying Expo fallback");
          const tokens = await initNotifications({ withExpoToken: true, maxRetries: 2 });
          if (cancelled) return;
          tokenToUse = (tokens as any)?.expo ?? (tokens as any)?.device ?? null;
          provider = "expo";
        }

        if (cancelled) return;

        if (!tokenToUse) {
          const isExpoGo =
            Constants.appOwnership === "expo" &&
            Constants.executionEnvironment === "storeClient";
          log.error("no push token available", {
            driverId: currentUserId,
            platform: Platform.OS,
            isExpoGo,
          });
          if (isExpoGo) {
            log.info("no push token in expo go (expected, remote push disabled)");
          }
          return;
        }

        log.info("push token acquired", {
          provider,
          tokenPreview: tokenToUse.substring(0, 20) + "...",
          driverId: currentUserId,
        });

        const key = currentUserId
          ? `push_token_${currentUserId}`
          : "push_token_default";

        try {
          const response = await registerPushToken({
            token: tokenToUse,
            driverId: currentUserId,
            provider,
          });
          await AsyncStorage.setItem(key, tokenToUse);
          log.success("push token registered on backend", { response, provider });
        } catch (e: any) {
          log.error("push token registration failed", {
            driverId: currentUserId,
            status: e?.response?.status,
            data: e?.response?.data,
            message: e?.message,
          });
        }
      } catch (e: any) {
        log.error("notification registration failed", {
          driverId: currentUserId,
          error: e?.message || String(e),
          status: e?.response?.status,
          data: e?.response?.data,
          platform: Platform.OS,
        });
        if (!cancelled) setPushFailed(true);
      } finally {
        registeringRef.current = false;
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [driver, isDriverAuthenticated, loading]);

  // FCM: register foreground handler, deep link handlers, token refresh
  useEffect(() => {
    if (Platform.OS === "web") return;
    if (loading || !isDriverAuthenticated || !driver) return;

    const currentUserId = driver?.id;

    const unsubForeground = registerForegroundHandler();
    const unsubOpened = registerNotificationOpenedHandler();
    const unsubNotifee = registerNotifeeForegroundHandler();

    setFCMNavigationHandler((deepLink: string) => {
      log.info("FCM deep link navigation", { deepLink });
      handleDeepLink(deepLink);
    });

    handleInitialNotification();

    const unsubTokenRefresh = onFCMTokenRefresh(async (newToken) => {
      if (!currentUserId) return;
      try {
        await registerPushToken({
          token: newToken,
          driverId: currentUserId,
          provider: "fcm",
        });
        log.success("FCM token refreshed and registered", { driverId: currentUserId });
      } catch (e: any) {
        log.error("FCM token refresh registration failed", { error: e?.message });
      }
    });

    return () => {
      unsubForeground();
      unsubOpened();
      unsubNotifee();
      unsubTokenRefresh();
    };
  }, [driver, isDriverAuthenticated, loading]);

  // A1: Vérifier l'optimisation batterie Samsung après login driver
  useEffect(() => {
    if (loading || !isDriverAuthenticated || !driver) return;
    if (Platform.OS !== "android") return;
    (async () => {
      try {
        const { needsExemption } = await checkBatteryOptimization();
        if (needsExemption) setShowBatteryGuide(true);
      } catch {
        // best-effort
      }
    })();
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
        log.info("initial deep link detected", { url });
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
      log.info("deep link received while app running", { url: event.url });
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
        log.info("starting adaptive gps tracking");
        await startAdaptiveLocationTracking();
        if (!cancelled) {
          log.success("adaptive gps tracking started");
        }
      } catch (e: any) {
        log.warn("adaptive gps tracking start error", {
          message: e?.message || String(e),
        });
      }
    })();

    return () => {
      cancelled = true;
      // Arrêter le tracking lors du démontage ou déconnexion
      stopAdaptiveLocationTracking();
    };
  }, [driver, isDriverAuthenticated, loading]);

  // ✅ UX : Afficher un écran de chargement pendant l'auto-login
  log.info("root nav loading state", {
    loading,
    versionLoading,
    isAuthenticated,
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
    mode,
    timestamp: new Date().toISOString(),
  });

  if (loading || versionLoading) {
    log.info("root nav showing loading screen");
    return <BrandedLoadingScreen />;
  }

  log.success("root nav ready (slot rendered)");

  return (
    <>
      <InAppNotificationToast />
      <OfflineBanner />
      {pushFailed && <PushFailureBanner />}
      <GpsDisabledBanner />
      <Slot />
      {updateStatus === "UPDATE_RECOMMENDED" && <UpdateRecommendedModal />}
      <BatteryOptimizationGuide
        visible={showBatteryGuide}
        onDismiss={() => setShowBatteryGuide(false)}
      />
    </>
  );
}

function BrandedLoadingScreen() {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(18)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        delay: 200,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 600,
        delay: 200,
        useNativeDriver: true,
      }),
    ]).start();
  }, [fadeAnim, slideAnim]);

  return (
    <View style={styles.loadingContainer}>
      <Image
        source={require("@/assets/images/icon-dark.png")}
        style={styles.loadingLogo}
        resizeMode="contain"
      />
      <Animated.View
        style={[
          styles.loadingContent,
          { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
        ]}
      >
        <ActivityIndicator size="small" color="rgba(255,255,255,0.9)" />
        <Text style={styles.loadingText}>Reconnexion en cours…</Text>
      </Animated.View>
      <Text style={styles.loadingBrand}>Liri Opérations</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0D7F72",
  },
  loadingLogo: {
    width: 90,
    height: 90,
    borderRadius: 20,
    marginBottom: 32,
  },
  loadingContent: {
    alignItems: "center",
    gap: 14,
  },
  loadingText: {
    fontSize: 15,
    fontWeight: "500",
    color: "rgba(255,255,255,0.85)",
    letterSpacing: 0.2,
  },
  loadingBrand: {
    position: "absolute",
    bottom: 48,
    fontSize: 13,
    fontWeight: "600",
    color: "rgba(255,255,255,0.4)",
    letterSpacing: 1.5,
    textTransform: "uppercase",
  },
});
