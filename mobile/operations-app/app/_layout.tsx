import * as Sentry from "@sentry/react-native";
import React, { useEffect, useRef, useState } from "react";
import { Slot, useSegments, useRouter } from "expo-router";
import * as Linking from "expo-linking";
import * as SplashScreen from "expo-splash-screen";
import * as Application from "expo-application";
import * as Updates from "expo-updates";
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
import { AppState, AppStateStatus, Platform, View, Text, Image, Animated, ActivityIndicator, StyleSheet, LogBox, InteractionManager } from "react-native";
import { GestureHandlerRootView } from "react-native-gesture-handler";

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
  reconcileBackgroundTrackingState,
  ensureBackgroundTrackingStopped,
  buildBgTrackingInputs,
  requestBackgroundPermissionIfNeeded,
} from "@/services/locationTracker";
import { MissionStateManager } from "@/services/missionState";
import { validateDeepLink } from "@/services/deepLinkHandler";
import {
  initNetworkStateCache,
  getNetworkStateSnapshot,
  subscribeToNetworkState,
} from "@/services/networkState";
import { getSyncEngine } from "@/services/syncEngine";
import { initLogContext } from "@/services/logContext";
import { getAuthKpiSnapshot, logAuthEvent } from "@/services/authLogging";
import { OfflineBanner } from "@/components/common/OfflineBanner";
import { PushFailureBanner } from "@/components/common/PushFailureBanner";
import { GpsDisabledBanner } from "@/components/common/GpsDisabledBanner";
import { InAppNotificationToast } from "@/components/common/InAppNotificationToast";
import { BatteryOptimizationGuide, BATTERY_GUIDE_DISMISSED_KEY } from "@/components/common/BatteryOptimizationGuide";
import { checkBatteryOptimization } from "@/services/batteryOptimization";
import { getLogger } from "@/utils/logger";
import { getConnectionState, subscribeSocketStatus } from "@/services/socket";

const log = getLogger("App");

/** Délai avant stop sync engine à l'unmount — évite stop immédiat lors de remount StrictMode. */
let pendingSyncEngineStopTimer: ReturnType<typeof setTimeout> | null = null;
const SYNC_ENGINE_STOP_DELAY_MS = 400;

SplashScreen.preventAutoHideAsync().catch(() => { });

// P0.2.C — Init cache réseau pour logs (corrélation logout ↔ offline)
initNetworkStateCache();
// P2.2 — Init log context (device_id hash pour corrélation multi-tenant)
// initLogContext();
// Version check - gestion des mises à jour obligatoires/recommandées
import { VersionProvider, useVersion } from "@/contexts/VersionContext";
import { AppAlertProvider } from "@/contexts/AppAlertContext";
import { UpdateRequiredScreen } from "@/components/version/UpdateRequiredScreen";
import { UpdateRecommendedModal } from "@/components/version/UpdateRecommendedModal";

// ✅ Enregistrer la tâche de localisation en arrière-plan (uniquement si le module natif est disponible)
// Workaround expo/expo#25325 : en __DEV__, les tâches background provoquent une boucle de reload.
if (Platform.OS !== "web" && !__DEV__) {
  try {
    require("@/tasks/locationTask");
  } catch (error) {
    log.info("task manager unavailable (expo go, needs dev build)", { error });
  }
}

// ✅ Phase 2.6: Définir la tâche de synchronisation silencieuse en arrière-plan
if (Platform.OS !== "web" && !__DEV__) {
  try {
    const { defineBackgroundSyncTask } = require("@/services/silentNotifications");
    defineBackgroundSyncTask();
    log.success("background sync task defined");
  } catch (error) {
    log.info("background sync task not available", { error });
  }
}

// Notifee onBackgroundEvent est enregistré dans index.js (avant expo-router/entry).

const SENTRY_DSN =
  process.env.EXPO_PUBLIC_SENTRY_DSN ||
  "https://500ea836dce2e802b27109d857cb3534@o4509736814772224.ingest.de.sentry.io/4509736867201104";
const SENTRY_APP_ID_PROD = "ch.liri.operations";
const SENTRY_IS_PROD_RUNTIME =
  !__DEV__ && Application.applicationId === SENTRY_APP_ID_PROD;
const SENTRY_SENSITIVE_KEY_RE =
  /(authorization|token|password|cookie|secret|api.?key|refresh|session)/i;

function redactSensitiveValue(value: unknown): unknown {
  if (value == null) return value;
  if (typeof value === "string") {
    if (value.length <= 12) return "***";
    return `${value.slice(0, 4)}...${value.slice(-4)}`;
  }
  if (Array.isArray(value)) return value.map((v) => redactSensitiveValue(v));
  if (typeof value !== "object") return value;
  const output: Record<string, unknown> = {};
  for (const [key, val] of Object.entries(value as Record<string, unknown>)) {
    if (SENTRY_SENSITIVE_KEY_RE.test(key)) {
      output[key] = typeof val === "string" ? redactSensitiveValue(val) : "***";
      continue;
    }
    output[key] = redactSensitiveValue(val);
  }
  return output;
}

function sanitizeSentryEvent(event: Sentry.ErrorEvent): Sentry.ErrorEvent {
  const cloned: Sentry.ErrorEvent = {
    ...event,
    request: event.request
      ? (redactSensitiveValue(event.request) as Record<string, unknown>)
      : event.request,
    contexts: event.contexts
      ? (redactSensitiveValue(event.contexts) as any)
      : event.contexts,
    extra: event.extra
      ? (redactSensitiveValue(event.extra) as Record<string, unknown>)
      : event.extra,
    user: event.user ? (redactSensitiveValue(event.user) as Sentry.User) : event.user,
    breadcrumbs: event.breadcrumbs?.map((b) => ({
      ...b,
      data: b.data
        ? (redactSensitiveValue(b.data) as Record<string, unknown>)
        : b.data,
    })),
  };
  return cloned;
}

Sentry.init({
  dsn: SENTRY_DSN,
  enabled: SENTRY_IS_PROD_RUNTIME,
  sendDefaultPii: false,
  tracesSampleRate: 0.2,
  profilesSampleRate: 0,
  beforeBreadcrumb(breadcrumb) {
    if (breadcrumb?.data) {
      return {
        ...breadcrumb,
        data: redactSensitiveValue(breadcrumb.data) as Record<string, unknown>,
      };
    }
    return breadcrumb;
  },
  beforeSend(event, hint) {
    const ex = hint?.originalException as Error | undefined;
    const msg = ex?.message ?? event?.message ?? "";
    if (
      typeof msg === "string" &&
      (msg.includes("Unable to activate keep awake") ||
        msg.includes("Unable to deactivate keep awake"))
    ) {
      return null;
    }
    return sanitizeSentryEvent(event);
  },
});

export default function RootLayout() {
  return (
    <GestureHandlerRootView style={styles.gestureRoot}>
      <VersionProvider>
        <AuthProvider>
          <AppAlertProvider>
            <RootNav />
          </AppAlertProvider>
        </AuthProvider>
      </VersionProvider>
    </GestureHandlerRootView>
  );
}

function RootNav() {
  const {
    mode,
    isAuthenticated,
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
    authSessionState,
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
    if (!SENTRY_IS_PROD_RUNTIME) return;
    const appVersion = Constants.expoConfig?.version ?? "unknown";
    const runtimeVersion =
      String(Updates.runtimeVersion ?? Constants.expoConfig?.runtimeVersion ?? "unknown");
    Sentry.setTag("app_version", appVersion);
    Sentry.setTag("runtime_version", runtimeVersion);
    Sentry.setTag("platform", Platform.OS);
    Sentry.setTag("native_app_id", Application.applicationId ?? "unknown");
  }, []);

  useEffect(() => {
    if (!SENTRY_IS_PROD_RUNTIME) return;
    Sentry.setContext("auth_state", {
      mode,
      is_authenticated: isAuthenticated,
      is_driver_authenticated: isDriverAuthenticated,
      is_enterprise_authenticated: isEnterpriseAuthenticated,
    });
  }, [mode, isAuthenticated, isDriverAuthenticated, isEnterpriseAuthenticated]);

  useEffect(() => {
    if (!SENTRY_IS_PROD_RUNTIME) return;
    const applyState = () => {
      const net = getNetworkStateSnapshot();
      Sentry.setContext("network_state", {
        isConnected: net?.isConnected ?? null,
        isInternetReachable: net?.isInternetReachable ?? null,
        type: net?.type ?? "unknown",
      });
    };
    applyState();
    const unsubscribe = subscribeToNetworkState(applyState);
    return unsubscribe;
  }, []);

  useEffect(() => {
    if (!SENTRY_IS_PROD_RUNTIME) return;
    const applySocket = () => {
      Sentry.setContext("socket_state", {
        connection_state: getConnectionState(),
      });
    };
    applySocket();
    const unsubscribe = subscribeSocketStatus((payload) => {
      Sentry.setContext("socket_state", {
        connection_state: payload.connectionState,
        reconnect_exhausted: payload.reconnectExhausted,
        auth_recovery_exhausted: payload.authRecoveryExhausted,
        role: payload.role,
      });
    });
    return unsubscribe;
  }, []);
  useEffect(() => {
    if (splashHiddenRef.current) return;
    if (loading || versionLoading) return;
    splashHiddenRef.current = true;
    SplashScreen.hideAsync().catch(() => { });
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
      log.debug("notification app mode set", { mode });
    }
  }, [isDriverAuthenticated, isEnterpriseAuthenticated]);

  const segments = useSegments();
  const router = useRouter();
  const registeringRef = useRef(false);
  const pushInitForDriverRef = useRef<number | null>(null);
  const syncEngineStartedRef = useRef(false);
  const adaptiveTrackingStartedRef = useRef(false);
  const syncEngineStopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const adaptiveTrackingStopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ✅ Phase 2 - Gestionnaire des actions de notifications
  useNotificationActions();

  // ✅ Deep link handler: parse atmr:// URLs and navigate to appropriate routes
  const isRuntimeInternalLink = React.useCallback((url: string) => {
    if (!url) return true;
    return (
      url.startsWith("exp://") ||
      url.startsWith("exp+") ||
      url.includes("expo-development-client") ||
      url.includes("/--/") // internal Expo Router runtime path
    );
  }, []);

  const handleDeepLink = React.useCallback((url: string) => {
    try {
      if (isRuntimeInternalLink(url)) {
        if (__DEV__) {
          log.debug("internal runtime deep link ignored", { url });
        }
        return;
      }
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
  }, [isDriverAuthenticated, loading, router, isRuntimeInternalLink]);

  // Si une mise à jour est REQUIRED, afficher l'écran bloquant
  // (avant même l'authentification)
  if (updateStatus === "UPDATE_REQUIRED") {
    return <UpdateRequiredScreen />;
  }

  // 🔐 Redirections selon l’état d’auth
  useEffect(() => {
    if (loading) return;
    if (
      authSessionState === "BOOTSTRAPPING" ||
      authSessionState === "RECOVERING" ||
      authSessionState === "DEGRADED"
    ) {
      return;
    }
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
    authSessionState,
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
    loading,
    mode,
    router,
    segments,
  ]);

  // KPI auth pilote: snapshot périodique agrégé pour Go/No-Go.
  useEffect(() => {
    const interval = setInterval(() => {
      const snapshot = getAuthKpiSnapshot();
      logAuthEvent("AUTH_KPI_SNAPSHOT", snapshot);
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  // 🔔 Config + enregistrement push (quand prêt)
  useEffect(() => {
    if (loading || versionLoading || !isDriverAuthenticated || !driver) return;
    const currentUserId = driver.id;
    if (pushInitForDriverRef.current === currentUserId) {
      return;
    }

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
    let registrationCompleted = false;

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
          tokenToUse = (tokens as any)?.expo ?? (tokens as any)?.device ?? null;
          provider = "expo";
        }

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
          pushInitForDriverRef.current = currentUserId;
          registrationCompleted = true;
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
      if (!registrationCompleted && pushInitForDriverRef.current === currentUserId) {
        // Sécurité: si un état concurrent invalide un run en cours, garder la possibilité de retry.
        pushInitForDriverRef.current = null;
      }
    };
  }, [driver, isDriverAuthenticated, loading, versionLoading]);

  useEffect(() => {
    if (loading || versionLoading) return;
    if (!isDriverAuthenticated) {
      pushInitForDriverRef.current = null;
    }
  }, [loading, versionLoading, isDriverAuthenticated]);

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
  // Respecte "Ne plus afficher" : on lit la préférence AVANT de décider d'afficher
  useEffect(() => {
    if (loading || !isDriverAuthenticated || !driver) return;
    if (Platform.OS !== "android") return;
    if (String(Constants.expoConfig?.extra?.APP_VARIANT || process.env.APP_VARIANT || "prod") === "prod") {
      return;
    }
    (async () => {
      try {
        const [dismissed, { needsExemption }] = await Promise.all([
          AsyncStorage.getItem(BATTERY_GUIDE_DISMISSED_KEY),
          checkBatteryOptimization(),
        ]);
        if (dismissed === "true") return;
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
        if (isRuntimeInternalLink(url)) {
          return;
        }
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
      if (isRuntimeInternalLink(event.url)) {
        return;
      }
      log.info("deep link received while app running", { url: event.url });
      if (isDriverAuthenticated && !loading) {
        handleDeepLink(event.url);
      }
    });

    return () => {
      subscription.remove();
    };
  }, [isDriverAuthenticated, loading, handleDeepLink, isRuntimeInternalLink]);

  // ✅ Plan 2G/3G : syncEngine démarre/arrête avec le driver
  // Important: ne pas lancer le tracking en mode entreprise (évite "no driver_id, skip enqueue"
  // et envoi de positions incorrectes quand l'utilisateur consulte le dashboard).
  useEffect(() => {
    if (loading || versionLoading) return;
    if (mode !== "driver" || !isDriverAuthenticated || !driver) {
      // Anti-flap: pendant l'hydratation auth, éviter stop/start immédiat.
      if (syncEngineStopTimerRef.current) {
        clearTimeout(syncEngineStopTimerRef.current);
      }
      syncEngineStopTimerRef.current = setTimeout(() => {
        if (syncEngineStartedRef.current) {
          getSyncEngine().stop();
          syncEngineStartedRef.current = false;
        }
      }, 1500);
      return;
    }
    // Remount rapide (StrictMode) : annuler le stop différé du cleanup
    if (pendingSyncEngineStopTimer) {
      clearTimeout(pendingSyncEngineStopTimer);
      pendingSyncEngineStopTimer = null;
    }
    if (syncEngineStopTimerRef.current) {
      clearTimeout(syncEngineStopTimerRef.current);
      syncEngineStopTimerRef.current = null;
    }
    if (!syncEngineStartedRef.current) {
      getSyncEngine().start();
      syncEngineStartedRef.current = true;
    }
  }, [driver?.id, isDriverAuthenticated, mode, loading, versionLoading]);

  // ✅ 4. Fréquence GPS Adaptative Mobile : Démarrer tracking adaptatif pour les drivers
  // Important: uniquement en mode driver (évite envoi de positions en mode entreprise).
  useEffect(() => {
    if (loading || versionLoading) return;
    if (mode !== "driver" || !isDriverAuthenticated || !driver) {
      if (adaptiveTrackingStopTimerRef.current) {
        clearTimeout(adaptiveTrackingStopTimerRef.current);
      }
      // Anti-flap auth pour éviter "tracking already active" au boot.
      adaptiveTrackingStopTimerRef.current = setTimeout(() => {
        if (adaptiveTrackingStartedRef.current) {
          stopAdaptiveLocationTracking();
          adaptiveTrackingStartedRef.current = false;
        }
      }, 1500);
      return;
    }
    if (adaptiveTrackingStopTimerRef.current) {
      clearTimeout(adaptiveTrackingStopTimerRef.current);
      adaptiveTrackingStopTimerRef.current = null;
    }
    if (adaptiveTrackingStartedRef.current) {
      return;
    }

    // Démarrer le tracking adaptatif pour le driver authentifié
    adaptiveTrackingStartedRef.current = true;

    (async () => {
      try {
        log.info("starting adaptive gps tracking");
        await startAdaptiveLocationTracking();
        log.success("adaptive gps tracking started");
      } catch (e: any) {
        adaptiveTrackingStartedRef.current = false;
        log.warn("adaptive gps tracking start error", {
          message: e?.message || String(e),
        });
      }
    })();
  }, [driver?.id, isDriverAuthenticated, mode, loading, versionLoading]);

  // ✅ Background GPS : réconciliation (mission-active only). Jamais de start au boot.
  // Règle B : réconciliation non-démarrante si pas de mission active.
  useEffect(() => {
    if (Platform.OS === "web") return;
    if (mode !== "driver") return;
    const sub = AppState.addEventListener("change", (state: AppStateStatus) => {
      if (state === "active") {
        InteractionManager.runAfterInteractions(() => {
          setTimeout(async () => {
            const inputs = await buildBgTrackingInputs({
              isAuthenticated: !!isDriverAuthenticated,
              role: "driver",
              hasActiveMission: MissionStateManager.isActive(),
            });
            await reconcileBackgroundTrackingState("app_resume", inputs);
          }, 150);
        });
        return;
      }
      // Quand l'utilisateur ouvre Google Maps (app en background),
      // forcer une réconciliation immédiate pour garantir le démarrage du task natif.
      setTimeout(async () => {
        const inputs = await buildBgTrackingInputs({
          isAuthenticated: !!isDriverAuthenticated,
          role: "driver",
          hasActiveMission: MissionStateManager.isActive(),
        });
        await reconcileBackgroundTrackingState("app_background", inputs);
      }, 0);
    });
    return () => sub.remove();
  }, [mode, isDriverAuthenticated]);

  // ✅ Réconciliation au logout : stop immédiat (contrat d'arrêt).
  useEffect(() => {
    if (Platform.OS === "web") return;
    if (isDriverAuthenticated) return;
    void ensureBackgroundTrackingStopped("logout");
  }, [isDriverAuthenticated]);

  // ✅ Background GPS : mission_started / mission_stopped → réconciliation
  useEffect(() => {
    if (Platform.OS === "web") return;
    if (mode !== "driver") return;
    const unsub = MissionStateManager.subscribe((event) => {
      // Réconcilier aussi sur transition_confirmed/state_changed pour démarrer le tracking
      // dès EN_ROUTE (mission-critical), avant que le chauffeur ouvre Maps et passe en arrière-plan.
      if (
        event !== "mission_started" &&
        event !== "mission_stopped" &&
        event !== "transition_confirmed" &&
        event !== "state_changed" &&
        event !== "reconciliation"
      )
        return;
      (async () => {
        if (event === "mission_started") {
          await requestBackgroundPermissionIfNeeded();
        }
        const inputs = await buildBgTrackingInputs({
          isAuthenticated: !!isDriverAuthenticated,
          role: "driver",
          hasActiveMission: MissionStateManager.isActive(),
        });
        await reconcileBackgroundTrackingState(event, inputs);
      })();
    });
    return unsub;
  }, [mode, isDriverAuthenticated]);

  // ✅ Réconciliation au login driver : corriger désalignement état métier vs natif.
  useEffect(() => {
    if (Platform.OS === "web") return;
    if (!isDriverAuthenticated || !driver || mode !== "driver") return;
    (async () => {
      const inputs = await buildBgTrackingInputs({
        isAuthenticated: true,
        role: "driver",
        hasActiveMission: MissionStateManager.isActive(),
      });
      await reconcileBackgroundTrackingState("session_change", inputs);
    })();
  }, [driver?.id, isDriverAuthenticated, mode]);

  // Cleanup global à l'unmount du layout
  useEffect(() => {
    return () => {
      if (pendingSyncEngineStopTimer) {
        clearTimeout(pendingSyncEngineStopTimer);
        pendingSyncEngineStopTimer = null;
      }
      if (syncEngineStartedRef.current) {
        // Délai pour remount StrictMode : si RootNav remonte vite, on évite le stop.
        pendingSyncEngineStopTimer = setTimeout(() => {
          pendingSyncEngineStopTimer = null;
          if (syncEngineStartedRef.current) {
            getSyncEngine().stop();
            syncEngineStartedRef.current = false;
          }
        }, SYNC_ENGINE_STOP_DELAY_MS);
      }
      if (syncEngineStopTimerRef.current) {
        clearTimeout(syncEngineStopTimerRef.current);
        syncEngineStopTimerRef.current = null;
      }
      if (adaptiveTrackingStartedRef.current) {
        stopAdaptiveLocationTracking();
        adaptiveTrackingStartedRef.current = false;
      }
      if (adaptiveTrackingStopTimerRef.current) {
        clearTimeout(adaptiveTrackingStopTimerRef.current);
        adaptiveTrackingStopTimerRef.current = null;
      }
    };
  }, []);

  // ✅ UX : journaliser uniquement les transitions d'état, pas chaque render.
  useEffect(() => {
    log.debug("root nav loading state", {
      loading,
      versionLoading,
      isAuthenticated,
      isDriverAuthenticated,
      isEnterpriseAuthenticated,
      mode,
      timestamp: new Date().toISOString(),
    });
  }, [
    loading,
    versionLoading,
    isAuthenticated,
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
    mode,
  ]);

  if (loading || versionLoading) {
    return <BrandedLoadingScreen />;
  }

  return (
    <>
      <InAppNotificationToast />
      <OfflineBanner />
      {(authSessionState === "RECOVERING" || authSessionState === "DEGRADED") && (
        <SessionRecoveryBanner state={authSessionState} />
      )}
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

function SessionRecoveryBanner({ state }: { state: "RECOVERING" | "DEGRADED" }) {
  const message =
    state === "RECOVERING"
      ? "Reconnexion en cours. La session reste active."
      : "Connexion dégradée. Nouvelle tentative automatique en cours.";
  return (
    <View style={styles.recoveryBanner}>
      <Text style={styles.recoveryBannerText}>{message}</Text>
    </View>
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
  gestureRoot: {
    flex: 1,
  },
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
  recoveryBanner: {
    backgroundColor: "#FFF4E5",
    borderBottomWidth: 1,
    borderBottomColor: "#F5D7A1",
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  recoveryBannerText: {
    color: "#8A5A00",
    fontSize: 12,
    fontWeight: "600",
    textAlign: "center",
  },
});
