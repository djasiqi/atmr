import { Platform } from "react-native";
import {
  FirebaseMessagingTypes,
  getMessaging,
  onMessage,
  getInitialNotification,
  onNotificationOpenedApp,
  requestPermission,
  AuthorizationStatus,
  getToken,
  onTokenRefresh,
} from "@react-native-firebase/messaging";
import { getApp, getApps } from "@react-native-firebase/app";
import notifee, { AndroidImportance, EventType } from "@notifee/react-native";
import { getLogger } from "@/utils/logger";
import { validateDeepLink } from "@/services/deepLinkHandler";

const log = getLogger("FCM");

/** FCM natif uniquement ; pas d’app [DEFAULT] sur web / Expo Go / config absente. */
let messagingMemo: FirebaseMessagingTypes.Module | null | undefined;

function getMessagingOrNull(): FirebaseMessagingTypes.Module | null {
  if (messagingMemo !== undefined) return messagingMemo;
  if (Platform.OS === "web") {
    messagingMemo = null;
    return null;
  }
  try {
    if (getApps().length === 0) {
      messagingMemo = null;
      log.warn("FCM désactivé: aucune app Firebase [DEFAULT] (google-services / dev build requis)");
      return null;
    }
    messagingMemo = getMessaging(getApp());
    return messagingMemo;
  } catch (e) {
    messagingMemo = null;
    log.warn("FCM désactivé: initialisation Firebase impossible", {
      message: e instanceof Error ? e.message : String(e),
    });
    return null;
  }
}

let permissionRequestPromise: Promise<boolean> | null = null;
let tokenRequestPromise: Promise<string | null> | null = null;

export type FCMData = {
  type: string;
  title?: string;
  body?: string;
  deepLink?: string;
  booking_id?: string;
  channelId?: string;
  trace_id?: string;
  recipient_role?: string;
};

type NavigationCallback = (deepLink: string) => void;

let _onNavigate: NavigationCallback | null = null;

export function setFCMNavigationHandler(cb: NavigationCallback): void {
  _onNavigate = cb;
}

function handleDeepLink(data: Record<string, string> | undefined): void {
  if (!data?.deepLink) return;
  const result = validateDeepLink(data.deepLink);
  if (!result.valid) {
    log.warn("invalid deep link from FCM", { deepLink: data.deepLink, error: result.error });
    return;
  }
  log.info("navigating from FCM deep link", { deepLink: data.deepLink, type: result.type });
  _onNavigate?.(data.deepLink);
}

async function ensureChannel(): Promise<void> {
  await notifee.createChannel({
    id: "missions_v2",
    name: "Missions",
    importance: AndroidImportance.HIGH,
    sound: "default",
  });
}

/**
 * Foreground message handler.
 * Android data-only: display via Notifee (same as background handler).
 * iOS notification+data: system already shows it — skip display to avoid duplicates.
 */
export function registerForegroundHandler(): () => void {
  const messaging = getMessagingOrNull();
  if (!messaging) return () => {};
  return onMessage(messaging, async (remoteMessage) => {
    log.info("foreground message received", {
      type: remoteMessage.data?.type,
      hasNotification: !!remoteMessage.notification,
    });

    if (Platform.OS === "android" && !remoteMessage.notification) {
      const data = remoteMessage.data as Record<string, string> | undefined;
      if (!data || data.type === "silent_update") return;

      await ensureChannel();
      // ID déterministe : évite doublons quand plusieurs FCM arrivent (ex: 2 DeviceTokens même appareil)
      const notifId = `mission_${data.booking_id || data.trace_id || Date.now()}_${data.type || "push"}`.replace(/\s/g, "_");
      await notifee.displayNotification({
        id: notifId,
        title: data.title || "Liri Opérations",
        body: data.body || "",
        data,
        android: {
          channelId: data.channelId || "missions_v2",
          importance: AndroidImportance.HIGH,
          sound: "default",
          pressAction: { id: "default" },
        },
      });
    }
  });
}

/**
 * Handles notification tap when app was killed (cold start).
 * Must be called early in the app lifecycle.
 */
export async function handleInitialNotification(): Promise<void> {
  const messaging = getMessagingOrNull();
  if (!messaging) return;
  try {
    const remoteMessage = await getInitialNotification(messaging);
    if (remoteMessage) {
      log.info("initial notification (killed->tap)", {
        type: remoteMessage.data?.type,
        booking_id: remoteMessage.data?.booking_id,
      });
      handleDeepLink(remoteMessage.data as Record<string, string> | undefined);
    }
  } catch (e) {
    log.error("getInitialNotification failed", { error: e });
  }
}

/**
 * Handles notification tap when app was in background.
 * Returns unsubscribe function.
 */
export function registerNotificationOpenedHandler(): () => void {
  const messaging = getMessagingOrNull();
  if (!messaging) return () => {};
  return onNotificationOpenedApp(messaging, (remoteMessage) => {
    log.info("notification opened (background->tap)", {
      type: remoteMessage.data?.type,
      booking_id: remoteMessage.data?.booking_id,
    });
    handleDeepLink(remoteMessage.data as Record<string, string> | undefined);
  });
}

/**
 * Handles Notifee foreground tap events (Android data-only displayed via Notifee).
 */
export function registerNotifeeForegroundHandler(): () => void {
  return notifee.onForegroundEvent(({ type, detail }) => {
    if (type === EventType.PRESS && detail.notification?.data) {
      log.info("notifee foreground press", {
        type: detail.notification.data.type,
      });
      handleDeepLink(detail.notification.data as Record<string, string>);
    }
  });
}

/**
 * Request FCM permission (mainly for iOS).
 * On Android 13+ POST_NOTIFICATIONS is handled by expo-notifications.
 */
export async function requestFCMPermission(): Promise<boolean> {
  if (permissionRequestPromise) {
    return permissionRequestPromise;
  }
  permissionRequestPromise = (async () => {
    const messaging = getMessagingOrNull();
    if (!messaging) return false;
    try {
      const authStatus = await requestPermission(messaging);
      const enabled =
        authStatus === AuthorizationStatus.AUTHORIZED ||
        authStatus === AuthorizationStatus.PROVISIONAL;
      log.info("FCM permission", { authStatus, enabled });
      return enabled;
    } catch (e) {
      log.error("FCM permission request failed", { error: e });
      return false;
    }
  })();

  try {
    return await permissionRequestPromise;
  } finally {
    permissionRequestPromise = null;
  }
}

/**
 * Get the FCM registration token for this device.
 */
export async function getFCMToken(): Promise<string | null> {
  if (tokenRequestPromise) {
    return tokenRequestPromise;
  }
  tokenRequestPromise = (async () => {
    const messaging = getMessagingOrNull();
    if (!messaging) return null;
    try {
      const token = await getToken(messaging);
      log.info("FCM token retrieved", { tokenPrefix: token?.substring(0, 12) });
      return token;
    } catch (e) {
      // En dev Android, l'absence de config Firebase complète ne doit pas afficher une erreur bloquante.
      const message =
        e instanceof Error
          ? e.message
          : typeof e === "string"
            ? e
            : Object.prototype.toString.call(e);
      log.warn("getFCMToken unavailable", { message, platform: Platform.OS });
      return null;
    }
  })();

  try {
    return await tokenRequestPromise;
  } finally {
    tokenRequestPromise = null;
  }
}

/**
 * Listen for token refresh events. Returns unsubscribe.
 */
export function onFCMTokenRefresh(
  callback: (token: string) => void
): () => void {
  const messaging = getMessagingOrNull();
  if (!messaging) return () => {};
  return onTokenRefresh(messaging, (token) => {
    log.info("FCM token refreshed", { tokenPrefix: token?.substring(0, 12) });
    callback(token);
  });
}
