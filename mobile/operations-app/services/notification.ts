// services/notification.ts
import { Platform, PermissionsAndroid } from "react-native";
import Constants from "expo-constants";
import * as Notifications from "expo-notifications";
import { getLogger } from "@/utils/logger";

const log = getLogger("Push");

export type PushTokens = {
  /** Android: FCM; iOS: APNs */
  device: string | null;
  /** Optionnel: si tu utilises le service Expo pour envoyer */
  expo?: string | null;
};

/** P0.5: Mode app pour filtrer les notifs au boot (driver vs enterprise).
 *  null = inconnu → filtre company par défaut (conservateur pour driver).
 */
let _notificationAppMode: "driver" | "enterprise" | null = null;

export function setNotificationAppMode(mode: "driver" | "enterprise" | null): void {
  _notificationAppMode = mode;
}

// --- 1) Handler global: quoi faire en foreground (P0.5: boot-level, avant useNotifications) ---
Notifications.setNotificationHandler({
  handleNotification: async (notification) => {
    const data = notification.request.content.data || {};
    const notificationType = data.type || "";
    const recipientRole = data.recipient_role as string | undefined;

    // ✅ Phase 2.6: Ne pas afficher les notifications silencieuses
    if (notificationType === "silent_update" || data["content-available"] === 1) {
      return {
        shouldShowAlert: false,
        shouldPlaySound: false,
        shouldSetBadge: false,
        shouldShowBanner: false,
        shouldShowList: false,
      };
    }

    // P0.5: Filtrer recipient_role=company quand on est en mode driver (ou inconnu = conservateur)
    if (recipientRole === "company" && _notificationAppMode !== "enterprise") {
      if (__DEV__) {
        log.info("ignored company notification on driver app", {
          trace_id: data.trace_id,
          recipient_role: recipientRole,
          app_mode: _notificationAppMode,
        });
      }
      return {
        shouldShowAlert: false,
        shouldPlaySound: false,
        shouldSetBadge: false,
        shouldShowBanner: false,
        shouldShowList: false,
      };
    }

    // Foreground: pas de notification système, uniquement in-app
    return {
      shouldShowAlert: false,
      shouldPlaySound: false,
      shouldSetBadge: false,
      shouldShowBanner: false,
      shouldShowList: false,
    };
  },
});

/**
 * Configure le canal Android et vérifie/obtient les permissions (iOS + Android < 13).
 * À appeler très tôt (App.tsx) avant initNotifications().
 */
export async function configureNotifications(): Promise<void> {
  try {
    // iOS & Android < 13: permission via Expo
    const { status } = await Notifications.getPermissionsAsync();
    if (status !== "granted") {
      const { status: newStatus } = await Notifications.requestPermissionsAsync();
      if (newStatus !== "granted") {
        log.warn("notifications denied by user");
        return;
      }
    }

    // Android: canal par défaut (à faire une fois)
    if (Platform.OS === "android") {
      await Notifications.setNotificationChannelAsync("default", {
        name: "default",
        importance: Notifications.AndroidImportance.MAX,
        sound: "default",
        vibrationPattern: [0, 250, 250, 250],
        lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
      });
    }

    log.success("notifications configured");
  } catch (error) {
    log.error("configure notifications failed", { error });
  }
}

/** Android 13+ (runtime) */
async function ensureAndroid13Permission(): Promise<boolean> {
  if (Platform.OS !== "android" || Platform.Version < 33) return true;
  const res = await PermissionsAndroid.request(
    "android.permission.POST_NOTIFICATIONS"
  );
  return res === PermissionsAndroid.RESULTS.GRANTED;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Récupère les tokens de push avec gestion d'erreur améliorée
 */
export async function initNotifications(
  opts: { withExpoToken?: boolean; maxRetries?: number } = {}
): Promise<PushTokens> {
  const maxRetries = opts.maxRetries ?? 3;
  const wantExpo = !!opts.withExpoToken;

  log.info("init notifications start", { maxRetries, wantExpo });

  // Android 13+: demander la permission runtime
  const granted13 = await ensureAndroid13Permission();
  if (!granted13) {
    log.warn("notification permission denied android 13");
    return { device: null, expo: null };
  }

  // Stratégie: sur un dev build Android (téléphone réel), le token "device" (FCM/APNs)
  // doit être prioritaire. Expo token reste optionnel (utile si tu envoies via Expo).
  let device: string | null = null;
  let expo: string | null = null;

  const projectId =
    (Constants.expoConfig as any)?.extra?.eas?.projectId ??
    (Constants as any)?.easConfig?.projectId ??
    null;

  const isExpoGo =
    Constants.appOwnership === "expo" &&
    (Constants.executionEnvironment === "storeClient" ||
      Constants.executionEnvironment === "standalone" /* fallback safe */);

  log.info("push env", {
    platform: Platform.OS,
    appOwnership: Constants.appOwnership,
    executionEnvironment: Constants.executionEnvironment,
    hasProjectId: !!projectId,
    isExpoGo,
  });

  // Expo Go: pas de push remote en SDK 53+ (Android)
  if (isExpoGo) {
    log.warn("expo go detected, remote push unavailable");
    return { device: null, expo: null };
  }

  // 1) Device token (FCM/APNs) avec retry
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      log.info("device token attempt", { attempt: attempt + 1, max: maxRetries + 1 });
      const tokenData = await Notifications.getDevicePushTokenAsync();
      device = tokenData?.data ?? null;
      if (device) {
        log.success("device token retrieved");
        break;
      }
      throw new Error("Token device vide");
    } catch (e: any) {
      const msg = String(e?.message || e);
      log.warn("device token failed", { attempt: attempt + 1, max: maxRetries + 1, msg });
      if (attempt < maxRetries) {
        const backoff = 400 * Math.pow(2, attempt);
        log.info("waiting before retry", { backoff });
        await sleep(backoff);
      }
    }
  }

  // 2) Expo token (optionnel)
  if (wantExpo) {
    try {
      log.info("fetching expo token");
      const expoToken = await Notifications.getExpoPushTokenAsync(
        projectId ? { projectId } : undefined
      );
      expo = expoToken?.data ?? null;
      log.success("expo token retrieved", { ok: !!expo });
    } catch (e: any) {
      const msg = String(e?.message || e);
      log.warn("expo token failed", { msg });
    }
  }

  // Résultat final
  const result = { device, expo };
  log.info("init notifications result", { device: !!device, expo: !!expo });

  return result;
}
