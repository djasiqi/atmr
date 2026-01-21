// services/notification.ts
import { Platform, PermissionsAndroid } from "react-native";
import Constants from "expo-constants";
import * as Notifications from "expo-notifications";

export type PushTokens = {
  /** Android: FCM; iOS: APNs */
  device: string | null;
  /** Optionnel: si tu utilises le service Expo pour envoyer */
  expo?: string | null;
};

// --- 1) Handler global: quoi faire en foreground ---
 Notifications.setNotificationHandler({
   handleNotification: async (notification) => {
     const data = notification.request.content.data || {};
     const notificationType = data.type || "";

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

     // Pour les autres notifications, afficher normalement
     return {
       shouldShowAlert: true,
       shouldPlaySound: false,
       shouldSetBadge: false,
       // iOS (SDK 5x) :
       shouldShowBanner: true,
       shouldShowList: true,
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
        console.warn("🚫 Notifications refusées par l'utilisateur");
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

    console.log("🔔 Notifications configurées");
  } catch (error) {
    console.error("❌ Erreur configureNotifications :", error);
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

  console.log("🔔 initNotifications START", { maxRetries, wantExpo });

  // Android 13+: demander la permission runtime
  const granted13 = await ensureAndroid13Permission();
  if (!granted13) {
    console.warn("🚫 Permission notifications refusée (Android 13+).");
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

  console.log("🔔 Push env:", {
    platform: Platform.OS,
    appOwnership: Constants.appOwnership,
    executionEnvironment: Constants.executionEnvironment,
    projectId: projectId ? "✅" : "❌",
    isExpoGo,
  });

  // Expo Go: pas de push remote en SDK 53+ (Android)
  if (isExpoGo) {
    console.warn("🚫 Expo Go détecté: push remote indisponible.");
    return { device: null, expo: null };
  }

  // 1) Device token (FCM/APNs) avec retry
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      console.log(`🔔 Tentative device token (${attempt + 1}/${maxRetries + 1})...`);
      const tokenData = await Notifications.getDevicePushTokenAsync();
      device = tokenData?.data ?? null;
      if (device) {
        console.log("✅ Device token récupéré");
        break;
      }
      throw new Error("Token device vide");
    } catch (e: any) {
      const msg = String(e?.message || e);
      console.warn(`⚠️ Device token échec (${attempt + 1}/${maxRetries + 1}): ${msg}`);
      if (attempt < maxRetries) {
        const backoff = 400 * Math.pow(2, attempt);
        console.log(`⏳ Attente ${backoff}ms avant retry...`);
        await sleep(backoff);
      }
    }
  }

  // 2) Expo token (optionnel)
  if (wantExpo) {
    try {
      console.log("🔔 Tentative récupération Expo token...");
      const expoToken = await Notifications.getExpoPushTokenAsync(
        projectId ? { projectId } : undefined
      );
      expo = expoToken?.data ?? null;
      console.log("✅ Expo token récupéré:", expo ? "OK" : "VIDE");
    } catch (e: any) {
      const msg = String(e?.message || e);
      console.warn("⚠️ Expo token échec:", msg);
    }
  }

  // Résultat final
  const result = { device, expo };
  console.log("🔔 initNotifications RESULT:", {
    device: device ? "✅" : "❌",
    expo: expo ? "✅" : "❌",
  });

  return result;
}
