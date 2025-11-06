// services/notification.ts
import { Platform, PermissionsAndroid } from "react-native";
import * as Notifications from "expo-notifications";

export type PushTokens = {
  /** Android: FCM; iOS: APNs */
  device: string | null;
  /** Optionnel: si tu utilises le service Expo pour envoyer */
  expo?: string | null;
};

// --- 1) Handler global: quoi faire en foreground ---
 Notifications.setNotificationHandler({
   handleNotification: async () => ({
     shouldShowAlert: true,
     shouldPlaySound: false,
     shouldSetBadge: false,
     // iOS (SDK 5x) :
     shouldShowBanner: true,
     shouldShowList: true,
   }),
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

  // Stratégie: essayer d'abord Expo token (plus stable), puis device token
  let device: string | null = null;
  let expo: string | null = null;

  // 1. Essayer le token Expo en premier (souvent plus stable)
  if (wantExpo || Platform.OS === 'android') {
    try {
      console.log("🔔 Tentative récupération Expo token...");
      const expoToken = await Notifications.getExpoPushTokenAsync();
      expo = expoToken?.data ?? null;
      console.log("✅ Expo token récupéré:", expo ? "OK" : "VIDE");
    } catch (e: any) {
      console.warn("⚠️ Expo token échec:", e?.message);
    }
  }

  // 2. Essayer le device token avec retry
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      console.log(`🔔 Tentative device token (${attempt + 1}/${maxRetries + 1})...`);
      
      // Sur Android, si Firebase échoue, utiliser le token Expo comme fallback
      if (Platform.OS === 'android' && attempt > 0 && expo) {
        console.log("🔔 Utilisation du token Expo comme device token (fallback)");
        device = expo;
        break;
      }

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
      
      // Si c'est une erreur Firebase et qu'on a un token Expo, l'utiliser
      if (msg.includes('FIS_AUTH_ERROR') && expo) {
        console.log("🔔 FIS_AUTH_ERROR détecté, utilisation du token Expo");
        device = expo;
        break;
      }
      
      if (attempt < maxRetries) {
        const backoff = 400 * Math.pow(2, attempt);
        console.log(`⏳ Attente ${backoff}ms avant retry...`);
        await sleep(backoff);
      }
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
