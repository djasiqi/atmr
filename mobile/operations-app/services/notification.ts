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

  // Stratégie: essayer d'abord Expo token (plus stable), puis device token
  let device: string | null = null;
  let expo: string | null = null;

  // Détecter si on est en mode dev (via extra config, package name, ou environnement)
  const expoExtra = Constants.expoConfig?.extra || {};
  const packageName = Constants.expoConfig?.android?.package || "";
  const appVariant = expoExtra.APP_VARIANT || process.env.APP_VARIANT || "";
  const isDevVariant = appVariant === "dev" || packageName.includes(".dev");
  
  // Détecter si on est en développement local (pas un build EAS)
  const isLocalDev = __DEV__ || Constants.executionEnvironment === "storeClient" || Constants.appOwnership === "expo";
  
  const shouldSkipFirebase = isDevVariant || isLocalDev;
  
  console.log("🔔 Détection variante:", { 
    packageName, 
    appVariant, 
    isDevVariant,
    isLocalDev,
    shouldSkipFirebase,
    hasGoogleServices: !!Constants.expoConfig?.android?.googleServicesFile,
    executionEnvironment: Constants.executionEnvironment,
    appOwnership: Constants.appOwnership
  });

  // Pour la variante dev ou le dev local, ignorer complètement Firebase
  // Car Expo Push peut aussi essayer Firebase en arrière-plan sur Android
  if (shouldSkipFirebase && Platform.OS === "android") {
    console.log("🔔 Mode dev/développement local détecté - Firebase désactivé, utilisation Expo Push uniquement");
    // Si Expo échoue aussi avec FIS_AUTH_ERROR, on accepte l'échec gracieusement
  }

  // 1. Essayer le token Expo en premier (souvent plus stable)
  if (wantExpo || Platform.OS === 'android') {
    try {
      console.log("🔔 Tentative récupération Expo token...");
      const expoToken = await Notifications.getExpoPushTokenAsync({
        projectId: Constants.expoConfig?.extra?.eas?.projectId,
      });
      expo = expoToken?.data ?? null;
      console.log("✅ Expo token récupéré:", expo ? "OK" : "VIDE");
    } catch (e: any) {
      const msg = String(e?.message || e);
      console.warn("⚠️ Expo token échec:", msg);
      
      // Si on est en dev/local et que Expo échoue avec FIS_AUTH_ERROR, c'est normal
      // Firebase n'est pas configuré ou accessible en développement local
      if (shouldSkipFirebase && msg.includes('FIS_AUTH_ERROR')) {
        console.log("ℹ️ FIS_AUTH_ERROR en dev/local - Firebase non accessible, c'est normal");
      }
    }
  }
  
  // 2. Essayer le device token avec retry (ignorer Firebase pour la variante dev/local)
  if (shouldSkipFirebase && Platform.OS === "android") {
    // Pour la variante dev/local Android, utiliser uniquement le token Expo (Firebase non accessible)
    console.log("🔔 Mode dev/local - skip Firebase device token, utilisation Expo uniquement");
    if (expo) {
      device = expo;
      console.log("✅ Token Expo utilisé comme device token pour le mode dev/local");
    } else {
      console.warn("⚠️ Mode dev/local: aucun token Expo disponible");
      // Accepter gracieusement l'absence de notifications en dev local
      console.log("ℹ️ Notifications désactivées en développement local - normal si Firebase non configuré");
    }
  } else {
    // Pour production ou iOS, essayer Firebase/APNs normalement
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
  }

  // Résultat final
  const result = { device, expo };
  console.log("🔔 initNotifications RESULT:", {
    device: device ? "✅" : "❌",
    expo: expo ? "✅" : "❌",
  });

  return result;
}
