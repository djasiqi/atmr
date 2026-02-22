// mobile/operations-app/services/richNotifications.ts
import * as Notifications from "expo-notifications";
import { downloadAsync, readDirectoryAsync, deleteAsync } from "expo-file-system";
import { Platform } from "react-native";
import { getLogger } from "@/utils/logger";

const log = getLogger("RichNotif");

// Chemin de cache pour les images de notifications
const CACHE_DIR = "/notificationImages/";

/**
 * Configuration pour Rich Media dans les notifications
 * Phase 2 - Enrichissement
 */

/**
 * Télécharge une image et la met en cache pour utilisation dans une notification
 *
 * @param imageUrl URL de l'image à télécharger
 * @param timeout Timeout en ms (défaut: 3000ms)
 * @returns URI locale de l'image ou null si échec
 */
export async function downloadImageForNotification(
  imageUrl: string,
  timeout: number = 3000
): Promise<string | null> {
  try {
    // Créer un nom de fichier unique basé sur l'URL
    const filename = `notif_${Date.now()}_${imageUrl.split("/").pop()}`;
    const fileUri = `${CACHE_DIR}${filename}`;

    log.info("downloading notification image", { imageUrl });

    // Télécharger l'image avec timeout
    const downloadPromise = downloadAsync(imageUrl, fileUri);
    const timeoutPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error("Timeout")), timeout)
    );

    const result: any = await Promise.race([downloadPromise, timeoutPromise]);

    if (result && result.status === 200) {
      log.success("image downloaded", { fileUri });
      return fileUri;
    } else {
      log.warn("image download failed", { status: result?.status });
      return null;
    }
  } catch (error) {
    log.error("download notification image failed", { error });
    return null;
  }
}

/**
 * Crée une notification avec image (BigPictureStyle sur Android)
 *
 * @param title Titre de la notification
 * @param body Corps du message
 * @param imageUrl URL de l'image à afficher
 * @param data Données additionnelles
 * @param channelId Canal Android à utiliser
 * @returns Notification ID ou null si échec
 */
export async function scheduleNotificationWithImage(
  title: string,
  body: string,
  imageUrl: string,
  data?: Record<string, any>,
  channelId?: string
): Promise<string | null> {
  try {
    // Télécharger l'image
    const localImageUri = await downloadImageForNotification(imageUrl);

    if (!localImageUri) {
      log.warn("image unavailable, notification without image");
      // Fallback: notification sans image
      return await Notifications.scheduleNotificationAsync({
        content: {
          title,
          body,
          data: data || {},
          ...(Platform.OS === "android" && channelId && { channelId }),
        },
        trigger: null,
      });
    }

    // Notification avec image
    const notificationId = await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data: data || {},
        ...(Platform.OS === "android" && channelId && { channelId }),

        // ✅ Attachments (iOS et Android)
        attachments: [
          {
            identifier: "image",
            url: localImageUri,
            type: "image",
            typeHint: "public.png", // ou "public.jpeg"
          },
        ],

        // ✅ Android: BigPictureStyle
        ...(Platform.OS === "android" && {
          // @ts-ignore - Expo types incomplets pour Android-specific
          android: {
            bigPicture: localImageUri,
            bigLargeIcon: null, // Cache l'icône par défaut pour BigPicture
          },
        }),
      },
      trigger: null,
    });

    log.success("notification with image created", { notificationId });
    return notificationId;
  } catch (error) {
    log.error("create notification with image failed", { error });
    return null;
  }
}

/**
 * Crée une notification style Inbox (Android uniquement)
 * Utile pour regrouper plusieurs messages/notifications
 *
 * @param title Titre principal
 * @param messages Liste des messages à afficher
 * @param data Données additionnelles
 * @param channelId Canal Android
 */
export async function scheduleInboxStyleNotification(
  title: string,
  messages: string[],
  data?: Record<string, any>,
  channelId?: string
): Promise<string | null> {
  if (Platform.OS !== "android") {
    log.warn("inbox style only on android");
    // Fallback iOS: notification simple avec compteur
    const body = `${messages.length} nouveaux messages`;
    return await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data: data || {},
      },
      trigger: null,
    });
  }

  try {
    const notificationId = await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body: `${messages.length} nouveaux messages`,
        data: data || {},
        ...(channelId && { channelId }),

        // @ts-ignore - Android-specific
        android: {
          style: "inbox",
          lines: messages.slice(0, 7), // Max 7 lignes
          summaryText: `+${Math.max(0, messages.length - 7)} autres`,
        },
      },
      trigger: null,
    });

    log.success("inbox style notification created", { notificationId });
    return notificationId;
  } catch (error) {
    log.error("create inbox style notification failed", { error });
    return null;
  }
}

/**
 * Crée une notification avec BigTextStyle (Android)
 * Utile pour afficher des textes longs avec expand
 *
 * @param title Titre
 * @param shortText Texte court (collapsed)
 * @param longText Texte long (expanded)
 * @param data Données additionnelles
 * @param channelId Canal Android
 */
export async function scheduleBigTextNotification(
  title: string,
  shortText: string,
  longText: string,
  data?: Record<string, any>,
  channelId?: string
): Promise<string | null> {
  if (Platform.OS !== "android") {
    // Fallback iOS: utiliser le texte court
    return await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body: shortText,
        data: data || {},
      },
      trigger: null,
    });
  }

  try {
    const notificationId = await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body: shortText,
        data: data || {},
        ...(channelId && { channelId }),

        // @ts-ignore - Android-specific
        android: {
          style: "bigtext",
          bigText: longText,
        },
      },
      trigger: null,
    });

    log.success("big text style notification created", { notificationId });
    return notificationId;
  } catch (error) {
    log.error("create big text notification failed", { error });
    return null;
  }
}

/**
 * Crée une notification de progression (Android)
 * Utile pour afficher l'état d'un téléchargement, trajet en cours, etc.
 *
 * @param title Titre
 * @param body Message
 * @param progress Progression (0-100)
 * @param indeterminate Si true, affiche une barre infinie
 * @param data Données additionnelles
 * @param channelId Canal Android
 */
export async function scheduleProgressNotification(
  title: string,
  body: string,
  progress: number,
  indeterminate: boolean = false,
  data?: Record<string, any>,
  channelId?: string
): Promise<string | null> {
  if (Platform.OS !== "android") {
    // iOS ne supporte pas les progress bars natives
    return await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body: `${body} (${progress}%)`,
        data: data || {},
      },
      trigger: null,
    });
  }

  try {
    const notificationId = await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data: data || {},
        ...(channelId && { channelId }),

        // @ts-ignore - Android-specific
        android: {
          progress: {
            max: 100,
            current: Math.min(100, Math.max(0, progress)),
            indeterminate,
          },
          ongoing: progress < 100, // Empêche de swiper si pas terminé
        },
      },
      trigger: null,
    });

    log.success("progress notification created", { notificationId, progress });
    return notificationId;
  } catch (error) {
    log.error("create progress notification failed", { error });
    return null;
  }
}

/**
 * Nettoie les images en cache utilisées pour les notifications
 * À appeler périodiquement pour libérer de l'espace
 */
export async function cleanupNotificationImageCache(): Promise<void> {
  try {
    const files = await readDirectoryAsync(CACHE_DIR);

    // Supprimer les fichiers commençant par "notif_"
    let cleaned = 0;
    for (const file of files) {
      if (file.startsWith("notif_")) {
        await deleteAsync(`${CACHE_DIR}${file}`, {
          idempotent: true,
        });
        cleaned++;
      }
    }

    if (cleaned > 0) {
      log.info("notification images cleaned", { cleaned });
    }
  } catch (error) {
    log.error("cleanup notification image cache failed", { error });
  }
}

/**
 * Vérifie si une URL d'image est valide et accessible
 *
 * @param imageUrl URL à vérifier
 * @param timeout Timeout en ms
 * @returns true si l'image est accessible
 */
export async function isImageAccessible(
  imageUrl: string,
  timeout: number = 2000
): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(imageUrl, {
      method: "HEAD",
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    return response.ok;
  } catch (error) {
    return false;
  }
}
