// mobile/operations-app/services/silentNotifications.ts
/**
 * Service de gestion des notifications silencieuses (data-only)
 * 
 * Phase 2.6 - Notifications Silencieuses
 * 
 * Permet de:
 * - Synchroniser données missions en arrière-plan
 * - Précharger cartes et itinéraires
 * - Mettre à jour profil chauffeur sans déranger
 * 
 * NOTE: Depuis SDK récents, expo-background-fetch est déprécié.
 * Cette implémentation n'en dépend plus en runtime pour éviter les warnings.
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import { getLogger } from "@/utils/logger";
import { scheduleMissionReminder } from "./localNotifications";
import { trackNotificationEvent } from "./notificationAnalytics";

const log = getLogger("SilentNotif");

let TaskManager: any = null;

try {
  TaskManager = require("expo-task-manager");
} catch (e) {
  getLogger("SilentNotif").warn("expo-task-manager not installed, background limited");
}

/**
 * Types de synchronisation disponibles
 */
export enum SyncType {
  MISSIONS = "missions",
  PROFILE = "profile",
  MAPS = "maps",
  CONFIG = "config",
}

/**
 * Résultat d'une synchronisation
 */
export enum BackgroundFetchResult {
  NewData = "NewData",
  NoData = "NoData",
  Failed = "Failed",
}

/**
 * Définition de la tâche de synchronisation background
 */
const BACKGROUND_SYNC_TASK = "background-data-sync";
let backgroundSyncConfigured = false;
let backgroundTaskDefined = false;

/**
 * Configure la synchronisation en arrière-plan
 * À appeler au démarrage de l'app
 */
export async function setupBackgroundSync(): Promise<void> {
  if (!TaskManager) {
    log.warn("background sync unavailable (task manager missing)");
    return;
  }
  if (backgroundSyncConfigured) {
    log.debug("background sync already configured (in-memory)");
    return;
  }

  try {
    // Le sync background est piloté par silent push + TaskManager.
    // Pas d'enregistrement BackgroundFetch (déprécié).
    backgroundSyncConfigured = true;
    log.success("background sync configured");
  } catch (error) {
    log.error("background sync configuration failed", { error });
  }
}

/**
 * Annule la synchronisation en arrière-plan
 */
export async function unregisterBackgroundSync(): Promise<void> {
  backgroundSyncConfigured = false;
  log.info("background sync disabled");
}

/**
 * Gère les notifications silencieuses (data-only)
 */
export async function handleSilentNotification(
  data: any
): Promise<BackgroundFetchResult> {
  const startTime = Date.now();

  try {
    log.info("silent notification received", { sync_type: data.sync_type });

    const syncType = data.sync_type as SyncType;
    const payload = data.payload || {};

    switch (syncType) {
      case SyncType.MISSIONS:
        await syncMissions(payload);
        break;

      case SyncType.PROFILE:
        await syncDriverProfile(payload);
        break;

      case SyncType.MAPS:
        await precacheMaps(payload);
        break;

      case SyncType.CONFIG:
        await syncAppConfig(payload);
        break;

      default:
        log.warn("unknown sync type", { sync_type: data.sync_type });
        return BackgroundFetchResult.NoData;
    }

    const duration = Date.now() - startTime;
    log.success("sync completed", { syncType, durationMs: duration });

    return BackgroundFetchResult.NewData;
  } catch (error) {
    log.error("silent notification failed", { error });
    return BackgroundFetchResult.Failed;
  }
}

/**
 * Synchronise les missions en arrière-plan
 */
async function syncMissions(payload: any): Promise<void> {
  try {
    const missions = payload.missions || [];

    if (missions.length === 0) {
      log.info("no missions to sync");
      return;
    }

    // Sauvegarder en local
    await AsyncStorage.setItem("cached_missions", JSON.stringify(missions));

    // Planifier rappels pour nouvelles missions
    let remindersScheduled = 0;
    for (const mission of missions) {
      if (mission.scheduled_time && mission.id) {
        const reminder = await scheduleMissionReminder(
          {
            id: mission.id,
            scheduled_time: mission.scheduled_time,
            pickup_location: mission.pickup_location || mission.pickup_address,
            dropoff_location: mission.dropoff_location || mission.dropoff_address,
            passenger_name: mission.passenger_name,
          },
          30
        );
        if (reminder) remindersScheduled++;
      }
    }

    log.success("missions synced, reminders scheduled", {
      missionsCount: missions.length,
      remindersScheduled,
    });
  } catch (error) {
    log.error("sync missions failed", { error });
    throw error;
  }
}

/**
 * Synchronise le profil chauffeur
 */
async function syncDriverProfile(payload: any): Promise<void> {
  try {
    const profile = payload.profile || {};

    if (Object.keys(profile).length === 0) {
      log.info("no profile to sync");
      return;
    }

    // Sauvegarder profil en local
    await AsyncStorage.setItem("cached_driver_profile", JSON.stringify(profile));

    // Mettre à jour stats si présentes
    if (payload.stats) {
      await AsyncStorage.setItem("cached_driver_stats", JSON.stringify(payload.stats));
    }

    log.success("driver profile synced");
  } catch (error) {
    log.error("sync profile failed", { error });
    throw error;
  }
}

/**
 * Précharge les cartes pour itinéraires
 */
async function precacheMaps(payload: any): Promise<void> {
  try {
    const routes = payload.routes || [];

    if (routes.length === 0) {
      log.info("no maps to precache");
      return;
    }

    // Sauvegarder itinéraires pour préchargement
    await AsyncStorage.setItem("cached_routes", JSON.stringify(routes));

    // Note: Le téléchargement des tiles de carte nécessiterait
    // une bibliothèque de cartes spécifique (ex: react-native-maps)
    // Pour l'instant, on sauvegarde juste les métadonnées

    log.success("routes cached", { count: routes.length });
  } catch (error) {
    log.error("precache maps failed", { error });
    throw error;
  }
}

/**
 * Synchronise la configuration de l'app
 */
async function syncAppConfig(payload: any): Promise<void> {
  try {
    const config = payload.config || {};

    if (Object.keys(config).length === 0) {
      log.info("no config to sync");
      return;
    }

    // Sauvegarder config en local
    await AsyncStorage.setItem("app_config", JSON.stringify(config));

    log.success("app config synced");
  } catch (error) {
    log.error("sync config failed", { error });
    throw error;
  }
}

/**
 * Récupère les missions cachées
 */
export async function getCachedMissions(): Promise<any[]> {
  try {
    const cached = await AsyncStorage.getItem("cached_missions");
    return cached ? JSON.parse(cached) : [];
  } catch (error) {
    log.error("read cached missions failed", { error });
    return [];
  }
}

/**
 * Récupère le profil chauffeur caché
 */
export async function getCachedDriverProfile(): Promise<any | null> {
  try {
    const cached = await AsyncStorage.getItem("cached_driver_profile");
    return cached ? JSON.parse(cached) : null;
  } catch (error) {
    log.error("read cached profile failed", { error });
    return null;
  }
}

/**
 * Récupère les stats chauffeur cachées
 */
export async function getCachedDriverStats(): Promise<any | null> {
  try {
    const cached = await AsyncStorage.getItem("cached_driver_stats");
    return cached ? JSON.parse(cached) : null;
  } catch (error) {
    log.error("read cached stats failed", { error });
    return null;
  }
}

/**
 * Nettoie les données cachées
 */
export async function clearCachedData(): Promise<void> {
  try {
    await AsyncStorage.multiRemove([
      "cached_missions",
      "cached_driver_profile",
      "cached_driver_stats",
      "cached_routes",
      "app_config",
    ]);
    log.success("cache cleared");
  } catch (error) {
    log.error("clear cache failed", { error });
  }
}

/**
 * Définit la tâche TaskManager pour sync périodique
 * À appeler au niveau global (avant registerRootComponent)
 */
export function defineBackgroundSyncTask(): void {
  if (!TaskManager) {
    return;
  }
  if (backgroundTaskDefined) {
    return;
  }
  backgroundTaskDefined = true;
  TaskManager.defineTask(BACKGROUND_SYNC_TASK, async () => {
    try {
      log.info("background sync task started");

      // Ici, on pourrait faire un appel API pour récupérer les données
      // Pour l'instant, on considère que les données arrivent via silent notifications

      // Vérifier si des données sont en attente de sync
      const hasCachedData = await AsyncStorage.getItem("cached_missions");

      if (hasCachedData) {
        log.info("cached data available");
        return;
      }

      log.info("no data pending");
      return;
    } catch (error) {
      log.error("background task failed", { error });
      return;
    }
  });
}
