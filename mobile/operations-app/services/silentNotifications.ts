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
 * NOTE: Nécessite l'installation de expo-background-fetch:
 * npx expo install expo-background-fetch expo-task-manager
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Notifications from "expo-notifications";
import { Platform } from "react-native";
import { scheduleMissionReminder } from "./localNotifications";
import { trackNotificationEvent } from "./notificationAnalytics";

// Import conditionnel de BackgroundFetch (optionnel)
let BackgroundFetch: any = null;
let TaskManager: any = null;

try {
  BackgroundFetch = require("expo-background-fetch");
  TaskManager = require("expo-task-manager");
} catch (e) {
  console.warn(
    "⚠️ expo-background-fetch non installé. Fonctionnalités background limitées."
  );
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

/**
 * Configure la synchronisation en arrière-plan
 * À appeler au démarrage de l'app
 */
export async function setupBackgroundSync(): Promise<void> {
  if (!BackgroundFetch || !TaskManager) {
    console.warn(
      "⚠️ Background sync non disponible (expo-background-fetch manquant)"
    );
    return;
  }

  try {
    // Vérifier si déjà enregistrée
    const isRegistered = await TaskManager.isTaskRegisteredAsync(
      BACKGROUND_SYNC_TASK
    );

    if (!isRegistered) {
      // Enregistrer la tâche
      await BackgroundFetch.registerTaskAsync(BACKGROUND_SYNC_TASK, {
        minimumInterval: 15 * 60, // 15 minutes
        stopOnTerminate: false, // Continue après fermeture app
        startOnBoot: true, // Démarre au boot device
      });

      console.log("✅ Background sync configuré avec succès");
    } else {
      console.log("ℹ️ Background sync déjà configuré");
    }
  } catch (error) {
    console.error("❌ Erreur configuration background sync:", error);
  }
}

/**
 * Annule la synchronisation en arrière-plan
 */
export async function unregisterBackgroundSync(): Promise<void> {
  try {
    await BackgroundFetch.unregisterTaskAsync(BACKGROUND_SYNC_TASK);
    console.log("✅ Background sync désactivé");
  } catch (error) {
    console.error("❌ Erreur désactivation background sync:", error);
  }
}

/**
 * Gère les notifications silencieuses (data-only)
 */
export async function handleSilentNotification(
  data: any
): Promise<BackgroundFetchResult> {
  const startTime = Date.now();

  try {
    console.log("📥 Silent notification reçue:", data.sync_type);

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
        console.warn("⚠️ Type sync inconnu:", data.sync_type);
        return BackgroundFetchResult.NoData;
    }

    const duration = Date.now() - startTime;
    console.log(`✅ Sync ${syncType} complétée en ${duration}ms`);

    return BackgroundFetchResult.NewData;
  } catch (error) {
    console.error("❌ Erreur silent notification:", error);
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
      console.log("ℹ️ Aucune mission à synchroniser");
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

    console.log(
      `✅ ${missions.length} missions synchronisées, ${remindersScheduled} rappels planifiés`
    );
  } catch (error) {
    console.error("❌ Erreur sync missions:", error);
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
      console.log("ℹ️ Aucun profil à synchroniser");
      return;
    }

    // Sauvegarder profil en local
    await AsyncStorage.setItem("cached_driver_profile", JSON.stringify(profile));

    // Mettre à jour stats si présentes
    if (payload.stats) {
      await AsyncStorage.setItem("cached_driver_stats", JSON.stringify(payload.stats));
    }

    console.log("✅ Profil chauffeur synchronisé");
  } catch (error) {
    console.error("❌ Erreur sync profil:", error);
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
      console.log("ℹ️ Aucune carte à précharger");
      return;
    }

    // Sauvegarder itinéraires pour préchargement
    await AsyncStorage.setItem("cached_routes", JSON.stringify(routes));

    // Note: Le téléchargement des tiles de carte nécessiterait
    // une bibliothèque de cartes spécifique (ex: react-native-maps)
    // Pour l'instant, on sauvegarde juste les métadonnées

    console.log(`✅ ${routes.length} itinéraires cachés`);
  } catch (error) {
    console.error("❌ Erreur préchargement cartes:", error);
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
      console.log("ℹ️ Aucune config à synchroniser");
      return;
    }

    // Sauvegarder config en local
    await AsyncStorage.setItem("app_config", JSON.stringify(config));

    console.log("✅ Configuration app synchronisée");
  } catch (error) {
    console.error("❌ Erreur sync config:", error);
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
    console.error("❌ Erreur lecture missions cachées:", error);
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
    console.error("❌ Erreur lecture profil caché:", error);
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
    console.error("❌ Erreur lecture stats cachées:", error);
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
    console.log("✅ Cache nettoyé");
  } catch (error) {
    console.error("❌ Erreur nettoyage cache:", error);
  }
}

/**
 * Définit la tâche TaskManager pour sync périodique
 * À appeler au niveau global (avant registerRootComponent)
 */
export function defineBackgroundSyncTask(): void {
  TaskManager.defineTask(BACKGROUND_SYNC_TASK, async () => {
    try {
      console.log("🔄 Tâche background sync démarrée");

      // Ici, on pourrait faire un appel API pour récupérer les données
      // Pour l'instant, on considère que les données arrivent via silent notifications

      // Vérifier si des données sont en attente de sync
      const hasCachedData = await AsyncStorage.getItem("cached_missions");

      if (hasCachedData) {
        console.log("✅ Données cachées disponibles");
        return BackgroundFetch.BackgroundFetchResult.NewData;
      }

      console.log("ℹ️ Aucune donnée en attente");
      return BackgroundFetch.BackgroundFetchResult.NoData;
    } catch (error) {
      console.error("❌ Erreur tâche background:", error);
      return BackgroundFetch.BackgroundFetchResult.Failed;
    }
  });
}
