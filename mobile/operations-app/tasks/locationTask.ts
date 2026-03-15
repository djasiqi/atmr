// tasks/locationTask.ts
// Tâche en arrière-plan pour le tracking de localisation

import * as Location from "expo-location";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { getLogger } from "@/utils/logger";
import { enqueueLocationBatch, type QueuedLocation } from "../services/locationQueue";

const log = getLogger("LocationTask");

// Vérifier si le module natif est disponible
let TaskManager: any = null;
try {
  TaskManager = require("expo-task-manager");
  // Vérifier que le module est bien chargé (pas juste un stub)
  if (TaskManager && typeof TaskManager.defineTask === "function") {
    log.success("task manager loaded");
  } else {
    log.warn("task manager methods unavailable");
    TaskManager = null;
  }
} catch (error: any) {
  log.warn("expo-task-manager unavailable", { message: error?.message || error });
  log.info("native rebuild required");
  // En mode Expo Go ou sans rebuild, on exporte quand même les constantes
  // mais la tâche ne sera pas active
}

const LOCATION_TASK_NAME = "background-location-task";

// Buffer pour les positions (batching)
let positionBuffer: Array<{
  latitude: number;
  longitude: number;
  speed: number;
  heading: number;
  accuracy: number;
  timestamp: number;
}> = [];

const BATCH_SIZE = 3;
const BATCH_INTERVAL_MS = 15000; // 15 secondes
let flushInterval: ReturnType<typeof setInterval> | null = null;

// Envoyer le batch de positions
async function flushPositionBatch() {
  if (positionBuffer.length === 0) {
    return;
  }

  try {
    // Récupérer le driver_id depuis le storage
    const driverIdStr = await AsyncStorage.getItem("driver_id");
    if (!driverIdStr) {
      log.warn("driver id not found");
      positionBuffer = [];
      return;
    }

    const driverId = parseInt(driverIdStr, 10);
    const batch = [...positionBuffer];
    positionBuffer = []; // Clear buffer

    const queued: QueuedLocation[] = batch.map((p) => ({
      latitude: p.latitude,
      longitude: p.longitude,
      speed: p.speed,
      heading: p.heading,
      accuracy: p.accuracy,
      timestamp: p.timestamp,
      driver_id: driverId,
    }));

    // Une seule opération AsyncStorage au lieu de N (évite Background ANR)
    await enqueueLocationBatch(queued);

    log.info("positions enqueued", { count: queued.length, driverId });

    // ✅ Fallback HTTP en arrière-plan : fire-and-forget pour éviter ANR.
    // Android ~5–10s max pour les tâches background — on ne bloque pas.
    const latest = queued[queued.length - 1];
    if (latest) {
      (async () => {
        try {
          const { updateDriverLocation } = await import("../services/api");
          await Promise.race([
            updateDriverLocation({
              latitude: latest.latitude,
              longitude: latest.longitude,
              speed: latest.speed,
              heading: latest.heading,
              accuracy: latest.accuracy,
              timestamp: latest.timestamp,
            }),
            new Promise<never>((_, reject) =>
              setTimeout(() => reject(new Error("timeout")), 5000)
            ),
          ]);
          log.info("position sent via HTTP (background fallback)", { driverId });
        } catch (e: any) {
          log.warn("HTTP location fallback failed", {
            error: e?.message ?? String(e),
            driverId,
          });
        }
      })();
    }
  } catch (error) {
    log.error("enqueue batch failed", { error });
  }
}

// Variable pour indiquer si la tâche est enregistrée
let taskRegistered = false;

// ✅ PROTECTION : Vérifier si la tâche est déjà définie (évite les doubles appels)
// Cette protection est importante car React StrictMode peut monter/démonter les composants deux fois
let taskDefinitionAttempted = false;

// Définir la tâche en arrière-plan (uniquement si TaskManager est disponible)
// Workaround expo/expo#25325 : en __DEV__, éviter defineTask pour stopper la boucle de reload
if (TaskManager && !taskDefinitionAttempted && !__DEV__) {
  taskDefinitionAttempted = true;
  try {
    // Définir la tâche directement (defineTask est idempotent mais on protège quand même)
    TaskManager.defineTask(LOCATION_TASK_NAME, async ({ data, error }: { data?: { locations: Location.LocationObject[] }; error?: Error }) => {
    log.info("task called");

    if (error) {
      log.error("task error", { error });
      return;
    }

    if (data) {
      const { locations } = data;
      log.info("locations received", { locations });

      // Récupérer le driver_id
      try {
        const driverId = await AsyncStorage.getItem("driver_id");
        log.info("driver id fetched", { driverId });
      } catch (e) {
        log.warn("driver id fetch error", { error: e });
      }

      for (const location of locations) {
        const { latitude, longitude, speed, heading, accuracy } = location.coords;
        const timestamp = location.timestamp || Date.now();

        // Ajouter au buffer
        positionBuffer.push({
          latitude: Number(latitude),
          longitude: Number(longitude),
          speed: Number(speed || 0),
          heading: Number(heading || 0),
          accuracy: Number(accuracy || 10),
          timestamp,
        });

        log.info("position buffered", { bufferLength: positionBuffer.length, batchSize: BATCH_SIZE });

        // Flush si buffer plein
        if (positionBuffer.length >= BATCH_SIZE) {
          await flushPositionBatch();
        }
      }

      // ✅ Envoyer le batch restant à la fin (mode event-driven, pas besoin d'interval)
      // Android appelle la task par batch de locations, on peut envoyer directement
      if (positionBuffer.length > 0) {
        log.info("final batch flush", { remaining: positionBuffer.length });
        await flushPositionBatch();
      }
    }
    });
    taskRegistered = true;
    log.success("task registered");
  } catch (error: any) {
    log.error("task registration failed", { error: error?.message || error });
    taskRegistered = false;
    taskDefinitionAttempted = false; // Permettre de réessayer en cas d'erreur
  }
} else if (!TaskManager) {
  log.warn("task manager unavailable");
} else {
  log.info("task definition skip duplicate");
}

// Fonction pour vérifier si la tâche est enregistrée
export function isTaskRegistered(): boolean {
  return taskRegistered && TaskManager !== null;
}

// Exporter TaskManager pour utilisation dans useLocation
export { TaskManager };

// Démarrer le flush périodique
function startPeriodicFlush() {
  if (flushInterval) {
    return; // Déjà démarré
  }

  flushInterval = setInterval(async () => {
    log.info("periodic flush", { bufferLength: positionBuffer.length });
    await flushPositionBatch();
  }, BATCH_INTERVAL_MS);
}

// Arrêter le flush périodique
function stopPeriodicFlush() {
  if (flushInterval) {
    clearInterval(flushInterval);
    flushInterval = null;
  }
}

// Ne pas initialiser automatiquement - sera fait par la tâche quand nécessaire
// initSocket() et startPeriodicFlush() seront appelés quand la première position arrive

export { LOCATION_TASK_NAME, startPeriodicFlush, stopPeriodicFlush };

