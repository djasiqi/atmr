// tasks/locationTask.ts
// Tâche en arrière-plan pour le tracking de localisation

import * as Location from "expo-location";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { getLogger } from "@/utils/logger";
import { enqueueLocationBatch, type QueuedLocation } from "../services/locationQueue";
import { MissionStateManager } from "../services/missionState";
import { resolveMissionContext } from "../services/locationMissionContext";

const log = getLogger("LocationTask");
const trackLog = getLogger("TRACK");

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

/** 3 lectures max (1 + 2 retries) pour laisser AsyncStorage s’aligner après setDriverId. */
const DRIVER_ID_RETRY_DELAY_MS = 120;
const DRIVER_ID_MAX_ATTEMPT_INDEX = 2;

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
    await MissionStateManager.ensureHydrated({ skipNetwork: true });

    let driverIdStr: string | null = null;
    for (let attempt = 0; attempt <= DRIVER_ID_MAX_ATTEMPT_INDEX; attempt++) {
      driverIdStr = await AsyncStorage.getItem("driver_id");
      if (driverIdStr) break;
      trackLog.warn("driver_id missing", {
        retryAttempt: attempt,
        bufferSize: positionBuffer.length,
      });
      if (attempt < DRIVER_ID_MAX_ATTEMPT_INDEX) {
        await new Promise((r) => setTimeout(r, DRIVER_ID_RETRY_DELAY_MS));
      }
    }
    if (!driverIdStr) {
      return;
    }

    const driverId = parseInt(driverIdStr, 10);
    if (!Number.isFinite(driverId) || driverId <= 0) {
      trackLog.warn("driver_id missing", {
        retryAttempt: DRIVER_ID_MAX_ATTEMPT_INDEX,
        bufferSize: positionBuffer.length,
        invalid: true,
      });
      return;
    }

    const batch = [...positionBuffer];
    positionBuffer = [];
    const { missionId, mode } = resolveMissionContext();

    const queued: QueuedLocation[] = batch.map((p) => ({
      latitude: p.latitude,
      longitude: p.longitude,
      speed: p.speed,
      heading: p.heading,
      accuracy: p.accuracy,
      timestamp: p.timestamp,
      driver_id: driverId,
      location_mode: mode,
      mission_id: missionId,
      recorded_at: new Date(p.timestamp || Date.now()).toISOString(),
      sent_at: new Date().toISOString(),
      is_background: true,
    }));

    await enqueueLocationBatch(queued);

    log.info("positions enqueued", { count: queued.length, driverId });

    const latest = queued[queued.length - 1];
    if (latest) {
      (async () => {
        try {
          const { updateDriverLocation } = await import("../services/api");
          await Promise.race([
            updateDriverLocation({
              lat: latest.latitude,
              lon: latest.longitude,
              speed_mps: latest.speed,
              heading: latest.heading,
              accuracy_m: latest.accuracy,
              recorded_at:
                latest.recorded_at ||
                new Date(latest.timestamp || Date.now()).toISOString(),
              sent_at: new Date().toISOString(),
              is_background: true,
              location_mode: latest.location_mode,
              mission_id: latest.mission_id ?? null,
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

