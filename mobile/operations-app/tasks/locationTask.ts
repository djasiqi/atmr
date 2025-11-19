// tasks/locationTask.ts
// Tâche en arrière-plan pour le tracking de localisation

import * as Location from "expo-location";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { io, Socket } from "socket.io-client";

// Vérifier si le module natif est disponible
let TaskManager: any = null;
try {
  TaskManager = require("expo-task-manager");
  // Vérifier que le module est bien chargé (pas juste un stub)
  if (TaskManager && typeof TaskManager.defineTask === "function") {
    console.log("[LocationTask] ✅ TaskManager chargé et disponible");
  } else {
    console.warn("[LocationTask] ⚠️ TaskManager chargé mais méthodes non disponibles");
    TaskManager = null;
  }
} catch (error: any) {
  console.warn("[LocationTask] ⚠️ expo-task-manager non disponible:", error?.message || error);
  console.warn("[LocationTask] ℹ️ Nécessite un rebuild natif (npx expo prebuild + rebuild)");
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

// Socket.IO pour l'envoi des positions
let socket: Socket | null = null;

// Initialiser le socket depuis le storage (utilise la même config que services/socket.ts)
async function initSocket() {
  try {
    const token = await AsyncStorage.getItem("token") || await AsyncStorage.getItem("authToken");
    if (!token) {
      console.log("[LocationTask] ⚠️ Pas de token, socket non initialisé");
      return;
    }

    // Utiliser la même logique de configuration que services/socket.ts
    const Constants = require("expo-constants");
    const { baseURL } = require("../services/api");
    
    // Flask-SocketIO vit à la racine (/socket.io). On enlève le suffixe /api ou /api/vX.
    const SOCKET_ORIGIN = baseURL.replace(/\/api(?:\/v\d+)?$/, "");
    const IS_SECURE = SOCKET_ORIGIN.startsWith("https://");
    const IS_DEV = __DEV__;

    socket = io(SOCKET_ORIGIN, {
      path: "/socket.io",
      auth: { token },
      extraHeaders: { Authorization: `Bearer ${token}` },
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
      forceNew: true,
      transports: IS_SECURE ? ["websocket"] : IS_DEV ? ["websocket", "polling"] : ["websocket"],
      upgrade: true,
      rememberUpgrade: true,
      secure: IS_SECURE,
    });

    socket.on("connect", () => {
      console.log("[LocationTask] ✅ Socket connecté");
    });

    socket.on("disconnect", (reason) => {
      console.log(`[LocationTask] ⚠️ Socket déconnecté (raison: ${reason})`);
    });

    socket.on("reconnect", (attemptNumber) => {
      console.log(`[LocationTask] 🔄 Socket reconnecté (tentative ${attemptNumber})`);
    });

    socket.on("connect_error", (error) => {
      console.error("[LocationTask] ❌ Erreur connexion socket:", error);
    });
  } catch (error) {
    console.error("[LocationTask] ❌ Erreur initialisation socket:", error);
  }
}

// Envoyer le batch de positions
async function flushPositionBatch() {
  if (positionBuffer.length === 0) {
    return;
  }

  // Initialiser le socket si nécessaire
  if (!socket || !socket.connected) {
    console.log("[LocationTask] 🔌 Socket non connecté, initialisation...");
    await initSocket();
    // Attendre un peu pour la connexion
    await new Promise((resolve) => setTimeout(resolve, 2000));
    if (!socket || !socket.connected) {
      console.log("[LocationTask] ⚠️ Socket non disponible après initialisation, positions mises en attente");
      console.log(`[LocationTask] 📊 État: socket=${!!socket}, connected=${socket?.connected}`);
      return;
    }
  }

  try {
    // Récupérer le driver_id depuis le storage
    const driverIdStr = await AsyncStorage.getItem("driver_id");
    if (!driverIdStr) {
      console.log("[LocationTask] ⚠️ Driver ID non trouvé");
      return;
    }

    const driverId = parseInt(driverIdStr, 10);
    const batch = [...positionBuffer];
    positionBuffer = []; // Clear buffer

    const payload = {
      positions: batch,
      driver_id: driverId,
    };

    console.log(`[LocationTask] 📍 Envoi batch: ${batch.length} positions, driver_id=${driverId}`);

    socket.emit("driver_location_batch", payload);
    console.log(`[LocationTask] ✅ Batch envoyé: ${batch.length} positions`);
  } catch (error) {
    console.error("[LocationTask] ❌ Erreur envoi batch:", error);
  }
}

// Variable pour indiquer si la tâche est enregistrée
let taskRegistered = false;

// ✅ PROTECTION : Vérifier si la tâche est déjà définie (évite les doubles appels)
// Cette protection est importante car React StrictMode peut monter/démonter les composants deux fois
let taskDefinitionAttempted = false;

// Définir la tâche en arrière-plan (uniquement si TaskManager est disponible)
if (TaskManager && !taskDefinitionAttempted) {
  taskDefinitionAttempted = true;
  try {
    // Définir la tâche directement (defineTask est idempotent mais on protège quand même)
    TaskManager.defineTask(LOCATION_TASK_NAME, async ({ data, error }: { data?: { locations: Location.LocationObject[] }; error?: Error }) => {
    // ✅ Log explicite pour diagnostiquer si la tâche est appelée en arrière-plan
    console.log(`[LocationTask] 🔔 Task appelée`);
    
    if (error) {
      console.log(`[LocationTask] ❌ Erreur dans la tâche :`, error);
      return;
    }

    if (data) {
      const { locations } = data;
      console.log(`[LocationTask] 📍 Locations reçues :`, JSON.stringify(locations));

      // Récupérer le driver_id
      try {
        const driverId = await AsyncStorage.getItem("driver_id");
        console.log(`[LocationTask] ℹ️ driver_id récupéré:`, driverId);
      } catch (e) {
        console.log(`[LocationTask] ⚠️ Erreur récupération driver_id:`, e);
      }

      // Initialiser le socket au premier appel (pas de setInterval, mode event-driven)
      if (!socket) {
        await initSocket();
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

        console.log(`[LocationTask] 📍 Position ajoutée au buffer: ${positionBuffer.length}/${BATCH_SIZE}`);

        // Flush si buffer plein
        if (positionBuffer.length >= BATCH_SIZE) {
          await flushPositionBatch();
        }
      }

      // ✅ Envoyer le batch restant à la fin (mode event-driven, pas besoin d'interval)
      // Android appelle la task par batch de locations, on peut envoyer directement
      if (positionBuffer.length > 0) {
        console.log(`[LocationTask] 📤 Envoi batch final (${positionBuffer.length} positions restantes)`);
        await flushPositionBatch();
      }
    }
    });
    taskRegistered = true;
    console.log(`[LocationTask] ✅ Tâche "${LOCATION_TASK_NAME}" enregistrée avec succès`);
  } catch (error: any) {
    console.error(`[LocationTask] ❌ Erreur lors de l'enregistrement de la tâche:`, error?.message || error);
    taskRegistered = false;
    taskDefinitionAttempted = false; // Permettre de réessayer en cas d'erreur
  }
} else if (!TaskManager) {
  console.warn("[LocationTask] TaskManager non disponible - la tâche en arrière-plan ne sera pas active");
} else {
  console.log(`[LocationTask] ℹ️ Tentative de définition de tâche déjà effectuée → skip (protection double appel)`);
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
    console.log(`[LocationTask] ⏰ Flush périodique (buffer=${positionBuffer.length})`);
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

