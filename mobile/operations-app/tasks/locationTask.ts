// tasks/locationTask.ts
// Tâche en arrière-plan pour le tracking de localisation

import * as TaskManager from "expo-task-manager";
import * as Location from "expo-location";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { io, Socket } from "socket.io-client";

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
let flushInterval: NodeJS.Timeout | null = null;

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

    socket.on("disconnect", () => {
      console.log("[LocationTask] ⚠️ Socket déconnecté");
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
    await initSocket();
    // Attendre un peu pour la connexion
    await new Promise((resolve) => setTimeout(resolve, 1000));
    if (!socket || !socket.connected) {
      console.log("[LocationTask] ⚠️ Socket non disponible, positions mises en attente");
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

// Définir la tâche en arrière-plan
TaskManager.defineTask(LOCATION_TASK_NAME, async ({ data, error }) => {
  if (error) {
    console.error("[LocationTask] ❌ Erreur:", error);
    return;
  }

  if (data) {
    const { locations } = data as { locations: Location.LocationObject[] };

    // Initialiser le socket et le flush périodique au premier appel
    if (!socket) {
      await initSocket();
      startPeriodicFlush();
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
  }
});

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

