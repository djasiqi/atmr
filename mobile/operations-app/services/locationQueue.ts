// services/locationQueue.ts
// ✅ P2-1: Mode Offline Mobile - Persister queue GPS + resync au reconnect

import AsyncStorage from "@react-native-async-storage/async-storage";

const LOCATION_QUEUE_KEY = "@atmr:location_queue";
const MAX_QUEUE_SIZE = 1000; // Limiter la taille de la queue pour éviter l'overflow

export interface QueuedLocation {
  latitude: number;
  longitude: number;
  speed: number;
  heading: number;
  accuracy: number;
  timestamp: number;
  driver_id: number;
}

/**
 * Ajoute une position à la queue persistante.
 * ✅ P2: Bug #8 - Filtrer positions > 24h
 * @param location Position GPS à ajouter
 */
export async function enqueueLocation(
  location: QueuedLocation
): Promise<void> {
  try {
    const queue = await getLocationQueue();
    
    // ✅ P2: Bug #8 - Filtrer les positions > 24h
    const now = Date.now();
    const MAX_AGE_MS = 24 * 60 * 60 * 1000;  // 24 heures
    
    const validQueue = queue.filter(loc => {
      const age = now - loc.timestamp;
      return age < MAX_AGE_MS;
    });
    
    // Ajouter la nouvelle position
    validQueue.push(location);

    // Limiter la taille de la queue
    if (validQueue.length > MAX_QUEUE_SIZE) {
      // Garder les positions les plus récentes
      validQueue.splice(0, validQueue.length - MAX_QUEUE_SIZE);
    }

    // Logger si des positions ont été expirées
    const expiredCount = queue.length - validQueue.length + 1;  // +1 car on vient d'ajouter
    if (expiredCount > 0) {
      console.log(`🗑️ [locationQueue] ${expiredCount - 1} positions expirées (> 24h)`);
    }

    await AsyncStorage.setItem(LOCATION_QUEUE_KEY, JSON.stringify(validQueue));
  } catch (error) {
    console.error("❌ [locationQueue] Erreur lors de l'ajout à la queue:", error);
  }
}

/**
 * Récupère toutes les positions en queue.
 * @returns Liste des positions en queue
 */
export async function getLocationQueue(): Promise<QueuedLocation[]> {
  try {
    const data = await AsyncStorage.getItem(LOCATION_QUEUE_KEY);
    if (!data) {
      return [];
    }
    return JSON.parse(data) as QueuedLocation[];
  } catch (error) {
    console.error("❌ [locationQueue] Erreur lors de la lecture de la queue:", error);
    return [];
  }
}

/**
 * Supprime les positions envoyées avec succès de la queue.
 * @param sentLocations Positions qui ont été envoyées avec succès
 */
export async function removeSentLocations(
  sentLocations: QueuedLocation[]
): Promise<void> {
  try {
    const queue = await getLocationQueue();
    const sentTimestamps = new Set(
      sentLocations.map((loc) => loc.timestamp)
    );

    const remaining = queue.filter(
      (loc) => !sentTimestamps.has(loc.timestamp)
    );

    await AsyncStorage.setItem(LOCATION_QUEUE_KEY, JSON.stringify(remaining));
  } catch (error) {
    console.error(
      "❌ [locationQueue] Erreur lors de la suppression de la queue:",
      error
    );
  }
}

/**
 * Vide complètement la queue.
 */
export async function clearLocationQueue(): Promise<void> {
  try {
    await AsyncStorage.removeItem(LOCATION_QUEUE_KEY);
  } catch (error) {
    console.error("❌ [locationQueue] Erreur lors du vidage de la queue:", error);
  }
}

/**
 * Récupère le nombre de positions en queue.
 * @returns Nombre de positions en queue
 */
export async function getQueueSize(): Promise<number> {
  const queue = await getLocationQueue();
  return queue.length;
}

/**
 * ✅ P2-1: Synchronise la queue GPS avec le serveur via Socket.IO.
 * Appelée automatiquement lors de la reconnexion.
 * @param socket Instance Socket.IO connectée
 */
export async function syncLocationQueue(socket: any): Promise<void> {
  try {
    const queue = await getLocationQueue();
    if (queue.length === 0) {
      console.log("📦 [locationQueue] Queue vide, pas de resync nécessaire");
      return;
    }

    if (!socket || !socket.connected) {
      console.warn("📦 [locationQueue] Socket non connecté, resync reporté");
      return;
    }

    console.log(`📦 [locationQueue] Resync: ${queue.length} positions en queue`);

    // Grouper par driver_id (au cas où)
    const byDriver = new Map<number, QueuedLocation[]>();
    for (const loc of queue) {
      const driverQueue = byDriver.get(loc.driver_id) || [];
      driverQueue.push(loc);
      byDriver.set(loc.driver_id, driverQueue);
    }

    // Envoyer chaque groupe de positions
    for (const [driverId, locations] of byDriver.entries()) {
      try {
        const payload = {
          positions: locations.map((loc) => ({
            latitude: loc.latitude,
            longitude: loc.longitude,
            speed: loc.speed,
            heading: loc.heading,
            accuracy: loc.accuracy,
            timestamp: loc.timestamp,
          })),
          driver_id: driverId,
        };

        console.log(
          `📤 [locationQueue] Envoi batch resync: ${locations.length} positions pour driver ${driverId}`
        );

        // ✅ P1: Attendre ACK du serveur avant de supprimer
        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => {
            reject(new Error('Timeout waiting for ACK'));
          }, 5000);
          
          socket.emit("driver_location_batch", payload, (ack: any) => {
            clearTimeout(timeout);
            if (ack?.success) {
              resolve();
            } else {
              reject(new Error(ack?.error || 'ACK failed'));
            }
          });
        });

        // ✅ P1: Supprimer SEULEMENT après ACK de succès
        await removeSentLocations(locations);

        console.log(
          `✅ [locationQueue] Resync réussi: ${locations.length} positions confirmées et supprimées`
        );
      } catch (error) {
        console.error(
          `❌ [locationQueue] Erreur resync pour driver ${driverId}:`,
          error
        );
        // ✅ P1: Ne PAS supprimer, sera retenté plus tard
      }
    }

    const remaining = await getQueueSize();
    if (remaining > 0) {
      console.log(
        `⚠️ [locationQueue] ${remaining} positions restantes en queue après resync`
      );
    } else {
      console.log("✅ [locationQueue] Queue vidée avec succès");
    }
  } catch (error) {
    console.error("❌ [locationQueue] Erreur lors de la resync:", error);
  }
}

