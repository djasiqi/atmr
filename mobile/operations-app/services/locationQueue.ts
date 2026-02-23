// services/locationQueue.ts
// ✅ P2-1: Mode Offline Mobile - Persister queue GPS + resync au reconnect

import AsyncStorage from "@react-native-async-storage/async-storage";
import { getLogger } from "@/utils/logger";
import { getSocket, getSocketRole } from "./socket";

const log = getLogger("LocationQ");

const LOCATION_QUEUE_KEY = "@atmr:location_queue";
const MAX_QUEUE_SIZE = 1000; // Limiter la taille de la queue pour éviter l'overflow

// ✅ Client-side stabilisation (rate-limit + singleflight)
const RESYNC_CHUNK_SIZE = 50; // éviter des payloads énormes
const MIN_DELAY_BETWEEN_EMITS_MS = 5500; // serveur: ~1 event / 5s → on met une marge

// ✅ Compteur d'échecs batch consécutifs — au delà de 3, fallback individuel
let consecutiveBatchFailures = 0;
const MAX_BATCH_FAILURES_BEFORE_FALLBACK = 3;

let resyncInFlight: Promise<void> | null = null;
let nextAllowedEmitAt = 0; // timestamp (ms)

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function isRateLimitError(err: unknown): boolean {
  const msg =
    err instanceof Error ? err.message : typeof err === "string" ? err : "";
  return msg.toLowerCase().includes("rate limit");
}

function getRetryAfterSeconds(err: unknown): number | null {
  // on attache souvent retry_after sur l'Error (voir useLocation / ci-dessous)
  const ra =
    typeof err === "object" && err !== null ? (err as any).retry_after : null;
  if (typeof ra === "number" && Number.isFinite(ra) && ra > 0) return ra;
  return null;
}

/**
 * ✅ Résoudre le meilleur socket disponible.
 * Priorité: socket global (getSocket) > socket passé en paramètre.
 * Le socket global est mis à jour lors des reconnexions automatiques,
 * alors que le socket du hook React peut être obsolète.
 */
function resolveSocket(socketParam: any): any {
  if (getSocketRole() !== "driver") {
    return null;
  }
  const globalSocket = getSocket();
  if (globalSocket && globalSocket.connected) {
    return globalSocket;
  }
  if (socketParam && socketParam.connected) {
    return socketParam;
  }
  return null;
}

/**
 * ✅ Fallback: envoyer les positions une par une via driver_location (sans ACK)
 * quand le batch avec callback échoue systématiquement.
 */
function emitIndividualFallback(socket: any, payload: any): void {
  const positions = payload.positions || [];
  const driverId = payload.driver_id;
  for (const pos of positions) {
    socket.emit("driver_location", {
      latitude: pos.latitude,
      longitude: pos.longitude,
      speed: pos.speed,
      heading: pos.heading,
      accuracy: pos.accuracy,
      timestamp: pos.timestamp,
      driver_id: driverId,
    });
  }
  log.info("individual fallback sent", { count: positions.length });
}

async function emitBatchWithAck(socketParam: any, payload: any): Promise<void> {
  // throttle global: éviter d'enchaîner plusieurs emits trop vite (rate limit server)
  const now = Date.now();
  if (now < nextAllowedEmitAt) {
    const waitMs = nextAllowedEmitAt - now;
    throw Object.assign(new Error("Rate limit exceeded"), {
      retry_after: Math.ceil(waitMs / 1000),
    });
  }

  // ✅ Résoudre le meilleur socket (global > param)
  const socket = resolveSocket(socketParam);
  if (!socket) {
    throw new Error("Socket not connected");
  }

  // ✅ Si trop de failures batch consécutives, utiliser le fallback individuel
  if (consecutiveBatchFailures >= MAX_BATCH_FAILURES_BEFORE_FALLBACK) {
    log.warn("batch failures fallback", { consecutiveBatchFailures });
    emitIndividualFallback(socket, payload);
    consecutiveBatchFailures = 0; // Reset pour réessayer le batch plus tard
    nextAllowedEmitAt = Date.now() + MIN_DELAY_BETWEEN_EMITS_MS;
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      consecutiveBatchFailures++;
      reject(new Error("Timeout waiting for ACK"));
    }, 15000);

    socket.emit("driver_location_batch", payload, (ack: any) => {
      clearTimeout(timeout);
      if (ack?.success) {
        consecutiveBatchFailures = 0; // Reset sur succès
        resolve();
      } else {
        consecutiveBatchFailures++;
        const e = new Error(ack?.error || "ACK failed");
        if (typeof ack?.retry_after === "number") {
          (e as any).retry_after = ack.retry_after;
        }
        reject(e);
      }
    });
  });

  // respecter un minimum d'espacement entre emits
  nextAllowedEmitAt = Date.now() + MIN_DELAY_BETWEEN_EMITS_MS;
}

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
      log.info("expired positions dropped", { count: expiredCount - 1 });
    }

    await AsyncStorage.setItem(LOCATION_QUEUE_KEY, JSON.stringify(validQueue));
  } catch (error) {
    log.error("enqueue failed", { error });
  }
}

/**
 * Ajoute plusieurs positions en une seule opération AsyncStorage.
 * Utilisé par la tâche background pour éviter N lectures/écritures séquentielles.
 */
export async function enqueueLocationBatch(
  locations: QueuedLocation[]
): Promise<void> {
  if (locations.length === 0) return;
  try {
    const queue = await getLocationQueue();
    const now = Date.now();
    const MAX_AGE_MS = 24 * 60 * 60 * 1000;

    const merged = [...queue, ...locations].filter(
      (loc) => now - loc.timestamp < MAX_AGE_MS
    );

    if (merged.length > MAX_QUEUE_SIZE) {
      merged.splice(0, merged.length - MAX_QUEUE_SIZE);
    }

    await AsyncStorage.setItem(LOCATION_QUEUE_KEY, JSON.stringify(merged));
    log.info("batch enqueued", { added: locations.length, total: merged.length });
  } catch (error) {
    log.error("batch enqueue failed", { error });
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
    log.error("queue read failed", { error });
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
    log.error("remove sent failed", { error });
  }
}

/**
 * Vide complètement la queue.
 */
export async function clearLocationQueue(): Promise<void> {
  try {
    await AsyncStorage.removeItem(LOCATION_QUEUE_KEY);
  } catch (error) {
    log.error("clear queue failed", { error });
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
  // ✅ Singleflight: si un resync est déjà en cours, réutiliser la promesse
  if (resyncInFlight) {
    return await resyncInFlight;
  }

  resyncInFlight = (async () => {
    const queue = await getLocationQueue();
    if (queue.length === 0) {
      log.info("queue empty no resync");
      return;
    }

    if (!socket || !socket.connected) {
      log.warn("socket not connected resync deferred");
      throw new Error("Socket not connected");
    }

    // Important: `queue` est un snapshot. Pendant le resync, de nouvelles positions peuvent
    // être ajoutées (ex: task background) → la queue peut rester non vide, c'est normal.
    log.info("resync snapshot", { count: queue.length });

    // Grouper par driver_id (au cas où)
    const byDriver = new Map<number, QueuedLocation[]>();
    for (const loc of queue) {
      const driverQueue = byDriver.get(loc.driver_id) || [];
      driverQueue.push(loc);
      byDriver.set(loc.driver_id, driverQueue);
    }

    // Envoyer chaque groupe de positions
    for (const [driverId, locations] of byDriver.entries()) {
      // chunker pour limiter le payload et permettre un retry progressif
      for (let i = 0; i < locations.length; i += RESYNC_CHUNK_SIZE) {
        const chunk = locations.slice(i, i + RESYNC_CHUNK_SIZE);
        const payload = {
          positions: chunk.map((loc) => ({
            latitude: loc.latitude,
            longitude: loc.longitude,
            speed: loc.speed,
            heading: loc.heading,
            accuracy: loc.accuracy,
            timestamp: loc.timestamp,
          })),
          driver_id: driverId,
        };

        log.info("batch resync send", {
          chunkLength: chunk.length,
          chunkIndex: Math.floor(i / RESYNC_CHUNK_SIZE) + 1,
          totalChunks: Math.ceil(locations.length / RESYNC_CHUNK_SIZE),
          driverId,
        });

        try {
          // si on vient de faire un emit, attendre un peu (marge)
          const now = Date.now();
          if (now < nextAllowedEmitAt) {
            await sleep(nextAllowedEmitAt - now);
          }

          await emitBatchWithAck(socket, payload);
          await removeSentLocations(chunk);
        } catch (error) {
          const retryAfter = getRetryAfterSeconds(error);
          if (isRateLimitError(error) || retryAfter) {
            const seconds = retryAfter || 5;
            nextAllowedEmitAt = Math.max(
              nextAllowedEmitAt,
              Date.now() + (seconds + 1) * 1000
            );
          }

          log.error("resync failed", { driverId, error });
          // ✅ IMPORTANT: propager l'erreur pour que le caller backoff correctement
          throw error;
        }
      }
    }

    const remaining = await getQueueSize();
    if (remaining > 0) {
      log.info("remaining after resync", { remaining });
      // Ne pas lever d'erreur: en background/offline/ratelimit, ou si la queue reçoit de
      // nouvelles positions pendant l'envoi, elle peut légitimement ne pas être vide.
      return;
    } else {
      log.success("queue flushed");
    }
  })()
    .finally(() => {
      resyncInFlight = null;
    });

  return await resyncInFlight;
}

