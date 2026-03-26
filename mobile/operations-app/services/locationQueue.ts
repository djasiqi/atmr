// services/locationQueue.ts
// ✅ P2-1: Mode Offline Mobile - Persister queue GPS + resync au reconnect

import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Crypto from "expo-crypto";
import { getLogger } from "@/utils/logger";
import { updateDriverLocation } from "./api";
import { getSocket, getSocketRole } from "./socket";

const log = getLogger("LocationQ");
const trackLog = getLogger("TRACK");

const LOCATION_QUEUE_KEY = "@atmr:location_queue";
const MAX_QUEUE_SIZE = 1000; // hard-cap globale
const MAX_MISSION_QUEUE_SIZE = 20; // v2.1: mission_live garde un buffer court

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
  const positions = (payload.positions || []).filter(
    (pos: any) => normalizeMode(pos.location_mode) !== "availability_presence"
  );
  const driverId = payload.driver_id;
  for (const pos of positions) {
    const timestampIso = new Date(pos.timestamp || Date.now()).toISOString();
    socket.emit("driver_location", {
      latitude: pos.latitude,
      longitude: pos.longitude,
      speed: pos.speed,
      heading: pos.heading,
      accuracy: pos.accuracy,
      timestamp: pos.timestamp,
      location_mode: normalizeMode(pos.location_mode) || "availability_presence",
      recorded_at: pos.recorded_at || timestampIso,
      sent_at: pos.sent_at || new Date().toISOString(),
      is_background: pos.is_background ?? false,
      mission_id: pos.mission_id ?? null,
      driver_id: driverId,
      location_event_id: pos.location_event_id,
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
  // IMPORTANT: throw pour que le caller ne retire PAS les positions (pas d'ACK)
  if (consecutiveBatchFailures >= MAX_BATCH_FAILURES_BEFORE_FALLBACK) {
    log.warn("batch failures fallback", { consecutiveBatchFailures });
    emitIndividualFallback(socket, payload);
    consecutiveBatchFailures = 0; // Reset pour réessayer le batch plus tard
    nextAllowedEmitAt = Date.now() + MIN_DELAY_BETWEEN_EMITS_MS;
    throw new Error("Fallback sent without ACK - positions kept in queue");
  }

  const canonicalPositions = (payload.positions || [])
    .filter(
      (pos: any) => normalizeMode(pos.location_mode) !== "availability_presence"
    )
    .map((pos: any) => {
      const ts = pos.timestamp || Date.now();
      return {
        latitude: pos.latitude,
        longitude: pos.longitude,
        speed: pos.speed ?? 0,
        heading: pos.heading ?? 0,
        accuracy: pos.accuracy ?? 0,
        timestamp: ts,
        location_mode: normalizeMode(pos.location_mode) || "availability_presence",
        recorded_at: pos.recorded_at || new Date(ts).toISOString(),
        sent_at: pos.sent_at || new Date().toISOString(),
        is_background: pos.is_background ?? false,
        mission_id: pos.mission_id ?? null,
        location_event_id: pos.location_event_id,
      };
    });

  if (canonicalPositions.length === 0) {
    return;
  }

  // Garantie stricte : aucune position sans champs requis (évite payload cassé en prod)
  for (let i = 0; i < canonicalPositions.length; i++) {
    const p = canonicalPositions[i];
    if (!p.location_mode || !p.recorded_at) {
      log.error("position missing required fields before emit", {
        index: i,
        has_location_mode: !!p.location_mode,
        has_recorded_at: !!p.recorded_at,
        sample: { latitude: p.latitude, longitude: p.longitude },
      });
      throw new Error(`Position ${i} missing location_mode or recorded_at`);
    }
  }

  if (__DEV__ || canonicalPositions.length <= 2) {
    const sample = canonicalPositions[0];
    log.info("batch emit sample", {
      has_location_mode: !!sample.location_mode,
      has_recorded_at: !!sample.recorded_at,
      count: canonicalPositions.length,
    });
  }

  const filteredPayload = {
    ...payload,
    positions: canonicalPositions,
  };

  log.info("batch emit pre-send", {
    count: canonicalPositions.length,
    sample_keys: Object.keys(canonicalPositions[0] ?? {}),
    has_location_mode: !!(canonicalPositions[0] as any)?.location_mode,
    has_recorded_at: !!(canonicalPositions[0] as any)?.recorded_at,
  });

  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      consecutiveBatchFailures++;
      reject(new Error("Timeout waiting for ACK"));
    }, 15000);

    socket.emit("driver_location_batch", filteredPayload, (ack: any) => {
      clearTimeout(timeout);
      if (ack?.success) {
        const posCount = ack?.positions_count ?? 0;
        const rejCount = ack?.rejected_count ?? 0;
        if (posCount === 0 && rejCount > 0) {
          consecutiveBatchFailures++;
          reject(new Error("All positions rejected"));
          return;
        }
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
  location_mode?: "mission_live" | "availability_presence" | "passive_last_known";
  recorded_at?: string;
  sent_at?: string;
  is_background?: boolean;
  mission_id?: number | null;
  /** Identité stable du point (corrélation backend / retries) */
  location_event_id?: string;
}

function normalizeMode(
  mode?: string
): "mission_live" | "availability_presence" | "passive_last_known" {
  if (mode === "mission_live" || mode === "availability_presence" || mode === "passive_last_known") {
    return mode;
  }
  return "availability_presence";
}

const _UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function ensureLocationEventId(loc: QueuedLocation): string {
  const id = loc.location_event_id?.trim();
  if (id && _UUID_RE.test(id)) {
    return id;
  }
  if (typeof Crypto.randomUUID === "function") {
    return Crypto.randomUUID();
  }
  return `loc-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

function applyQueueRetention(queue: QueuedLocation[]): QueuedLocation[] {
  const missionItems = queue.filter((q) => normalizeMode(q.location_mode) === "mission_live");
  const missionTrimmed =
    missionItems.length > MAX_MISSION_QUEUE_SIZE
      ? missionItems.slice(missionItems.length - MAX_MISSION_QUEUE_SIZE)
      : missionItems;

  const latestAvailabilityByDriver = new Map<number, QueuedLocation>();
  for (const item of queue) {
    if (normalizeMode(item.location_mode) !== "availability_presence") continue;
    latestAvailabilityByDriver.set(item.driver_id, item);
  }

  const passiveItems = queue.filter(
    (q) => normalizeMode(q.location_mode) === "passive_last_known"
  );

  const merged = [...missionTrimmed, ...latestAvailabilityByDriver.values(), ...passiveItems].sort(
    (a, b) => a.timestamp - b.timestamp
  );
  if (merged.length > MAX_QUEUE_SIZE) {
    return merged.slice(merged.length - MAX_QUEUE_SIZE);
  }
  return merged;
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
    
    const normalized: QueuedLocation = {
      ...location,
      location_mode: normalizeMode(location.location_mode),
      recorded_at:
        location.recorded_at || new Date(location.timestamp || Date.now()).toISOString(),
      sent_at: location.sent_at || new Date().toISOString(),
      location_event_id: ensureLocationEventId(location),
    };
    validQueue.push(normalized);

    // Limiter la taille de la queue
    const retainedQueue = applyQueueRetention(validQueue);

    // Logger si des positions ont été expirées
    const expiredCount = queue.length - validQueue.length + 1;  // +1 car on vient d'ajouter
    if (expiredCount > 0) {
      log.info("expired positions dropped", { count: expiredCount - 1 });
    }

    await AsyncStorage.setItem(LOCATION_QUEUE_KEY, JSON.stringify(retainedQueue));
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
    ).map((loc) => ({
      ...loc,
      location_mode: normalizeMode(loc.location_mode),
      recorded_at: loc.recorded_at || new Date(loc.timestamp || Date.now()).toISOString(),
      sent_at: loc.sent_at || new Date().toISOString(),
      location_event_id: ensureLocationEventId(loc),
    }));
    const retained = applyQueueRetention(merged);

    await AsyncStorage.setItem(LOCATION_QUEUE_KEY, JSON.stringify(retained));
    if (__DEV__ && locations.length > 0) {
      const s = locations[locations.length - 1];
      trackLog.debug("enqueue", {
        mode: normalizeMode(s.location_mode),
        missionId: s.mission_id ?? null,
        isBackground: !!s.is_background,
        batchSize: locations.length,
      });
    }
    log.info("batch enqueued", { added: locations.length, total: retained.length });
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
    const raw = JSON.parse(data) as QueuedLocation[];
    return raw.map((loc) => ({
      ...loc,
      location_mode: normalizeMode(loc.location_mode),
      recorded_at: loc.recorded_at || new Date(loc.timestamp || Date.now()).toISOString(),
      sent_at: loc.sent_at || new Date().toISOString(),
    }));
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

const HTTP_FALLBACK_TIMEOUT_MS = 5000;
let flushHttpInFlight: Promise<void> | null = null;

/**
 * Envoie la dernière position en queue via HTTP (fallback quand socket déconnecté).
 * Maintient le chauffeur "en ligne" côté entreprise pendant l'arrière-plan.
 * Timeout 5s pour éviter ANR si l'app est en transition background.
 * Singleflight : évite les appels concurrents qui peuvent provoquer ANR.
 */
export async function flushLatestPositionViaHttp(): Promise<void> {
  if (flushHttpInFlight) return flushHttpInFlight;
  flushHttpInFlight = (async () => {
    try {
      const queue = await getLocationQueue();
      if (queue.length === 0) return;
      const latest = queue[queue.length - 1];
      const { updateDriverLocation } = await import("./api");
      const httpPromise = updateDriverLocation({
        lat: latest.latitude,
        lon: latest.longitude,
        speed_mps: latest.speed,
        heading: latest.heading,
        accuracy_m: latest.accuracy,
        recorded_at:
          latest.recorded_at || new Date(latest.timestamp || Date.now()).toISOString(),
        sent_at: new Date().toISOString(),
        is_background: true,
        // Contrat backend : mission explicite → mission_live (évite un défaut HTTP mission_live ambigu)
        location_mode:
          latest.mission_id != null
            ? "mission_live"
            : normalizeMode(latest.location_mode),
        mission_id: latest.mission_id ?? null,
      });
      const result = await Promise.race([
        httpPromise,
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("HTTP location timeout")), HTTP_FALLBACK_TIMEOUT_MS)
        ),
      ]);
      if ((result as any)?.ok === false) {
        log.warn("HTTP location fallback rejected", { message: (result as any)?.message });
      } else {
        log.info("latest position sent via HTTP fallback");
      }
    } catch (e: any) {
      log.warn("HTTP location fallback failed", { error: e?.message ?? String(e) });
    } finally {
      flushHttpInFlight = null;
    }
  })();
  return flushHttpInFlight;
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

    const presenceOnly = queue.filter(
      (loc) => normalizeMode(loc.location_mode) === "availability_presence"
    );
    if (presenceOnly.length > 0) {
      const latestByDriver = new Map<number, QueuedLocation>();
      for (const loc of presenceOnly) {
        latestByDriver.set(loc.driver_id, loc);
      }
      let presenceAccepted = true;
      for (const loc of latestByDriver.values()) {
        const presResult = await updateDriverLocation({
          lat: loc.latitude,
          lon: loc.longitude,
          speed_mps: loc.speed,
          heading: loc.heading,
          accuracy_m: loc.accuracy,
          recorded_at: loc.recorded_at || new Date(loc.timestamp || Date.now()).toISOString(),
          sent_at: new Date().toISOString(),
          is_background: true,
          location_mode: "availability_presence",
          mission_id: null,
          location_event_id: loc.location_event_id,
        });
        if (presResult?.ok === false) {
          log.warn("presence location rejected", { message: presResult.message, driver_id: loc.driver_id });
          presenceAccepted = false;
        }
      }
      if (presenceAccepted) {
        await removeSentLocations(presenceOnly);
      }
    }

    const socketQueue = queue.filter(
      (loc) => normalizeMode(loc.location_mode) !== "availability_presence"
    );
    if (socketQueue.length === 0) {
      return;
    }

    // Grouper par driver_id (au cas où)
    const byDriver = new Map<number, QueuedLocation[]>();
    for (const loc of socketQueue) {
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
          positions: chunk.map((loc) => {
            const ts = loc.timestamp || Date.now();
            return {
              latitude: loc.latitude,
              longitude: loc.longitude,
              speed: loc.speed ?? 0,
              heading: loc.heading ?? 0,
              accuracy: loc.accuracy ?? 0,
              timestamp: ts,
              location_mode: normalizeMode(loc.location_mode) || "availability_presence",
              recorded_at: loc.recorded_at || new Date(ts).toISOString(),
              sent_at: loc.sent_at || new Date().toISOString(),
              is_background: loc.is_background ?? false,
              mission_id: loc.mission_id ?? null,
              location_event_id: loc.location_event_id,
            };
          }),
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

