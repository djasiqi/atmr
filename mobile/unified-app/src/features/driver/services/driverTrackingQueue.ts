import AsyncStorage from "@react-native-async-storage/async-storage";
import { AppStateStatus } from "react-native";
import { DriverLocationPayload } from "../types";
import { sendDriverLocation } from "../api/driverHttp";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";

export type TrackingDeliveryState =
  | "queued"
  | "socket_emitted"
  | "backend_acked"
  | "retry_pending"
  | "dropped"
  | "expired"
  | "compacted";

export type DriverTrackingMode = "mission_live" | "availability_presence" | "observability_only";

export type DriverTrackingQueueItem = {
  id: string;
  sequenceId: number;
  trackingSessionId: string;
  batchId: string;
  positionId: string;
  missionId: number | null;
  appState: AppStateStatus;
  locationMode: DriverTrackingMode;
  payload: DriverLocationPayload;
  queuedAt: number;
  retryCount: number;
  deliveryState: TrackingDeliveryState;
  lastAttemptAt: number | null;
  ackedAt: number | null;
  lastError: string | null;
};

export type DriverTrackingFlushResult = {
  sent: number;
  backendAcked: number;
  socketEmitted: number;
  dropped: number;
  retried: number;
  queueDepth: number;
  flushPathUsed: "http_fallback" | "socket_batch";
  lastBackendAckAt: number | null;
  oldestItemAgeMs: number | null;
  networkProfile: "offline" | "poor" | "normal";
};

type DriverTrackingQueueSnapshot = {
  queueDepth: number;
  oldestQueuedAt: number | null;
  newestQueuedAt: number | null;
  oldestItemAgeMs: number | null;
};

const STORAGE_KEY = "driver_tracking_delivery_queue_v1";
const MAX_QUEUE_ITEMS = Number(process.env.EXPO_PUBLIC_DRIVER_TRACKING_QUEUE_MAX_ITEMS ?? "1000");
const MAX_QUEUE_AGE_MS = Number(process.env.EXPO_PUBLIC_DRIVER_TRACKING_QUEUE_MAX_AGE_MS ?? "86400000");
const MAX_RETRIES = 6;
// 24h aligné sur ops locationQueue — couvre les missions longues (nuit, TGV, aéroport).
const REPLAY_WINDOW_MS = 24 * 60 * 60 * 1000;
const SOCKET_ACK_DEFAULT_STALE_MS = 75_000;
const COMPACTION_MEDIUM_SPACING_MS = 20_000;
const COMPACTION_HIGH_SPACING_MS = 45_000;
const SPEED_DELTA_PIVOT_MS = 4;
const HEADING_DELTA_PIVOT_DEG = 30;
const SOCKET_BATCH_MAX_POINTS = Number(process.env.EXPO_PUBLIC_DRIVER_SOCKET_BATCH_MAX_POINTS ?? "20");
const DRAIN_BATCH_SIZE = Number(process.env.EXPO_PUBLIC_DRIVER_TRACKING_DRAIN_BATCH_SIZE ?? "50");
const DRAIN_INTERVAL_MS = Number(process.env.EXPO_PUBLIC_DRIVER_TRACKING_DRAIN_INTERVAL_MS ?? "500");
const MAX_DRAIN_POSITIONS_PER_MINUTE = Number(
  process.env.EXPO_PUBLIC_DRIVER_TRACKING_MAX_DRAIN_POSITIONS_PER_MINUTE ?? "1200"
);
const TRACKING_SESSION_TTL_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_TRACKING_SESSION_TTL_SEC ?? "1800"
) * 1000;
const SESSION_STORAGE_KEY = "driver_tracking_session_v1";
const inMemoryStorage = new Map<string, string>();

function nowMs(): number {
  return Date.now();
}

function safeJsonParse<T>(raw: string | null): T | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

function buildQueueId(): string {
  return `trk_${nowMs()}_${Math.random().toString(36).slice(2, 10)}`;
}

class DriverTrackingQueue {
  private loaded = false;
  private isFlushing = false;
  private items: DriverTrackingQueueItem[] = [];
  private sequenceCounter = 0;
  private trackingSessionId = "";
  private sessionCreatedAt = 0;
  private drainedInCurrentMinute = 0;
  private drainMinuteBucket = 0;

  private async readStorage(key: string): Promise<string | null> {
    const storage = AsyncStorage as unknown as {
      getItem?: (input: string) => Promise<string | null>;
    };
    if (typeof storage?.getItem === "function") {
      return storage.getItem(key);
    }
    return inMemoryStorage.get(key) ?? null;
  }

  private async writeStorage(key: string, value: string): Promise<void> {
    const storage = AsyncStorage as unknown as {
      setItem?: (input: string, output: string) => Promise<void>;
    };
    if (typeof storage?.setItem === "function") {
      await storage.setItem(key, value);
      return;
    }
    inMemoryStorage.set(key, value);
  }

  private async ensureLoaded() {
    if (this.loaded) return;
    const parsed = safeJsonParse<DriverTrackingQueueItem[]>(await this.readStorage(STORAGE_KEY));
    this.items = Array.isArray(parsed) ? parsed : [];
    this.items.sort((a, b) => (a.sequenceId ?? 0) - (b.sequenceId ?? 0));
    this.sequenceCounter = this.items.reduce((max, item) => Math.max(max, item.sequenceId ?? 0), 0);
    const sessionRaw = safeJsonParse<{ trackingSessionId: string; createdAt: number }>(
      await this.readStorage(SESSION_STORAGE_KEY)
    );
    if (
      sessionRaw?.trackingSessionId &&
      typeof sessionRaw.createdAt === "number" &&
      nowMs() - sessionRaw.createdAt < TRACKING_SESSION_TTL_MS
    ) {
      this.trackingSessionId = sessionRaw.trackingSessionId;
      this.sessionCreatedAt = sessionRaw.createdAt;
    } else {
      this.rotateTrackingSession();
    }
    this.loaded = true;
  }

  private async persist() {
    await this.writeStorage(STORAGE_KEY, JSON.stringify(this.items));
  }

  private async persistSession() {
    await this.writeStorage(
      SESSION_STORAGE_KEY,
      JSON.stringify({
        trackingSessionId: this.trackingSessionId,
        createdAt: this.sessionCreatedAt,
      })
    );
  }

  private rotateTrackingSession() {
    this.trackingSessionId = `trk_sess_${nowMs()}_${Math.random().toString(36).slice(2, 10)}`;
    this.sessionCreatedAt = nowMs();
  }

  private ensureSessionFresh() {
    if (!this.trackingSessionId || nowMs() - this.sessionCreatedAt >= TRACKING_SESSION_TTL_MS) {
      this.rotateTrackingSession();
      void this.persistSession();
    }
  }

  private isExpired(item: DriverTrackingQueueItem): boolean {
    const maxAge = Math.min(REPLAY_WINDOW_MS, MAX_QUEUE_AGE_MS);
    return nowMs() - item.queuedAt > maxAge;
  }

  private headingDeltaDegrees(previous?: number, next?: number): number {
    if (typeof previous !== "number" || typeof next !== "number") return 0;
    const raw = Math.abs(previous - next);
    return Math.min(raw, 360 - raw);
  }

  private shouldKeepAsPivot(previous: DriverTrackingQueueItem, current: DriverTrackingQueueItem): boolean {
    if (previous.locationMode !== current.locationMode) return true;
    const speedDelta = Math.abs((previous.payload.speed ?? 0) - (current.payload.speed ?? 0));
    if (speedDelta >= SPEED_DELTA_PIVOT_MS) return true;
    const headingDelta = this.headingDeltaDegrees(previous.payload.heading, current.payload.heading);
    return headingDelta >= HEADING_DELTA_PIVOT_DEG;
  }

  private compactQueueBySpacing(spacingMs: number): number {
    if (this.items.length <= 2) return 0;
    const sorted = [...this.items].sort((a, b) => a.queuedAt - b.queuedAt);
    const firstAnchor = sorted[0];
    const newestAnchor = sorted[sorted.length - 1];
    const kept: DriverTrackingQueueItem[] = [firstAnchor];
    let lastKept = firstAnchor;
    for (let index = 1; index < sorted.length - 1; index += 1) {
      const candidate = sorted[index]!;
      const isPivot = this.shouldKeepAsPivot(lastKept, candidate);
      const hasSpacing = candidate.queuedAt - lastKept.queuedAt >= spacingMs;
      if (isPivot || hasSpacing) {
        candidate.deliveryState = "compacted";
        kept.push(candidate);
        lastKept = candidate;
      }
    }
    if (kept[kept.length - 1]?.id !== newestAnchor.id) {
      kept.push(newestAnchor);
    }
    const dropped = sorted.length - kept.length;
    this.items = kept;
    return dropped;
  }

  private compactQueueIfNeeded() {
    if (!isFeatureEnabled("tracking_queue_compaction_enabled")) return;
    if (this.items.length <= MAX_QUEUE_ITEMS) return;
    const ratio = this.items.length / MAX_QUEUE_ITEMS;
    const spacingMs = ratio >= 1.25 ? COMPACTION_HIGH_SPACING_MS : COMPACTION_MEDIUM_SPACING_MS;
    const compactedCount = this.compactQueueBySpacing(spacingMs);
    if (compactedCount <= 0) return;
    emitDriverTelemetry("tracking.queue.compacted", {
      source: "driver.tracking.queue",
      compacted_count: compactedCount,
      queue_depth: this.items.length,
      spacing_ms: spacingMs,
    });
  }

  private trimIfNeeded() {
    if (this.items.length <= MAX_QUEUE_ITEMS) return;
    const overflow = this.items.length - MAX_QUEUE_ITEMS;
    const firstAnchor = this.items[0];
    const mutable = firstAnchor ? this.items.slice(1) : [...this.items];
    const dropped = mutable.splice(0, overflow);
    this.items = firstAnchor ? [firstAnchor, ...mutable] : mutable;
    dropped.forEach((item) => {
      emitDriverTelemetry("tracking.queue.dropped", {
        source: "driver.tracking.queue",
        mission_id: item.missionId,
        reason: "max_queue_size",
        queue_replay_window_ms: REPLAY_WINDOW_MS,
      });
    });
  }

  async enqueue(entry: {
    missionId: number | null;
    appState: AppStateStatus;
    locationMode: DriverTrackingMode;
    payload: DriverLocationPayload;
  }): Promise<DriverTrackingQueueItem> {
    await this.ensureLoaded();
    this.ensureSessionFresh();
    const sequenceId = this.sequenceCounter + 1;
    this.sequenceCounter = sequenceId;
    const positionId = `trk_pos_${sequenceId}_${Math.random().toString(36).slice(2, 8)}`;
    const batchId = `trk_batch_${Math.floor(sequenceId / Math.max(1, DRAIN_BATCH_SIZE))}_${nowMs()}`;
    const item: DriverTrackingQueueItem = {
      id: buildQueueId(),
      sequenceId,
      trackingSessionId: this.trackingSessionId,
      batchId,
      positionId,
      missionId: entry.missionId,
      appState: entry.appState,
      locationMode: entry.locationMode,
      payload: {
        ...entry.payload,
        locationMode: entry.locationMode,
        trackingEventId: undefined,
      },
      queuedAt: nowMs(),
      retryCount: 0,
      deliveryState: "queued",
      lastAttemptAt: null,
      ackedAt: null,
      lastError: null,
    };
    item.payload.trackingEventId = item.id;
    this.items.push(item);
    this.compactQueueIfNeeded();
    this.trimIfNeeded();
    await this.persist();
    await this.persistSession();
    const oldestQueuedAt = this.items.length > 0 ? this.items[0]?.queuedAt ?? null : null;
    emitDriverTelemetry("tracking.queue.enqueued", {
      source: "driver.tracking.queue",
      mission_id: entry.missionId,
      app_state: entry.appState,
      queue_depth: this.items.length,
      oldest_item_age_ms: oldestQueuedAt ? Math.max(0, nowMs() - oldestQueuedAt) : null,
      location_mode: entry.locationMode,
      sequence_id: sequenceId,
      tracking_session_id: this.trackingSessionId,
    });
    return item;
  }

  async flush(options?: {
    ackStaleMs?: number;
    networkProfile?: "offline" | "poor" | "normal";
    forceHttpFallback?: boolean;
  }): Promise<DriverTrackingFlushResult> {
    await this.ensureLoaded();
    this.ensureSessionFresh();
    const ackStaleMs = options?.ackStaleMs ?? SOCKET_ACK_DEFAULT_STALE_MS;
    const networkProfile = options?.networkProfile ?? "normal";
    if (this.isFlushing) {
      return {
        sent: 0,
        backendAcked: 0,
        socketEmitted: 0,
        dropped: 0,
        retried: 0,
        queueDepth: this.items.length,
        flushPathUsed: "http_fallback",
        lastBackendAckAt: null,
        oldestItemAgeMs: this.items[0] ? Math.max(0, nowMs() - this.items[0].queuedAt) : null,
        networkProfile,
      };
    }
    this.isFlushing = true;
    let sent = 0;
    let backendAcked = 0;
    let socketEmitted = 0;
    let dropped = 0;
    let retried = 0;
    let lastBackendAckAt: number | null = null;
    let flushPathUsed: DriverTrackingFlushResult["flushPathUsed"] = "http_fallback";
    try {
      const enableRealAck = isFeatureEnabled("tracking_real_ack_semantics_enabled");
      const remaining: DriverTrackingQueueItem[] = [];
      this.items.sort((a, b) => (a.sequenceId ?? 0) - (b.sequenceId ?? 0));
      const oldestQueuedAt = this.items[0]?.queuedAt ?? null;
      const minuteBucket = Math.floor(nowMs() / 60_000);
      if (this.drainMinuteBucket !== minuteBucket) {
        this.drainMinuteBucket = minuteBucket;
        this.drainedInCurrentMinute = 0;
      }
      const remainingBudget = Math.max(0, MAX_DRAIN_POSITIONS_PER_MINUTE - this.drainedInCurrentMinute);
      const maxDrainNow = Math.max(0, Math.min(remainingBudget, DRAIN_BATCH_SIZE));
      if (maxDrainNow <= 0) {
        emitDriverTelemetry("tracking.queue.backpressure", {
          source: "driver.tracking.queue",
          queue_depth: this.items.length,
          max_drain_positions_per_minute: MAX_DRAIN_POSITIONS_PER_MINUTE,
          drain_interval_ms: DRAIN_INTERVAL_MS,
        });
        await this.persist();
        return {
          sent: 0,
          backendAcked: 0,
          socketEmitted: 0,
          dropped: 0,
          retried: 0,
          queueDepth: this.items.length,
          flushPathUsed,
          lastBackendAckAt,
          oldestItemAgeMs: oldestQueuedAt ? Math.max(0, nowMs() - oldestQueuedAt) : null,
          networkProfile,
        };
      }

      // Envoi batch socket reel (N points par emission) pour reduire la pression reseau.
      if (!options?.forceHttpFallback && realtimeManager.isDriverSocketReady()) {
        const socketCandidates = this.items.filter(
          (item) => !this.isExpired(item) && item.deliveryState !== "socket_emitted"
        );
        let sentThisFlush = 0;
        for (let index = 0; index < socketCandidates.length; index += SOCKET_BATCH_MAX_POINTS) {
          if (sentThisFlush >= maxDrainNow) break;
          const chunk = socketCandidates.slice(
            index,
            Math.min(index + SOCKET_BATCH_MAX_POINTS, index + (maxDrainNow - sentThisFlush))
          );
          const sentViaSocket = realtimeManager.sendDriverLocationBatch(
            chunk.map((item) => ({
              tracking_event_id: item.id,
              sequence_id: item.sequenceId,
              tracking_session_id: item.trackingSessionId,
              position_id: item.positionId,
              batch_id: item.batchId,
              mission_id: item.missionId,
              latitude: item.payload.latitude,
              longitude: item.payload.longitude,
              accuracy: item.payload.accuracy,
              heading: item.payload.heading,
              speed: item.payload.speed,
              timestamp: item.payload.timestamp,
              location_mode: item.locationMode,
              is_background: item.payload.isBackground,
            }))
          );
          if (!sentViaSocket) break;
          for (const item of chunk) {
            sent += 1;
            sentThisFlush += 1;
            this.drainedInCurrentMinute += 1;
            socketEmitted += 1;
            flushPathUsed = "socket_batch";
            item.deliveryState = enableRealAck ? "socket_emitted" : "backend_acked";
            item.lastAttemptAt = nowMs();
            item.lastError = null;
            if (!enableRealAck) {
              backendAcked += 1;
              item.ackedAt = nowMs();
              lastBackendAckAt = item.ackedAt;
            }
          }
        }
      }

      for (const item of this.items) {
        if (sent >= maxDrainNow) {
          remaining.push(item);
          continue;
        }
        if (this.isExpired(item)) {
          dropped += 1;
          emitDriverTelemetry("tracking.queue.expired", {
            source: "driver.tracking.queue",
            mission_id: item.missionId,
            queue_replay_window_ms: REPLAY_WINDOW_MS,
          });
          continue;
        }
        if (item.deliveryState === "backend_acked") {
          continue;
        }

        try {
          const canTrySocket =
            !options?.forceHttpFallback &&
            (item.deliveryState === "queued" || item.deliveryState === "retry_pending");
          if (canTrySocket) {
            const socketSent = realtimeManager.sendDriverLocationBatch([
              {
                tracking_event_id: item.id,
                sequence_id: item.sequenceId,
                tracking_session_id: item.trackingSessionId,
                position_id: item.positionId,
                batch_id: item.batchId,
                mission_id: item.missionId,
                latitude: item.payload.latitude,
                longitude: item.payload.longitude,
                accuracy: item.payload.accuracy,
                heading: item.payload.heading,
                speed: item.payload.speed,
                timestamp: item.payload.timestamp,
                location_mode: item.locationMode,
                is_background: item.payload.isBackground,
              },
            ]);
            if (socketSent) {
              sent += 1;
              this.drainedInCurrentMinute += 1;
              socketEmitted += 1;
              flushPathUsed = "socket_batch";
              item.deliveryState = enableRealAck ? "socket_emitted" : "backend_acked";
              item.lastAttemptAt = nowMs();
              item.lastError = null;
              if (!enableRealAck) {
                backendAcked += 1;
                item.ackedAt = nowMs();
                lastBackendAckAt = item.ackedAt;
                emitDriverTelemetry("tracking.ingest.ack", {
                  source: "driver.tracking.queue",
                  mission_id: item.missionId,
                  flush_path: "socket_batch",
                  queue_item_id: item.id,
                  ack_status: "accepted",
                });
                continue;
              }
              emitDriverTelemetry("tracking.socket.emit_without_backend_ack", {
                source: "driver.tracking.queue",
                mission_id: item.missionId,
                queue_item_id: item.id,
              });
              remaining.push(item);
              continue;
            }
          }

          const socketEmitStale =
            item.deliveryState === "socket_emitted" &&
            item.lastAttemptAt !== null &&
            nowMs() - item.lastAttemptAt < ackStaleMs;
          if (socketEmitStale && !options?.forceHttpFallback) {
            remaining.push(item);
            continue;
          }

          item.deliveryState = "retry_pending";
          item.lastAttemptAt = nowMs();
          const ack = await sendDriverLocation({
            ...item.payload,
            locationMode: item.locationMode,
            trackingEventId: item.id,
          });
          sent += 1;
          if (ack.ack_status === "accepted" || ack.ack_status === "duplicate") {
            item.deliveryState = "backend_acked";
            item.ackedAt = nowMs();
            item.lastError = null;
            backendAcked += 1;
            lastBackendAckAt = item.ackedAt;
            emitDriverTelemetry("tracking.ingest.ack", {
              source: "driver.tracking.queue",
              mission_id: item.missionId,
              flush_path: "http_fallback",
              queue_item_id: item.id,
              ack_status: ack.ack_status,
              accept_reason: ack.accept_reason ?? null,
              trace_id: ack.trace_id ?? null,
            });
            continue;
          }
          item.deliveryState = ack.ack_status === "stale" ? "expired" : "dropped";
          dropped += 1;
          emitDriverTelemetry("tracking.queue.dropped", {
            source: "driver.tracking.queue",
            mission_id: item.missionId,
            reason: `ack_${ack.ack_status}`,
            queue_item_id: item.id,
            accept_reason: ack.accept_reason ?? null,
            trace_id: ack.trace_id ?? null,
          });
        } catch (error) {
          item.deliveryState = "retry_pending";
          item.lastError = error instanceof Error ? error.message : "send_failed";
          item.retryCount += 1;
          retried += 1;
          if (item.retryCount >= MAX_RETRIES) {
            dropped += 1;
            item.deliveryState = "dropped";
            emitDriverTelemetry("tracking.queue.dropped", {
              source: "driver.tracking.queue",
              mission_id: item.missionId,
              reason: "max_retries",
              retry_count: item.retryCount,
            });
          } else {
            remaining.push(item);
          }
        }
      }
      this.items = remaining;
      await this.persist();
      if (this.items.length > 0 && DRAIN_INTERVAL_MS > 0) {
        setTimeout(() => {
          void this.flush(options);
        }, DRAIN_INTERVAL_MS);
      }
      emitDriverTelemetry("tracking.queue.flush", {
        source: "driver.tracking.queue",
        sent,
        backend_acked: backendAcked,
        socket_emitted: socketEmitted,
        dropped,
        retried,
        queue_depth: this.items.length,
        oldest_item_age_ms: oldestQueuedAt ? Math.max(0, nowMs() - oldestQueuedAt) : null,
        tracking_queue_oldest_age_ms: oldestQueuedAt ? Math.max(0, nowMs() - oldestQueuedAt) : null,
        network_profile_active: networkProfile,
        flush_path: flushPathUsed,
      });
      return {
        sent,
        backendAcked,
        socketEmitted,
        dropped,
        retried,
        queueDepth: this.items.length,
        flushPathUsed,
        lastBackendAckAt,
        oldestItemAgeMs: oldestQueuedAt ? Math.max(0, nowMs() - oldestQueuedAt) : null,
        networkProfile,
      };
    } finally {
      this.isFlushing = false;
    }
  }

  async getSnapshot(): Promise<DriverTrackingQueueSnapshot> {
    await this.ensureLoaded();
    if (this.items.length === 0) {
      return {
        queueDepth: 0,
        oldestQueuedAt: null,
        newestQueuedAt: null,
        oldestItemAgeMs: null,
      };
    }
    const sorted = [...this.items].sort((a, b) => a.queuedAt - b.queuedAt);
    const oldestQueuedAt = sorted[0]?.queuedAt ?? null;
    return {
      queueDepth: this.items.length,
      oldestQueuedAt,
      newestQueuedAt: sorted[sorted.length - 1]?.queuedAt ?? null,
      oldestItemAgeMs: oldestQueuedAt ? Math.max(0, nowMs() - oldestQueuedAt) : null,
    };
  }

  async markBackendAckedByIds(ids: string[]): Promise<number> {
    await this.ensureLoaded();
    if (!ids.length) return 0;
    const idSet = new Set(ids);
    const beforeCount = this.items.length;
    this.items = this.items.filter((item) => !idSet.has(item.id));
    const ackedCount = beforeCount - this.items.length;
    if (ackedCount > 0) {
      await this.persist();
      emitDriverTelemetry("tracking.ingest.ack", {
        source: "driver.tracking.queue",
        flush_path: "socket_batch",
        ack_status: "accepted",
        backend_acked_count: ackedCount,
      });
    }
    return ackedCount;
  }

  async markBackendAckedByWatermark(ackLastSequenceId: number): Promise<number> {
    await this.ensureLoaded();
    if (!Number.isFinite(ackLastSequenceId) || ackLastSequenceId <= 0) return 0;
    const beforeCount = this.items.length;
    this.items = this.items.filter((item) => item.sequenceId > ackLastSequenceId);
    const ackedCount = beforeCount - this.items.length;
    if (ackedCount > 0) {
      await this.persist();
      emitDriverTelemetry("tracking.ingest.ack", {
        source: "driver.tracking.queue",
        flush_path: "socket_batch",
        ack_status: "accepted",
        backend_acked_count: ackedCount,
        ack_last_sequence_id: ackLastSequenceId,
      });
    }
    return ackedCount;
  }
}

export const driverTrackingQueue = new DriverTrackingQueue();
