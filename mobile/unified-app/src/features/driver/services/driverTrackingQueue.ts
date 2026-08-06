import AsyncStorage from "@react-native-async-storage/async-storage";
import { AppStateStatus, Platform } from "react-native";
import { DriverLocationPayload } from "../types";
import type { DriverLocationAckStatus } from "../types";
import { sendDriverLocation } from "../api/driverHttp";
import { formatTrackingSendError } from "./driverTrackingSendErrorFormat";
import {
  QueueSuspendReason,
  QueueSuspendState,
  resolveQueueSuspendMs,
} from "./driverTrackingQueueBackoff";
import { onAuthRefreshSuccess } from "../../../core/auth/authRefreshListeners";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import {
  canEmitSocketBatchNow,
  getSocketBatchCooldownRemainingMs,
  recordSocketBatchRateLimited,
  recordSocketBatchSent,
} from "./socketBatchPacing";
import { trackingQueueStore } from "./trackingQueueStore";
import { fetchTrackingWatermark, registerTrackingSession } from "./trackingSessionsApi";

function allowMemoryFallback(): boolean {
  return Platform.OS === "web" || typeof jest !== "undefined";
}

export type TrackingDeliveryState =
  | "queued"
  | "socket_emitted"
  | "backend_acked"
  | "retry_pending"
  | "dropped"
  | "expired"
  | "compacted";

export type DriverTrackingMode = "mission_live" | "availability_presence" | "observability_only";

/** availability_presence = HTTP only (socket interdit côté backend). */
function isSocketEligibleLocationMode(mode: DriverTrackingMode): boolean {
  return mode !== "availability_presence";
}

export type DriverTrackingQueueItem = {
  id: string;
  sequenceId: number;
  trackingSessionId: string;
  /** Génération autoritaire backend (nullable jusqu'à POST /sessions). */
  sessionGeneration: number | null;
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
  /** État de conservation jusqu'au watermark PG (Annexe A.4). */
  persistState?:
    | "non_ingested"
    | "ingested_non_persisted"
    | "persisted"
    | "rejected"
    | "tombstone";
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
  lastBackendAckStatus: DriverLocationAckStatus | null;
  /** ID local de l’élément effectivement envoyé (jamais inventé côté serveur). */
  lastBackendAckRequestEventId: string | null;
  /** ID renvoyé par le serveur — null si absent, sans fallback sur item.id. */
  lastBackendAckServerEventId: string | null;
  oldestItemAgeMs: number | null;
  networkProfile: "offline" | "poor" | "normal";
};

type DriverTrackingQueueSnapshot = {
  queueDepth: number;
  oldestQueuedAt: number | null;
  newestQueuedAt: number | null;
  oldestItemAgeMs: number | null;
  trackingSessionId?: string;
  sequenceCounter?: number;
  sessionGeneration?: number | null;
  suspendReason?: string | null;
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
const DRAIN_INTERVAL_MS = Number(process.env.EXPO_PUBLIC_DRIVER_TRACKING_DRAIN_INTERVAL_MS ?? "2000");
const MAX_DRAIN_POSITIONS_PER_MINUTE = Number(
  process.env.EXPO_PUBLIC_DRIVER_TRACKING_MAX_DRAIN_POSITIONS_PER_MINUTE ?? "1200"
);
/** Au-delà de ce seuil, bascule HTTP immédiate (évite la famine socket_emitted). */
const BACKLOG_FORCE_HTTP_THRESHOLD = Number(
  process.env.EXPO_PUBLIC_DRIVER_TRACKING_BACKLOG_FORCE_HTTP ?? "30"
);
const TRACKING_SESSION_TTL_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_TRACKING_SESSION_TTL_SEC ?? "1800"
) * 1000;
const SESSION_STORAGE_KEY = "driver_tracking_session_v1";
const QUEUE_SUSPEND_STORAGE_KEY = "driver_tracking_queue_suspend_v1";
const inMemoryStorage = new Map<string, string>();

function normalizeTrackingEnqueueMode(
  locationMode: DriverTrackingMode,
  missionId: number | null
): DriverTrackingMode {
  if (locationMode === "mission_live" && missionId == null) {
    return "availability_presence";
  }
  return locationMode;
}

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
  private pendingFlushOptions: {
    ackStaleMs?: number;
    networkProfile?: "offline" | "poor" | "normal";
    forceHttpFallback?: boolean;
  } | null = null;
  private authListenerRegistered = false;
  private queueSuspend: QueueSuspendState | null = null;
  private items: DriverTrackingQueueItem[] = [];
  private sequenceCounter = 0;
  private trackingSessionId = "";
  private sessionGeneration: number | null = null;
  private sessionCreatedAt = 0;
  private identityKey: string | null = null;
  private drainedInCurrentMinute = 0;
  private drainMinuteBucket = 0;
  private drainTimer: ReturnType<typeof setTimeout> | null = null;
  /** IDs explicitement ingérés (calcul contigu local — pas d'état serveur). */
  private ingestedEventIds = new Set<string>();
  private watermarkTimer: ReturnType<typeof setTimeout> | null = null;
  private watermarkInFlight = false;
  private static readonly WATERMARK_POLL_MS = 4_000;

  private clearDrainTimer(): void {
    if (this.drainTimer) {
      clearTimeout(this.drainTimer);
      this.drainTimer = null;
    }
  }

  /**
   * Planifie au plus un drain : évite la tempête de flush quand des points
   * attendent un ACK socket (état socket_emitted non stale).
   */
  private scheduleDrainIfNeeded(
    options: {
      ackStaleMs?: number;
      networkProfile?: "offline" | "poor" | "normal";
      forceHttpFallback?: boolean;
    },
    ackStaleMs: number
  ): void {
    if (this.items.length === 0) {
      this.clearDrainTimer();
      return;
    }
    const delayMs = this.computeNextDrainDelayMs(ackStaleMs);
    if (delayMs == null) {
      this.clearDrainTimer();
      return;
    }
    const suspendDelay =
      this.suspendActive() && this.queueSuspend
        ? Math.max(delayMs, this.queueSuspend.untilMs - nowMs())
        : delayMs;
    if (this.drainTimer) {
      return;
    }
    this.drainTimer = setTimeout(() => {
      this.drainTimer = null;
      void this.flush(options);
    }, suspendDelay);
  }

  private computeNextDrainDelayMs(ackStaleMs: number): number | null {
    const now = nowMs();
    let hasSendable = false;
    let nextWakeAt: number | null = null;

    for (const item of this.items) {
      if (item.deliveryState === "backend_acked") continue;
      if (item.deliveryState === "queued" || item.deliveryState === "retry_pending") {
        hasSendable = true;
        continue;
      }
      if (item.deliveryState === "socket_emitted" && item.lastAttemptAt != null) {
        const staleAt = item.lastAttemptAt + ackStaleMs;
        if (now >= staleAt) {
          hasSendable = true;
        } else {
          nextWakeAt = nextWakeAt == null ? staleAt : Math.min(nextWakeAt, staleAt);
        }
      }
    }

    if (!hasSendable) {
      return nextWakeAt == null ? null : Math.max(DRAIN_INTERVAL_MS, nextWakeAt - now);
    }

    const pacingRemaining = getSocketBatchCooldownRemainingMs(now);
    if (pacingRemaining > 0) {
      const pacingWake = now + pacingRemaining;
      nextWakeAt = nextWakeAt == null ? pacingWake : Math.min(nextWakeAt, pacingWake);
    }

    if (nextWakeAt != null) {
      return Math.max(DRAIN_INTERVAL_MS, nextWakeAt - now);
    }
    return DRAIN_INTERVAL_MS;
  }

  /**
   * Débloque la file quand le socket est mort ou la backlog explose :
   * les points `socket_emitted` sans ACK ne doivent pas bloquer le repli HTTP 75 s.
   */
  private async prepareFlushTransport(): Promise<{
    socketReady: boolean;
    backlogPressure: boolean;
    releasedCount: number;
  }> {
    const socketReady = realtimeManager.isDriverSocketReady();
    const backlogPressure = this.items.length >= BACKLOG_FORCE_HTTP_THRESHOLD;
    let releasedCount = 0;
    const socketEmittedCount = this.items.filter((i) => i.deliveryState === "socket_emitted").length;
    if (!socketReady || backlogPressure) {
      releasedCount = await this.releaseSocketEmittedForHttpRetry();
    }
    if (releasedCount > 0 || backlogPressure) {
      emitDriverTelemetry("tracking.queue.transport_unblock", {
        source: "driver.tracking.queue",
        queue_depth: this.items.length,
        socket_ready: socketReady,
        backlog_pressure: backlogPressure,
        released_count: releasedCount,
        socket_emitted_count: socketEmittedCount,
      });
    }
    return { socketReady, backlogPressure, releasedCount };
  }

  private tryEmitSocketBatch(chunk: DriverTrackingQueueItem[]): boolean {
    if (!canEmitSocketBatchNow()) {
      return false;
    }
    // Un batch = une seule tracking_session_id (Phase 0A multi-session)
    const sessionId = chunk[0]?.trackingSessionId;
    if (!sessionId || chunk.some((i) => i.trackingSessionId !== sessionId)) {
      return false;
    }
    const sentViaSocket = realtimeManager.sendDriverLocationBatch(
      chunk.map((item) => ({
        // Canonique + alias rétrocompat (plan v5)
        location_event_id: item.id,
        tracking_event_id: item.id,
        sequence_id: item.sequenceId,
        tracking_session_id: item.trackingSessionId,
        session_generation: item.sessionGeneration,
        position_id: item.positionId,
        batch_id: item.batchId,
        mission_id: item.missionId,
        latitude: item.payload.latitude,
        longitude: item.payload.longitude,
        accuracy: item.payload.accuracy,
        heading: item.payload.heading,
        speed: item.payload.speed,
        timestamp: item.payload.timestamp,
        recorded_at: item.payload.timestamp,
        location_mode: item.locationMode,
        is_background: item.payload.isBackground,
        platform: Platform.OS === "ios" ? "ios" : "android",
      }))
    );
    if (!sentViaSocket) {
      return false;
    }
    recordSocketBatchSent();
    return true;
  }

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
    await trackingQueueStore.init();

    // Source de vérité : SQLite (natif) ; AsyncStorage = legacy one-shot uniquement.
    const legacyParsed = safeJsonParse<DriverTrackingQueueItem[]>(
      await this.readStorage(STORAGE_KEY)
    );
    const legacyItems = Array.isArray(legacyParsed) ? legacyParsed : [];

    if (legacyItems.length > 0) {
      const imported = await trackingQueueStore.importLegacyOnce(
        legacyItems.map((item) => ({
          locationEventId: item.id,
          trackingSessionId: item.trackingSessionId,
          sessionGeneration: item.sessionGeneration ?? null,
          sequenceId: item.sequenceId,
          payloadJson: JSON.stringify(item.payload),
          state: item.persistState ?? "non_ingested",
          queuedAt: item.queuedAt,
          lastAttemptAt: item.lastAttemptAt,
          retryCount: item.retryCount,
          deliveryState: item.deliveryState,
          missionId: item.missionId,
          locationMode: item.locationMode,
          batchId: item.batchId,
          positionId: item.positionId,
          appState: String(item.appState),
          lastError: item.lastError,
          ackedAt: item.ackedAt,
        }))
      );
      if (imported) {
        await this.writeStorage(STORAGE_KEY, "");
      }
    } else {
      await trackingQueueStore.importLegacyOnce([]);
    }

    const activeRows = await trackingQueueStore.listActive();
    if (activeRows.length > 0 || trackingQueueStore.isDurableBackendAvailable()) {
      this.items = activeRows.map((row) => ({
        id: row.locationEventId,
        sequenceId: row.sequenceId,
        trackingSessionId: row.trackingSessionId,
        sessionGeneration: row.sessionGeneration,
        batchId: row.batchId,
        positionId: row.positionId,
        missionId: row.missionId,
        appState: row.appState as AppStateStatus,
        locationMode: row.locationMode as DriverTrackingMode,
        payload: safeJsonParse<DriverLocationPayload>(row.payloadJson) ?? ({} as DriverLocationPayload),
        queuedAt: row.queuedAt,
        retryCount: row.retryCount,
        deliveryState: row.deliveryState as TrackingDeliveryState,
        lastAttemptAt: row.lastAttemptAt,
        ackedAt: row.ackedAt,
        lastError: row.lastError,
        persistState: row.state,
      }));
    } else if (allowMemoryFallback()) {
      this.items = legacyItems;
    } else {
      this.items = [];
    }

    this.items.sort((a, b) => {
      const ga = a.sessionGeneration ?? 0;
      const gb = b.sessionGeneration ?? 0;
      if (ga !== gb) return ga - gb;
      if (a.trackingSessionId !== b.trackingSessionId) {
        return a.trackingSessionId.localeCompare(b.trackingSessionId);
      }
      return (a.sequenceId ?? 0) - (b.sequenceId ?? 0);
    });
    const sessionRaw = safeJsonParse<{
      trackingSessionId: string;
      createdAt: number;
      sessionGeneration?: number | null;
    }>(await this.readStorage(SESSION_STORAGE_KEY));
    if (
      sessionRaw?.trackingSessionId &&
      typeof sessionRaw.createdAt === "number" &&
      nowMs() - sessionRaw.createdAt < TRACKING_SESSION_TTL_MS
    ) {
      this.trackingSessionId = sessionRaw.trackingSessionId;
      this.sessionCreatedAt = sessionRaw.createdAt;
      this.sessionGeneration = sessionRaw.sessionGeneration ?? null;
      const sameSession = this.items.filter(
        (i) => i.trackingSessionId === this.trackingSessionId
      );
      this.sequenceCounter = sameSession.reduce(
        (max, item) => Math.max(max, item.sequenceId ?? 0),
        0
      );
    } else {
      this.rotateTrackingSession();
    }
    await this.loadSuspendState();
    this.registerAuthRefreshListener();
    this.loaded = true;
  }

  private async persist() {
    // Natif : SQLite autoritaire — chaque item actif déjà upserté à l'enqueue.
    // AsyncStorage n'est plus la source durable ; écriture legacy désactivée
    // dès que SQLite est disponible.
    if (trackingQueueStore.isDurableBackendAvailable()) {
      return;
    }
    if (trackingQueueStore.isDurableUnavailable() && Platform.OS !== "web") {
      // Ne pas prétendre conserver via AsyncStorage comme durable.
      return;
    }
    await this.writeStorage(STORAGE_KEY, JSON.stringify(this.items));
  }

  private async persistSession() {
    await this.writeStorage(
      SESSION_STORAGE_KEY,
      JSON.stringify({
        trackingSessionId: this.trackingSessionId,
        createdAt: this.sessionCreatedAt,
        sessionGeneration: this.sessionGeneration,
      })
    );
  }

  private registerAuthRefreshListener() {
    if (this.authListenerRegistered) return;
    this.authListenerRegistered = true;
    onAuthRefreshSuccess(() => {
      void this.clearAuthSuspensionOnly("auth_refresh_success");
    });
  }

  private async loadSuspendState() {
    const parsed = safeJsonParse<QueueSuspendState>(
      await this.readStorage(QUEUE_SUSPEND_STORAGE_KEY)
    );
    if (
      parsed &&
      typeof parsed.untilMs === "number" &&
      typeof parsed.reason === "string" &&
      parsed.untilMs > nowMs()
    ) {
      this.queueSuspend = parsed;
    } else {
      this.queueSuspend = null;
    }
  }

  private async persistSuspendState() {
    if (!this.queueSuspend) {
      await this.writeStorage(QUEUE_SUSPEND_STORAGE_KEY, "");
      return;
    }
    await this.writeStorage(QUEUE_SUSPEND_STORAGE_KEY, JSON.stringify(this.queueSuspend));
  }

  private suspendActive(): boolean {
    return this.queueSuspend !== null && nowMs() < this.queueSuspend.untilMs;
  }

  async clearSuspension(source = "manual"): Promise<void> {
    if (!this.queueSuspend) return;
    this.queueSuspend = null;
    await this.persistSuspendState();
    emitDriverTelemetry("tracking.queue.suspension_cleared", {
      source: "driver.tracking.queue",
      reason: source,
    });
  }

  async clearAuthSuspensionOnly(source = "auth_refresh_success"): Promise<void> {
    if (this.queueSuspend?.reason !== "auth") return;
    await this.clearSuspension(source);
  }

  private async activateSuspension(
    suspendMs: number,
    reason: QueueSuspendReason
  ): Promise<void> {
    const untilMs = nowMs() + Math.max(1_000, suspendMs);
    this.queueSuspend = { untilMs, reason };
    await this.persistSuspendState();
    emitDriverTelemetry("tracking.queue.suspended", {
      source: "driver.tracking.queue",
      reason,
      suspend_ms: suspendMs,
      until_ms: untilMs,
    });
  }

  private createLocalTrackingSession(): void {
    const createdAt = nowMs();
    this.trackingSessionId = `trk_sess_${createdAt}_${Math.random().toString(36).slice(2, 10)}`;
    this.sessionCreatedAt = createdAt;
    this.sessionGeneration = null;
    this.sequenceCounter = 0;
  }

  private rotateTrackingSession() {
    this.createLocalTrackingSession();
    void this.persistSession();
    void this.registerSessionWithBackend();
  }

  private async registerSessionWithBackend(): Promise<void> {
    const sessionIdAtStart = this.trackingSessionId;
    if (!sessionIdAtStart) return;
    try {
      const res = await registerTrackingSession({
        tracking_session_id: sessionIdAtStart,
        tracking_session_started_at: new Date(this.sessionCreatedAt).toISOString(),
      });
      // Ignorer réponse tardive d'une session A si session active = B
      if (this.trackingSessionId !== sessionIdAtStart) {
        emitDriverTelemetry("tracking.session.register_stale_ignored", {
          source: "driver.tracking.queue",
          tracking_session_id: sessionIdAtStart,
          active_tracking_session_id: this.trackingSessionId,
        });
        return;
      }
      this.sessionGeneration = res.session_generation;
      if (typeof res.first_sequence_id === "number" && res.first_sequence_id > 0) {
        this.sequenceCounter = Math.max(
          this.sequenceCounter,
          res.first_sequence_id - 1
        );
      }
      for (const item of this.items) {
        if (item.trackingSessionId === sessionIdAtStart) {
          item.sessionGeneration = res.session_generation;
        }
      }
      await this.persistSession();
      await this.persist();
    } catch {
      // Offline : capture locale continue ; génération injectée au retour réseau
      emitDriverTelemetry("tracking.session.register_deferred", {
        source: "driver.tracking.queue",
        tracking_session_id: sessionIdAtStart,
      });
    }
  }

  private ensureSessionFresh() {
    if (!this.trackingSessionId || nowMs() - this.sessionCreatedAt >= TRACKING_SESSION_TTL_MS) {
      this.rotateTrackingSession();
      void this.persistSession();
    }
  }

  /**
   * Quarantaine logout — ne purge jamais les points non ACKés.
   * Réconciliation uniquement si la même identité se reconnecte.
   * `lifecycleOperationId` permet la compensation si le logout devient stale.
   */
  async quarantineOnLogout(identity: {
    userId: number | string;
    driverId: number | string;
    companyId: number | string;
    lifecycleOperationId?: string;
  }): Promise<void> {
    await this.ensureLoaded();
    const key = `${identity.userId}:${identity.driverId}:${identity.companyId}`;
    this.identityKey = key;
    await trackingQueueStore.quarantineForIdentity(key, identity.lifecycleOperationId ?? null);
    emitDriverTelemetry("tracking.queue.quarantined", {
      source: "driver.tracking.queue",
      queue_depth: this.items.length,
      lifecycle_operation_id: identity.lifecycleOperationId ?? null,
    });
  }

  async clearQuarantineIfOperationMatches(lifecycleOperationId: string): Promise<boolean> {
    await this.ensureLoaded();
    return trackingQueueStore.clearQuarantineIfOperationMatches(lifecycleOperationId);
  }

  async resumeAfterLogin(identity: {
    userId: number | string;
    driverId: number | string;
    companyId: number | string;
  }): Promise<boolean> {
    return this.resumeAfterAuthRecovery(identity, {
      authSessionChanged: false,
      preservePendingTrackingSessions: true,
    });
  }

  /**
   * Reprise après recovery auth.
   * Les points déjà persistés conservent tracking_session_id / sequence_id / event_id.
   * Une nouvelle tracking session n'est créée que si authSessionChanged et qu'aucun backlog
   * n'est en attente — sinon on flush d'abord avec les IDs existants.
   */
  async resumeAfterAuthRecovery(
    identity: {
      userId: number | string;
      driverId: number | string;
      companyId: number | string;
    },
    options: {
      authSessionChanged?: boolean;
      preservePendingTrackingSessions?: boolean;
      beginNewSessionIfIdle?: boolean;
    } = {}
  ): Promise<boolean> {
    await this.ensureLoaded();
    const key = `${identity.userId}:${identity.driverId}:${identity.companyId}`;
    const ok = await trackingQueueStore.clearQuarantineIfMatch(key);
    if (!ok) {
      emitDriverTelemetry("tracking.queue.quarantine_identity_mismatch", {
        source: "driver.tracking.queue",
      });
      return false;
    }
    this.identityKey = key;

    const preserve = options.preservePendingTrackingSessions !== false;
    const pendingCount = this.items.filter(
      (i) => (i.persistState ?? "non_ingested") !== "persisted"
    ).length;

    emitDriverTelemetry("tracking.queue.resume_after_auth", {
      source: "driver.tracking.queue",
      pending_count: pendingCount,
      auth_session_changed: Boolean(options.authSessionChanged),
      preserve_pending: preserve,
    });

    // Ne jamais réécrire tracking_session_id / sequence des points existants.
    if (preserve && pendingCount > 0) {
      void this.flush().catch(() => undefined);
      return true;
    }

    const shouldBeginNew =
      options.beginNewSessionIfIdle === true
      || (options.authSessionChanged === true && pendingCount === 0);

    if (shouldBeginNew) {
      await this.beginNewTrackingSession();
    } else if (this.sessionGeneration == null && this.trackingSessionId) {
      await this.registerSessionWithBackend();
    }
    return true;
  }

  private scheduleWatermarkReconcile(): void {
    if (this.watermarkTimer) return;
    this.watermarkTimer = setTimeout(() => {
      this.watermarkTimer = null;
      void this.reconcileWatermarks();
    }, DriverTrackingQueue.WATERMARK_POLL_MS);
  }

  /**
   * Poll watermark multi-session pour les items ingested_non_persisted.
   * Tombstone uniquement seq <= contiguous OU location_event_id dans out_of_order_persisted.
   */
  private async reconcileWatermarks(): Promise<void> {
    if (this.watermarkInFlight) return;
    await this.ensureLoaded();
    const pending = this.items.filter(
      (i) => (i.persistState ?? "non_ingested") === "ingested_non_persisted"
    );
    if (pending.length === 0) return;
    this.watermarkInFlight = true;
    try {
      const bySession = new Map<string, DriverTrackingQueueItem[]>();
      for (const item of pending) {
        const sid = item.trackingSessionId;
        if (!sid) continue;
        const list = bySession.get(sid) ?? [];
        list.push(item);
        bySession.set(sid, list);
      }
      const toPersist: string[] = [];
      for (const [sessionId, sessionItems] of bySession) {
        let cursor: string | null = null;
        let contiguous = 0;
        const oooIds = new Set<string>();
        const missing = new Set<number>();
        try {
          do {
            const wm = await fetchTrackingWatermark(sessionId, cursor);
            contiguous = Math.max(contiguous, wm.contiguous_persisted_through ?? 0);
            for (const row of wm.out_of_order_persisted ?? []) {
              oooIds.add(row.location_event_id);
            }
            for (const range of wm.missing_ranges ?? []) {
              const [from, to] = range;
              for (let s = from; s <= to; s += 1) missing.add(s);
            }
            cursor = wm.next_cursor;
          } while (cursor);
        } catch (err) {
          const meta = formatTrackingSendError(err);
          if (meta.http_status === 401 || meta.http_status === 429 || meta.http_status === 503) {
            const plan = resolveQueueSuspendMs(meta, meta.retry_after_seconds);
            if (plan) {
              await this.activateSuspension(plan.suspendMs, plan.reason);
            }
          }
          emitDriverTelemetry("tracking.watermark.poll_failed", {
            source: "driver.tracking.queue",
            tracking_session_id: sessionId,
            http_status: meta.http_status,
          });
          continue;
        }
        for (const item of sessionItems) {
          if (missing.has(item.sequenceId)) continue;
          const inContiguous = item.sequenceId > 0 && item.sequenceId <= contiguous;
          const inOoo = oooIds.has(item.id);
          if (inContiguous || inOoo) {
            item.persistState = "persisted";
            item.deliveryState = "backend_acked";
            item.ackedAt = nowMs();
            item.lastError = null;
            toPersist.push(item.id);
          }
        }
      }
      if (toPersist.length > 0) {
        try {
          await trackingQueueStore.markState(toPersist, "persisted", {
            ackedAt: nowMs(),
            deliveryState: "backend_acked",
          });
        } catch {
          /* best-effort */
        }
        this.items = this.items.filter(
          (i) => !toPersist.includes(i.id) || (i.persistState ?? "") !== "persisted"
        );
        // Retirer les persisted de la file mémoire
        this.items = this.items.filter((i) => !toPersist.includes(i.id));
        await this.persist();
      }
    } finally {
      this.watermarkInFlight = false;
      const stillPending = this.items.some(
        (i) => (i.persistState ?? "non_ingested") === "ingested_non_persisted"
      );
      if (stillPending) this.scheduleWatermarkReconcile();
    }
  }

  /**
   * Ouvre une nouvelle session tracking pour les points futurs uniquement.
   * N'altère pas les items déjà en file. Offline-first : jamais bloqué par le réseau.
   */
  async beginNewTrackingSession(): Promise<void> {
    await this.ensureLoaded();
    this.createLocalTrackingSession();
    await this.persistSession();
    void this.registerSessionWithBackend();
    emitDriverTelemetry("tracking.queue.new_session_begun", {
      source: "driver.tracking.queue",
      tracking_session_id: this.trackingSessionId,
      session_generation: this.sessionGeneration,
    });
  }

  /** Contigu ingested local (serveur ne maintient pas cet état). */
  async applyIngestedEventIds(eventIds: string[]): Promise<number> {
    await this.ensureLoaded();
    let marked = 0;
    const toMark: string[] = [];
    for (const id of eventIds) {
      this.ingestedEventIds.add(id);
      const item = this.items.find((i) => i.id === id);
      if (item && (item.persistState ?? "non_ingested") === "non_ingested") {
        item.persistState = "ingested_non_persisted";
        toMark.push(id);
        marked += 1;
      }
    }
    if (marked > 0) {
      try {
        await trackingQueueStore.markState(toMark, "ingested_non_persisted");
      } catch {
        emitDriverTelemetry("tracking.queue.mark_state_failed", {
          source: "driver.tracking.queue",
          state: "ingested_non_persisted",
          count: toMark.length,
        });
      }
      await this.persist();
      const sessionItems = this.items
        .filter(
          (i) =>
            i.trackingSessionId === this.trackingSessionId &&
            (i.persistState === "ingested_non_persisted" ||
              i.persistState === "persisted" ||
              this.ingestedEventIds.has(i.id))
        )
        .map((i) => i.sequenceId)
        .sort((a, b) => a - b);
      let contiguous = 0;
      for (const seq of sessionItems) {
        if (seq === contiguous + 1) contiguous = seq;
        else break;
      }
      if (contiguous > 0) {
        await trackingQueueStore.setContiguousIngested(
          this.trackingSessionId,
          contiguous
        );
      }
    }
    return marked;
  }

  private isExpired(item: DriverTrackingQueueItem): boolean {
    // ingested_non_persisted : jamais expiré automatiquement (Annexe A.4)
    if (item.persistState === "ingested_non_persisted") {
      return false;
    }
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
    // Ne jamais évincer ingested_non_persisted ; tombstone audité pour non_ingested
    const evictable = this.items.filter(
      (i) => (i.persistState ?? "non_ingested") === "non_ingested"
    );
    const dropped = evictable.slice(0, overflow);
    const dropIds = new Set(dropped.map((d) => d.id));
    this.items = this.items.filter((i) => !dropIds.has(i.id));
    dropped.forEach((item) => {
      void trackingQueueStore.recordGap({
        trackingSessionId: item.trackingSessionId,
        sequenceFrom: item.sequenceId,
        sequenceTo: item.sequenceId,
        reason: "capacity_tombstone",
        createdAt: nowMs(),
      });
      emitDriverTelemetry("tracking.queue.dropped", {
        source: "driver.tracking.queue",
        mission_id: item.missionId,
        reason: "max_queue_size_tombstone",
        queue_replay_window_ms: REPLAY_WINDOW_MS,
        alert_visible: true,
      });
    });
    if (this.items.length > MAX_QUEUE_ITEMS) {
      emitDriverTelemetry("tracking.queue.saturation_alert", {
        source: "driver.tracking.queue",
        queue_depth: this.items.length,
        max_items: MAX_QUEUE_ITEMS,
      });
    }
  }

  async enqueue(entry: {
    missionId: number | null;
    appState: AppStateStatus;
    locationMode: DriverTrackingMode;
    payload: DriverLocationPayload;
  }): Promise<DriverTrackingQueueItem> {
    await this.ensureLoaded();
    this.ensureSessionFresh();
    const locationMode = normalizeTrackingEnqueueMode(
      entry.locationMode,
      entry.missionId
    );
    const sequenceId = this.sequenceCounter + 1;
    this.sequenceCounter = sequenceId;
    const positionId = `trk_pos_${sequenceId}_${Math.random().toString(36).slice(2, 8)}`;
    const batchId = `trk_batch_${Math.floor(sequenceId / Math.max(1, DRAIN_BATCH_SIZE))}_${nowMs()}`;
    const item: DriverTrackingQueueItem = {
      id: buildQueueId(),
      sequenceId,
      trackingSessionId: this.trackingSessionId,
      sessionGeneration: this.sessionGeneration,
      batchId,
      positionId,
      missionId: entry.missionId,
      appState: entry.appState,
      locationMode,
      payload: {
        ...entry.payload,
        locationMode,
        trackingEventId: undefined,
      },
      queuedAt: nowMs(),
      retryCount: 0,
      deliveryState: "queued",
      lastAttemptAt: null,
      ackedAt: null,
      lastError: null,
      persistState: "non_ingested",
    };
    item.payload.trackingEventId = item.id;

    // SQLite INSERT avant de considérer la capture conservée (natif).
    try {
      await trackingQueueStore.upsert({
        locationEventId: item.id,
        trackingSessionId: item.trackingSessionId,
        sessionGeneration: item.sessionGeneration ?? null,
        sequenceId: item.sequenceId,
        payloadJson: JSON.stringify(item.payload),
        state: "non_ingested",
        queuedAt: item.queuedAt,
        lastAttemptAt: item.lastAttemptAt,
        retryCount: item.retryCount,
        deliveryState: item.deliveryState,
        missionId: item.missionId,
        locationMode: item.locationMode,
        batchId: item.batchId,
        positionId: item.positionId,
        appState: String(item.appState),
        lastError: item.lastError,
        ackedAt: item.ackedAt,
      });
    } catch (err) {
      // Rollback séquence : capture non conservée
      this.sequenceCounter = Math.max(0, sequenceId - 1);
      emitDriverTelemetry("tracking.queue.durable_unavailable", {
        source: "driver.tracking.queue",
        error: String(err),
        location_event_id: item.id,
      });
      throw err;
    }

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
      location_mode: locationMode,
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
      this.pendingFlushOptions = {
        ...this.pendingFlushOptions,
        ...options,
        forceHttpFallback:
          options?.forceHttpFallback === true || this.pendingFlushOptions?.forceHttpFallback === true,
      };
      emitDriverTelemetry("tracking.queue.flush_coalesced", {
        source: "driver.tracking.queue",
        queue_depth: this.items.length,
        force_http_fallback: this.pendingFlushOptions?.forceHttpFallback === true,
      });
      return {
        sent: 0,
        backendAcked: 0,
        socketEmitted: 0,
        dropped: 0,
        retried: 0,
        queueDepth: this.items.length,
        flushPathUsed: "http_fallback",
        lastBackendAckAt: null,
        lastBackendAckStatus: null,
        lastBackendAckRequestEventId: null,
        lastBackendAckServerEventId: null,
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
    let lastBackendAckStatus: DriverTrackingFlushResult["lastBackendAckStatus"] = null;
    let lastBackendAckRequestEventId: string | null = null;
    let lastBackendAckServerEventId: string | null = null;
    let flushPathUsed: DriverTrackingFlushResult["flushPathUsed"] = "http_fallback";
    try {
      await this.loadSuspendState();
      const transport = await this.prepareFlushTransport();
      const effectiveForceHttp =
        options?.forceHttpFallback === true ||
        transport.backlogPressure ||
        !transport.socketReady;
      if (this.suspendActive() && this.queueSuspend) {
        const waitMs = Math.max(0, this.queueSuspend.untilMs - nowMs());
        emitDriverTelemetry("tracking.queue.suspend_wait", {
          source: "driver.tracking.queue",
          reason: this.queueSuspend.reason,
          wait_ms: waitMs,
          queue_depth: this.items.length,
        });
        if (this.items.length > 0 && waitMs > 0) {
          setTimeout(() => {
            void this.flush(options);
          }, waitMs);
        }
        return {
          sent: 0,
          backendAcked: 0,
          socketEmitted: 0,
          dropped: 0,
          retried: 0,
          queueDepth: this.items.length,
          flushPathUsed,
          lastBackendAckAt: null,
          lastBackendAckStatus: null,
          lastBackendAckRequestEventId: null,
          lastBackendAckServerEventId: null,
          oldestItemAgeMs: this.items[0]
            ? Math.max(0, nowMs() - this.items[0].queuedAt)
            : null,
          networkProfile,
        };
      }

      const enableRealAck = isFeatureEnabled("tracking_real_ack_semantics_enabled");
      const remaining: DriverTrackingQueueItem[] = [];
      this.items.sort((a, b) => {
        const ga = a.sessionGeneration ?? 0;
        const gb = b.sessionGeneration ?? 0;
        if (ga !== gb) return ga - gb;
        if (a.trackingSessionId !== b.trackingSessionId) {
          return a.trackingSessionId.localeCompare(b.trackingSessionId);
        }
        return (a.sequenceId ?? 0) - (b.sequenceId ?? 0);
      });
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
          lastBackendAckStatus,
          oldestItemAgeMs: oldestQueuedAt ? Math.max(0, nowMs() - oldestQueuedAt) : null,
          networkProfile,
        };
      }

      // Envoi batch socket réel — un batch = une seule session (Phase 0A)
      if (!effectiveForceHttp && transport.socketReady) {
        const primarySession =
          this.items.find(
            (item) =>
              !this.isExpired(item) &&
              item.deliveryState !== "socket_emitted" &&
              isSocketEligibleLocationMode(item.locationMode)
          )?.trackingSessionId ?? this.trackingSessionId;
        const socketCandidates = this.items.filter(
          (item) =>
            !this.isExpired(item) &&
            item.deliveryState !== "socket_emitted" &&
            item.trackingSessionId === primarySession &&
            isSocketEligibleLocationMode(item.locationMode)
        );
        let sentThisFlush = 0;
        for (let index = 0; index < socketCandidates.length; index += SOCKET_BATCH_MAX_POINTS) {
          if (sentThisFlush >= maxDrainNow) break;
          if (!canEmitSocketBatchNow()) break;
          const chunk = socketCandidates.slice(
            index,
            Math.min(index + SOCKET_BATCH_MAX_POINTS, index + (maxDrainNow - sentThisFlush))
          );
          if (!this.tryEmitSocketBatch(chunk)) break;
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
            !effectiveForceHttp &&
            transport.socketReady &&
            isSocketEligibleLocationMode(item.locationMode) &&
            (item.deliveryState === "queued" || item.deliveryState === "retry_pending");
          if (canTrySocket && canEmitSocketBatchNow()) {
            if (this.tryEmitSocketBatch([item])) {
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
            nowMs() - item.lastAttemptAt < ackStaleMs &&
            transport.socketReady;
          if (socketEmitStale && !effectiveForceHttp) {
            remaining.push(item);
            continue;
          }

          item.deliveryState = "retry_pending";
          item.lastAttemptAt = nowMs();
          const ack = await sendDriverLocation({
            ...item.payload,
            locationMode: item.locationMode,
            trackingEventId: item.id,
            trackingSessionId: item.trackingSessionId,
            sessionGeneration: item.sessionGeneration,
            sequenceId: item.sequenceId,
          });
          sent += 1;
          lastBackendAckRequestEventId = item.id;
          lastBackendAckServerEventId = ack.tracking_event_id ?? null;
          lastBackendAckStatus = ack.ack_status;
          lastBackendAckAt = nowMs();

          if (
            ack.tracking_event_id != null &&
            ack.tracking_event_id !== item.id
          ) {
            item.deliveryState = "retry_pending";
            item.lastError = "ack_event_id_mismatch";
            retried += 1;
            remaining.push(item);
            emitDriverTelemetry("tracking.queue.ack_event_id_mismatch", {
              source: "driver.tracking.queue",
              mission_id: item.missionId,
              queue_item_id: item.id,
              server_event_id: ack.tracking_event_id,
            });
            continue;
          }

          if (ack.ack_status === "partially_ingested") {
            const ingestedIds = ack.ingested_event_ids ?? null;
            const retryIds = ack.retry_event_ids ?? null;
            if (ingestedIds == null && retryIds == null) {
              item.deliveryState = "retry_pending";
              item.lastError = "partially_ingested_lists_missing";
              retried += 1;
              remaining.push(item);
              continue;
            }
            const ingestedSet = new Set(ingestedIds ?? []);
            const retrySet = new Set(retryIds ?? []);
            let conflict = false;
            for (const id of ingestedSet) {
              if (retrySet.has(id)) {
                conflict = true;
                break;
              }
            }
            if (conflict) {
              item.deliveryState = "retry_pending";
              item.lastError = "partially_ingested_list_conflict";
              retried += 1;
              remaining.push(item);
              emitDriverTelemetry("tracking.queue.partial_ack_conflict", {
                source: "driver.tracking.queue",
                queue_item_id: item.id,
              });
              continue;
            }
            if (ingestedSet.size > 0) {
              await this.applyIngestedEventIds([...ingestedSet]);
            }
            const currentIngested = ingestedSet.has(item.id);
            const currentRetry = retrySet.has(item.id);
            if (currentIngested) {
              item.persistState = "ingested_non_persisted";
              item.deliveryState = "backend_acked";
              item.ackedAt = nowMs();
              item.lastError = null;
              backendAcked += 1;
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
            item.deliveryState = "retry_pending";
            item.lastError = currentRetry
              ? "partially_ingested_retry"
              : "partially_ingested_current_missing";
            retried += 1;
            remaining.push(item);
            continue;
          }

          if (
            ack.ack_status === "persisted" &&
            ack.durability === "persisted_sync" &&
            (!ack.location_event_id || ack.location_event_id === item.id || ack.tracking_event_id === item.id)
          ) {
            if (ack.ingested_event_ids?.length) {
              await this.applyIngestedEventIds(ack.ingested_event_ids);
            }
            item.persistState = "persisted";
            try {
              await trackingQueueStore.markState([item.id], "persisted", {
                ackedAt: nowMs(),
                deliveryState: "backend_acked",
              });
            } catch {
              emitDriverTelemetry("tracking.queue.mark_state_failed", {
                source: "driver.tracking.queue",
                state: "persisted",
                location_event_id: item.id,
              });
            }
            item.deliveryState = "backend_acked";
            item.ackedAt = nowMs();
            item.lastError = null;
            backendAcked += 1;
            emitDriverTelemetry("tracking.ingest.ack", {
              source: "driver.tracking.queue",
              mission_id: item.missionId,
              flush_path: "http_fallback",
              queue_item_id: item.id,
              ack_status: ack.ack_status,
              durability: ack.durability ?? null,
              accept_reason: ack.accept_reason ?? null,
              trace_id: ack.trace_id ?? null,
            });
            continue;
          }

          if (ack.ack_status === "duplicate") {
            item.persistState = "persisted";
            item.deliveryState = "backend_acked";
            item.ackedAt = nowMs();
            item.lastError = null;
            backendAcked += 1;
            try {
              await trackingQueueStore.markState([item.id], "persisted", {
                ackedAt: nowMs(),
                deliveryState: "backend_acked",
              });
            } catch {
              /* best-effort */
            }
            continue;
          }

          // 202 / queued_async / ingested* → conserver SQLite, attendre watermark
          if (
            ack.ack_status === "queued" ||
            ack.ack_status === "ingested" ||
            ack.ack_status === "ingested_non_persisted" ||
            ack.durability === "queued_async"
          ) {
            if (ack.ingested_event_ids?.length) {
              await this.applyIngestedEventIds(ack.ingested_event_ids);
            } else {
              await this.applyIngestedEventIds([item.id]);
            }
            item.persistState = "ingested_non_persisted";
            item.deliveryState = "retry_pending";
            item.lastError = "awaiting_watermark";
            remaining.push(item);
            emitDriverTelemetry("tracking.ingest.ack", {
              source: "driver.tracking.queue",
              mission_id: item.missionId,
              flush_path: "http_fallback",
              queue_item_id: item.id,
              ack_status: "ingested_non_persisted",
              durability: "queued_async",
              accept_reason: ack.accept_reason ?? null,
              trace_id: ack.trace_id ?? null,
            });
            void this.scheduleWatermarkReconcile();
            continue;
          }

          // accepted sans durability : ne pas tombstoner (compat backend ancien)
          if (ack.ack_status === "accepted" && ack.durability !== "persisted_sync") {
            item.persistState = "ingested_non_persisted";
            item.deliveryState = "retry_pending";
            item.lastError = "accepted_without_durability";
            remaining.push(item);
            continue;
          }

          if (ack.ack_status === "persisted") {
            // persisted sans durability explicite : exiger location_event_id match
            if (
              ack.location_event_id &&
              ack.location_event_id !== item.id &&
              ack.tracking_event_id !== item.id
            ) {
              remaining.push(item);
              continue;
            }
            item.persistState = "persisted";
            item.deliveryState = "backend_acked";
            item.ackedAt = nowMs();
            backendAcked += 1;
            try {
              await trackingQueueStore.markState([item.id], "persisted", {
                ackedAt: nowMs(),
                deliveryState: "backend_acked",
              });
            } catch {
              /* best-effort */
            }
            continue;
          }

          item.deliveryState = ack.ack_status === "stale" ? "expired" : "dropped";
          if (ack.ack_status === "rejected" || ack.ack_status === "ignored") {
            item.persistState = ack.ack_status === "rejected" ? "rejected" : "tombstone";
          }
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
          const meta = formatTrackingSendError(error);
          emitDriverTelemetry("tracking.queue.http_send_failure", {
            source: "driver.tracking.queue",
            mission_id: item.missionId,
            queue_item_id: item.id,
            app_state: item.appState,
            error_class: meta.error_class,
            error_message: meta.error_message,
            http_status: meta.http_status,
            api_error_code: meta.api_error_code,
            transport_code: meta.transport_code,
            retry_count: item.retryCount,
            queue_depth: this.items.length,
            force_http_fallback: effectiveForceHttp,
          });
          const suspendPlan = resolveQueueSuspendMs(meta, meta.retry_after_seconds);
          if (suspendPlan) {
            await this.activateSuspension(suspendPlan.suspendMs, suspendPlan.reason);
            item.deliveryState = "retry_pending";
            item.lastError = suspendPlan.reason;
            remaining.push(item);
            continue;
          }
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
      this.scheduleDrainIfNeeded(
        {
          ackStaleMs: options?.ackStaleMs,
          networkProfile: options?.networkProfile,
          forceHttpFallback: effectiveForceHttp,
        },
        ackStaleMs
      );
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
      emitDriverTelemetry("tracking.flush.transport", {
        source: "driver.tracking.queue",
        transport: flushPathUsed === "socket_batch" ? "socket" : "http",
        sent,
        socket_emitted: socketEmitted,
        backend_acked: backendAcked,
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
        lastBackendAckStatus,
        lastBackendAckRequestEventId,
        lastBackendAckServerEventId,
        oldestItemAgeMs: oldestQueuedAt ? Math.max(0, nowMs() - oldestQueuedAt) : null,
        networkProfile,
      };
    } finally {
      this.isFlushing = false;
      const pendingFlush = this.pendingFlushOptions;
      this.pendingFlushOptions = null;
      if (pendingFlush) {
        void this.flush(pendingFlush);
      }
    }
  }

  async getSnapshot(): Promise<DriverTrackingQueueSnapshot> {
    await this.ensureLoaded();
    const base = {
      trackingSessionId: this.trackingSessionId,
      sequenceCounter: this.sequenceCounter,
      sessionGeneration: this.sessionGeneration,
      suspendReason: this.queueSuspend?.reason ?? null,
    };
    if (this.items.length === 0) {
      return {
        queueDepth: 0,
        oldestQueuedAt: null,
        newestQueuedAt: null,
        oldestItemAgeMs: null,
        ...base,
      };
    }
    const sorted = [...this.items].sort((a, b) => a.queuedAt - b.queuedAt);
    const oldestQueuedAt = sorted[0]?.queuedAt ?? null;
    return {
      queueDepth: this.items.length,
      oldestQueuedAt,
      newestQueuedAt: sorted[sorted.length - 1]?.queuedAt ?? null,
      oldestItemAgeMs: oldestQueuedAt ? Math.max(0, nowMs() - oldestQueuedAt) : null,
      ...base,
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

  /**
   * Reprendre après ACK `session_conflict` : nouvelle session active.
   * Ne rebind PAS les points de l'ancienne session (multi-session / superseded).
   */
  async reconcileAfterSessionConflict(): Promise<string> {
    await this.ensureLoaded();
    const previousSessionId = this.trackingSessionId;
    this.rotateTrackingSession();
    let released = 0;
    for (const item of this.items) {
      if (item.trackingSessionId === previousSessionId) {
        if (item.deliveryState === "socket_emitted") {
          item.deliveryState = "retry_pending";
          item.lastAttemptAt = null;
          released += 1;
        }
      }
    }
    await this.persistSession();
    if (released > 0) {
      await this.persist();
    }
    emitDriverTelemetry("tracking.session.reconciled", {
      source: "driver.tracking.queue",
      previous_session_id: previousSessionId,
      tracking_session_id: this.trackingSessionId,
      rebound_count: 0,
      released_socket_count: released,
    });
    return this.trackingSessionId;
  }

  async releaseSocketEmittedForHttpRetry(): Promise<number> {
    await this.ensureLoaded();
    let released = 0;
    for (const item of this.items) {
      if (item.deliveryState === "socket_emitted") {
        item.deliveryState = "retry_pending";
        item.lastAttemptAt = null;
        released += 1;
      }
    }
    if (released > 0) {
      await this.persist();
      emitDriverTelemetry("tracking.queue.socket_release_for_http", {
        source: "driver.tracking.queue",
        released_count: released,
      });
    }
    return released;
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

  /** Réservé aux tests unitaires — vide la file en mémoire. */
  async resetForTests(): Promise<void> {
    this.items = [];
    this.isFlushing = false;
    this.pendingFlushOptions = null;
    this.ingestedEventIds.clear();
    this.queueSuspend = null;
    this.trackingSessionId = "";
    this.sessionGeneration = null;
    this.sequenceCounter = 0;
    this.sessionCreatedAt = 0;
    this.watermarkInFlight = false;
    if (this.watermarkTimer) {
      clearTimeout(this.watermarkTimer);
      this.watermarkTimer = null;
    }
    await this.persist();
  }

  /** Test-only : activer une suspension. */
  async activateSuspensionForTests(
    suspendMs: number,
    reason: QueueSuspendReason
  ): Promise<void> {
    await this.activateSuspension(suspendMs, reason);
  }
}

export const driverTrackingQueue = new DriverTrackingQueue();

export function clearDriverTrackingQueueSuspension(): Promise<void> {
  return driverTrackingQueue.clearAuthSuspensionOnly();
}
