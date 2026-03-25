import AsyncStorage from "@react-native-async-storage/async-storage";
import { getLogger } from "@/utils/logger";
import { getNetworkStateSnapshot } from "./networkState";
import { api } from "./api";
import type { MissionBarStatus } from "./missionState";

const log = getLogger("ActionQueue");

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PendingAction {
  id: string;
  bookingId: string;
  targetStatus: MissionBarStatus;
  timestamp: number;
  retryCount: number;
  lastRetryAt: number | null;
}

interface MissionStateManagerLike {
  onTransitionConfirmed(action: PendingAction): void;
  onTransitionFailed(action: PendingAction, reason: "conflict" | "error" | "stale"): void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STORAGE_KEY = "pending_mission_actions_v1";
const STORAGE_KEY_OLD = "pending_mission_actions";
const MAX_RETRIES = 10;
const BACKOFF_BASE_MS = 2000;
const BACKOFF_MAX_MS = 60_000;
const MUST_SEND_ALL_STEPS = true;
const STALE_ACTION_MS = 24 * 60 * 60 * 1000; // 24h

// ---------------------------------------------------------------------------
// UUID helper (no external dep)
// ---------------------------------------------------------------------------

function uuid(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

// ---------------------------------------------------------------------------
// PendingActionsQueue
// ---------------------------------------------------------------------------

export class PendingActionsQueue {
  private actions: PendingAction[] = [];
  private flushLocks = new Map<string, boolean>();
  private lastFlushAt = 0;
  private manager: MissionStateManagerLike;
  private loaded = false;
  constructor(manager: MissionStateManagerLike) {
    this.manager = manager;
  }

  // -- Lifecycle -----------------------------------------------------------
  // Plan 2G/3G : Les triggers (online, foreground, socket connect) sont gérés par syncEngine.
  // startListening/stopListening conservés pour compatibilité mais ne font plus rien.
  startListening(): void {
    // Nop — syncEngine appelle flush() via MissionStateManager.syncPendingActions()
  }

  stopListening(): void {
    // Nop
  }

  // -- Persistence ---------------------------------------------------------

  private async load(): Promise<void> {
    if (this.loaded) return;
    try {
      let raw = await AsyncStorage.getItem(STORAGE_KEY);
      if (!raw) {
        const old = await AsyncStorage.getItem(STORAGE_KEY_OLD);
        if (old) {
          raw = old;
          await AsyncStorage.setItem(STORAGE_KEY, old);
          await AsyncStorage.removeItem(STORAGE_KEY_OLD);
        }
      }
      if (raw) this.actions = JSON.parse(raw);
    } catch (e) {
      log.warn("load error", { error: e });
    }
    this.loaded = true;
  }

  private async save(): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(this.actions));
    } catch (e) {
      log.error("save error", { error: e });
    }
  }

  // -- Public API ----------------------------------------------------------

  async enqueue(params: { bookingId: string; targetStatus: MissionBarStatus }): Promise<PendingAction> {
    await this.load();

    const action: PendingAction = {
      id: uuid(),
      bookingId: params.bookingId,
      targetStatus: params.targetStatus,
      timestamp: Date.now(),
      retryCount: 0,
      lastRetryAt: null,
    };

    if (!MUST_SEND_ALL_STEPS) {
      this.actions = this.actions.filter((a) => a.bookingId !== params.bookingId);
    }

    this.actions.push(action);
    await this.save();

    log.info("enqueue", {
      event: "enqueue",
      booking_id: params.bookingId,
      status: params.targetStatus,
      queue_size: this.actions.length,
      operation_id: action.id,
    });

    // Envoi immédiat en arrière-plan : ne pas await flushBooking ici.
    // Sinon MissionStateManager.requestTransition (ex. « Confirmer » fin de mission) reste
    // bloqué jusqu'à ce que TOUTE la file pour ce booking soit flushée (plusieurs PUT si
    // MUST_SEND_ALL_STEPS), ce qui peut prendre des dizaines de secondes ou sembler infini.
    try {
      const net = getNetworkStateSnapshot();
      if (net?.isConnected === true && net?.isInternetReachable !== false) {
        void this.flushBooking(params.bookingId).catch((e: unknown) => {
          log.warn("flush after enqueue failed", {
            event: "flush_after_enqueue",
            booking_id: params.bookingId,
            error: e,
          });
        });
      }
    } catch {
      // will be retried on network reconnect
    }

    return action;
  }

  async purge(bookingId: string): Promise<void> {
    await this.load();
    this.actions = this.actions.filter((a) => a.bookingId !== bookingId);
    await this.save();
  }

  async flush(): Promise<void> {
    const now = Date.now();
    if (now - this.lastFlushAt < 500) return;
    this.lastFlushAt = now;

    await this.load();
    await this.pruneStale();
    if (this.actions.length === 0) return;

    const net = getNetworkStateSnapshot();
    if (net?.isConnected !== true || net?.isInternetReachable === false) return;

    const bookingIds = [...new Set(this.actions.map((a) => a.bookingId))];
    await Promise.all(bookingIds.map((id) => this.flushBooking(id)));
  }

  private async pruneStale(): Promise<void> {
    const now = Date.now();
    const stale = this.actions.filter((a) => now - a.timestamp > STALE_ACTION_MS);
    if (stale.length === 0) return;

    log.warn("prune stale", {
      event: "prune_stale",
      count: stale.length,
      booking_ids: stale.map((a) => a.bookingId),
    });
    for (const action of stale) {
      this.manager.onTransitionFailed(action, "stale");
    }
    this.actions = this.actions.filter((a) => now - a.timestamp <= STALE_ACTION_MS);
    await this.save();
  }

  hasPending(bookingId?: string): boolean {
    if (bookingId) return this.actions.some((a) => a.bookingId === bookingId);
    return this.actions.length > 0;
  }

  getPendingStatus(bookingId: string): "pending" | "offline" | "error" | null {
    const action = this.actions.find((a) => a.bookingId === bookingId);
    if (!action) return null;
    if (action.retryCount >= MAX_RETRIES) return "error";
    if (action.retryCount > 0) return "offline";
    return "pending";
  }

  // -- Flush per booking (mutex) -------------------------------------------

  private async flushBooking(bookingId: string): Promise<void> {
    if (this.flushLocks.get(bookingId)) return;
    this.flushLocks.set(bookingId, true);

    try {
      const actions = this.actions
        .filter((a) => a.bookingId === bookingId)
        .sort((a, b) => a.timestamp - b.timestamp);

      for (const action of actions) {
        if (!this.shouldRetry(action)) continue;

        const result = await this.sendStatusUpdate(action);

        if (result.status === 409) {
          log.warn("flush conflict", {
            event: "flush_conflict",
            booking_id: action.bookingId,
            status: action.targetStatus,
            operation_id: action.id,
            result: "conflict",
            server_data: result.data,
          });
          this.manager.onTransitionFailed(action, "conflict");
          this.removeAction(action.id);
          await this.save();
          break;
        }

        if (result.status === 200 || result.status === 201) {
          this.removeAction(action.id);
          log.info("flush ok", {
            event: "flush_ok",
            booking_id: action.bookingId,
            status: action.targetStatus,
            operation_id: action.id,
            result: result.data?.unchanged ? "unchanged" : "sent",
            queue_size: this.actions.length,
          });
          await this.save();
          this.manager.onTransitionConfirmed(action);
          continue;
        }

        // Network / server error — keep in queue, stop this booking's flush
        action.retryCount++;
        action.lastRetryAt = Date.now();
        await this.save();

        if (action.retryCount >= MAX_RETRIES) {
          this.manager.onTransitionFailed(action, "error");
        }
        break;
      }
    } finally {
      this.flushLocks.set(bookingId, false);
    }
  }

  private shouldRetry(action: PendingAction): boolean {
    if (action.retryCount >= MAX_RETRIES) return false;
    if (!action.lastRetryAt) return true;
    const delay = Math.min(BACKOFF_BASE_MS * Math.pow(2, action.retryCount), BACKOFF_MAX_MS);
    return Date.now() - action.lastRetryAt >= delay;
  }

  private async sendStatusUpdate(action: PendingAction): Promise<{ status: number; data?: any }> {
    try {
      const response = await api.put(
        `/driver/me/bookings/${action.bookingId}/status`,
        { status: action.targetStatus },
        {
          headers: { "X-Idempotency-Key": action.id },
          validateStatus: (s: number) => s < 500,
        }
      );
      return { status: response.status, data: response.data };
    } catch (error: any) {
      const status = error?.response?.status ?? 0;
      return { status, data: error?.response?.data };
    }
  }

  private removeAction(id: string): void {
    this.actions = this.actions.filter((a) => a.id !== id);
  }
}
