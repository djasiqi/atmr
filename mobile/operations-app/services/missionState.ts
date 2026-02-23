import AsyncStorage from "@react-native-async-storage/async-storage";
import type { Booking } from "./api";
import { getAssignedTrips } from "./api";
import type { BookingStatus } from "@/utils/bookingStatus";
import { normalizeBookingStatus } from "@/utils/bookingStatus";
import {
  PendingActionsQueue,
  type PendingAction,
} from "./pendingActionsQueue";
import { getLogger } from "@/utils/logger";
import { buildQuickActionLink, safeOpenURL, openNavigation } from "./deepLinks";

const log = getLogger("MissionState");

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MissionBarStatus = "ASSIGNED" | "EN_ROUTE" | "IN_PROGRESS" | "COMPLETED";

export interface BookingPreview {
  id: number;
  pickup_at: string;
  client_display: string;
  pickup_short: string;
  dropoff_short: string;
  can_show_identity: boolean;
}

export type ActionButton = {
  id: string;
  label: string;
  targetStatus?: MissionBarStatus;
  type: "status" | "call" | "incident";
};

export interface MissionState {
  activeMission: Booking | null;
  nextBookingPreview: BookingPreview | null;
  currentStatus: MissionBarStatus;
  allowedTransitions: MissionBarStatus[];
  allowedActions: ActionButton[];
  isNavigating: boolean;
  lastNavigationDestination: string | null;
  privacyMode: boolean;
}

type MissionEventType =
  | "state_changed"
  | "mission_started"
  | "mission_stopped"
  | "transition_requested"
  | "transition_confirmed"
  | "transition_failed"
  | "reconciliation";

type MissionEventListener = (event: MissionEventType, state: MissionState) => void;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STORAGE_KEY = "active_mission_state_v1";
const STORAGE_KEY_OLD = "active_mission_state";
const MISSIONS_CACHE_KEY = "missions_cache_v2";

const VALID_TRANSITIONS: Record<MissionBarStatus, MissionBarStatus[]> = {
  ASSIGNED: ["EN_ROUTE"],
  EN_ROUTE: ["IN_PROGRESS"],
  IN_PROGRESS: ["COMPLETED"],
  COMPLETED: [],
};

const STATUS_ACTIONS: Record<MissionBarStatus, ActionButton[]> = {
  ASSIGNED: [
    { id: "EN_ROUTE", label: "En route", targetStatus: "EN_ROUTE", type: "status" },
    { id: "CALL", label: "Appeler", type: "call" },
  ],
  EN_ROUTE: [
    { id: "IN_PROGRESS", label: "À bord", targetStatus: "IN_PROGRESS", type: "status" },
    { id: "CALL", label: "Appeler", type: "call" },
  ],
  IN_PROGRESS: [
    { id: "COMPLETED", label: "Terminer", targetStatus: "COMPLETED", type: "status" },
    { id: "CALL", label: "Appeler", type: "call" },
  ],
  COMPLETED: [],
};

// ---------------------------------------------------------------------------
// Singleton MissionStateManager
// ---------------------------------------------------------------------------

class MissionStateManagerImpl {
  private state: MissionState = this.emptyState();
  private listeners: Set<MissionEventListener> = new Set();
  private queue = new PendingActionsQueue(this);
  private hydrated = false;
  private reconciliationTimer: ReturnType<typeof setInterval> | null = null;
  private lastReconciliationAt = 0;
  private reconciliationInProgress = false;
  private lastMapsOpenAt = 0;
  private lastNavigatedTarget: string | null = null;
  private static RECONCILIATION_INTERVAL_MS = 3 * 60 * 1000;
  private static RECONCILIATION_COOLDOWN_MS = 20_000;
  private static MAPS_DEBOUNCE_MS = 3_000;

  // -- State helpers -------------------------------------------------------

  private emptyState(): MissionState {
    return {
      activeMission: null,
      nextBookingPreview: null,
      currentStatus: "ASSIGNED",
      allowedTransitions: [],
      allowedActions: [],
      isNavigating: false,
      lastNavigationDestination: null,
      privacyMode: false,
    };
  }

  private deriveFromStatus(status: MissionBarStatus): Pick<MissionState, "allowedTransitions" | "allowedActions"> {
    return {
      allowedTransitions: VALID_TRANSITIONS[status] ?? [],
      allowedActions: STATUS_ACTIONS[status] ?? [],
    };
  }

  getState(): MissionState {
    return { ...this.state };
  }

  getCallablePhone(): string | null {
    const m = this.state.activeMission;
    if (!m) return null;
    return m.client?.contact_phone ?? m.client?.phone ?? m.client_phone ?? null;
  }

  // -- Event bus -----------------------------------------------------------

  subscribe(listener: MissionEventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(event: MissionEventType) {
    const snapshot = this.getState();
    for (const l of this.listeners) {
      try {
        l(event, snapshot);
      } catch (e) {
        log.error("listener error", { error: e });
      }
    }
  }

  // -- Persistence ---------------------------------------------------------

  private async persist() {
    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({
        activeMission: this.state.activeMission,
        currentStatus: this.state.currentStatus,
        nextBookingPreview: this.state.nextBookingPreview,
        isNavigating: this.state.isNavigating,
        lastNavigationDestination: this.state.lastNavigationDestination,
        privacyMode: this.state.privacyMode,
      }));
    } catch (e) {
      log.error("persist error", { error: e });
    }
  }

  private async clearPersistence() {
    try {
      await AsyncStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      log.error("clear persistence error", { error: e });
    }
  }

  // -- Hydration (self-sufficient for headless) ----------------------------

  async ensureHydrated(options?: { skipNetwork?: boolean }): Promise<boolean> {
    if (this.hydrated && this.state.activeMission) return true;

    const skipNetwork = options?.skipNetwork ?? false;

    // Step 1: dedicated key (with migration from old key)
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
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved.activeMission) {
          this.applyHydration(saved);
          log.info("hydration from active_mission_state", {
            bookingId: saved.activeMission?.id,
            status: saved.currentStatus,
          });
          return true;
        }
      }
    } catch (e) {
      log.warn("hydration step 1 failed", { error: e });
    }

    // Step 2: general missions cache
    try {
      const raw = await AsyncStorage.getItem(MISSIONS_CACHE_KEY);
      if (raw) {
        const missions: Booking[] = JSON.parse(raw);
        const active = missions.find((m) => {
          const s = normalizeBookingStatus(m.status);
          return s === "ASSIGNED" || s === "EN_ROUTE" || s === "IN_PROGRESS";
        });
        if (active) {
          this.applyHydration({
            activeMission: active,
            currentStatus: normalizeBookingStatus(active.status) as MissionBarStatus,
          });
          log.info("hydration from missions_cache", { bookingId: active.id });
          return true;
        }
      }
    } catch (e) {
      log.warn("hydration step 2 failed", { error: e });
    }

    // Step 3: fetch from API — skipped in background/headless to avoid ANR
    if (!skipNetwork) {
      try {
        const bookings = await getAssignedTrips();
        const active = bookings.find((m) => {
          const s = normalizeBookingStatus(m.status);
          return s === "ASSIGNED" || s === "EN_ROUTE" || s === "IN_PROGRESS";
        });
        if (active) {
          this.applyHydration({
            activeMission: active,
            currentStatus: normalizeBookingStatus(active.status) as MissionBarStatus,
          });
          log.info("hydration from api_fetch", { bookingId: active.id });
          return true;
        }
      } catch (e) {
        log.warn("hydration step 3 fetch failed", { error: e });
      }
    }

    log.warn("hydration none all steps failed", { skipNetwork });

    // Step 4: fallback — open Quick Actions via deep link (only foreground)
    if (!skipNetwork) {
      try {
        await safeOpenURL(buildQuickActionLink({}));
      } catch {
        // best-effort
      }
    }
    return false;
  }

  private applyHydration(saved: Partial<{
    activeMission: Booking;
    currentStatus: MissionBarStatus;
    nextBookingPreview: BookingPreview | null;
    isNavigating: boolean;
    lastNavigationDestination: string | null;
    privacyMode: boolean;
  }>) {
    const status = (saved.currentStatus ?? "ASSIGNED") as MissionBarStatus;
    const mission = saved.activeMission ?? null;
    const privacyFromBooking = mission?.can_show_identity === false
      || (mission as any)?.institution_privacy_mode === true;
    this.state = {
      activeMission: mission,
      nextBookingPreview: saved.nextBookingPreview ?? null,
      currentStatus: status,
      ...this.deriveFromStatus(status),
      isNavigating: saved.isNavigating ?? false,
      lastNavigationDestination: saved.lastNavigationDestination ?? null,
      privacyMode: saved.privacyMode ?? privacyFromBooking ?? false,
    };
    this.hydrated = true;
  }

  // -- Public API ----------------------------------------------------------

  async startMission(mission: Booking, destination?: string): Promise<void> {
    const status = normalizeBookingStatus(mission.status) as MissionBarStatus;
    const privacyFromBooking = (mission as any).can_show_identity === false
      || (mission as any).institution_privacy_mode === true;
    this.state = {
      activeMission: mission,
      nextBookingPreview: this.state.nextBookingPreview,
      currentStatus: status,
      ...this.deriveFromStatus(status),
      isNavigating: true,
      lastNavigationDestination: destination ?? null,
      privacyMode: privacyFromBooking,
    };
    this.hydrated = true;
    await this.persist();
    this.startReconciliationTimer();
    this.emit("mission_started");
  }

  async requestTransition(targetStatus: MissionBarStatus): Promise<boolean> {
    await this.ensureHydrated();
    const bookingId = this.state.activeMission?.id;
    if (!bookingId) {
      log.warn("request transition no active mission", { event: "request_transition", result: "no_active_mission", status: targetStatus });
      return false;
    }

    if (this.state.currentStatus === targetStatus) {
      return true;
    }

    if (!this.state.allowedTransitions.includes(targetStatus)) {
      log.warn("request transition invalid", {
        event: "request_transition",
        booking_id: bookingId,
        status: targetStatus,
        result: "invalid",
        current: this.state.currentStatus,
      });
      return false;
    }

    log.info("request transition", {
      event: "request_transition",
      booking_id: bookingId,
      status: targetStatus,
      source: "requestTransition",
    });

    this.state.currentStatus = targetStatus;
    Object.assign(this.state, this.deriveFromStatus(targetStatus));
    this.emit("transition_requested");
    await this.persist();

    await this.queue.enqueue({
      bookingId: String(bookingId),
      targetStatus,
    });

    return true;
  }

  /**
   * Called by PendingActionsQueue after successful API call.
   * Re-open Maps when IN_PROGRESS is confirmed (destination switches to dropoff).
   */
  onTransitionConfirmed(action: PendingAction): void {
    this.emit("transition_confirmed");

    if (action.targetStatus === "IN_PROGRESS") {
      this.navigateToCurrentTarget();
    }
  }

  /**
   * Called by PendingActionsQueue when API returns 409, max retries, or stale.
   * On conflict/stale, trigger reconciliation to sync with server truth.
   */
  onTransitionFailed(action: PendingAction, reason: "conflict" | "error" | "stale"): void {
    log.warn("transition failed", {
      event: "transition_failed",
      booking_id: action.bookingId,
      status: action.targetStatus,
      operation_id: action.id,
      reason,
    });
    this.emit("transition_failed");

    if (reason === "conflict" || reason === "stale") {
      this.reconcileNow();
    }
  }

  /**
   * Reconcile local state with server data (socket events, foreground resume).
   */
  async updateFromServer(bookings: Booking[]): Promise<void> {
    if (!this.state.activeMission) return;

    const serverMission = bookings.find((b) => b.id === this.state.activeMission!.id);
    if (!serverMission) return;

    const serverStatus = normalizeBookingStatus(serverMission.status);

    if (serverStatus === "CANCELED" || serverStatus === "COMPLETED" || serverStatus === "RETURN_COMPLETED") {
      this.state.isNavigating = false;
      await this.queue.purge(String(this.state.activeMission.id));
      await this.stopMission();
      this.emit("reconciliation");
      return;
    }

    // Update local if server is ahead
    const current = this.state.currentStatus;
    if (serverStatus !== current && isAhead(serverStatus as MissionBarStatus, current)) {
      this.state.currentStatus = serverStatus as MissionBarStatus;
      this.state.activeMission = serverMission;
      Object.assign(this.state, this.deriveFromStatus(serverStatus as MissionBarStatus));
      await this.persist();
      this.emit("reconciliation");
    }

    // Update next booking preview from response metadata
    if ((serverMission as any).next_booking_preview) {
      this.state.nextBookingPreview = (serverMission as any).next_booking_preview;
    }
  }

  setNextBookingPreview(preview: BookingPreview | null): void {
    this.state.nextBookingPreview = preview;
    this.emit("state_changed");
  }

  async stopMission(): Promise<void> {
    this.stopReconciliationTimer();
    this.lastNavigatedTarget = null;
    this.lastMapsOpenAt = 0;
    this.state = this.emptyState();
    this.hydrated = false;
    await this.clearPersistence();
    this.emit("mission_stopped");
  }

  setNavigating(navigating: boolean): void {
    this.state.isNavigating = navigating;
    this.persist();
  }

  /**
   * Returns the destination the driver should navigate to based on current status.
   * Terminal statuses return null (no navigation needed).
   */
  getCurrentTarget(): string | null {
    const m = this.state.activeMission;
    if (!m) return null;
    const s = this.state.currentStatus;
    if (s === "COMPLETED") return null;
    if (s === "ASSIGNED" || s === "EN_ROUTE") {
      return m.pickup_location ?? null;
    }
    return m.dropoff_location ?? null;
  }

  /**
   * Open Maps toward the current target if the driver is navigating.
   * Debounced (3s) and skips if destination hasn't changed.
   */
  async navigateToCurrentTarget(): Promise<void> {
    if (!this.state.isNavigating) return;
    const target = this.getCurrentTarget();
    if (!target) return;

    const now = Date.now();
    if (now - this.lastMapsOpenAt < MissionStateManagerImpl.MAPS_DEBOUNCE_MS) return;
    if (target === this.lastNavigatedTarget) return;

    this.lastMapsOpenAt = now;
    try {
      await openNavigation(target);
      this.lastNavigatedTarget = target;
      this.state.lastNavigationDestination = target;
      await this.persist();
      log.info("maps reopen ok", {
        event: "maps_reopen_ok",
        booking_id: this.state.activeMission?.id,
        status: this.state.currentStatus,
        target,
      });
    } catch (e) {
      log.warn("maps reopen failed", {
        event: "maps_reopen_failed",
        booking_id: this.state.activeMission?.id,
        status: this.state.currentStatus,
        target,
      });
    }
  }

  async syncPendingActions(): Promise<void> {
    await this.queue.flush();
  }

  async reconcileNow(): Promise<void> {
    const now = Date.now();
    if (now - this.lastReconciliationAt < MissionStateManagerImpl.RECONCILIATION_COOLDOWN_MS) return;
    if (this.reconciliationInProgress) return;
    if (!this.state.activeMission) return;
    this.reconciliationInProgress = true;
    this.lastReconciliationAt = now;
    try {
      const bookings = await getAssignedTrips();
      await this.updateFromServer(bookings);
      await this.queue.flush();
    } catch {
      // best-effort
    } finally {
      this.reconciliationInProgress = false;
    }
  }

  init(): void {
    this.queue.startListening();
  }

  isActive(): boolean {
    return this.state.activeMission !== null;
  }

  private startReconciliationTimer(): void {
    this.stopReconciliationTimer();
    this.reconciliationTimer = setInterval(() => {
      this.reconcileNow();
    }, MissionStateManagerImpl.RECONCILIATION_INTERVAL_MS);
  }

  private stopReconciliationTimer(): void {
    if (this.reconciliationTimer) {
      clearInterval(this.reconciliationTimer);
      this.reconciliationTimer = null;
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STATUS_ORDER: Record<string, number> = {
  ASSIGNED: 0,
  EN_ROUTE: 1,
  IN_PROGRESS: 2,
  COMPLETED: 3,
};

function isAhead(a: MissionBarStatus, b: MissionBarStatus): boolean {
  return (STATUS_ORDER[a] ?? 0) > (STATUS_ORDER[b] ?? 0);
}

// ---------------------------------------------------------------------------
// Singleton export
// ---------------------------------------------------------------------------

export const MissionStateManager = new MissionStateManagerImpl();
MissionStateManager.init();
