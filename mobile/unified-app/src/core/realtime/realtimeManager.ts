import { io, Socket } from "socket.io-client";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import { appendSessionJournalEvent } from "../observability/sessionJournal";

const MAX_AUTH_REFRESH_ATTEMPTS = 5;
const MAX_RECONNECT_BACKOFF_MS = 30_000;
const RECONNECT_WINDOW_MS = 60_000;
const DEGRADED_HYSTERESIS_MS = Number(process.env.EXPO_PUBLIC_REALTIME_DEGRADED_HYSTERESIS_MS ?? "8000");

function reconnectWindowCap(): number {
  return Number(process.env.EXPO_PUBLIC_REALTIME_RECONNECT_WINDOW_CAP ?? "5");
}

// Codes renvoyés par le backend indiquant un état terminal — pas de retry.
const TERMINAL_AUTH_CODES = new Set([
  "session_revoked",
  "refresh_expired",
  "account_disabled",
  "tenant_access_revoked",
]);

type RealtimeState = {
  activeContextId: string | null;
  connected: boolean;
  mode: "idle" | "polling" | "socket";
  desiredTransport: "polling" | "socket";
  actualTransport: "idle" | "polling" | "socket";
  lastEventAt: string | null;
  lastError: string | null;
  reconnectAttempts: number;
  reconnectBackoffMs: number;
  authAttempts: number;
  authExhausted: boolean;
  authErrorCode: string | null;
  transportAuthority: "socket" | "polling" | "reconcile" | "degraded";
  degradedMode: boolean;
  degradedModeSince: string | null;
  reconnectWindowStartedAtMs: number | null;
  reconnectWindowAttempts: number;
};

type RealtimeLifecycleListener = (state: RealtimeState) => void;
type DriverEventListener = (event: unknown) => void;
type AuthExhaustedCallback = (reason: "exhausted" | "terminal", code?: string) => void;

class RealtimeManager {
  private state: RealtimeState = {
    activeContextId: null,
    connected: false,
    mode: "idle",
    desiredTransport: "polling",
    actualTransport: "idle",
    lastEventAt: null,
    lastError: null,
    reconnectAttempts: 0,
    reconnectBackoffMs: 0,
    authAttempts: 0,
    authExhausted: false,
    authErrorCode: null,
    transportAuthority: "polling",
    degradedMode: false,
    degradedModeSince: null,
    reconnectWindowStartedAtMs: null,
    reconnectWindowAttempts: 0,
  };
  private listeners = new Set<RealtimeLifecycleListener>();
  private driverEventListeners = new Set<DriverEventListener>();
  private authExhaustedCallbacks = new Set<AuthExhaustedCallback>();
  private socket: Socket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private degradedHysteresisTimer: ReturnType<typeof setTimeout> | null = null;

  private clearDegradedHysteresisTimer() {
    if (this.degradedHysteresisTimer) {
      clearTimeout(this.degradedHysteresisTimer);
      this.degradedHysteresisTimer = null;
    }
  }

  /** Re-évalue l’hystérésis dégradée sans autre événement réseau (passe setState). */
  private scheduleDegradedHysteresisRecheck(remainingMs: number) {
    this.clearDegradedHysteresisTimer();
    this.degradedHysteresisTimer = setTimeout(() => {
      this.degradedHysteresisTimer = null;
      this.setState({});
    }, Math.max(0, Math.ceil(remainingMs)));
  }

  private notify() {
    const snapshot = this.getSnapshot();
    this.listeners.forEach((listener) => {
      listener(snapshot);
    });
  }

  private setState(next: Partial<RealtimeState>) {
    const previous = this.state;
    this.state = { ...this.state, ...next };
    this.evaluateDegradedMode(previous);
    this.emitTransportStateTelemetry(previous);
    this.notify();
  }

  private emitTransportStateTelemetry(previous: RealtimeState) {
    if (
      previous.mode === this.state.mode &&
      previous.actualTransport === this.state.actualTransport &&
      previous.transportAuthority === this.state.transportAuthority &&
      previous.connected === this.state.connected
    ) {
      return;
    }
    emitDriverTelemetry("realtime.transport.state", {
      source: "core.realtime.manager",
      mode: this.state.mode,
      actual_transport: this.state.actualTransport,
      connected: this.state.connected,
      transport_authority: this.state.transportAuthority,
    });
  }

  private evaluateDegradedMode(previous: RealtimeState) {
    const shouldConsiderDegraded =
      this.state.desiredTransport === "socket" &&
      !this.state.authExhausted &&
      this.state.actualTransport !== "socket";
    if (!shouldConsiderDegraded) {
      this.clearDegradedHysteresisTimer();
      if (this.state.degradedMode) {
        const startedAt = this.state.degradedModeSince ? Date.parse(this.state.degradedModeSince) : null;
        emitDriverTelemetry("realtime.degraded.exited", {
          source: "core.realtime.manager",
          degraded_duration_ms: startedAt ? Date.now() - startedAt : null,
        });
      }
      this.state.degradedMode = false;
      this.state.degradedModeSince = null;
      if (this.state.transportAuthority === "degraded") {
        this.state.transportAuthority = this.state.actualTransport === "socket" ? "socket" : "polling";
      }
      return;
    }
    if (!this.state.degradedModeSince) {
      this.state.degradedModeSince = new Date(Date.now()).toISOString();
      this.scheduleDegradedHysteresisRecheck(DEGRADED_HYSTERESIS_MS);
      return;
    }
    const degradedElapsedMs = Date.now() - Date.parse(this.state.degradedModeSince);
    if (degradedElapsedMs < DEGRADED_HYSTERESIS_MS) {
      this.scheduleDegradedHysteresisRecheck(DEGRADED_HYSTERESIS_MS - degradedElapsedMs);
      return;
    }
    if (!this.state.degradedMode) {
      this.clearDegradedHysteresisTimer();
      this.state.degradedMode = true;
      this.state.transportAuthority = "degraded";
      emitDriverTelemetry("realtime.degraded.entered", {
        source: "core.realtime.manager",
        degraded_hysteresis_ms: DEGRADED_HYSTERESIS_MS,
      });
    }
  }

  connect(contextId: string, options?: { enableSocket?: boolean }) {
    const desiredTransport = options?.enableSocket ? "socket" : "polling";
    if (
      this.state.connected &&
      this.state.activeContextId === contextId &&
      this.state.desiredTransport === desiredTransport
    ) {
      return;
    }
    this.disconnect();
    void appendSessionJournalEvent("realtime.connect", {
      desired_transport: desiredTransport,
    }, contextId);
    const initialActual: RealtimeState["actualTransport"] = "polling";
    /* Connexion socket demandée : démarrage effectif encore en polling jusqu'à upgradeEngine */
    const socketPending = desiredTransport === "socket";
    this.setState({
      activeContextId: contextId,
      connected: true,
      mode: desiredTransport,
      desiredTransport,
      actualTransport: initialActual,
      transportAuthority: "polling",
      lastError: null,
      reconnectAttempts: 0,
      reconnectBackoffMs: 0,
      authAttempts: 0,
      authExhausted: false,
      authErrorCode: null,
      degradedMode: false,
      degradedModeSince: socketPending ? new Date().toISOString() : null,
      reconnectWindowStartedAtMs: null,
      reconnectWindowAttempts: 0,
    });
    if (desiredTransport === "socket") {
      this.connectSocket(contextId);
    }
  }

  disconnect() {
    this.clearDegradedHysteresisTimer();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }
    const shouldNotify =
      this.state.connected || this.state.activeContextId !== null || this.state.mode !== "idle";
    this.state = {
      activeContextId: null,
      connected: false,
      mode: "idle",
      desiredTransport: "polling",
      actualTransport: "idle",
      lastEventAt: null,
      lastError: null,
      reconnectAttempts: 0,
      reconnectBackoffMs: 0,
      authAttempts: 0,
      authExhausted: false,
      authErrorCode: null,
      transportAuthority: "polling",
      degradedMode: false,
      degradedModeSince: null,
      reconnectWindowStartedAtMs: null,
      reconnectWindowAttempts: 0,
    };
    if (shouldNotify) {
      void appendSessionJournalEvent("realtime.disconnect", {
        desired_transport: this.state.desiredTransport,
        actual_transport: this.state.actualTransport,
      }, this.state.activeContextId);
      this.notify();
    }
  }

  onContextSwitch(nextContextId: string | null, options?: { enableSocket?: boolean }) {
    this.disconnect();
    if (!nextContextId) return;
    if (typeof options?.enableSocket === "boolean") {
      this.connect(nextContextId, options);
      return;
    }
    this.connect(nextContextId, {
      enableSocket: this.state.desiredTransport === "socket",
    });
  }

  getSnapshot() {
    return this.state;
  }

  subscribe(listener: RealtimeLifecycleListener) {
    this.listeners.add(listener);
    listener(this.getSnapshot());
    return () => {
      this.listeners.delete(listener);
    };
  }

  subscribeDriverEvents(listener: DriverEventListener) {
    this.driverEventListeners.add(listener);
    return () => {
      this.driverEventListeners.delete(listener);
    };
  }

  isDriverSocketReady() {
    return Boolean(this.socket?.connected) && this.state.actualTransport === "socket";
  }

  setTransportAuthority(authority: RealtimeState["transportAuthority"], reason?: string) {
    if (this.state.transportAuthority === authority) return;
    this.setState({ transportAuthority: authority });
    emitDriverTelemetry("realtime.transport.authority", {
      source: "core.realtime.manager",
      transport_authority: authority,
      reason: reason ?? null,
    });
  }

  sendDriverLocationBatch(
    payload: {
      tracking_event_id: string;
      mission_id: number | null;
      latitude: number;
      longitude: number;
      accuracy?: number;
      heading?: number;
      speed?: number;
      timestamp?: string;
      location_mode?: string;
      is_background?: boolean;
    }[]
  ): boolean {
    if (!this.isDriverSocketReady() || !this.socket) return false;
    this.socket.emit("driver_location_batch", payload);
    return true;
  }

  onAuthExhausted(callback: AuthExhaustedCallback): () => void {
    this.authExhaustedCallbacks.add(callback);
    return () => {
      this.authExhaustedCallbacks.delete(callback);
    };
  }

  private notifyAuthExhausted(reason: "exhausted" | "terminal", code?: string) {
    this.authExhaustedCallbacks.forEach((cb) => cb(reason, code));
  }

  private emitDriverEvent(event: unknown) {
    this.driverEventListeners.forEach((listener) => {
      listener(event);
    });
  }

  private connectSocket(contextId: string) {
    const socketUrl = process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL;
    if (!socketUrl) {
      this.setState({
        mode: "polling",
        lastError: "EXPO_PUBLIC_DRIVER_SOCKET_URL not configured",
      });
      return;
    }
    this.socket = io(socketUrl, {
      transports: ["websocket"],
      reconnection: false, // géré manuellement pour contrôler l'auth recovery
      timeout: 10000,
      query: { context_id: contextId, surface: "driver" },
    });

    this.socket.on("connect", () => {
      this.setState({
        connected: true,
        mode: "socket",
        actualTransport: "socket",
        transportAuthority: "socket",
        lastError: null,
        authAttempts: 0,
        authExhausted: false,
        authErrorCode: null,
        reconnectWindowStartedAtMs: null,
        reconnectWindowAttempts: 0,
      });
      emitDriverTelemetry("realtime.socket.reconnect", {
        source: "core.realtime.manager",
        context_id: contextId,
      });
      void appendSessionJournalEvent("realtime.socket.connected", undefined, contextId);
    });

    this.socket.on("disconnect", () => {
      this.setState({
        connected: false,
        mode: "polling",
        actualTransport: "polling",
        transportAuthority: "degraded",
      });
      emitDriverTelemetry("realtime.socket.disconnect", {
        source: "core.realtime.manager",
        context_id: contextId,
      });
      void appendSessionJournalEvent("realtime.socket.disconnected", undefined, contextId);
      this.emitTransportMismatchIfNeeded(contextId);
      if (!this.state.authExhausted) {
        this.scheduleReconnect(contextId);
      }
    });

    this.socket.on("connect_error", (error) => {
      const errMessage = error.message ?? "";
      const errData = (error as unknown as { data?: { code?: string } }).data;
      const errCode = errData?.code ?? "";
      const isAuthError =
        errMessage.includes("401") ||
        errMessage.includes("403") ||
        errMessage.includes("Unauthorized") ||
        errMessage.includes("Forbidden");
      const isTerminal = TERMINAL_AUTH_CODES.has(errCode);

      this.setState({
        connected: false,
        mode: "polling",
        actualTransport: "polling",
        transportAuthority: "degraded",
        lastError: errMessage,
        authErrorCode: errCode || null,
      });
      emitDriverTelemetry("realtime.socket.disconnect", {
        source: "core.realtime.manager",
        context_id: contextId,
        reason: errMessage,
      });
      void appendSessionJournalEvent("realtime.socket.error", {
        reason: errMessage,
        error_code: errCode || null,
      }, contextId);
      this.emitTransportMismatchIfNeeded(contextId);

      if (isAuthError || isTerminal) {
        const nextAuthAttempts = this.state.authAttempts + 1;
        emitDriverTelemetry("realtime.auth.retry", {
          source: "core.realtime.manager",
          context_id: contextId,
          retry_count: nextAuthAttempts,
          reason: errMessage,
          error_code: errCode || null,
          terminal: isTerminal,
        });

        if (isTerminal || nextAuthAttempts >= MAX_AUTH_REFRESH_ATTEMPTS) {
          this.setState({
            authAttempts: nextAuthAttempts,
            authExhausted: true,
            transportAuthority: "degraded",
          });
          emitDriverTelemetry("realtime.auth.exhausted", {
            source: "core.realtime.manager",
            context_id: contextId,
            reason: isTerminal ? "terminal" : "exhausted",
            retry_count: nextAuthAttempts,
            error_code: errCode || null,
          });
          this.notifyAuthExhausted(isTerminal ? "terminal" : "exhausted", errCode || undefined);
          return;
        }

        this.setState({ authAttempts: nextAuthAttempts });
      }

      this.scheduleReconnect(contextId);
    });

    this.socket.on("driver_mission_event", (event: unknown) => {
      this.setState({ lastEventAt: new Date().toISOString() });
      this.emitDriverEvent(event);
    });

    this.socket.on("eta_changed", (event: unknown) => {
      this.setState({ lastEventAt: new Date().toISOString() });
      this.emitDriverEvent(event);
    });

    this.socket.on("driver_location_batch_ack", (event: unknown) => {
      this.setState({ lastEventAt: new Date().toISOString() });
      this.emitDriverEvent({
        event_type: "driver_location_batch_ack",
        payload: event,
      });
    });
  }

  private scheduleReconnect(contextId: string) {
    if (this.reconnectTimer) return;
    const now = Date.now();
    const windowStart = this.state.reconnectWindowStartedAtMs ?? now;
    const inSameWindow = now - windowStart <= RECONNECT_WINDOW_MS;
    const windowAttempts = inSameWindow ? this.state.reconnectWindowAttempts + 1 : 1;
    const nextWindowStart = inSameWindow ? windowStart : now;
    if (windowAttempts > reconnectWindowCap()) {
      this.setState({
        reconnectWindowStartedAtMs: nextWindowStart,
        reconnectWindowAttempts: windowAttempts,
        transportAuthority: "degraded",
      });
      emitDriverTelemetry("realtime.reconnect.cap_reached", {
        source: "core.realtime.manager",
        context_id: contextId,
        reconnect_attempt_window_cap: reconnectWindowCap(),
        reconnect_window_ms: RECONNECT_WINDOW_MS,
        reconnect_attempt_window_count: windowAttempts,
      });
      return;
    }
    const nextAttempts = this.state.reconnectAttempts + 1;
    const baseBackoff = Math.min(2 ** nextAttempts * 1000, MAX_RECONNECT_BACKOFF_MS);
    // Jitter pour eviter les reconnect storms synchrones.
    const jitter = baseBackoff * 0.3 * (Math.random() - 0.5);
    const backoffMs = Math.max(500, Math.floor(baseBackoff + jitter));
    this.setState({
      reconnectAttempts: nextAttempts,
      reconnectBackoffMs: backoffMs,
      reconnectWindowStartedAtMs: nextWindowStart,
      reconnectWindowAttempts: windowAttempts,
    });
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.state.activeContextId || this.state.authExhausted) return;
      this.connectSocket(contextId);
    }, backoffMs);
  }

  private emitTransportMismatchIfNeeded(contextId: string) {
    if (this.state.desiredTransport === this.state.actualTransport) return;
    emitDriverTelemetry("realtime.transport.mismatch", {
      source: "core.realtime.manager",
      context_id: contextId,
      desired_transport: this.state.desiredTransport,
      actual_transport: this.state.actualTransport,
    });
  }
}

export const realtimeManager = new RealtimeManager();
