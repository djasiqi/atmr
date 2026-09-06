import { io, Socket } from "socket.io-client";
import { isFeatureEnabled } from "../featureFlags/registry";
import { mobileReconnectCircuitBreaker } from "./reconnectCircuitBreaker";
import { resolveDriverSocketUrl } from "./resolveDriverSocketUrl";
import { getWsCanaryExtraHeaders } from "./wsCanary";
import {
  observeConnectionAuthority,
  type AuthorityPayload,
} from "./connectionAuthority";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import {
  recordDriverSocketConnected,
  recordSocketReconnect,
} from "../observability/perfKpi";
import {
  recordRealtimeNotify,
  recordSocketEventByChannel,
  type SocketPerfChannel,
} from "../observability/perfInstrumentation";
import { isDriverSessionNetworkReady } from "../network/driverSessionNetworkGate";
import { appendSessionJournalEvent } from "../observability/sessionJournal";
import {
  recordSocketConnectTotal,
  recordSocketDisconnectTotal,
  recordSocketReconnectFailedTotal,
  recordSocketReconnectTotal,
} from "../observability/runtimeStabilityMetrics";

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
type TeamChatEvent = { type: "team_chat_message" | "team_chat_typing"; payload: unknown };
type TeamChatEventListener = (event: TeamChatEvent) => void;
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
  private teamChatEventListeners = new Set<TeamChatEventListener>();
  private authExhaustedCallbacks = new Set<AuthExhaustedCallback>();
  private socket: Socket | null = null;
  /** Incrémenté à chaque teardown pour invalider les callbacks d'un socket obsolète. */
  private socketGeneration = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private degradedHysteresisTimer: ReturnType<typeof setTimeout> | null = null;
  private hasSocketConnectedOnce = false;

  private isCurrentSocket(socket: Socket | null, generation: number): boolean {
    return socket !== null && this.socket === socket && this.socketGeneration === generation;
  }

  private disposeSocketInstance(socket: Socket): void {
    socket.removeAllListeners();
    socket.io.removeAllListeners?.();
    socket.disconnect();
  }

  private teardownActiveSocket(): void {
    if (this.socket) {
      this.disposeSocketInstance(this.socket);
      this.socket = null;
    }
    this.socketGeneration += 1;
  }

  /** Epoch auth courant (login/logout/refresh) — permet d'abandonner un refresh devenu obsolète. */
  private captureAuthEpoch(): number {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { getAuthEpoch } = require("../auth/authCredentialStore") as {
        getAuthEpoch: () => number;
      };
      return getAuthEpoch();
    } catch {
      return -1;
    }
  }

  private isAuthEpochStillCurrent(epoch: number): boolean {
    if (epoch === -1) return true;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { isCurrentAuthEpoch } = require("../auth/authCredentialStore") as {
        isCurrentAuthEpoch: (e: number) => boolean;
      };
      return isCurrentAuthEpoch(epoch);
    } catch {
      return true;
    }
  }

  /** Évite qu'un refresh token async réécrive l'auth d'un socket déjà remplacé (logout/reconnect). */
  private applyFreshAuthToken(socket: Socket, generation: number): boolean {
    if (!this.isCurrentSocket(socket, generation)) return false;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { getAuthAccessToken } = require("../api/client") as {
        getAuthAccessToken: () => string | null;
      };
      const token = getAuthAccessToken();
      if (!token) return false;
      if (socket.auth && typeof socket.auth === "object") {
        (socket.auth as { token?: string }).token = token;
      }
      return true;
    } catch {
      return false;
    }
  }

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
    const started = Date.now();
    const snapshot = this.getSnapshot();
    const listenerCount = this.listeners.size;
    this.listeners.forEach((listener) => {
      listener(snapshot);
    });
    recordRealtimeNotify(Date.now() - started, listenerCount);
  }

  private touchLastEventAtSilent() {
    this.state.lastEventAt = new Date().toISOString();
  }

  private onSocketPayload(channel: SocketPerfChannel, handler: () => void) {
    recordSocketEventByChannel(channel);
    handler();
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
    if (contextId.startsWith("driver:") && options?.enableSocket) {
      if (!isDriverSessionNetworkReady()) {
        emitDriverTelemetry("realtime.connect.blocked_before_session_ready", {
          source: "core.realtime.manager",
          context_id: contextId,
        });
        return;
      }
    }
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
    this.teardownActiveSocket();
    const prevDesiredTransport = this.state.desiredTransport;
    const prevActualTransport = this.state.actualTransport;
    const prevContextId = this.state.activeContextId;
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
        desired_transport: prevDesiredTransport,
        actual_transport: prevActualTransport,
      }, prevContextId);
      this.notify();
    }
  }

  onContextSwitch(nextContextId: string | null, options?: { enableSocket?: boolean }) {
    if (!nextContextId) {
      this.disconnect();
      return;
    }
    const desiredTransport =
      typeof options?.enableSocket === "boolean"
        ? options.enableSocket
          ? "socket"
          : "polling"
        : this.state.desiredTransport;

    if (
      this.socket?.connected &&
      desiredTransport === "socket" &&
      this.state.desiredTransport === "socket" &&
      !this.state.authExhausted &&
      this.state.activeContextId !== nextContextId
    ) {
      const prevContextId = this.state.activeContextId;
      this.state.activeContextId = nextContextId;
      if (this.socket.io.opts?.query && typeof this.socket.io.opts.query === "object") {
        (this.socket.io.opts.query as Record<string, string>).context_id = nextContextId;
      }
      try {
        this.socket.emit("join_driver_room", {});
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const { recordJoinRoomCount } = require("../observability/perfInstrumentation") as {
          recordJoinRoomCount: () => void;
        };
        recordJoinRoomCount();
      } catch {
        // ignore
      }
      void appendSessionJournalEvent(
        "realtime.connect.keepalive",
        {
          previous_context_id: prevContextId,
          next_context_id: nextContextId,
        },
        nextContextId
      );
      this.notify();
      return;
    }

    const previousDesiredTransport = this.state.desiredTransport;
    this.disconnect();
    if (typeof options?.enableSocket === "boolean") {
      this.connect(nextContextId, options);
      return;
    }
    this.connect(nextContextId, {
      enableSocket: previousDesiredTransport === "socket",
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

  subscribeTeamChatEvents(listener: TeamChatEventListener) {
    this.teamChatEventListeners.add(listener);
    return () => {
      this.teamChatEventListeners.delete(listener);
    };
  }

  /** Force le transport socket pour le contexte actif (ex. ouverture écran chat). */
  ensureDriverSocket(contextId: string) {
    void this.ensureDriverSocketAsync(contextId);
  }

  private async ensureDriverSocketAsync(contextId: string) {
    if (!contextId) return;
    const snap = this.getSnapshot();
    if (snap.authExhausted) return;

    if (snap.activeContextId !== contextId || snap.desiredTransport !== "socket") {
      this.connect(contextId, { enableSocket: true });
      return;
    }

    if (!this.socket) {
      this.connectSocket(contextId);
      return;
    }

    if (this.socket.connected) return;

    const socketBeforeAwait = this.socket;
    const generationBeforeAwait = this.socketGeneration;
    const epochAtStart = this.captureAuthEpoch();
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { refreshAuthTokenNow } = require("../api/client") as {
        refreshAuthTokenNow: () => Promise<boolean>;
        getAuthAccessToken: () => string | null;
      };
      await refreshAuthTokenNow().catch(() => false);

      // disconnect() / connect() concurrent peuvent annuler this.socket pendant l'await
      if (!this.isCurrentSocket(socketBeforeAwait, generationBeforeAwait)) {
        if (!this.socket && this.state.activeContextId === contextId) {
          this.connectSocket(contextId);
        }
        return;
      }
      // Epoch changé pendant l'await (logout/login concurrent) : abandonner ce refresh.
      if (!this.isAuthEpochStillCurrent(epochAtStart)) return;

      if (!this.applyFreshAuthToken(socketBeforeAwait, generationBeforeAwait)) return;
      if (!socketBeforeAwait.connected) {
        socketBeforeAwait.connect();
      }
    } catch {
      if (!this.isCurrentSocket(socketBeforeAwait, generationBeforeAwait) && !this.socket) {
        this.connectSocket(contextId);
      }
    }
  }

  emitTeamChatMessage(payload: Record<string, unknown>): boolean {
    if (!this.isDriverSocketReady() || !this.socket) return false;
    this.socket.emit("team_chat_message", payload);
    return true;
  }

  emitTeamChatTyping(payload: Record<string, unknown> = { surface: "driver" }): boolean {
    if (!this.isDriverSocketReady() || !this.socket) return false;
    this.socket.emit("team_chat_typing", payload);
    return true;
  }

  isDriverSocketReady() {
    return (
      Boolean(this.socket?.connected) &&
      this.state.desiredTransport === "socket" &&
      !this.state.authExhausted
    );
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
      sequence_id?: number;
      tracking_session_id?: string;
      batch_id?: string;
      position_id?: string;
      mission_id: number | null;
      latitude: number;
      longitude: number;
      accuracy?: number;
      heading?: number;
      speed?: number;
      timestamp?: string;
      location_mode?: string;
      is_background?: boolean;
      capture_id?: string | null;
    }[]
  ): boolean {
    if (!this.isDriverSocketReady() || !this.socket) return false;
    const first = payload[0];
    this.socket.emit("driver_location_batch", {
      tracking_session_id: first?.tracking_session_id,
      batch_id: first?.batch_id,
      positions: payload,
    });
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

  private emitTeamChatEvent(event: TeamChatEvent) {
    this.teamChatEventListeners.forEach((listener) => {
      listener(event);
    });
  }

  private connectSocket(contextId: string) {
    if (this.state.activeContextId && this.state.activeContextId !== contextId) {
      return;
    }
    this.teardownActiveSocket();
    const generation = this.socketGeneration;

    const socketUrl = resolveDriverSocketUrl();
    if (!socketUrl) {
      this.setState({
        mode: "polling",
        lastError: "Driver socket URL not configured",
      });
      return;
    }

    let accessToken: string | null = null;
    const handshakeAuth = { token: "" };
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { getAuthAccessToken } = require("../api/client") as {
        refreshAuthTokenNow: () => Promise<boolean>;
        getAuthAccessToken: () => string | null;
      };
      accessToken = getAuthAccessToken();
      handshakeAuth.token = accessToken ?? "";
    } catch {
      accessToken = null;
    }
    const canaryHeaders = getWsCanaryExtraHeaders();
    const socketOptions: NonNullable<Parameters<typeof io>[1]> = {
      transports: ["websocket", "polling"],
      reconnection: false, // géré manuellement pour contrôler l'auth recovery
      timeout: 10000,
      path: "/socket.io",
      query: { context_id: contextId, surface: "driver" },
      auth: handshakeAuth,
      ...(Object.keys(canaryHeaders).length > 0
        ? { extraHeaders: canaryHeaders }
        : {}),
    };

    const hasAccessToken = Boolean(accessToken);

    if (typeof __DEV__ !== "undefined" && __DEV__) {
      console.log("[DriverSocket]", {
        socketUrl,
        urlEnvSource: process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL?.trim()
          ? "EXPO_PUBLIC_DRIVER_SOCKET_URL"
          : "api_origin_fallback",
        transports: socketOptions.transports,
        path: socketOptions.path,
        contextId,
        surface: "driver",
        hasToken: hasAccessToken,
      });
    }

    const socket = io(socketUrl, socketOptions);
    this.socket = socket;

    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { refreshAuthTokenNow } = require("../api/client") as {
        refreshAuthTokenNow: () => Promise<boolean>;
      };
      const epochAtStart = this.captureAuthEpoch();
      void refreshAuthTokenNow()
        .then(() => {
          if (!this.isCurrentSocket(socket, generation)) return;
          if (!this.isAuthEpochStillCurrent(epochAtStart)) return;
          this.applyFreshAuthToken(socket, generation);
        })
        .catch(() => undefined);
    } catch {
      // ignore: handshake utilise déjà le token courant
    }

    socket.io.on("reconnect_attempt", () => {
      if (!this.isCurrentSocket(socket, generation)) return;
      const epochAtStart = this.captureAuthEpoch();
      void (async () => {
        try {
          // eslint-disable-next-line @typescript-eslint/no-require-imports
          const { refreshAuthTokenNow } = require("../api/client") as {
            refreshAuthTokenNow: () => Promise<boolean>;
            getAuthAccessToken: () => string | null;
          };
          await refreshAuthTokenNow();
          if (!this.isCurrentSocket(socket, generation)) return;
          if (!this.isAuthEpochStillCurrent(epochAtStart)) return;
          this.applyFreshAuthToken(socket, generation);
        } catch {
          // ignore: reconnect continuera en mode degrade/polling
        }
      })();
    });

    socket.on("connect", () => {
      if (!this.isCurrentSocket(socket, generation)) return;
      recordDriverSocketConnected(true);
      recordSocketConnectTotal({ context_id: contextId });
      if (this.hasSocketConnectedOnce) {
        recordSocketReconnectTotal({ context_id: contextId });
        recordSocketReconnect("driver_socket_reconnect", "driver");
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const { recordSocketReconnectCount } = require("../observability/perfInstrumentation") as {
          recordSocketReconnectCount: () => void;
        };
        recordSocketReconnectCount();
      }
      this.hasSocketConnectedOnce = true;
      try {
        socket.emit("join_driver_room", {});
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const { recordJoinRoomCount } = require("../observability/perfInstrumentation") as {
          recordJoinRoomCount: () => void;
        };
        recordJoinRoomCount();
      } catch {
        // ignore: rooms déjà jointes côté serveur au connect
      }
      if (isFeatureEnabled("realtime_reconnect_circuit_breaker_enabled")) {
        mobileReconnectCircuitBreaker.recordSuccess();
      }
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

    // Phase 2 PR B/C — gate D3.3 (fix G4) : observabilité ws-service vs backend.
    socket.on("connection.authority", (payload: unknown) => {
      if (!this.isCurrentSocket(socket, generation)) return;
      observeConnectionAuthority(payload as AuthorityPayload | undefined);
    });

    socket.on("disconnect", () => {
      if (!this.isCurrentSocket(socket, generation)) return;
      recordDriverSocketConnected(false);
      recordSocketDisconnectTotal({ context_id: contextId });
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

    socket.on("connect_error", (error) => {
      if (!this.isCurrentSocket(socket, generation)) return;
      recordSocketReconnectFailedTotal({ context_id: contextId });
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

      if (isFeatureEnabled("realtime_reconnect_circuit_breaker_enabled")) {
        mobileReconnectCircuitBreaker.recordFailure(socket);
      }
      this.scheduleReconnect(contextId);
    });

    socket.on("driver_mission_event", (event: unknown) => {
      this.onSocketPayload("driver_mission_event", () => {
        this.touchLastEventAtSilent();
        this.emitDriverEvent(event);
      });
    });

    socket.on("eta_changed", (event: unknown) => {
      this.onSocketPayload("eta_changed", () => {
        this.touchLastEventAtSilent();
        this.emitDriverEvent(event);
      });
    });

    socket.on("driver_location_batch_ack", (event: unknown) => {
      this.onSocketPayload("driver_location_batch_ack", () => {
        this.touchLastEventAtSilent();
        this.emitDriverEvent({
          event_type: "driver_location_batch_ack",
          payload: event,
        });
      });
    });

    socket.on("rate_limit_exceeded", (payload: unknown) => {
      this.onSocketPayload("rate_limit_exceeded", () => {
        this.emitDriverEvent({
          event_type: "rate_limit_exceeded",
          payload,
        });
      });
    });

    socket.on("team_chat_message", (payload: unknown) => {
      this.onSocketPayload("team_chat_message", () => {
        this.touchLastEventAtSilent();
        this.emitTeamChatEvent({ type: "team_chat_message", payload });
      });
    });

    socket.on("conversation_message", (payload: unknown) => {
      this.onSocketPayload("conversation_message", () => {
        this.touchLastEventAtSilent();
        this.emitTeamChatEvent({ type: "team_chat_message", payload });
      });
    });

    socket.on("team_chat_typing", (payload: unknown) => {
      this.onSocketPayload("other", () => {
        this.emitTeamChatEvent({ type: "team_chat_typing", payload });
      });
    });
  }

  private scheduleReconnect(contextId: string) {
    if (this.reconnectTimer) return;
    if (
      isFeatureEnabled("realtime_reconnect_circuit_breaker_enabled") &&
      !mobileReconnectCircuitBreaker.shouldAllowReconnectAttempt()
    ) {
      emitDriverTelemetry("realtime.reconnect.circuit_breaker_blocked", {
        source: "core.realtime.manager",
        context_id: contextId,
      });
      return;
    }
    const now = Date.now();
    const windowStart = this.state.reconnectWindowStartedAtMs;
    const inSameWindow = windowStart !== null && now - windowStart <= RECONNECT_WINDOW_MS;
    const windowAttempts = inSameWindow ? this.state.reconnectWindowAttempts + 1 : 1;
    const nextWindowStart = inSameWindow && windowStart !== null ? windowStart : now;
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
      if (this.state.activeContextId !== contextId) return;
      if (this.socket?.connected) return;
      void (async () => {
        const reconnectGeneration = this.socketGeneration;
        const epochAtStart = this.captureAuthEpoch();
        try {
          // eslint-disable-next-line @typescript-eslint/no-require-imports
          const { refreshAuthTokenNow } = require("../api/client") as {
            refreshAuthTokenNow: () => Promise<boolean>;
          };
          await refreshAuthTokenNow();
        } catch {
          // ignore: reconnect attempt still proceeds
        }
        if (this.state.activeContextId !== contextId || this.state.authExhausted) return;
        if (this.socketGeneration !== reconnectGeneration && this.socket?.connected) return;
        // Epoch changé pendant l'await (logout/login concurrent) : ne pas reconnecter avec un contexte obsolète.
        if (!this.isAuthEpochStillCurrent(epochAtStart)) return;
        this.connectSocket(contextId);
      })();
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
