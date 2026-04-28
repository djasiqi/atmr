import { io, Socket } from "socket.io-client";
import { Platform } from "react-native";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { contextRealtimeRouter } from "../../../core/realtime/contextRealtimeRouter";
import { getRealtimeChannelsForSurface } from "../../../core/realtime/contextRegistry";
import { normalizeCompanyEventType } from "../../../core/realtime/eventContracts";
import {
  CompanyRealtimeSnapshot,
  CompanyRealtimeStatus,
  reduceCompanyRealtimeStatus,
} from "./companyRealtimeState";

type CompanyRealtimeListener = (snapshot: CompanyRealtimeSnapshot) => void;

const SOCKET_EVENTS = getRealtimeChannelsForSurface("company");

/**
 * Résout l'URL Socket.IO. Priorité : URL explicite company/driver, sinon
 * l'origine de `EXPO_PUBLIC_API_BASE_URL` (même hôte:port que l’API Flask).
 * Exporté pour l’écran de diagnostic (paramètres entreprise).
 */
export function getResolvedCompanySocketUrl(): string {
  const a =
    process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL?.trim() ||
    process.env.EXPO_PUBLIC_DRIVER_SOCKET_URL?.trim() ||
    "";
  if (a) {
    return a;
  }
  const base = process.env.EXPO_PUBLIC_API_BASE_URL;
  if (base) {
    try {
      return new URL(base).origin;
    } catch {
      return "";
    }
  }
  return "";
}

const MAP_SILENCE_RESYNC_MS = 120_000;

/** Attente du JWT après bootstrap / refresh (évite un failed définitif si connect() est un peu tôt). */
const TOKEN_WAIT_MAX = 12;
const TOKEN_WAIT_MS = 350;
const TOKEN_WAIT_FIRST_MS = 80;

class CompanyRealtimeBridge {
  private socket: Socket | null = null;
  private listeners = new Set<CompanyRealtimeListener>();
  private silenceTimer: ReturnType<typeof setInterval> | null = null;
  private tokenWaitTimer: ReturnType<typeof setTimeout> | null = null;
  private tokenWaitAttempt = 0;
  private hasConnectedOnce = false;
  private snapshot: CompanyRealtimeSnapshot = {
    status: "idle",
    connected: false,
    contextId: null,
    lastEventAt: null,
    lastError: null,
  };

  private notify() {
    const current = this.getSnapshot();
    this.listeners.forEach((listener) => listener(current));
  }

  private clearTokenWait() {
    if (this.tokenWaitTimer) {
      clearTimeout(this.tokenWaitTimer);
      this.tokenWaitTimer = null;
    }
    this.tokenWaitAttempt = 0;
  }

  /** Attend le Bearer (quelques centaines de ms) avant d’abandonner. */
  private startTokenWaitAndConnect(contextId: string) {
    this.clearTokenWait();
    const tick = () => {
      this.tokenWaitTimer = null;
      let token: string | null = null;
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const { getAuthAccessToken } = require("../../../core/api/client") as {
          getAuthAccessToken: () => string | null;
        };
        token = getAuthAccessToken();
      } catch {
        token = null;
      }
      if (token) {
        this.tokenWaitAttempt = 0;
        this.connect(contextId);
        return;
      }
      this.tokenWaitAttempt += 1;
      if (this.tokenWaitAttempt >= TOKEN_WAIT_MAX) {
        this.tokenWaitAttempt = 0;
        this.setStatus(
          "failed",
          "Jeton d’authentification absent pour le WebSocket. Reconnectez-vous ou rechargez l’app."
        );
        return;
      }
      this.tokenWaitTimer = setTimeout(tick, TOKEN_WAIT_MS);
    };
    this.tokenWaitTimer = setTimeout(tick, this.tokenWaitAttempt === 0 ? TOKEN_WAIT_FIRST_MS : TOKEN_WAIT_MS);
  }

  private setStatus(nextStatus: CompanyRealtimeStatus, error?: string | null) {
    this.snapshot = {
      ...this.snapshot,
      status: reduceCompanyRealtimeStatus(this.snapshot.status, nextStatus),
      connected: nextStatus === "healthy",
      lastError: error ?? this.snapshot.lastError,
    };
    this.notify();
  }

  private updateEventTimestamp() {
    this.snapshot = {
      ...this.snapshot,
      lastEventAt: new Date().toISOString(),
      connected: true,
      status: reduceCompanyRealtimeStatus(this.snapshot.status, "healthy"),
      lastError: null,
    };
    this.notify();
  }

  private resetSilenceTimer() {
    if (this.silenceTimer) {
      clearInterval(this.silenceTimer);
      this.silenceTimer = null;
    }
    this.silenceTimer = setInterval(() => {
      if (!this.snapshot.lastEventAt) return;
      const silenceMs = Date.now() - Date.parse(this.snapshot.lastEventAt);
      if (silenceMs >= MAP_SILENCE_RESYNC_MS) {
        this.setStatus("degraded", "socket_silence_resync_required");
      }
    }, 5_000);
  }

  private bindDispatchEvents(contextId: string, socket: Socket) {
    SOCKET_EVENTS.forEach((eventName) => {
      socket.on(eventName, (payload: unknown) => {
        if (!payload || typeof payload !== "object") {
          return;
        }
        this.updateEventTimestamp();
        contextRealtimeRouter.dispatch(
          contextId,
          {
            ...payload,
            event_type: normalizeCompanyEventType(eventName) ?? eventName,
            context_type: "company",
          },
          { contextType: "company" }
        );
      });
    });
  }

  connect(contextId: string) {
    if (!isFeatureEnabled("company_realtime_enabled")) {
      this.snapshot = {
        ...this.snapshot,
        contextId,
        status: "idle",
        connected: false,
        lastError: "company realtime disabled by feature flag",
      };
      this.notify();
      return;
    }

    this.disconnect();
    this.snapshot = {
      ...this.snapshot,
      contextId,
      status: "connecting",
      connected: false,
      lastError: null,
    };
    this.notify();

    const socketUrl = getResolvedCompanySocketUrl();
    if (!socketUrl) {
      this.setStatus(
        "failed",
        "Aucune URL socket : EXPO_PUBLIC_COMPANY_SOCKET_URL (ou EXPO_PUBLIC_DRIVER_SOCKET_URL) ou EXPO_PUBLIC_API_BASE_URL (origine HTTP)"
      );
      return;
    }

    // require paresseux : évite de charger l’api client au chargement du module (Jest, dépendances circulaires).
    let token: string | null = null;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { getAuthAccessToken } = require("../../../core/api/client") as {
        getAuthAccessToken: () => string | null;
      };
      token = getAuthAccessToken();
    } catch {
      token = null;
    }
    if (!token) {
      this.startTokenWaitAndConnect(contextId);
      return;
    }
    this.clearTokenWait();

    const socketOptions: NonNullable<Parameters<typeof io>[1]> = {
      // Polling d’abord = meilleure tenue (proxies, réseau mobile) puis upgrade WS
      transports: ["polling", "websocket"],
      reconnection: true,
      reconnectionAttempts: 25,
      reconnectionDelay: 500,
      reconnectionDelayMax: 12_000,
      randomizationFactor: 0.5,
      timeout: 20_000,
      path: "/socket.io",
      query: { context_id: contextId, surface: "company" },
      // `_extract_token` côté Flask : auth (token) et/ou header Bearer (natif en complément)
      auth: { token },
      ...(Platform.OS !== "web"
        ? { extraHeaders: { Authorization: `Bearer ${token}` } as Record<string, string> }
        : {}),
    };

    const socket = io(socketUrl, socketOptions);
    this.socket = socket;

    socket.on("connect", () => {
      const isReconnect = this.hasConnectedOnce;
      this.hasConnectedOnce = true;
      this.snapshot = {
        ...this.snapshot,
        status: "healthy",
        connected: true,
        lastEventAt: new Date().toISOString(),
        lastError: null,
      };
      this.notify();
      socket.emit("join_company");
      this.resetSilenceTimer();
      if (isReconnect) {
        contextRealtimeRouter.dispatch(
          contextId,
          {
            event_type: "company_socket_reconnected",
            context_type: "company",
            reconnected_at: new Date().toISOString(),
          },
          { contextType: "company" }
        );
      }
    });

    socket.on("disconnect", () => {
      this.setStatus("reconnecting");
    });

    socket.on("connect_error", (error) => {
      if (typeof __DEV__ !== "undefined" && __DEV__) {
        // Aide au diag : mauvaise EXPO_PUBLIC_API_BASE_URL, JWT, capacité, pare-feu, etc.
        console.warn("[CompanyRealtimeBridge] connect_error:", error?.message || error);
      }
      this.setStatus("reconnecting", error instanceof Error ? error.message : String(error));
    });

    socket.io.on("reconnect_attempt", () => {
      this.setStatus("reconnecting");
    });

    socket.io.on("reconnect_failed", () => {
      this.setStatus("failed", "reconnect_failed");
    });

    this.bindDispatchEvents(contextId, socket);
  }

  reconnect() {
    if (!this.snapshot.contextId) return;
    this.connect(this.snapshot.contextId);
  }

  disconnect() {
    this.clearTokenWait();
    if (this.silenceTimer) {
      clearInterval(this.silenceTimer);
      this.silenceTimer = null;
    }
    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }
    this.snapshot = {
      ...this.snapshot,
      status: "idle",
      connected: false,
      lastError: null,
      lastEventAt: null,
    };
    this.notify();
  }

  subscribe(listener: CompanyRealtimeListener) {
    this.listeners.add(listener);
    listener(this.getSnapshot());
    return () => {
      this.listeners.delete(listener);
    };
  }

  getSnapshot(): CompanyRealtimeSnapshot {
    return { ...this.snapshot };
  }
}

export const companyRealtimeBridge = new CompanyRealtimeBridge();
