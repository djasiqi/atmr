import { io, Socket } from "socket.io-client";
import { Platform } from "react-native";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { getResolvedApiBaseUrl } from "../../../core/api/client";
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
 * Natif : (0) WebSocket seul comme le flux driver ; (1) long-poll seul.
 * Important : avec `upgrade: false` au palier polling, Engine.IO n’essaie pas d’upgrader vers le WS
 * (sinon même « websocket error » en boucle alors qu’on a basculé sur le polling).
 */
const NATIVE_SOCKET_TRANSPORT_STEPS = [
  ["websocket"],
  ["polling"],
] as const;

/**
 * Résout l'URL Socket.IO pour le flux **company** uniquement.
 * Priorité : `EXPO_PUBLIC_COMPANY_SOCKET_URL`, puis l’origine de `EXPO_PUBLIC_API_BASE_URL`
 * (même hôte:port que l’API Flask). Pas de repli sur `EXPO_PUBLIC_DRIVER_SOCKET_URL` :
 * le realtime driver et le bridge company sont indépendants.
 * Exporté pour l’écran de diagnostic (paramètres entreprise).
 */
export function getResolvedCompanySocketUrl(): string {
  const company = process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL?.trim();
  if (company) {
    return company;
  }
  // Utiliser l'URL résolue (correction LAN IP Metro incluse en dev) plutôt que la variable d'env brute.
  // Sans ça, si l'IP LAN du PC change, l'API fonctionne (resolveBaseUrl la corrige) mais Socket.IO
  // pointe sur l'ancienne IP → ECONNREFUSED → description: 0.
  const resolved = getResolvedApiBaseUrl();
  if (resolved) {
    try {
      return new URL(resolved).origin;
    } catch {
      return "";
    }
  }
  return "";
}

/** Quelle env a résolu `getResolvedCompanySocketUrl()` (sans exposer les secrets — pour logs / diagnostics). */
export function getResolvedCompanySocketEnvSource():
  | "EXPO_PUBLIC_COMPANY_SOCKET_URL"
  | "EXPO_PUBLIC_API_BASE_URL"
  | "none" {
  if (process.env.EXPO_PUBLIC_COMPANY_SOCKET_URL?.trim()) {
    return "EXPO_PUBLIC_COMPANY_SOCKET_URL";
  }
  if (process.env.EXPO_PUBLIC_API_BASE_URL?.trim()) {
    return "EXPO_PUBLIC_API_BASE_URL";
  }
  return "none";
}

/** Identifiant entreprise nu (`"1"`) depuis `company:1` — query Socket.IO si le backend le lit à part `context_id`. */
export function getCompanyNumericIdFromContextId(contextId: string): string {
  const m = /^company:(.+)$/.exec(contextId.trim());
  return m?.[1]?.trim() ?? "";
}

const MAP_SILENCE_RESYNC_MS = 120_000;

/** Marge avant l’exp JWT : on rafraîchit le refresh pour réaligner Axios + auth Socket.IO. */
const PROACTIVE_ACCESS_REFRESH_BEFORE_EXP_MS = 90_000;

/** Attente du JWT après bootstrap / refresh (évite un failed définitif si connect() est un peu tôt). */
const TOKEN_WAIT_MAX = 12;
const TOKEN_WAIT_MS = 350;
const TOKEN_WAIT_FIRST_MS = 80;

const CONNECT_ERROR_DEV_LOG_COOLDOWN_MS = 8_000;

/** Champs utiles pour Metro / DevTools (sans jeton). Engine.IO renvoie souvent un message générique seul. */
/** `exp` JWT (ms) sans vérifier la signature — uniquement pour planifier un refresh avant coupure WS. */
function readJwtExpMs(token: string): number | null {
  try {
    const parts = token.split(".");
    if (parts.length < 2) return null;
    const seg = parts[1] ?? "";
    if (!seg) return null;
    const pad = (4 - (seg.length % 4)) % 4;
    const b64 = (seg + "=".repeat(pad)).replace(/-/g, "+").replace(/_/g, "/");
    if (typeof globalThis.atob !== "function") return null;
    const binary = globalThis.atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    const json = new TextDecoder().decode(bytes);
    const payload = JSON.parse(json) as { exp?: unknown };
    const exp = payload.exp;
    return typeof exp === "number" && Number.isFinite(exp) ? exp * 1000 : null;
  } catch {
    return null;
  }
}

function describeConnectErrorForDev(error: unknown): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (error instanceof Error) {
    out.name = error.name;
    out.message = error.message;
    const c = (error as Error & { cause?: unknown }).cause;
    if (c !== undefined) {
      out.cause = c instanceof Error ? c.message : String(c);
    }
  }
  if (typeof error === "object" && error !== null) {
    const e = error as Record<string, unknown>;
    for (const key of ["description", "type", "context"] as const) {
      const v = e[key];
      if (
        v !== undefined &&
        (typeof v === "string" || typeof v === "number" || typeof v === "boolean")
      ) {
        out[key] = v;
      }
    }
    const data = e.data;
    if (data && typeof data === "object" && !Array.isArray(data)) {
      const d = data as Record<string, unknown>;
      if (typeof d.code === "string" || typeof d.code === "number") {
        out.data_code = d.code;
      }
      if (typeof d.message === "string") {
        out.data_message = d.message;
      }
    }
  }
  return out;
}

class CompanyRealtimeBridge {
  private socket: Socket | null = null;
  private listeners = new Set<CompanyRealtimeListener>();
  private silenceTimer: ReturnType<typeof setInterval> | null = null;
  private proactiveAccessRefreshTimer: ReturnType<typeof setTimeout> | null = null;
  private tokenWaitTimer: ReturnType<typeof setTimeout> | null = null;
  private tokenWaitAttempt = 0;
  private hasConnectedOnce = false;
  /** iOS/Android : index dans `NATIVE_SOCKET_TRANSPORT_STEPS`. */
  private nativeTransportEscalation = 0;

  private lastNativeTransportStep(): number {
    return Math.min(
      this.nativeTransportEscalation,
      NATIVE_SOCKET_TRANSPORT_STEPS.length - 1,
    );
  }
  /** Limite le spam `console.warn` en dev lors de reconnexions en boucle. */
  private lastConnectErrorDevLogAt = 0;
  private lastConnectErrorDevLogMsg: string | null = null;
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

  private clearProactiveAccessRefreshTimer() {
    if (this.proactiveAccessRefreshTimer) {
      clearTimeout(this.proactiveAccessRefreshTimer);
      this.proactiveAccessRefreshTimer = null;
    }
  }

  /**
   * Met à jour le même objet `auth` passé au constructeur Socket.IO pour que chaque reconnexion
   * (après expiration JWT côté serveur) envoie le jeton courant d’Axios, pas seulement celui du 1er connect().
   */
  private syncHandshakeAuthFromSession(handshakeAuth: { token: string }) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { getAuthAccessToken } = require("../../../core/api/client") as {
        getAuthAccessToken: () => string | null;
      };
      const t = getAuthAccessToken();
      if (t) handshakeAuth.token = t;
    } catch {
      /* ignore */
    }
  }

  /** Rafraîchit le couple access/refresh puis réinjecte le Bearer dans l’objet auth du handshake. */
  private patchHandshakeAuthAfterRefresh(handshakeAuth: { token: string }) {
    void (async () => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const { getAuthAccessToken, refreshAuthTokenNow } = require("../../../core/api/client") as {
          getAuthAccessToken: () => string | null;
          refreshAuthTokenNow: () => Promise<boolean>;
        };
        await refreshAuthTokenNow();
        const t = getAuthAccessToken();
        if (t) handshakeAuth.token = t;
      } catch {
        /* ignore */
      }
    })();
  }

  private scheduleProactiveAccessRefresh(token: string, handshakeAuth: { token: string }) {
    this.clearProactiveAccessRefreshTimer();
    const expMs = readJwtExpMs(token);
    if (!expMs) return;
    const delay = expMs - Date.now() - PROACTIVE_ACCESS_REFRESH_BEFORE_EXP_MS;
    if (delay < 3_000 || delay > 24 * 60 * 60_000) return;
    this.proactiveAccessRefreshTimer = setTimeout(() => {
      this.proactiveAccessRefreshTimer = null;
      this.patchHandshakeAuthAfterRefresh(handshakeAuth);
    }, delay);
  }

  /** Coupe la socket sans remettre le bridge en « idle » (évite d’annuler une escalade transport en cours). */
  private softDisconnectSocket() {
    this.clearTokenWait();
    this.clearProactiveAccessRefreshTimer();
    if (this.silenceTimer) {
      clearInterval(this.silenceTimer);
      this.silenceTimer = null;
    }
    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }
  }

  /** Attend le Bearer (quelques centaines de ms) avant d’abandonner. */
  private startTokenWaitAndConnect(contextId: string) {
    this.clearTokenWait();
    const tick = () => {
      this.tokenWaitTimer = null;
      const finishAfterRefresh = async () => {
        try {
          // eslint-disable-next-line @typescript-eslint/no-require-imports
          const { getAuthAccessToken, refreshAuthTokenNow } = require("../../../core/api/client") as {
            getAuthAccessToken: () => string | null;
            refreshAuthTokenNow: () => Promise<boolean>;
          };
          await refreshAuthTokenNow();
          const after = getAuthAccessToken();
          if (after) {
            this.tokenWaitAttempt = 0;
            this.connect(contextId);
            return;
          }
        } catch {
          /* ignore */
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
      void finishAfterRefresh();
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

    const prevCtx = this.snapshot.contextId;
    if (prevCtx !== null && prevCtx !== contextId) {
      this.nativeTransportEscalation = 0;
    }

    this.softDisconnectSocket();
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
        "Aucune URL socket : définir EXPO_PUBLIC_COMPANY_SOCKET_URL ou EXPO_PUBLIC_API_BASE_URL (origine utilisée pour /socket.io)"
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

    const nativeStep =
      Platform.OS !== "web" ? this.lastNativeTransportStep() : 0;

    const transports: ("websocket" | "polling")[] =
      Platform.OS === "web"
        ? ["websocket", "polling"]
        : [...NATIVE_SOCKET_TRANSPORT_STEPS[nativeStep]];

    /** Palier polling : empêcher l’upgrade WS (cause typique du spam « websocket error » sur Android). */
    const disablePollingUpgrade =
      Platform.OS !== "web" && nativeStep === 1 && transports[0] === "polling";

    const companyNumericId = getCompanyNumericIdFromContextId(contextId);
    const socketQuery: Record<string, string> = {
      context_id: contextId,
      surface: "company",
      ...(companyNumericId.length > 0 ? { company_id: companyNumericId } : {}),
    };

    const handshakeAuth = { token };

    const socketOptions: NonNullable<Parameters<typeof io>[1]> = {
      transports,
      reconnection: true,
      reconnectionAttempts: 25,
      reconnectionDelay: 500,
      reconnectionDelayMax: 12_000,
      randomizationFactor: 0.5,
      timeout: 20_000,
      path: "/socket.io",
      query: socketQuery,
      // `_extract_token` côté Flask : auth (token) et/ou header Bearer (natif en complément)
      auth: handshakeAuth,
      ...(disablePollingUpgrade ? { upgrade: false as const } : {}),
      ...(Platform.OS !== "web"
        ? { extraHeaders: { Authorization: `Bearer ${token}` } as Record<string, string> }
        : {}),
    };

    if (typeof __DEV__ !== "undefined" && __DEV__) {
      const upgradeOpt = socketOptions.upgrade;
      console.log("[CompanySocket]", {
        socketUrl,
        urlEnvSource: getResolvedCompanySocketEnvSource(),
        transports,
        nativeTransportEscalation:
          Platform.OS !== "web" ? this.nativeTransportEscalation : undefined,
        disablePollingUpgrade,
        upgrade:
          upgradeOpt === false ? false : upgradeOpt === true ? true : "default",
        path: socketOptions.path,
        contextId,
        company_id: companyNumericId || undefined,
        surface: "company",
        hasToken: Boolean(token),
      });
    }

    const socket = io(socketUrl, socketOptions);
    this.socket = socket;

    this.scheduleProactiveAccessRefresh(token, handshakeAuth);

    socket.io.on("reconnect_attempt", () => {
      this.syncHandshakeAuthFromSession(handshakeAuth);
      if (!handshakeAuth.token?.trim()) {
        this.patchHandshakeAuthAfterRefresh(handshakeAuth);
      }
    });

    socket.on("connect", () => {
      this.lastConnectErrorDevLogMsg = null;
      this.nativeTransportEscalation = 0;
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
      this.syncHandshakeAuthFromSession(handshakeAuth);
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
      const msg =
        error instanceof Error
          ? error.message
          : typeof error === "object" &&
              error !== null &&
              "message" in error &&
              (error as { message: unknown }).message != null
            ? String((error as { message: unknown }).message)
            : String(error);
      const lower = msg.toLowerCase();
      const looksLikeTransportLayerIssue =
        lower.includes("xhr poll") ||
        lower.includes("xhr post") ||
        lower.includes("websocket");
      const fatalAuth = /\b401\b|\b403\b|unauthorized|forbidden/i.test(lower);

      if (
        Platform.OS !== "web" &&
        looksLikeTransportLayerIssue &&
        !fatalAuth &&
        this.nativeTransportEscalation < NATIVE_SOCKET_TRANSPORT_STEPS.length - 1
      ) {
        this.nativeTransportEscalation += 1;
        this.softDisconnectSocket();
        this.snapshot = {
          ...this.snapshot,
          contextId,
          status: "connecting",
          connected: false,
          lastError: null,
        };
        this.notify();
        setTimeout(() => this.connect(contextId), 0);
        return;
      }

      if (typeof __DEV__ !== "undefined" && __DEV__) {
        const now = Date.now();
        const cooled =
          now - this.lastConnectErrorDevLogAt >= CONNECT_ERROR_DEV_LOG_COOLDOWN_MS ||
          msg !== this.lastConnectErrorDevLogMsg;
        if (cooled) {
          console.warn("[CompanyRealtimeBridge] connect_error:", msg);
          const detail = describeConnectErrorForDev(error);
          if (Object.keys(detail).length > 0) {
            console.warn("[CompanySocket] connect_error detail", detail);
          }
          this.lastConnectErrorDevLogAt = now;
          this.lastConnectErrorDevLogMsg = msg;
        }
      }
      this.setStatus("reconnecting", msg);
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
    this.nativeTransportEscalation = 0;
    this.connect(this.snapshot.contextId);
  }

  disconnect() {
    this.nativeTransportEscalation = 0;
    this.softDisconnectSocket();
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
