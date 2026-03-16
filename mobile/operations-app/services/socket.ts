// services/socket.ts
import { getLogger } from "@/utils/logger";
import { io, type Socket } from "socket.io-client";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import Constants from "expo-constants";
import * as Application from "expo-application";
import { baseURL, getAssignedTrips, getCompanyMessages, type Booking, type Message } from "./api";
import { resolveBookingConflicts, resolveMessageConflicts, type Conflict } from "./conflictResolution";
import { getSessionDiagHeaderValue, pushSessionEvent, setConnectionStateSuffix } from "./sessionJournal";
import type { SessionEvent } from "./sessionJournal";
import { getNetworkStateSnapshot, subscribeToNetworkState } from "./networkState";
import { extractAuthStatus } from "./socketAuthUtils";
import { secureStorage } from "./storage";
import { logAuthEvent } from "./authLogging";
import { refreshDriverTokenOrchestrated } from "./driverTokenOrchestrator";

const log = getLogger("Socket");
const NATIVE_APP_ID = Application.applicationId || "";
const IS_NATIVE_PROD_APP_ID = NATIVE_APP_ID === "ch.liri.operations";
const EXECUTION_ENVIRONMENT = String(Constants.executionEnvironment || "");
const IS_PRODUCTION_RUNTIME = !__DEV__ && IS_NATIVE_PROD_APP_ID;

/** Plan 2G/3G : Listeners appelés à chaque connexion socket (pour syncEngine flush). */
const socketConnectListeners = new Set<() => void>();
export function addSocketConnectListener(cb: () => void): () => void {
  socketConnectListeners.add(cb);
  return () => socketConnectListeners.delete(cb);
}

/** Rôle socket stable — source unique de vérité (éviter strings magiques). */
export type SocketRole = "driver" | "enterprise";

/** P0.3+ État connexion socket pour UI (ONLINE / RECONNECTING / OFFLINE). Jamais de logout sur disconnect. */
export type ConnectionState = "ONLINE" | "RECONNECTING" | "OFFLINE";
let connectionState: ConnectionState = "OFFLINE";
let explicitDisconnect = false;

/** P2.1.1 — Backoff + limite tentatives pour connect_error 401/403 (driver) */
const MAX_DRIVER_AUTH_REFRESH_ATTEMPTS = 5;
let driverConnectErrorAuthAttempts = 0;
let driverConnectErrorAuthBackoffMs = 2000;
/** P2.1.1b — Anti-concurrence : une seule tentative manuelle en vol. */
let manualReconnectInProgress = false;
/** P2.1.1b — Après 5 échecs : UI peut afficher "Connexion temps réel indisponible". */
let authRecoveryExhausted = false;

/** P2.1.2 — Enterprise : après 10 tentatives auto, Socket.IO stoppe → bouton Reconnecter. */
let reconnectExhausted = false;

export function getAuthRecoveryExhausted(): boolean {
  return authRecoveryExhausted;
}

export function getReconnectExhausted(): boolean {
  return reconnectExhausted;
}

export function getSocketRole(): SocketRole | null {
  return socketRole;
}

// ✅ EventEmitter simple pour notifier les composants du resync
class SimpleEventEmitter {
  private listeners: Map<string, Set<Function>> = new Map();

  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: Function) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.delete(callback);
    }
  }

  emit(event: string, ...args: any[]) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach((callback) => {
        try {
          callback(...args);
        } catch (error) {
          log.warn("listener error", { event, error });
        }
      });
    }
  }
}

const resyncEmitter = new SimpleEventEmitter();

// ✅ EventEmitter pour les événements de booking (new_booking, booking_updated, booking_cancelled)
const bookingEmitter = new SimpleEventEmitter();

// ✅ EventEmitter pour les événements de messages (team_chat_message)
const messageEmitter = new SimpleEventEmitter();

/** P2.1.2b — État socket pour UI (subscribe au lieu de polling). */
export type SocketStatusPayload = {
  role: SocketRole | null;
  connectionState: ConnectionState;
  reconnectExhausted: boolean;
  authRecoveryExhausted: boolean;
};

type SocketStatusListener = (payload: SocketStatusPayload) => void;
const socketStatusListeners = new Set<SocketStatusListener>();

function getSocketStatusPayload(): SocketStatusPayload {
  return {
    role: socketRole,
    connectionState,
    reconnectExhausted,
    authRecoveryExhausted,
  };
}

/** Shallow hash pour dedupe — n'émettre que si l'état a changé. */
function payloadHash(p: SocketStatusPayload): string {
  return `${p.role}|${p.connectionState}|${p.reconnectExhausted}|${p.authRecoveryExhausted}`;
}
let lastEmittedHash: string | null = null;

function emitSocketStatusChange() {
  const payload = getSocketStatusPayload();
  const hash = payloadHash(payload);
  if (hash === lastEmittedHash) return;
  lastEmittedHash = hash;
  // Itérer sur copie pour éviter reentrancy (ex: listener appelle reconnectSocketManually)
  const copy = Array.from(socketStatusListeners);
  copy.forEach((cb) => {
    try {
      cb(payload);
    } catch (e) {
      log.warn("socket status listener error", { error: e });
    }
  });
}

export function subscribeSocketStatus(listener: SocketStatusListener): () => void {
  socketStatusListeners.add(listener);
  listener(getSocketStatusPayload());
  return () => {
    socketStatusListeners.delete(listener); // Idempotent : delete 2× safe
  };
}

// ✅ Set pour déduplication des event_id (persistant entre reconnexions)
const seenEventIds = new Set<string>();
const MAX_SEEN_EVENTS = 1000;

// ✅ Helper pour déduplication
function checkAndAddEventId(eventId: string | undefined | null, eventName: string): boolean {
  if (!eventId || typeof eventId !== "string") {
    return true; // Pas d'event_id, continuer normalement (backward compatible)
  }
  
  if (seenEventIds.has(eventId)) {
    log.warn("duplicate event ignored", { event_id: eventId, event_name: eventName });
    return false; // Doublon, ignorer
  }
  
  seenEventIds.add(eventId);
  // Limiter la taille du Set (garder les 1000 derniers - FIFO)
  if (seenEventIds.size > MAX_SEEN_EVENTS) {
    const first = seenEventIds.values().next().value;
    if (first && typeof first === "string") {
      seenEventIds.delete(first);
    }
  }
  
  return true; // Nouvel événement, continuer
}

// ✅ Résoudre l'URL Socket.IO depuis EXPO_PUBLIC_SOCKET_URL ou fallback
const getSocketOrigin = (): string => {
  const rawAppVariant = String(
    Constants.expoConfig?.extra?.APP_VARIANT || process.env.APP_VARIANT || "prod"
  );
  const nativeAppId = Application.applicationId || "";
  const appVariant = nativeAppId === "ch.liri.operations" ? "prod" : rawAppVariant;
  const isProduction =
    appVariant === "prod" &&
    !__DEV__ &&
    Constants.executionEnvironment === "standalone";
  // PRIORITÉ 1: Variable d'environnement dédiée
  const socketUrl = process.env.EXPO_PUBLIC_SOCKET_URL;
  if (socketUrl) {
    if (!isProduction) {
      // Guardrail dev: if API and Socket hosts diverge, prefer API host to avoid endless reconnect loop.
      try {
        const apiUrl = new URL(baseURL);
        const socketParsed = new URL(socketUrl);
        if (apiUrl.host && socketParsed.host && apiUrl.host !== socketParsed.host) {
          const fallbackOrigin = `${apiUrl.protocol}//${apiUrl.host}`.replace(/\/+$/, "");
          log.warn("socket url host mismatch in dev, fallback to api host", {
            socketHost: socketParsed.host,
            apiHost: apiUrl.host,
            fallbackOrigin,
          });
          return fallbackOrigin;
        }
      } catch (e) {
        log.warn("socket url parse mismatch guard skipped", {
          error: e instanceof Error ? e.message : String(e),
        });
      }
    }
    // Validation en production : doit être HTTPS
    if (isProduction && !socketUrl.startsWith("https://")) {
      log.error("invalid socket url in production", { socketUrl });
      throw new Error("EXPO_PUBLIC_SOCKET_URL invalide en production");
    }
    // ✅ Normaliser : supprimer slash final pour éviter //socket.io
    return socketUrl.replace(/\/+$/, "");
  }

  // PRIORITÉ 2: Depuis app.config.js extra.socketUrl
  const expoExtra = Constants.expoConfig?.extra || {};
  const configSocketUrl = expoExtra.socketUrl;
  if (configSocketUrl) {
    if (
      isProduction &&
      (String(configSocketUrl).includes("localhost") ||
        String(configSocketUrl).includes("127.0.0.1") ||
        !String(configSocketUrl).startsWith("https://"))
    ) {
      throw new Error("socketUrl doit être HTTPS et non-localhost en production");
    }
    // ✅ Normaliser : supprimer slash final pour éviter //socket.io
    return configSocketUrl.replace(/\/+$/, "");
  }

  // En production: pas de fallback implicite
  if (isProduction) {
    throw new Error("EXPO_PUBLIC_SOCKET_URL est requis en production");
  }

  // PRIORITÉ 3: Fallback vers baseURL (dev uniquement)
  // ✅ Utiliser URL API pour extraire origin (plus robuste)
  try {
    const apiUrl = new URL(baseURL);
    const socketOrigin = `${apiUrl.protocol}//${apiUrl.host}`;
    // ✅ Log pour debug
    if (__DEV__) {
      log.info("socket origin from baseurl", { socketOrigin, baseURL });
    }
    // ✅ Normaliser : supprimer slash final (déjà normalisé mais on s'assure)
    return socketOrigin.replace(/\/+$/, "");
  } catch (e) {
    // Fallback: regex si URL malformée
    log.warn("baseurl parse error, using fallback", { baseURL, error: e });
    let socketOrigin = baseURL.replace(/\/api(?:\/v\d+)?$/, "");
    // ✅ Normaliser : supprimer slash final pour éviter //socket.io
    socketOrigin = socketOrigin.replace(/\/+$/, "");
    if (__DEV__) {
      log.info("socket origin regex fallback", { socketOrigin, baseURL });
    }
    return socketOrigin;
  }

  // PRIORITÉ 4: Fallback pour développement uniquement (si baseURL n'est pas disponible)
  if (__DEV__) {
    // Android emulator utilise 10.0.2.2 pour accéder à localhost de la machine hôte
    const platform = require("react-native").Platform.OS;
    if (platform === "android") {
      return "http://10.0.2.2:5000";
    }
    // iOS simulator et web utilisent localhost
    return "http://localhost:5000";
  }

  throw new Error("Impossible de résoudre l'URL socket");
};

let SOCKET_ORIGIN = "";
try {
  SOCKET_ORIGIN = getSocketOrigin();
  // ✅ Normalisation finale : s'assurer qu'il n'y a pas de slash final
  // Utiliser une regex plus robuste pour supprimer tous les slashes finaux
  SOCKET_ORIGIN = SOCKET_ORIGIN.replace(/\/+$/, "").trim();
  // ✅ Double vérification : s'assurer qu'il n'y a pas de slash final après trim
  if (SOCKET_ORIGIN.endsWith("/")) {
    SOCKET_ORIGIN = SOCKET_ORIGIN.slice(0, -1);
  }
} catch (error) {
  // Fail-safe bootstrap: ne jamais faire crasher l'app au chargement.
  SOCKET_ORIGIN = "";
  log.error("socket origin resolution failed", {
    error: error instanceof Error ? error.message : String(error),
  });
}

// ✅ Détection Android emulator
const isAndroidEmulator = Platform.OS === "android" && 
  (Constants.isDevice === false || __DEV__);

// ✅ Ajuster SOCKET_ORIGIN pour Android emulator
if (isAndroidEmulator && SOCKET_ORIGIN.includes("localhost")) {
  SOCKET_ORIGIN = SOCKET_ORIGIN.replace("localhost", "10.0.2.2");
  if (__DEV__) {
    log.info("android emulator origin adjusted", { SOCKET_ORIGIN });
  }
}

const IS_SECURE = SOCKET_ORIGIN.startsWith("https://");

// (Optionnel) logs verbeux en dev pour socket.io
let enableSocketIODebug = () => {};
try {
  enableSocketIODebug = () =>
    require("debug").enable("socket.io-client:*,engine.io-client:*");
} catch {}

let socket: Socket | null = null;
let socketRole: SocketRole | null = null;
let connectPromise: Promise<Socket> | null = null;
let connectPreparationPromise: Promise<void> | null = null;
let connectPreparationRole: SocketRole | null = null;
let lastHeartbeat = Date.now(); // Track last heartbeat timestamp
let heartbeatInterval: ReturnType<typeof setInterval> | null = null; // Interval for sending pings
let lastHeartbeatTimeoutWarningAt = 0;
let networkUnsubscribe: (() => void) | null = null; // networkState unsubscribe
let offlineDisconnectTimer: ReturnType<typeof setTimeout> | null = null;

const IS_DEV = __DEV__;

/** R1: payload auth socket pour corrélation backend (device_id, session_diag). */
export type SocketAuthExtras = { device_id?: string; session_diag?: string | null };

function buildOptions(
  token: string,
  role: SocketRole = "driver",
  extras?: SocketAuthExtras
) {
  // ✅ P2.1.3: Jitter anti-storm (0–2000ms) pour éviter thundering herd
  const jitterDelay = Math.random() * 2000; // 0–2000ms
  const jitterMax = Math.random() * 2000; // 0–2000ms
  // ✅ P0.3: driver = retry infini; enterprise = 10 tentatives
  const reconnectionAttempts = role === "driver" ? Infinity : 10;
  const auth: Record<string, unknown> = { token };
  if (extras?.device_id != null) auth.device_id = extras.device_id;
  if (extras?.session_diag != null) auth.session_diag = extras.session_diag;
  const useWebsocketOnly = IS_PRODUCTION_RUNTIME;
  const base = {
    path: "/socket.io", // ⚠️ sans slash final
    auth,
    extraHeaders: { Authorization: `Bearer ${token}` },
    reconnection: true,
    reconnectionAttempts,
    reconnectionDelay: 5000 + jitterDelay,
    reconnectionDelayMax: 30000 + jitterMax,
    timeout: 20000,
    forceNew: false,
    // En production mobile, forcer websocket pour eviter les boucles xhr post/poll (HTTP 400).
    // En dev, garder polling+websocket pour compatibilite reseaux locaux.
    transports: useWebsocketOnly ? ["websocket"] : ["polling", "websocket"],
    upgrade: !useWebsocketOnly,
    rememberUpgrade: true,
    secure: IS_SECURE,
  };
  return base;
}

export async function connectSocket(
  token: string,
  role: SocketRole = "driver"
): Promise<Socket | null> {
  if (!SOCKET_ORIGIN) {
    log.error("socket disabled: invalid or missing socket origin", {
      role,
      appVariant: String(
        Constants.expoConfig?.extra?.APP_VARIANT || process.env.APP_VARIANT || "prod"
      ),
    });
    return null;
  }
  // ✅ Feature flag : désactiver Socket.IO si EXPO_PUBLIC_SOCKET_ENABLED=false
  const socketEnabled = process.env.EXPO_PUBLIC_SOCKET_ENABLED !== 'false';
  if (!socketEnabled) {
    log.info("socket disabled by env", {});
    return null;
  }
  
  if (!token) {
    log.warn("no token provided", {});
    return null;
  }

  if (socket && socketRole && socketRole !== role) {
    try {
      socket.off();
      socket.disconnect();
    } catch {}
    socket = null;
    connectPromise = null;
  }

  // ✅ IMPORTANT: si une connexion est déjà en cours, ne pas "nettoyer" un socket
  // simplement parce qu'il est encore `disconnected` (état normal pendant le handshake).
  // Sinon on perd la référence globale et `getSocket()` retourne null (observé en logs).
  if (connectPromise && socketRole === role) {
    return connectPromise;
  }

  // Singleflight dès l'entrée: éviter plusieurs initialisations parallèles
  // avant que connectPromise soit assigné (fenêtre de course observée en logs).
  if (!connectPromise && connectPreparationPromise && connectPreparationRole === role) {
    await connectPreparationPromise;
    if (connectPromise && socketRole === role) {
      return connectPromise;
    }
    if (socket && socket.connected && socketRole === role) {
      return socket;
    }
  }

  // ✅ Vérifier que socket n'est pas déjà connecté
  if (socket && socket.connected && socketRole === role) {
    return socket;  // Réutiliser connexion existante
  }

  // ✅ Si socket existe mais déconnecté, nettoyer avant de créer nouveau
  // ⚠️ Ne faire ça que si aucune connexion n'est en cours (cf. connectPromise ci-dessus)
  if (socket && socketRole === role && socket.disconnected) {
    try {
      socket.off();
      socket.disconnect();
    } catch {}
    socket = null;
  }
  if (IS_DEV) enableSocketIODebug();

  socketRole = role;
  emitSocketStatusChange();
  let resolvePreparation: (() => void) | undefined;
  connectPreparationRole = role;
  connectPreparationPromise = new Promise<void>((resolve) => {
    resolvePreparation = resolve;
  });

  // ✅ Setup centralized booking event listeners
  function setupBookingListeners(s: Socket | null) {
    if (!s || !s.connected) {
      log.warn("cannot setup booking listeners", { reason: "socket not connected" });
      return;
    }

    // Remove existing listeners to avoid duplicates
    s.off("new_booking");
    s.off("booking_updated");
    s.off("booking_cancelled");
    s.off("booking_reassigned");

    // Listen to new_booking event
    s.on("new_booking", (data: Booking) => {
      // ✅ Déduplication par event_id
      const eventId = (data as any)?.event_id;
      if (!checkAndAddEventId(eventId, "new_booking")) {
        return; // Doublon, ignorer
      }
      
      log.info("booking new received", { booking_id: data?.id, event_id: eventId });
      bookingEmitter.emit("new_booking", data);
    });

    // Listen to booking_updated event
    s.on("booking_updated", (data: Booking) => {
      // ✅ Déduplication par event_id
      const eventId = (data as any)?.event_id;
      if (!checkAndAddEventId(eventId, "booking_updated")) {
        return; // Doublon, ignorer
      }
      
      log.info("booking updated received", { booking_id: data?.id, status: data?.status, event_id: eventId });
      bookingEmitter.emit("booking_updated", data);
    });

    // Listen to booking_cancelled event
    s.on("booking_cancelled", (data: { id: number } | Booking) => {
      // ✅ Déduplication par event_id
      const eventId = typeof data === "object" && "event_id" in data ? (data as any).event_id : undefined;
      if (!checkAndAddEventId(eventId, "booking_cancelled")) {
        return; // Doublon, ignorer
      }
      
      const bookingId = typeof data === "object" && "id" in data ? data.id : null;
      log.info("booking cancelled received", { booking_id: bookingId, event_id: eventId });
      bookingEmitter.emit("booking_cancelled", data);
    });

    // Listen to booking_reassigned event (mission retirée à ce chauffeur)
    s.on("booking_reassigned", (data: any) => {
      const eventId = (data as any)?.event_id;
      if (!checkAndAddEventId(eventId, "booking_reassigned")) {
        return;
      }

      const bookingId = (data as any)?.booking_id ?? (data as any)?.id ?? null;
      log.info("booking reassigned received", {
        booking_id: bookingId,
        new_driver_id: (data as any)?.new_driver_id,
        event_id: eventId,
      });
      bookingEmitter.emit("booking_reassigned", data);
    });

    log.info("booking listeners setup complete", {});
  }

  // ✅ P2: Setup centralized message event listeners
  function setupMessageListeners(s: Socket | null) {
    if (!s || !s.connected) {
      log.warn("cannot setup message listeners", { reason: "socket not connected" });
      return;
    }

    // Remove existing listeners to avoid duplicates
    s.off("team_chat_message");

    // Listen to team_chat_message event
    s.on("team_chat_message", (message: Message) => {
      // ✅ Déduplication par event_id
      const eventId = (message as any)?.event_id;
      if (!checkAndAddEventId(eventId, "team_chat_message")) {
        return; // Doublon, ignorer
      }
      
      log.info("team chat message received", {
        message_id: message?.id,
        sender_id: message?.sender_id,
        event_id: eventId,
      });
      messageEmitter.emit("team_chat_message", message);
    });

    log.info("message listeners setup complete", {});
  }

  // ✅ P0.3: Un disconnect ultérieur est "explicite" seulement si disconnectSocket() a été appelé
  explicitDisconnect = false;

  if (socketRole === "driver") {
    pushSessionEvent("SOCKET_CONNECTING");
    setConnectionStateSuffix("RECONN");
    connectionState = "RECONNECTING";
    emitSocketStatusChange();
  }

  try {
    // ✅ R1: device_id + session_diag dans socket.auth pour corrélation backend (connect/disconnect)
    let device_id: string | undefined;
    let session_diag: string | null = null;
    if (socketRole === "driver") {
      try {
        const { asyncStorage } = await import("./storage");
        device_id = await asyncStorage.getOrCreateDeviceId();
      } catch {
        // R3: storage edge-case — pas de logout, socket continue sans device_id, event pour enquête
        device_id = undefined;
        pushSessionEvent("DEVICE_ID_ERROR");
      }
      session_diag = getSessionDiagHeaderValue(); // peut être null, auth envoyé quand même (token seul minimum)
    }

    log.info("before socket creation", {
      SOCKET_ORIGIN,
      execution_environment: EXECUTION_ENVIRONMENT,
      is_production_runtime: IS_PRODUCTION_RUNTIME,
    });

    const authExtras: SocketAuthExtras =
      socketRole === "driver" ? { device_id, session_diag: session_diag ?? undefined } : {};

    connectPromise = new Promise<Socket>((resolve, reject) => {
      try {
      // ✅ Normalisation finale avant l'appel io() pour éviter //socket.io
      const normalizedOrigin = SOCKET_ORIGIN.replace(/\/+$/, "").trim();
      const opts = buildOptions(token, socketRole ?? "driver", authExtras);
      log.info("calling io()", { SOCKET_ORIGIN: normalizedOrigin });
      // ⚠️ Utiliser l'origine sans /api sinon 404 sur /api/socket.io
      socket = io(normalizedOrigin, opts);
      log.info("socket created", { id: socket ? (socket.id || "pending") : null });

      socket.on("connect", async () => {
        lastHeartbeatTimeoutWarningAt = 0;
        // ✅ P2.1.1: Reset backoff après connexion réussie
        if (socketRole === "driver") {
          driverConnectErrorAuthAttempts = 0;
          driverConnectErrorAuthBackoffMs = 2000;
          authRecoveryExhausted = false;
        }
        // ✅ P2.1.2: Reset reconnectExhausted après connexion réussie (enterprise)
        if (socketRole === "enterprise") {
          reconnectExhausted = false;
        }
        // ✅ P0.3: SessionEvents + état UI (ONLINE / RECONN / OFF) — jamais de logout sur disconnect
        if (socketRole === "driver") {
          connectionState = "ONLINE";
          setConnectionStateSuffix("ONLINE");
          pushSessionEvent("SOCKET_CONNECTED");
        } else if (socketRole === "enterprise") {
          connectionState = "ONLINE";
          setConnectionStateSuffix("ONLINE");
          pushSessionEvent("SOCKET_CONNECTED");
        }
        logAuthEvent("SOCKET_CONNECT", { role: socketRole, outcome: "success" });
        emitSocketStatusChange();
        socketConnectListeners.forEach((cb) => {
          try {
            cb();
          } catch (e) {
            log.warn("socketConnect listener error", { error: e });
          }
        });
        log.info("connected", { socket_id: socket?.id, user_type: socketRole || "unknown" });
        
        if (socketRole === "driver") {
          await joinDriverRoom().catch(() => {});  // ✅ Corrigé : enlever le + (syntaxe incorrecte)
          
          // ✅ Setup centralized booking event listeners
          setupBookingListeners(socket);
          
          // ✅ P2: Setup centralized message event listeners
          setupMessageListeners(socket);
          
          // ✅ Resync automatique à la connexion
          try {
            const lastSync = await AsyncStorage.getItem("last_sync_timestamp");
            const now = Date.now();
            
            // Si pas de timestamp ou si > 5 minutes depuis dernière sync, faire resync
            const shouldResync = !lastSync || (now - Number(lastSync)) > 5 * 60 * 1000;
            
            if (shouldResync) {
              // ✅ Utiliser le timestamp de dernière sync pour resync incrémental
              const since = lastSync ? new Date(Number(lastSync)).toISOString() : undefined;
              
              log.info("resync start", {
                last_sync: lastSync ? new Date(Number(lastSync)).toISOString() : null,
                since: since || null,
              });
              
              try {
                // ✅ Charger les données serveur avec filtre "since" pour resync incrémental
                const serverBookings = await getAssignedTrips({ since });
                
                // ✅ Charger les données locales depuis le cache
                const MISSIONS_CACHE_KEY = "missions_cache_v2";
                const localBookingsRaw = await AsyncStorage.getItem(MISSIONS_CACHE_KEY);
                let localBookings: Booking[] = [];
                
                if (localBookingsRaw) {
                  try {
                    localBookings = JSON.parse(localBookingsRaw);
                  } catch (parseError) {
                    log.warn("resync parse local error", {
                      error: parseError instanceof Error ? parseError.message : String(parseError),
                    });
                  }
                }
                
                // ✅ Détecter et résoudre les conflits
                const resolutionResult = resolveBookingConflicts(localBookings, serverBookings);
                
                // ✅ Resync incrémental (since) : si le serveur retourne vide, cela signifie
                // "aucun changement depuis last_sync", pas "toutes les missions supprimées".
                // On conserve les données locales ou on fait un full fetch si pas de cache.
                let finalResolved = resolutionResult.resolved;
                if (since && serverBookings.length === 0) {
                  if (localBookings.length > 0) {
                    finalResolved = localBookings;
                    log.info("resync incremental empty server, keeping local", {
                      local_count: localBookings.length,
                    });
                  } else {
                    const fullBookings = await getAssignedTrips();
                    finalResolved = fullBookings;
                    log.info("resync incremental empty server, full fetch", {
                      full_count: fullBookings.length,
                    });
                  }
                }
                
                // ✅ Logger les conflits détectés
                if (resolutionResult.hasConflicts) {
                  log.info("resync conflicts detected", {
                    conflicts_count: resolutionResult.conflicts.length,
                    conflicts: resolutionResult.conflicts.map(c => ({
                      id: c.id,
                      type: c.type,
                      conflictingFields: c.conflictingFields,
                      resolution: c.resolution
                    })),
                  });
                  
                  // ✅ Stocker l'historique des conflits (pour debugging)
                  try {
                    const conflictHistoryRaw = await AsyncStorage.getItem("conflict_history");
                    const conflictHistory = conflictHistoryRaw ? JSON.parse(conflictHistoryRaw) : [];
                    conflictHistory.push({
                      timestamp: new Date().toISOString(),
                      conflicts: resolutionResult.conflicts.map(c => ({
                        id: c.id,
                        type: c.type,
                        conflictingFields: c.conflictingFields,
                        resolution: c.resolution
                      }))
                    });
                    // Garder seulement les 50 derniers conflits
                    const trimmedHistory = conflictHistory.slice(-50);
                    await AsyncStorage.setItem("conflict_history", JSON.stringify(trimmedHistory));
                  } catch (historyError) {
                    // Ignorer les erreurs de stockage de l'historique
                    log.warn("resync conflict history error", {
                      error: historyError instanceof Error ? historyError.message : String(historyError),
                    });
                  }
                }
                
                // ✅ Émettre les données résolues (pas les données serveur brutes)
                resyncEmitter.emit("bookings:resync", finalResolved);
                log.info("resync complete", {
                  bookings_count: finalResolved.length,
                  conflicts_count: resolutionResult.conflicts.length,
                  has_conflicts: resolutionResult.hasConflicts,
                });
                
                // ✅ Mettre à jour le timestamp de dernière synchronisation
                await AsyncStorage.setItem("last_sync_timestamp", now.toString());
              } catch (resyncError) {
                log.warn("resync error", {
                  error: resyncError instanceof Error ? resyncError.message : String(resyncError),
                });
              }
            } else {
              log.info("resync skipped", { reason: "recent_sync", last_sync: new Date(Number(lastSync)).toISOString() });
            }

            // ✅ P2: Resync messages manquant
            try {
              const lastMessagesSync = await AsyncStorage.getItem("last_messages_sync_timestamp");
              const nowMessages = Date.now();
              
              // Si pas de timestamp ou si > 5 minutes depuis dernière sync, faire resync
              const shouldResyncMessages = !lastMessagesSync || (nowMessages - Number(lastMessagesSync)) > 5 * 60 * 1000;
              
              if (shouldResyncMessages) {
                // Récupérer le company_id depuis AsyncStorage (stocké lors de la connexion)
                const storedCompanyId = await AsyncStorage.getItem("driver_company_id");
                
                if (storedCompanyId) {
                  const companyId = parseInt(storedCompanyId, 10);
                  if (!isNaN(companyId)) {
                    log.info("messages resync start", {
                      last_sync: lastMessagesSync ? new Date(Number(lastMessagesSync)).toISOString() : null,
                    });
                    
                    try {
                      // ✅ Charger les données serveur
                      const serverMessages = await getCompanyMessages(companyId);
                      
                      // ✅ Charger les données locales depuis le cache (si disponible)
                      const MESSAGES_CACHE_KEY = "messages_cache_v1";
                      const localMessagesRaw = await AsyncStorage.getItem(MESSAGES_CACHE_KEY);
                      let localMessages: Message[] = [];
                      
                      if (localMessagesRaw) {
                        try {
                          localMessages = JSON.parse(localMessagesRaw);
                        } catch (parseError) {
                          log.warn("messages resync parse local error", {
                            error: parseError instanceof Error ? parseError.message : String(parseError),
                          });
                        }
                      }
                      
                      // ✅ Détecter et résoudre les conflits (server-wins pour les messages)
                      const resolutionResult = resolveMessageConflicts(localMessages, serverMessages);
                      
                      // ✅ Logger les conflits détectés (rare pour les messages)
                      if (resolutionResult.hasConflicts) {
                        log.info("messages resync conflicts detected", {
                          conflicts_count: resolutionResult.conflicts.length,
                          conflicts: resolutionResult.conflicts.map(c => ({
                            id: c.id,
                            type: c.type,
                            conflictingFields: c.conflictingFields,
                            resolution: c.resolution
                          })),
                        });
                      }
                      
                      // ✅ Émettre les données résolues
                      messageEmitter.emit("messages:resync", resolutionResult.resolved);
                      log.info("messages resync complete", {
                        messages_count: resolutionResult.resolved.length,
                        conflicts_count: resolutionResult.conflicts.length,
                        has_conflicts: resolutionResult.hasConflicts,
                      });
                      
                      // ✅ Mettre à jour le timestamp de dernière synchronisation
                      await AsyncStorage.setItem("last_messages_sync_timestamp", nowMessages.toString());
                    } catch (messagesResyncError) {
                      log.warn("messages resync error", {
                        error: messagesResyncError instanceof Error ? messagesResyncError.message : String(messagesResyncError),
                      });
                    }
                  }
                } else {
                  log.info("messages resync skipped", { reason: "no_company_id" });
                }
              } else {
                log.info("messages resync skipped", {
                  reason: "recent_sync",
                  last_sync: new Date(Number(lastMessagesSync)).toISOString(),
                });
              }
            } catch (messagesError) {
              log.warn("messages resync setup error", {
                error: messagesError instanceof Error ? messagesError.message : String(messagesError),
              });
            }
          } catch (error) {
            log.warn("resync setup error", {
              error: error instanceof Error ? error.message : String(error),
            });
          }
        } else if (socketRole === "enterprise") {
          joinCompanyRoom().catch(() => {});  // ✅ Corrigé : enlever le + (syntaxe incorrecte)
        }
        
        // ✅ Heartbeat applicatif actif : envoyer ping toutes les 30s
        lastHeartbeat = Date.now();
        
        // Écouter les pong du serveur (off d'abord pour éviter l'empilement sur reconnexion)
        if (socket) {
          socket.off("pong");
          socket.on("pong", (data: any) => {
            lastHeartbeat = Date.now();
            log.info("heartbeat pong", { timestamp: data?.timestamp });
          });
        }
        
        // Envoyer des ping toutes les 30s
        if (heartbeatInterval) {
          clearInterval(heartbeatInterval);
        }
        heartbeatInterval = setInterval(() => {
          const currentSocket = socket;
          if (currentSocket && currentSocket.connected) {
            const timeSinceLastHeartbeat = Date.now() - lastHeartbeat;
            // Ne pas forcer disconnect/reconnect ici: Socket.IO gère déjà heartbeat
            // transport. Un timeout applicatif peut être un faux positif et couper
            // une connexion saine (vu sur Android en conditions réelles).
            if (timeSinceLastHeartbeat > 60000) {
              const now = Date.now();
              if (now - lastHeartbeatTimeoutWarningAt > 5 * 60 * 1000) {
                lastHeartbeatTimeoutWarningAt = now;
                log.debug("heartbeat stale (no forced reconnect)", {
                  time_since_last_ms: timeSinceLastHeartbeat,
                  threshold_ms: 60000,
                });
              }
            }
            currentSocket.emit("ping");
            log.info("heartbeat ping", {});
          }
        }, 30000); // Toutes les 30s
        
        // ✅ Plan 2G/3G : Détection réseau via networkState (source unique)
        if (!networkUnsubscribe) {
          networkUnsubscribe = subscribeToNetworkState(() => {
            const state = getNetworkStateSnapshot();
            const isConnected =
              state?.isConnected === true && state?.isInternetReachable !== false;
            const currentSocket = socket;

            if (!isConnected && currentSocket?.connected) {
              if (!offlineDisconnectTimer) {
                offlineDisconnectTimer = setTimeout(() => {
                  offlineDisconnectTimer = null;
                  const latestState = getNetworkStateSnapshot();
                  const stillOffline =
                    latestState?.isConnected !== true ||
                    latestState?.isInternetReachable === false;
                  if (stillOffline && currentSocket.connected) {
                    log.info("network offline (debounced)", {});
                    currentSocket.disconnect();
                  }
                }, 5000);
              }
            }

            if (isConnected && offlineDisconnectTimer) {
              clearTimeout(offlineDisconnectTimer);
              offlineDisconnectTimer = null;
            }

            if (isConnected && !currentSocket?.connected) {
              log.info("network online", {});
              try {
                currentSocket?.connect();
              } catch (e) {
                log.warn("network online reconnect failed", {
                  error: e instanceof Error ? e.message : String(e),
                });
              }
            }
          });
        }
        
        resolve(socket as Socket);
      });

      socket.on("disconnect", (reason) => {
        logAuthEvent("SOCKET_DISCONNECT", {
          role: socketRole,
          reason: String(reason),
          explicit: explicitDisconnect,
        });
        // ✅ P0.3: SessionEvents — jamais de logout sur disconnect
        if (socketRole === "driver") {
          pushSessionEvent(("SOCKET_DISCONNECTED:" + String(reason)) as SessionEvent);
          if (!explicitDisconnect) {
            connectionState = "RECONNECTING";
            setConnectionStateSuffix("RECONN");
            emitSocketStatusChange();
          }
        }
        log.info("disconnected", { reason });
        connectPromise = null;
        
        // ✅ Ne pas reconnecter si serveur a déconnecté volontairement (io server disconnect / unauthorized)
        const reasonStr = String(reason);
        if (reasonStr === "io server disconnect" || reasonStr === "unauthorized") {
          log.info("disconnect stop reconnect", { reason: reasonStr });
          if (socket && socket.io) {
            (socket.io.opts as any).reconnection = false;
          }
          return;
        }
      });

      // ✅ P0.3 + P2.1.2: reconnect_attempt (Manager) — SessionEvent + fallback enterprise exhausted
      let enterpriseReconnectAttemptCount = 0;
      const ENTERPRISE_MAX_ATTEMPTS = 10;
      (socket as any).io?.on?.("reconnect_attempt", (n: number) => {
        if (socketRole === "driver") {
          pushSessionEvent(("SOCKET_RECONNECT_ATTEMPT:" + n) as SessionEvent);
        }
        if (socketRole === "enterprise") {
          enterpriseReconnectAttemptCount = n;
          if (n >= ENTERPRISE_MAX_ATTEMPTS && !(socket?.connected)) {
            if (!reconnectExhausted) {
              logAuthEvent("SOCKET_RECONNECT_EXHAUSTED", { role: "enterprise" });
            }
            reconnectExhausted = true;
            pushSessionEvent("SOCKET_RECONNECT_FAILED");
            emitSocketStatusChange();
          }
        }
      });

      // ✅ P2.1.2: reconnect_failed (Manager) — enterprise : 10 tentatives épuisées → bouton Reconnecter
      (socket as any).io?.on?.("reconnect_failed", () => {
        if (socketRole === "enterprise") {
          if (!reconnectExhausted) {
            logAuthEvent("SOCKET_RECONNECT_EXHAUSTED", { role: "enterprise" });
          }
          reconnectExhausted = true;
          pushSessionEvent("SOCKET_RECONNECT_FAILED");
          log.info("reconnect failed", { role: "enterprise", reconnectExhausted: true });
        }
      });

      // ✅ Handler reconnect : rejoin automatiquement les rooms après reconnexion
      socket.on("reconnect", (attempt) => {
        if (socketRole === "driver") {
          pushSessionEvent("SOCKET_RECONNECT_SUCCESS");
          connectionState = "ONLINE";
          setConnectionStateSuffix("ONLINE");
        }
        // ✅ P2.1.2: Reset reconnectExhausted après reconnexion réussie (enterprise)
        if (socketRole === "enterprise") {
          reconnectExhausted = false;
          enterpriseReconnectAttemptCount = 0;
        }
        emitSocketStatusChange();
        log.info("reconnected", { attempt });
        lastHeartbeat = Date.now();
        
        if (socketRole === "driver") {
          joinDriverRoom().catch(() => {});
          // ✅ Re-setup booking listeners on reconnect
          setupBookingListeners(socket);
          // ✅ P2: Re-setup message listeners on reconnect
          setupMessageListeners(socket);
          log.info("joined room", { room: "driver" });
        } else if (socketRole === "enterprise") {
          joinCompanyRoom().catch(() => {});
          log.info("joined room", { room: "company" });
        }
      });

      socket.on("connect_error", (err: any) => {
        const errorMsg = (err?.message || String(err)).slice(0, 200);
        const authStatus = extractAuthStatus(err);
        logAuthEvent("SOCKET_CONNECT_ERROR", {
          role: socketRole,
          message_truncated: errorMsg,
          ...(authStatus != null ? { status: authStatus } : {}),
        });
        if (socketRole === "driver") {
          pushSessionEvent(("SOCKET_CONNECT_ERROR:" + errorMsg) as SessionEvent);
          setConnectionStateSuffix("RECONN");
          connectionState = "RECONNECTING";
          emitSocketStatusChange();
        }
        const isRateLimit = errorMsg.includes("rate limit") ||
                           errorMsg.includes("Trop de tentatives") ||
                           errorMsg.includes("retry_after");

        const isTransportError =
          errorMsg.toLowerCase().includes("websocket error") ||
          errorMsg.toLowerCase().includes("transport");
        if (isTransportError) {
          log.warn("connect transport issue", {
            error: errorMsg,
            is_rate_limit: isRateLimit,
          });
        } else {
          log.error("connect error", { error: errorMsg, is_rate_limit: isRateLimit });
        }

        // ✅ Si rate limit, désactiver reconnexion automatique pour éviter boucle
        if (isRateLimit && socket && socket.io) {
          log.warn("rate limit detected, disabling auto reconnect", {});
          (socket.io.opts as any).reconnection = false;
          // Réactiver après 30 secondes (le retry_after du serveur)
          setTimeout(() => {
            if (socket && socket.io) {
              (socket.io.opts as any).reconnection = true;
              log.info("reconnect reenabled", {});
            }
          }, 30000);
        }

        // ✅ P2.1.1: connect_error 401/403 → refresh singleflight + reconnect (driver, sans logout)
        const network = getNetworkStateSnapshot();
        const isOnline = network?.isConnected !== false;
        // Garde offline : ne pas tenter refresh si réseau down (battery drain)
        const isDriverAuthError =
          socketRole === "driver" &&
          (authStatus === 401 || authStatus === 403) &&
          isOnline &&
          driverConnectErrorAuthAttempts < MAX_DRIVER_AUTH_REFRESH_ATTEMPTS &&
          !manualReconnectInProgress;

        if (isDriverAuthError && socket && socket.io) {
          manualReconnectInProgress = true;
          driverConnectErrorAuthAttempts++;
          const wasReconnection = (socket.io.opts as any).reconnection;
          (socket.io.opts as any).reconnection = false;
          const backoffMs = Math.min(driverConnectErrorAuthBackoffMs, 60000);
          driverConnectErrorAuthBackoffMs = Math.min(driverConnectErrorAuthBackoffMs * 2, 60000);

          pushSessionEvent("SOCKET_AUTH_REFRESH_ATTEMPT");
          pushSessionEvent(`SOCKET_RECONNECT_BACKOFF:${backoffMs}` as SessionEvent);

          setTimeout(async () => {
            try {
              const newToken = await refreshDriverTokenOrchestrated("socket_connect_error");
              pushSessionEvent("SOCKET_AUTH_REFRESH_SUCCESS");
              if (socket?.auth && typeof socket.auth === "object") {
                (socket.auth as Record<string, unknown>).token = newToken;
              }
              if (!socket?.connected) {
                socket?.connect();
              }
            } catch (refreshErr: unknown) {
              const refreshStatus = (refreshErr as { response?: { status?: number } })?.response?.status;
              const isAuthInvalid = (refreshErr as { reason?: string })?.reason?.includes("refresh_rejected");
              if (refreshStatus === 401 || refreshStatus === 403 || isAuthInvalid) {
                connectPromise = null;
                reject(refreshErr);
              } else {
                connectPromise = null;
                reject(err);
              }
            } finally {
              manualReconnectInProgress = false;
              if (driverConnectErrorAuthAttempts >= MAX_DRIVER_AUTH_REFRESH_ATTEMPTS) {
                if (!authRecoveryExhausted) {
                  logAuthEvent("SOCKET_AUTH_RECOVERY_EXHAUSTED", { role: "driver" });
                }
                authRecoveryExhausted = true;
                pushSessionEvent("SOCKET_AUTH_RECOVERY_EXHAUSTED");
                emitSocketStatusChange();
                if (socket?.io) {
                  (socket.io.opts as any).reconnection = wasReconnection ?? true;
                }
              } else {
                if (socket?.io) {
                  (socket.io.opts as any).reconnection = wasReconnection ?? true;
                }
              }
            }
          }, backoffMs);
          return;
        }

        connectPromise = null;
        reject(err);
      });

      socket.on("unauthorized", async (data: any) => {
        const errorMsg = data?.error || String(data);
        const isTokenExpired = errorMsg.includes("Token expiré") || 
                              errorMsg.includes("expiré") ||
                              errorMsg.includes("expired");
        
        log.error("unauthorized", { error: errorMsg, is_token_expired: isTokenExpired });
        
        // ✅ Si token expiré, arrêter reconnexion auto et déclencher refresh
        if (isTokenExpired && socket && socket.io) {
          log.warn("token expired, disabling auto reconnect", {});
          
          // Désactiver reconnexion automatique pour éviter boucle
          (socket.io.opts as any).reconnection = false;
          
          // Déconnecter proprement
          try {
            socket.disconnect();
          } catch (e) {
            // Ignorer erreurs de déconnexion
          }
          
          // Essayer de rafraîchir le token
          try {
            log.info("refreshing token", {});
            const newToken = await refreshDriverTokenOrchestrated("socket_unauthorized");
            if (socket?.auth && typeof socket.auth === "object") {
              (socket.auth as Record<string, unknown>).token = newToken;
            }
            log.info("token refreshed, will reconnect manually", {});
            // Réactiver reconnexion et reconnecter manuellement après un court délai
            setTimeout(() => {
              if (socket && socket.io) {
                (socket.io.opts as any).reconnection = true;
                socket.connect();
              }
            }, 1000);
          } catch (refreshError: any) {
            log.error("token refresh error", {
              error: refreshError?.message || String(refreshError),
            });
            // En cas d'erreur de refresh, ne pas reconnecter automatiquement
            // L'utilisateur devra se reconnecter manuellement
          }
        }
      });

      socket.on("error", (e: any) => {
        const errorMsg = e?.message || String(e);
        const errorData = e?.data || {};
        const isRateLimit = errorMsg.includes("rate limit") || 
                           errorMsg.includes("Trop de tentatives") ||
                           errorData?.error?.includes("Trop de tentatives") ||
                           errorData?.retry_after !== undefined;
        
        log.error("socket error", {
          error: errorMsg,
          error_data: errorData,
          is_rate_limit: isRateLimit,
          retry_after: errorData?.retry_after,
        });
        
        // ✅ Si rate limit, désactiver reconnexion automatique temporairement
        if (isRateLimit && socket && socket.io) {
          const retryAfter = errorData?.retry_after || 30;
          log.warn("rate limit error, disabling auto reconnect", { retry_after: retryAfter });
          (socket.io.opts as any).reconnection = false;
          // Réactiver après retry_after + marge de sécurité
          setTimeout(() => {
            if (socket && socket.io) {
              (socket.io.opts as any).reconnection = true;
              log.info("reconnect reenabled after rate limit", {});
            }
          }, (retryAfter + 5) * 1000); // +5s marge de sécurité
        }
      });

      socket.on("connected", (data: any) => {
        log.info("handshake ok", { data });
      });

      // ✅ Événements supplémentaires du backend
      socket.on("joined_room", (data: any) => {
        log.info("joined room", { rooms: data?.rooms });
      });

      socket.on("joined_company", (data: any) => {
        log.info("joined company", { company_id: data?.company_id, room: data?.room });
      });

      socket.on("typing_indicator", (data: any) => {
        log.info("user typing", { user_id: data?.user_id });
        // TODO: Afficher indicateur de frappe dans UI
      });

      socket.on("stop_typing", (data: any) => {
        log.info("user stop typing", { user_id: data?.user_id });
      });

      // ✅ Événements d'arrivée driver
      socket.on("driver_arrived_at_pickup", (data: any) => {
        log.info("driver arrived pickup", { driver_id: data?.driver_id });
        // TODO: Afficher notification "Driver arrivé au point de départ"
      });

      socket.on("driver_arrived_at_dropoff", (data: any) => {
        log.info("driver arrived dropoff", { driver_id: data?.driver_id });
        // TODO: Afficher notification "Driver arrivé à destination"
      });
      } catch (e) {
        connectPromise = null;
        reject(e);
      }
    });

    return connectPromise;
  } finally {
    resolvePreparation?.();
    if (connectPreparationRole === role) {
      connectPreparationPromise = null;
      connectPreparationRole = null;
    }
  }
}

export function getSocket(): Socket | null {
  return socket;
}

export function disconnectSocket() {
  // ✅ P0.3: Logout explicite uniquement — jamais de logout sur simple disconnect
  explicitDisconnect = true;
  connectionState = "OFFLINE";
  setConnectionStateSuffix("OFF");
  try {
    socket?.off();
    socket?.disconnect();
    
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
      heartbeatInterval = null;
    }
    if (networkUnsubscribe) {
      networkUnsubscribe();
      networkUnsubscribe = null;
    }
    if (offlineDisconnectTimer) {
      clearTimeout(offlineDisconnectTimer);
      offlineDisconnectTimer = null;
    }
    
    log.info("disconnect cleanup", {});
  } catch (err) {
    log.error("disconnect error", {
      error: err instanceof Error ? err.message : String(err),
    });
  } finally {
    socket = null;
    socketRole = null;
    connectPromise = null;
    lastHeartbeat = Date.now();
    lastHeartbeatTimeoutWarningAt = 0;
    reconnectExhausted = false; // P2.1.2: reset on explicit disconnect
  }
}

/** P0.3: État connexion pour l'UI (ONLINE / RECONNECTING / OFFLINE). OFFLINE = logout explicite ou jamais connecté. */
export function getConnectionState(): ConnectionState {
  return connectionState;
}

/** P2.1.2: Reconnexion manuelle après reconnect_failed (enterprise) ou authRecoveryExhausted (driver). */
export async function reconnectSocketManually(role: SocketRole): Promise<Socket | null> {
  let token: string | null = null;
  if (role === "driver") {
    token = await secureStorage.getAccessToken();
  } else {
    token = await secureStorage.getEnterpriseToken();
  }
  if (!token) {
    log.warn("reconnect manual no token", { role });
    return null;
  }
  // Réinitialiser flags avant tentative
  if (role === "enterprise") {
    reconnectExhausted = false;
  } else {
    authRecoveryExhausted = false;
  }
  connectionState = "RECONNECTING";
  setConnectionStateSuffix("RECONN");
  emitSocketStatusChange();
  return connectSocket(token, role);
}

// ✅ Heartbeat métier : envoie métadonnées métier toutes les 60s
export async function sendDriverHeartbeat(payload: {
  last_mission_id?: number;
  location?: { lat: number; lon: number };
}) {
  if (socketRole !== "driver") return;
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s || !s.connected) {
    log.warn("driver heartbeat skipped", { reason: "socket_not_connected" });
    return;
  }
  
  s.emit("driver:heartbeat", {
    ...payload,
    timestamp: Date.now(),
  });
  
  log.info("driver heartbeat sent", {
    last_mission_id: payload.last_mission_id,
    has_location: !!payload.location,
  });
}

// Helpers côté driver
export async function joinDriverRoom() {
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s) {
    log.warn("cannot join driver room", { reason: "socket not connected" });
    return;
  }

  try {
    const idStr = await AsyncStorage.getItem("driver_id");
    const driver_id = idStr ? Number(idStr) : undefined;
    // ✅ FIX: Validation stricte du driver_id
    if (driver_id && Number.isFinite(driver_id) && driver_id > 0) {
      s?.emit("join_driver_room", { driver_id });
      log.info("join driver room emitted", { driver_id });
    } else {
      s?.emit("join_driver_room");
      log.info("join driver room emitted", { driver_id: null });
    }
  } catch {
    s?.emit("join_driver_room");
    log.info("join driver room emitted", { reason: "async storage error" });
  }
}

export async function joinCompanyRoom() {
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s) {
    log.warn("cannot join company room", { reason: "socket not connected" });
    return;
  }
  s.emit("join_company");
  log.info("join company emitted", {});
}

export async function sendDriverLocation(payload: {
  latitude: number;
  longitude: number;
  speed?: number;
  heading?: number;
  accuracy?: number;
  timestamp?: number | string;
}) {
  if (socketRole !== "driver") {
    log.warn("cannot send location", { reason: "socket_not_driver", currentRole: socketRole });
    return;
  }
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s) {
    log.warn("cannot send location", { reason: "socket not connected" });
    return;
  }

  try {
    const idStr = await AsyncStorage.getItem("driver_id");
    const driver_id = idStr ? Number(idStr) : undefined;
    const body =
      driver_id && Number.isFinite(driver_id) && driver_id > 0
        ? { ...payload, driver_id }
        : payload;
    s?.emit("driver_location", body);
    const has_driver_id =
      "driver_id" in (body as Record<string, unknown>) &&
      typeof (body as any).driver_id === "number";
    log.info("driver location emitted", {
      has_driver_id,
      lat: payload.latitude,
      lon: payload.longitude,
    });
  } catch {
    s?.emit("driver_location", payload);
    log.info("driver location emitted", { reason: "no driver_id" });
  }
}

/** Plan 2G/3G Phase 6 : Déclenche resync missions depuis syncEngine (fallback 60s). */
export async function triggerMissionResync(forceResync = false): Promise<void> {
  if (socketRole !== "driver") return;
  try {
    const lastSync = await AsyncStorage.getItem("last_sync_timestamp");
    const now = Date.now();
    const shouldResync = forceResync || !lastSync || (now - Number(lastSync)) > 5 * 60 * 1000;
    if (!shouldResync) return;

    const since = lastSync && !forceResync ? new Date(Number(lastSync)).toISOString() : undefined;
    const serverBookings = await getAssignedTrips({ since });

    const MISSIONS_CACHE_KEY = "missions_cache_v2";
    const localBookingsRaw = await AsyncStorage.getItem(MISSIONS_CACHE_KEY);
    let localBookings: Booking[] = [];
    if (localBookingsRaw) {
      try {
        localBookings = JSON.parse(localBookingsRaw);
      } catch {
        /* ignore */
      }
    }

    const resolutionResult = resolveBookingConflicts(localBookings, serverBookings);
    let finalResolved = resolutionResult.resolved;
    if (since && serverBookings.length === 0) {
      if (localBookings.length > 0) {
        finalResolved = localBookings;
        log.info("triggerMissionResync incremental empty server, keeping local", {
          local_count: localBookings.length,
        });
      } else {
        const fullBookings = await getAssignedTrips();
        finalResolved = fullBookings;
        log.info("triggerMissionResync incremental empty server, full fetch", {
          full_count: fullBookings.length,
        });
      }
    }
    resyncEmitter.emit("bookings:resync", finalResolved);
    await AsyncStorage.setItem("last_sync_timestamp", now.toString());
  } catch (e) {
    log.warn("triggerMissionResync error", { error: e });
  }
}

// ✅ Export de l'EventEmitter pour que les composants puissent écouter les événements de resync
export function onBookingsResync(callback: (bookings: Booking[]) => void) {
  resyncEmitter.on("bookings:resync", callback);
  return () => {
    resyncEmitter.off("bookings:resync", callback);
  };
}

// ✅ Export des fonctions pour écouter les événements de booking
export function onBookingNew(callback: (booking: Booking) => void) {
  bookingEmitter.on("new_booking", callback);
  return () => {
    bookingEmitter.off("new_booking", callback);
  };
}

export function onBookingUpdated(callback: (booking: Booking) => void) {
  bookingEmitter.on("booking_updated", callback);
  return () => {
    bookingEmitter.off("booking_updated", callback);
  };
}

export function onBookingCancelled(callback: (data: { id: number } | Booking) => void) {
  bookingEmitter.on("booking_cancelled", callback);
  return () => {
    bookingEmitter.off("booking_cancelled", callback);
  };
}

export function onBookingReassigned(callback: (data: any) => void) {
  bookingEmitter.on("booking_reassigned", callback);
  return () => {
    bookingEmitter.off("booking_reassigned", callback);
  };
}

// ✅ P2: Export des fonctions pour écouter les événements de messages
export function onTeamChatMessage(callback: (message: Message) => void) {
  messageEmitter.on("team_chat_message", callback);
  return () => {
    messageEmitter.off("team_chat_message", callback);
  };
}

export function onMessagesResync(callback: (messages: Message[]) => void) {
  messageEmitter.on("messages:resync", callback);
  return () => {
    messageEmitter.off("messages:resync", callback);
  };
}
