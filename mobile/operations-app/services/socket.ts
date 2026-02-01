// services/socket.ts
import { io, type Socket } from "socket.io-client";
import AsyncStorage from "@react-native-async-storage/async-storage";
import NetInfo from "@react-native-community/netinfo";
import { Platform } from "react-native";
import Constants from "expo-constants";
import { baseURL, getAssignedTrips, getCompanyMessages, refreshDriverTokenSingleflight, type Booking, type Message } from "./api";
import { resolveBookingConflicts, resolveMessageConflicts, type Conflict } from "./conflictResolution";
import { getSessionDiagHeaderValue, pushSessionEvent, setConnectionStateSuffix } from "./sessionJournal";
import type { SessionEvent } from "./sessionJournal";
import { getNetworkStateSnapshot } from "./networkState";
import { extractAuthStatus } from "./socketAuthUtils";
import { secureStorage } from "./storage";
import { logAuthEvent } from "./authLogging";

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
          console.warn(`[EventEmitter] Error in listener for ${event}:`, error);
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
      console.warn("[socket] SocketStatus listener error:", e);
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
    console.warn(
      JSON.stringify({
        event: "duplicate_event_ignored",
        event_id: eventId,
        event_name: eventName,
        timestamp: new Date().toISOString()
      })
    );
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
  // PRIORITÉ 1: Variable d'environnement dédiée
  const socketUrl = process.env.EXPO_PUBLIC_SOCKET_URL;
  if (socketUrl) {
    // Validation en production : doit être HTTPS
    const isProduction = !__DEV__ || Constants.expoConfig?.extra?.APP_VARIANT === "prod";
    if (isProduction && !socketUrl.startsWith("https://")) {
      console.error(
        "[Socket.IO] EXPO_PUBLIC_SOCKET_URL doit commencer par 'https://' en production. " +
        `Valeur actuelle: ${socketUrl}`
      );
      throw new Error("EXPO_PUBLIC_SOCKET_URL invalide en production");
    }
    // ✅ Normaliser : supprimer slash final pour éviter //socket.io
    return socketUrl.replace(/\/+$/, "");
  }

  // PRIORITÉ 2: Depuis app.config.js extra.socketUrl
  const expoExtra = Constants.expoConfig?.extra || {};
  const configSocketUrl = expoExtra.socketUrl;
  if (configSocketUrl) {
    // ✅ Normaliser : supprimer slash final pour éviter //socket.io
    return configSocketUrl.replace(/\/+$/, "");
  }

  // PRIORITÉ 3: Fallback vers baseURL (plus robuste avec URL API)
  // ✅ Utiliser URL API pour extraire origin (plus robuste)
  try {
    const apiUrl = new URL(baseURL);
    const socketOrigin = `${apiUrl.protocol}//${apiUrl.host}`;
    // ✅ Log pour debug
    if (__DEV__) {
      console.log(`[Socket] SOCKET_ORIGIN=${socketOrigin} (from baseURL=${baseURL})`);
    }
    // ✅ Normaliser : supprimer slash final (déjà normalisé mais on s'assure)
    return socketOrigin.replace(/\/+$/, "");
  } catch (e) {
    // Fallback: regex si URL malformée
    console.warn(`[Socket] Erreur parsing baseURL=${baseURL}, fallback regex:`, e);
    let socketOrigin = baseURL.replace(/\/api(?:\/v\d+)?$/, "");
    // ✅ Normaliser : supprimer slash final pour éviter //socket.io
    socketOrigin = socketOrigin.replace(/\/+$/, "");
    if (__DEV__) {
      console.log(`[Socket] SOCKET_ORIGIN=${socketOrigin} (from baseURL=${baseURL}, regex fallback)`);
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

  // PRIORITÉ 5: Production - erreur si non défini
  // ✅ AMÉLIORATION: Logger l'erreur mais utiliser baseURL comme dernier recours
  console.error(
    "[Socket.IO] ⚠️ EXPO_PUBLIC_SOCKET_URL non défini en production. " +
    "Utilisation de baseURL comme fallback (non recommandé)."
  );
  // Utiliser baseURL comme dernier recours au lieu de throw
  try {
    const apiUrl = new URL(baseURL);
    const socketOrigin = `${apiUrl.protocol}//${apiUrl.host}`;
    // ✅ Normaliser : supprimer slash final pour éviter //socket.io
    return socketOrigin.replace(/\/+$/, "");
  } catch {
    throw new Error("EXPO_PUBLIC_SOCKET_URL est requis en production");
  }
};

let SOCKET_ORIGIN = getSocketOrigin();
// ✅ Normalisation finale : s'assurer qu'il n'y a pas de slash final
// Utiliser une regex plus robuste pour supprimer tous les slashes finaux
SOCKET_ORIGIN = SOCKET_ORIGIN.replace(/\/+$/, "").trim();
// ✅ Double vérification : s'assurer qu'il n'y a pas de slash final après trim
if (SOCKET_ORIGIN.endsWith("/")) {
  SOCKET_ORIGIN = SOCKET_ORIGIN.slice(0, -1);
}

// ✅ Détection Android emulator
const isAndroidEmulator = Platform.OS === "android" && 
  (Constants.isDevice === false || __DEV__);

// ✅ Ajuster SOCKET_ORIGIN pour Android emulator
if (isAndroidEmulator && SOCKET_ORIGIN.includes("localhost")) {
  SOCKET_ORIGIN = SOCKET_ORIGIN.replace("localhost", "10.0.2.2");
  if (__DEV__) {
    console.log(`[Socket] Android emulator détecté, SOCKET_ORIGIN ajusté: ${SOCKET_ORIGIN}`);
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
let lastHeartbeat = Date.now(); // Track last heartbeat timestamp
let heartbeatInterval: ReturnType<typeof setInterval> | null = null; // Interval for sending pings
let networkUnsubscribe: (() => void) | null = null; // NetInfo unsubscribe function

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
    transports: ["websocket", "polling"],
    upgrade: true,
    rememberUpgrade: true,
    secure: IS_SECURE,
  };
  return base;
}

export async function connectSocket(
  token: string,
  role: SocketRole = "driver"
): Promise<Socket | null> {
  // ✅ Feature flag : désactiver Socket.IO si EXPO_PUBLIC_SOCKET_ENABLED=false
  const socketEnabled = process.env.EXPO_PUBLIC_SOCKET_ENABLED !== 'false';
  if (!socketEnabled) {
    console.log('[Socket] Socket.IO désactivé (EXPO_PUBLIC_SOCKET_ENABLED=false)');
    return null;
  }
  
  if (!token) {
    console.warn("❌ Aucun token fourni à connectSocket");
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

  // ✅ Setup centralized booking event listeners
  function setupBookingListeners(s: Socket | null) {
    if (!s || !s.connected) {
      console.warn("⚠️ Cannot setup booking listeners: socket not connected");
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
      
      console.log(JSON.stringify({
        event: "booking_new_received",
        booking_id: data?.id,
        event_id: eventId,
        timestamp: new Date().toISOString()
      }));
      bookingEmitter.emit("new_booking", data);
    });

    // Listen to booking_updated event
    s.on("booking_updated", (data: Booking) => {
      // ✅ Déduplication par event_id
      const eventId = (data as any)?.event_id;
      if (!checkAndAddEventId(eventId, "booking_updated")) {
        return; // Doublon, ignorer
      }
      
      console.log(JSON.stringify({
        event: "booking_updated_received",
        booking_id: data?.id,
        status: data?.status,
        event_id: eventId,
        timestamp: new Date().toISOString()
      }));
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
      console.log(JSON.stringify({
        event: "booking_cancelled_received",
        booking_id: bookingId,
        event_id: eventId,
        timestamp: new Date().toISOString()
      }));
      bookingEmitter.emit("booking_cancelled", data);
    });

    // Listen to booking_reassigned event (mission retirée à ce chauffeur)
    s.on("booking_reassigned", (data: any) => {
      const eventId = (data as any)?.event_id;
      if (!checkAndAddEventId(eventId, "booking_reassigned")) {
        return;
      }

      const bookingId = (data as any)?.booking_id ?? (data as any)?.id ?? null;
      console.log(
        JSON.stringify({
          event: "booking_reassigned_received",
          booking_id: bookingId,
          new_driver_id: (data as any)?.new_driver_id,
          event_id: eventId,
          timestamp: new Date().toISOString(),
        })
      );
      bookingEmitter.emit("booking_reassigned", data);
    });

    console.log("✅ Booking event listeners setup complete");
  }

  // ✅ P2: Setup centralized message event listeners
  function setupMessageListeners(s: Socket | null) {
    if (!s || !s.connected) {
      console.warn("⚠️ Cannot setup message listeners: socket not connected");
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
      
      console.log(JSON.stringify({
        event: "team_chat_message_received",
        message_id: message?.id,
        sender_id: message?.sender_id,
        event_id: eventId,
        timestamp: new Date().toISOString()
      }));
      messageEmitter.emit("team_chat_message", message);
    });

    console.log("✅ Message event listeners setup complete");
  }

  // ✅ P0.3: Un disconnect ultérieur est "explicite" seulement si disconnectSocket() a été appelé
  explicitDisconnect = false;

  if (socketRole === "driver") {
    pushSessionEvent("SOCKET_CONNECTING");
    setConnectionStateSuffix("RECONN");
    connectionState = "RECONNECTING";
    emitSocketStatusChange();
  }

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

  console.log(`[connectSocket] 📍 Avant création socket, SOCKET_ORIGIN=${SOCKET_ORIGIN}`);
  
  const authExtras: SocketAuthExtras =
    socketRole === "driver" ? { device_id, session_diag: session_diag ?? undefined } : {};
  
  connectPromise = new Promise<Socket>((resolve, reject) => {
    try {
      // ✅ Normalisation finale avant l'appel io() pour éviter //socket.io
      const normalizedOrigin = SOCKET_ORIGIN.replace(/\/+$/, "").trim();
      const opts = buildOptions(token, socketRole ?? "driver", authExtras);
      console.log(`[connectSocket] 📍 Appel io() avec SOCKET_ORIGIN=${normalizedOrigin}`);
      // ⚠️ Utiliser l'origine sans /api sinon 404 sur /api/socket.io
      socket = io(normalizedOrigin, opts);
      console.log(`[connectSocket] ✅ Socket créé:`, socket ? `id=${socket.id || 'pending'}` : 'NULL');

      socket.on("connect", async () => {
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
        // ✅ Logs structurés
        console.log(JSON.stringify({
          event: "socket_connect",
          socket_id: socket?.id,
          timestamp: new Date().toISOString(),
          user_type: socketRole || "unknown"
        }));
        
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
              
              console.log(JSON.stringify({
                event: "resync_start",
                last_sync: lastSync ? new Date(Number(lastSync)).toISOString() : null,
                since: since || null,
                timestamp: new Date().toISOString()
              }));
              
              try {
                // ✅ Charger les données serveur avec filtre "since" pour resync incrémental
                const serverBookings = await getAssignedTrips({ since });
                
                // ✅ Charger les données locales depuis le cache
                const MISSIONS_CACHE_KEY = "missions_cache_v1";
                const localBookingsRaw = await AsyncStorage.getItem(MISSIONS_CACHE_KEY);
                let localBookings: Booking[] = [];
                
                if (localBookingsRaw) {
                  try {
                    localBookings = JSON.parse(localBookingsRaw);
                  } catch (parseError) {
                    console.warn(JSON.stringify({
                      event: "resync_parse_local_error",
                      error: parseError instanceof Error ? parseError.message : String(parseError),
                      timestamp: new Date().toISOString()
                    }));
                  }
                }
                
                // ✅ Détecter et résoudre les conflits
                const resolutionResult = resolveBookingConflicts(localBookings, serverBookings);
                
                // ✅ Logger les conflits détectés
                if (resolutionResult.hasConflicts) {
                  console.log(JSON.stringify({
                    event: "resync_conflicts_detected",
                    conflicts_count: resolutionResult.conflicts.length,
                    conflicts: resolutionResult.conflicts.map(c => ({
                      id: c.id,
                      type: c.type,
                      conflictingFields: c.conflictingFields,
                      resolution: c.resolution
                    })),
                    timestamp: new Date().toISOString()
                  }));
                  
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
                    console.warn(JSON.stringify({
                      event: "resync_conflict_history_error",
                      error: historyError instanceof Error ? historyError.message : String(historyError),
                      timestamp: new Date().toISOString()
                    }));
                  }
                }
                
                // ✅ Émettre les données résolues (pas les données serveur brutes)
                resyncEmitter.emit("bookings:resync", resolutionResult.resolved);
                console.log(JSON.stringify({
                  event: "resync_complete",
                  bookings_count: resolutionResult.resolved.length,
                  conflicts_count: resolutionResult.conflicts.length,
                  has_conflicts: resolutionResult.hasConflicts,
                  timestamp: new Date().toISOString()
                }));
                
                // ✅ Mettre à jour le timestamp de dernière synchronisation
                await AsyncStorage.setItem("last_sync_timestamp", now.toString());
              } catch (resyncError) {
                console.warn(JSON.stringify({
                  event: "resync_error",
                  error: resyncError instanceof Error ? resyncError.message : String(resyncError),
                  timestamp: new Date().toISOString()
                }));
              }
            } else {
              console.log(JSON.stringify({
                event: "resync_skipped",
                reason: "recent_sync",
                last_sync: new Date(Number(lastSync)).toISOString(),
                timestamp: new Date().toISOString()
              }));
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
                    console.log(JSON.stringify({
                      event: "messages_resync_start",
                      last_sync: lastMessagesSync ? new Date(Number(lastMessagesSync)).toISOString() : null,
                      timestamp: new Date().toISOString()
                    }));
                    
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
                          console.warn(JSON.stringify({
                            event: "messages_resync_parse_local_error",
                            error: parseError instanceof Error ? parseError.message : String(parseError),
                            timestamp: new Date().toISOString()
                          }));
                        }
                      }
                      
                      // ✅ Détecter et résoudre les conflits (server-wins pour les messages)
                      const resolutionResult = resolveMessageConflicts(localMessages, serverMessages);
                      
                      // ✅ Logger les conflits détectés (rare pour les messages)
                      if (resolutionResult.hasConflicts) {
                        console.log(JSON.stringify({
                          event: "messages_resync_conflicts_detected",
                          conflicts_count: resolutionResult.conflicts.length,
                          conflicts: resolutionResult.conflicts.map(c => ({
                            id: c.id,
                            type: c.type,
                            conflictingFields: c.conflictingFields,
                            resolution: c.resolution
                          })),
                          timestamp: new Date().toISOString()
                        }));
                      }
                      
                      // ✅ Émettre les données résolues
                      messageEmitter.emit("messages:resync", resolutionResult.resolved);
                      console.log(JSON.stringify({
                        event: "messages_resync_complete",
                        messages_count: resolutionResult.resolved.length,
                        conflicts_count: resolutionResult.conflicts.length,
                        has_conflicts: resolutionResult.hasConflicts,
                        timestamp: new Date().toISOString()
                      }));
                      
                      // ✅ Mettre à jour le timestamp de dernière synchronisation
                      await AsyncStorage.setItem("last_messages_sync_timestamp", nowMessages.toString());
                    } catch (messagesResyncError) {
                      console.warn(JSON.stringify({
                        event: "messages_resync_error",
                        error: messagesResyncError instanceof Error ? messagesResyncError.message : String(messagesResyncError),
                        timestamp: new Date().toISOString()
                      }));
                    }
                  }
                } else {
                  console.log(JSON.stringify({
                    event: "messages_resync_skipped",
                    reason: "no_company_id",
                    timestamp: new Date().toISOString()
                  }));
                }
              } else {
                console.log(JSON.stringify({
                  event: "messages_resync_skipped",
                  reason: "recent_sync",
                  last_sync: new Date(Number(lastMessagesSync)).toISOString(),
                  timestamp: new Date().toISOString()
                }));
              }
            } catch (messagesError) {
              console.warn(JSON.stringify({
                event: "messages_resync_setup_error",
                error: messagesError instanceof Error ? messagesError.message : String(messagesError),
                timestamp: new Date().toISOString()
              }));
            }
          } catch (error) {
            console.warn(JSON.stringify({
              event: "resync_setup_error",
              error: error instanceof Error ? error.message : String(error),
              timestamp: new Date().toISOString()
            }));
          }
        } else if (socketRole === "enterprise") {
          joinCompanyRoom().catch(() => {});  // ✅ Corrigé : enlever le + (syntaxe incorrecte)
        }
        
        // ✅ Heartbeat applicatif actif : envoyer ping toutes les 30s
        lastHeartbeat = Date.now();
        
        // Écouter les pong du serveur
        if (socket) {
          socket.on("pong", (data: any) => {
            lastHeartbeat = Date.now();
            console.log(JSON.stringify({
              event: "heartbeat_pong",
              timestamp: data?.timestamp,
              received_at: new Date().toISOString()
            }));
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
            // Si pas de pong depuis >60s, forcer reconnexion
            if (timeSinceLastHeartbeat > 60000) {
              console.warn(JSON.stringify({
                event: "heartbeat_timeout",
                time_since_last_ms: timeSinceLastHeartbeat,
                threshold_ms: 60000,
                action: "force_reconnect",
                timestamp: new Date().toISOString()
              }));
              currentSocket.disconnect();
              currentSocket.connect();
              lastHeartbeat = Date.now();
            } else {
              currentSocket.emit("ping");
              console.log(JSON.stringify({
                event: "heartbeat_ping",
                timestamp: new Date().toISOString()
              }));
            }
          }
        }, 30000); // Toutes les 30s
        
        // ✅ Détection réseau (NetInfo)
        if (!networkUnsubscribe) {
          networkUnsubscribe = NetInfo.addEventListener((state: { isConnected: boolean | null }) => {
            const isConnected = state.isConnected ?? false;
            const currentSocket = socket;
            
            if (!isConnected && currentSocket?.connected) {
              console.log(JSON.stringify({
                event: "network_offline",
                timestamp: new Date().toISOString()
              }));
              currentSocket.disconnect();
            }
            
            if (isConnected && !currentSocket?.connected) {
              console.log(JSON.stringify({
                event: "network_online",
                timestamp: new Date().toISOString()
              }));
              // Reconnecter explicitement (Socket.IO ne le fait pas toujours après disconnect manuel)
              try {
                currentSocket?.connect();
              } catch (e) {
                console.warn(JSON.stringify({
                  event: "network_online_reconnect_failed",
                  error: e instanceof Error ? e.message : String(e),
                  timestamp: new Date().toISOString()
                }));
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
        // ✅ Logs structurés
        console.log(JSON.stringify({
          event: "socket_disconnect",
          reason: reason,
          timestamp: new Date().toISOString()
        }));
        connectPromise = null;
        
        // ✅ Ne pas reconnecter si serveur a déconnecté volontairement (io server disconnect / unauthorized)
        const reasonStr = String(reason);
        if (reasonStr === "io server disconnect" || reasonStr === "unauthorized") {
          console.log(JSON.stringify({
            event: "socket_disconnect_stop_reconnect",
            reason: reasonStr,
            timestamp: new Date().toISOString()
          }));
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
          console.log(JSON.stringify({
            event: "socket_reconnect_failed",
            role: "enterprise",
            reconnectExhausted: true,
            timestamp: new Date().toISOString()
          }));
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
        // ✅ Logs structurés
        console.log(JSON.stringify({
          event: "socket_reconnect",
          attempt: attempt,
          timestamp: new Date().toISOString()
        }));
        lastHeartbeat = Date.now();
        
        if (socketRole === "driver") {
          joinDriverRoom().catch(() => {});
          // ✅ Re-setup booking listeners on reconnect
          setupBookingListeners(socket);
          // ✅ P2: Re-setup message listeners on reconnect
          setupMessageListeners(socket);
          console.log(JSON.stringify({
            event: "socket_rejoin_room",
            room: "driver",
            timestamp: new Date().toISOString()
          }));
        } else if (socketRole === "enterprise") {
          joinCompanyRoom().catch(() => {});
          console.log(JSON.stringify({
            event: "socket_rejoin_room",
            room: "company",
            timestamp: new Date().toISOString()
          }));
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

        console.error(JSON.stringify({
          event: "socket_connect_error",
          error: errorMsg,
          is_rate_limit: isRateLimit,
          timestamp: new Date().toISOString()
        }));

        // ✅ Si rate limit, désactiver reconnexion automatique pour éviter boucle
        if (isRateLimit && socket && socket.io) {
          console.warn(JSON.stringify({
            event: "socket_rate_limit_detected",
            action: "disabling_auto_reconnect",
            timestamp: new Date().toISOString()
          }));
          (socket.io.opts as any).reconnection = false;
          // Réactiver après 30 secondes (le retry_after du serveur)
          setTimeout(() => {
            if (socket && socket.io) {
              (socket.io.opts as any).reconnection = true;
              console.log(JSON.stringify({
                event: "socket_reconnect_reenabled",
                timestamp: new Date().toISOString()
              }));
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
              const newToken = await refreshDriverTokenSingleflight();
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
        
        console.error(JSON.stringify({
          event: "socket_unauthorized",
          error: errorMsg,
          is_token_expired: isTokenExpired,
          timestamp: new Date().toISOString()
        }));
        
        // ✅ Si token expiré, arrêter reconnexion auto et déclencher refresh
        if (isTokenExpired && socket && socket.io) {
          console.warn(JSON.stringify({
            event: "socket_token_expired_detected",
            action: "disabling_auto_reconnect_and_refreshing_token",
            timestamp: new Date().toISOString()
          }));
          
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
            const { refreshAccessToken } = await import("./api");
            const { secureStorage } = await import("./storage");
            
            const refreshToken = await secureStorage.getRefreshToken();
            if (refreshToken) {
              console.log(JSON.stringify({
                event: "socket_refreshing_token",
                timestamp: new Date().toISOString()
              }));
              
              const refreshResponse = await refreshAccessToken(refreshToken);
              
              if (refreshResponse.access_token) {
                // Sauvegarder le nouveau token
                await secureStorage.setAccessToken(refreshResponse.access_token);
                if (refreshResponse.refresh_token) {
                  await secureStorage.setRefreshToken(refreshResponse.refresh_token);
                }
                
                console.log(JSON.stringify({
                  event: "socket_token_refreshed_successfully",
                  action: "will_reconnect_manually",
                  timestamp: new Date().toISOString()
                }));
                
                // Réactiver reconnexion et reconnecter manuellement après un court délai
                setTimeout(() => {
                  if (socket && socket.io) {
                    (socket.io.opts as any).reconnection = true;
                    socket.connect();
                  }
                }, 1000);
              } else {
                console.error(JSON.stringify({
                  event: "socket_token_refresh_failed_no_token",
                  timestamp: new Date().toISOString()
                }));
              }
            } else {
              console.error(JSON.stringify({
                event: "socket_token_refresh_failed_no_refresh_token",
                timestamp: new Date().toISOString()
              }));
            }
          } catch (refreshError: any) {
            console.error(JSON.stringify({
              event: "socket_token_refresh_error",
              error: refreshError?.message || String(refreshError),
              timestamp: new Date().toISOString()
            }));
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
        
        console.error(JSON.stringify({
          event: "socket_error",
          error: errorMsg,
          error_data: errorData,
          is_rate_limit: isRateLimit,
          retry_after: errorData?.retry_after,
          timestamp: new Date().toISOString()
        }));
        
        // ✅ Si rate limit, désactiver reconnexion automatique temporairement
        if (isRateLimit && socket && socket.io) {
          const retryAfter = errorData?.retry_after || 30;
          console.warn(JSON.stringify({
            event: "socket_rate_limit_error_detected",
            action: "disabling_auto_reconnect",
            retry_after: retryAfter,
            timestamp: new Date().toISOString()
          }));
          (socket.io.opts as any).reconnection = false;
          // Réactiver après retry_after + marge de sécurité
          setTimeout(() => {
            if (socket && socket.io) {
              (socket.io.opts as any).reconnection = true;
              console.log(JSON.stringify({
                event: "socket_reconnect_reenabled_after_rate_limit",
                timestamp: new Date().toISOString()
              }));
            }
          }, (retryAfter + 5) * 1000); // +5s marge de sécurité
        }
      });

      socket.on("connected", (data: any) => {
        console.log(JSON.stringify({
          event: "socket_handshake_ok",
          data: data,
          timestamp: new Date().toISOString()
        }));
      });

      // ✅ Événements supplémentaires du backend
      socket.on("joined_room", (data: any) => {
        console.log(JSON.stringify({
          event: "socket_joined_room",
          rooms: data?.rooms,
          timestamp: new Date().toISOString()
        }));
      });

      socket.on("joined_company", (data: any) => {
        console.log(JSON.stringify({
          event: "socket_joined_company",
          company_id: data?.company_id,
          room: data?.room,
          timestamp: new Date().toISOString()
        }));
      });

      socket.on("typing_indicator", (data: any) => {
        console.log(JSON.stringify({
          event: "user_typing",
          user_id: data?.user_id,
          timestamp: new Date().toISOString()
        }));
        // TODO: Afficher indicateur de frappe dans UI
      });

      socket.on("stop_typing", (data: any) => {
        console.log(JSON.stringify({
          event: "user_stop_typing",
          user_id: data?.user_id,
          timestamp: new Date().toISOString()
        }));
      });

      // ✅ Événements d'arrivée driver
      socket.on("driver_arrived_at_pickup", (data: any) => {
        console.log(JSON.stringify({
          event: "driver_arrived_pickup",
          driver_id: data?.driver_id,
          timestamp: new Date().toISOString()
        }));
        // TODO: Afficher notification "Driver arrivé au point de départ"
      });

      socket.on("driver_arrived_at_dropoff", (data: any) => {
        console.log(JSON.stringify({
          event: "driver_arrived_dropoff",
          driver_id: data?.driver_id,
          timestamp: new Date().toISOString()
        }));
        // TODO: Afficher notification "Driver arrivé à destination"
      });
    } catch (e) {
      connectPromise = null;
      reject(e);
    }
  });

  return connectPromise;
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
    
    console.log(JSON.stringify({
      event: "socket_disconnect_cleanup",
      timestamp: new Date().toISOString()
    }));
  } catch (err) {
    console.error(JSON.stringify({
      event: "socket_disconnect_error",
      error: err instanceof Error ? err.message : String(err),
      timestamp: new Date().toISOString()
    }));
  } finally {
    socket = null;
    socketRole = null;
    connectPromise = null;
    lastHeartbeat = Date.now();
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
    console.warn(JSON.stringify({
      event: "reconnect_socket_manual_no_token",
      role,
      timestamp: new Date().toISOString()
    }));
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
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s || !s.connected) {
    console.warn(JSON.stringify({
      event: "driver_heartbeat_skipped",
      reason: "socket_not_connected",
      timestamp: new Date().toISOString()
    }));
    return;
  }
  
  s.emit("driver:heartbeat", {
    ...payload,
    timestamp: Date.now(),
  });
  
  console.log(JSON.stringify({
    event: "driver_heartbeat_sent",
    last_mission_id: payload.last_mission_id,
    has_location: !!payload.location,
    timestamp: new Date().toISOString()
  }));
}

// Helpers côté driver
export async function joinDriverRoom() {
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s) {
    console.warn("⚠️ Socket non connecté, impossible de rejoindre la room");
    return;
  }

  try {
    const idStr = await AsyncStorage.getItem("driver_id");
    const driver_id = idStr ? Number(idStr) : undefined;
    // ✅ FIX: Validation stricte du driver_id
    if (driver_id && Number.isFinite(driver_id) && driver_id > 0) {
      s?.emit("join_driver_room", { driver_id });
      console.log(`📍 join_driver_room émis avec driver_id=${driver_id}`);
    } else {
      s?.emit("join_driver_room");
      console.log("📍 join_driver_room émis sans driver_id (fallback JWT)");
    }
  } catch {
    s?.emit("join_driver_room");
    console.log(
      "📍 join_driver_room émis sans driver_id (erreur AsyncStorage)"
    );
  }
}

export async function joinCompanyRoom() {
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s) {
    console.warn("⚠️ Socket non connecté, impossible de rejoindre la room entreprise");
    return;
  }
  s.emit("join_company");
  console.log("🏢 join_company émis");
}

export async function sendDriverLocation(payload: {
  latitude: number;
  longitude: number;
  speed?: number;
  heading?: number;
  accuracy?: number;
  timestamp?: number | string;
}) {
  const s = socket ?? (connectPromise ? await connectPromise : null);
  if (!s) {
    console.warn(
      "⚠️ Socket non connecté, impossible d'envoyer la localisation"
    );
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
    console.log(`📍 driver_location émis:`, {
      has_driver_id,
      lat: payload.latitude,
      lon: payload.longitude,
    });
  } catch {
    s?.emit("driver_location", payload);
    console.log("📍 driver_location émis sans driver_id (erreur)");
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
