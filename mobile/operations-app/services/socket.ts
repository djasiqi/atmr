// services/socket.ts
import { io, type Socket } from "socket.io-client";
import AsyncStorage from "@react-native-async-storage/async-storage";
import NetInfo from "@react-native-community/netinfo";
import { Platform } from "react-native";
import Constants from "expo-constants";
import { baseURL, getAssignedTrips, getCompanyMessages, type Booking, type Message } from "./api";
import { resolveBookingConflicts, resolveMessageConflicts, type Conflict } from "./conflictResolution";

type SocketRole = "driver" | "enterprise";

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
    return socketUrl;
  }

  // PRIORITÉ 2: Depuis app.config.js extra.socketUrl
  const expoExtra = Constants.expoConfig?.extra || {};
  const configSocketUrl = expoExtra.socketUrl;
  if (configSocketUrl) {
    return configSocketUrl;
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
    return socketOrigin;
  } catch (e) {
    // Fallback: regex si URL malformée
    console.warn(`[Socket] Erreur parsing baseURL=${baseURL}, fallback regex:`, e);
    const socketOrigin = baseURL.replace(/\/api(?:\/v\d+)?$/, "");
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
    return `${apiUrl.protocol}//${apiUrl.host}`;
  } catch {
    throw new Error("EXPO_PUBLIC_SOCKET_URL est requis en production");
  }
};

let SOCKET_ORIGIN = getSocketOrigin();

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

function buildOptions(token: string) {
  // ✅ Jitter anti-storm: ajouter variation aléatoire pour éviter reconnexions simultanées
  const jitterDelay = Math.random() * 1000; // 0-1000ms (augmenté)
  const jitterMax = Math.random() * 2000; // 0-2000ms (augmenté)
  
  const base = {
    path: "/socket.io", // ⚠️ sans slash final
    auth: { token },
    extraHeaders: { Authorization: `Bearer ${token}` },
    reconnection: true,
    reconnectionAttempts: 10,  // Max 10 tentatives (au lieu de Infinity)
    reconnectionDelay: 5000 + jitterDelay,  // ✅ Augmenté de 1s à 5s + jitter (5000-6000ms)
    reconnectionDelayMax: 30000 + jitterMax,  // ✅ Augmenté de 10s à 30s + jitter (30000-32000ms)
    timeout: 20000,
    forceNew: false,  // Réutiliser connexion existante
    // ✅ FIX: Toujours inclure polling comme fallback (même en HTTPS)
    // React Native peut avoir des problèmes avec websocket sur certains réseaux
    transports: ["websocket", "polling"], // ✅ Toujours inclure polling
    upgrade: true,
    rememberUpgrade: true,
    secure: IS_SECURE,
  };
  // ✅ Polling toujours inclus dans base, plus besoin de logique conditionnelle
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

  // ✅ Vérifier que socket n'est pas déjà connecté
  if (socket && socket.connected && socketRole === role) {
    return socket;  // Réutiliser connexion existante
  }

  // ✅ Si socket existe mais déconnecté, nettoyer avant de créer nouveau
  if (socket && socket.disconnected) {
    try {
      socket.off();
      socket.disconnect();
    } catch {}
    socket = null;
  }

  if (connectPromise && socketRole === role) {
    return connectPromise;
  }
  if (IS_DEV) enableSocketIODebug();

  socketRole = role;

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

  console.log(`[connectSocket] 📍 Avant création socket, SOCKET_ORIGIN=${SOCKET_ORIGIN}`);
  
  connectPromise = new Promise<Socket>((resolve, reject) => {
    try {
      console.log(`[connectSocket] 📍 Appel io() avec options:`, buildOptions(token));
      // ⚠️ Utiliser l'origine sans /api sinon 404 sur /api/socket.io
      socket = io(SOCKET_ORIGIN, buildOptions(token));
      console.log(`[connectSocket] ✅ Socket créé:`, socket ? `id=${socket.id || 'pending'}` : 'NULL');

      socket.on("connect", async () => {
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
              // Socket.IO reconnecte automatiquement
            }
          });
        }
        
        resolve(socket as Socket);
      });

      socket.on("disconnect", (reason) => {
        // ✅ Logs structurés
        console.log(JSON.stringify({
          event: "socket_disconnect",
          reason: reason,
          timestamp: new Date().toISOString()
        }));
        connectPromise = null;
        
        // ✅ Ne pas reconnecter si serveur a déconnecté volontairement
        const reasonStr = String(reason);
        if (reasonStr === "io server disconnect" || reasonStr === "unauthorized") {
          console.log(JSON.stringify({
            event: "socket_disconnect_stop_reconnect",
            reason: reasonStr,
            timestamp: new Date().toISOString()
          }));
          // Désactiver la reconnexion automatique
          if (socket && socket.io) {
            (socket.io.opts as any).reconnection = false;
          }
          return; // Pas de reconnexion auto
        }
      });

      // ✅ Handler reconnect : rejoin automatiquement les rooms après reconnexion
      socket.on("reconnect", (attempt) => {
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
        console.error(JSON.stringify({
          event: "socket_connect_error",
          error: err?.message || String(err),
          timestamp: new Date().toISOString()
        }));
        connectPromise = null;
        reject(err);
      });

      socket.on("unauthorized", (data: any) => {
        console.error(JSON.stringify({
          event: "socket_unauthorized",
          error: data?.error || String(data),
          timestamp: new Date().toISOString()
        }));
      });

      socket.on("error", (e: any) => {
        console.error(JSON.stringify({
          event: "socket_error",
          error: e?.message || String(e),
          timestamp: new Date().toISOString()
        }));
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
  try {
    socket?.off();
    socket?.disconnect();
    
    // ✅ Nettoyer heartbeat interval
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
      heartbeatInterval = null;
    }
    
    // ✅ Nettoyer listener réseau
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
  }
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
