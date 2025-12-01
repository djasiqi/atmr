// services/socket.ts
import { io, type Socket } from "socket.io-client";
import AsyncStorage from "@react-native-async-storage/async-storage";
import NetInfo from "@react-native-community/netinfo";
import { baseURL } from "./api"; // ← réutilise l'URL déjà déduite (Expo dev/prod)

type SocketRole = "driver" | "enterprise";

// Flask-SocketIO vit à la racine (/socket.io). On enlève le suffixe /api ou /api/vX.
const SOCKET_ORIGIN = baseURL.replace(/\/api(?:\/v\d+)?$/, "");
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
  const jitterDelay = Math.random() * 100; // 0-100ms
  const jitterMax = Math.random() * 500; // 0-500ms
  
  const base = {
    path: "/socket.io", // ⚠️ sans slash final
    auth: { token },
    extraHeaders: { Authorization: `Bearer ${token}` },
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 1000 + jitterDelay,  // ✅ Jitter: 1000-1100ms
    reconnectionDelayMax: 10000 + jitterMax,  // ✅ Jitter: 10000-10500ms
    timeout: 20000,
    forceNew: true,
    transports: IS_SECURE ? ["websocket"] : ["websocket", "polling"],
    upgrade: true,
    rememberUpgrade: true,
    secure: IS_SECURE,
  };
  // En dev non sécurisé (HTTP), on garde polling en secours
  if (IS_DEV && !IS_SECURE) {
    return { ...base, transports: ["websocket", "polling"] };
  }
  return base;
}

export async function connectSocket(
  token: string,
  role: SocketRole = "driver"
): Promise<Socket | null> {
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

  if (socket?.connected && socketRole === role) {
    return socket;
  }
  if (connectPromise && socketRole === role) {
    return connectPromise;
  }
  if (IS_DEV) enableSocketIODebug();

  socketRole = role;

  connectPromise = new Promise<Socket>((resolve, reject) => {
    try {
      // ⚠️ Utiliser l’origine sans /api sinon 404 sur /api/socket.io
      socket = io(SOCKET_ORIGIN, buildOptions(token));

      socket.on("connect", () => {
        // ✅ Logs structurés
        console.log(JSON.stringify({
          event: "socket_connect",
          socket_id: socket?.id,
          timestamp: new Date().toISOString(),
          user_type: socketRole || "unknown"
        }));
        
        if (socketRole === "driver") {
          joinDriverRoom().catch(() => {});  // ✅ Corrigé : enlever le + (syntaxe incorrecte)
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
