// services/socket.ts
import { io, type Socket } from "socket.io-client";
import { baseURL } from "./api"; // ← réutilise l’URL déjà déduite (Expo dev/prod)

// (Optionnel) logs verbeux en dev pour socket.io
let enableSocketIODebug = () => {};
try {
  enableSocketIODebug = () =>
    require("debug").enable("socket.io-client:*,engine.io-client:*");
} catch {}

let socket: Socket | null = null;
let connectPromise: Promise<Socket> | null = null;

const IS_DEV = __DEV__;

function buildOptions(token: string) {
  const base = {
    path: "/socket.io", // ⚠️ sans slash final
    auth: { token },
    extraHeaders: { Authorization: `Bearer ${token}` },
    reconnection: true,
    reconnectionAttempts: Infinity,
    // reconnectionAttempts: Number.POSITIVE_INFINITY,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 20000,
    forceNew: true,
  };
  // En dev mobile: polling-only + pas d’upgrade (Android/proxy-friendly)
  return IS_DEV ? { ...base, transports: ["polling"], upgrade: false } : base;
}

export async function connectSocket(token: string): Promise<Socket | null> {
  if (!token) {
    console.warn("❌ Aucun token fourni à connectSocket");
    return null;
  }
  if (socket?.connected) return socket;
  if (connectPromise) return connectPromise;
  if (IS_DEV) enableSocketIODebug();

  connectPromise = new Promise<Socket>((resolve, reject) => {
    try {
      socket = io(baseURL, buildOptions(token));

      socket.on("connect", () => {
        console.log("✅ Socket connecté, sid:", socket?.id);
        // rejoindre la room chauffeur à la connexion
        socket?.emit("join_driver_room");
        resolve(socket as Socket);
      });

      socket.on("disconnect", (reason) => {
        console.log("🔌 Socket déconnecté:", reason);
        connectPromise = null;
      });

      socket.on("connect_error", (err: any) => {
        console.error("❗ connect_error:", err?.message || err);
        connectPromise = null;
        reject(err);
      });

      socket.on("unauthorized", (data: any) => {
        console.error("⛔ unauthorized:", data);
      });

      socket.on("error", (e: any) => {
        console.error("❌ socket error:", e);
      });

      socket.on("connected", (data: any) => {
        console.log("🤝 handshake OK:", data);
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
  } finally {
    socket = null;
    connectPromise = null;
  }
}

// Helpers côté driver
export async function joinDriverRoom() {
  const s = socket ?? (connectPromise ? await connectPromise : null);
  s?.emit("join_driver_room");
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
  s?.emit("driver_location", payload);
}