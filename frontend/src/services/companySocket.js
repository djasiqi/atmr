// src/services/companySocket.js
import { io } from "socket.io-client";
import { getAccessToken } from "../hooks/useAuthToken";

let socket = null;
let connectPromise = null;
const listeners = new Map(); // event -> callback

const API_URL = process.env.REACT_APP_API_BASE_URL;
const IS_DEV = process.env.NODE_ENV === "development";

function buildSocketOptions(token) {
  // Dev: polling-only + no upgrade (Android/proxy-friendly)
  // Prod: laissez Socket.IO faire (WS prioritaire, fallback polling)
  const base = {
    path: "/socket.io", // ⚠️ sans slash final
    auth: { token },
    extraHeaders: { Authorization: `Bearer ${token}` },
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 500,
    reconnectionDelayMax: 5000,
    timeout: 20000,
    forceNew: true,
  };
  return IS_DEV
    ? { ...base, transports: ["polling"], upgrade: false }
    : base; // en prod: upgrade WS autorisé par défaut
}

export function getCompanySocket() {
  if (socket && socket.connected) return socket;

  if (!connectPromise) {
    const token = getAccessToken();
    if (!token) {
      console.warn("❌ Aucun token pour WebSocket");
      return null;
    }
    connectPromise = new Promise((resolve, reject) => {
      try {
        socket = io(API_URL, buildSocketOptions(token));

        socket.on("connect", () => {
          console.log("✅ WebSocket connecté (company)", socket.id);
          resolve(socket);
        });

        socket.on("disconnect", (reason) => {
          console.log("🔌 WebSocket déconnecté:", reason);
          connectPromise = null;
        });

        socket.on("connect_error", (err) => {
          console.error("⛔ Erreur de connexion WebSocket:", err?.message || err);
          connectPromise = null;
          reject(err);
        });

        socket.on("unauthorized", (err) => {
          console.error("⛔ Unauthorized WebSocket:", err);
        });
      } catch (e) {
        console.error("❌ Socket init error:", e);
        connectPromise = null;
        reject(e);
      }
    });
  }
  return socket || null;
}

export async function ensureCompanySocket() {
  const existing = getCompanySocket();
  if (existing && existing.connected) return existing;
  if (!connectPromise) return null;
  return connectPromise;
}

// ✅ Rejoindre une room d’entreprise (legacy no-op: le backend joint déjà la room à la connexion côté 'company')
export async function joinCompanyRoom(companyId) {
  const s = await ensureCompanySocket();
  if (!s) return;
  // Optionnel: si un handler existe côté serveur
  s.emit("join_company_room", { company_id: companyId });
}

// ✅ Quitter la room (optionnel si le serveur expose un handler)
export async function leaveCompanyRoom(companyId) {
  const s = await ensureCompanySocket();
  if (!s) return;
  s.emit("leave_company_room", { company_id: companyId });
}

// ✅ Écouter les mises à jour de localisation des chauffeurs
export async function onDriverLocationUpdate(callback) {
  const s = await ensureCompanySocket();
  if (!s) return;
  // Remplace l’éventuel listener existant pour éviter les doublons
  const evt = "driver_location_update";
  const prev = listeners.get(evt);
  if (prev) s.off(evt, prev);
  s.on(evt, callback);
  listeners.set(evt, callback);
}

// ✅ Arrêter d’écouter les mises à jour
export async function offDriverLocationUpdate() {
  const s = await ensureCompanySocket();
  if (!s) return;
  const evt = "driver_location_update";
  const prev = listeners.get(evt);
  if (prev) {
    s.off(evt, prev);
    listeners.delete(evt);
  }
}

// ✅ Fermeture propre (ex. au logout)
export function disconnectCompanySocket() {
  try {
    listeners.forEach((cb, evt) => {
      socket?.off(evt, cb);
    });
    listeners.clear();
    connectPromise = null;
    if (socket) {
      socket.disconnect();
      socket = null;
    }
  } catch {}
}