// frontend/src/hooks/useCompanySocket.js
import { useEffect, useState } from "react";
import { io } from "socket.io-client";
import { getAccessToken } from "./useAuthToken";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

// --- Singleton au niveau module ---
let SOCKET_SINGLETON = null;
let BASE_LISTENERS_ATTACHED = false;

export default function useCompanySocket() {
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const token = getAccessToken();
    if (!token) {
      console.warn("❌ Aucun token JWT disponible pour WebSocket.");
      return;
    }
    if (!SOCKET_SINGLETON) {
      SOCKET_SINGLETON = io(API_URL, {
        transports: ["websocket", "polling"],
        withCredentials: true,
        autoConnect: false,
        reconnection: true,
        reconnectionAttempts: Infinity,
        reconnectionDelay: 500,
        reconnectionDelayMax: 5000,
        timeout: 8000,
        path: "/socket.io",
      });
    }
    // met à jour le token et (re)connecte si besoin
    SOCKET_SINGLETON.auth = { token };
    if (!SOCKET_SINGLETON.connected && !SOCKET_SINGLETON.connecting) {
      SOCKET_SINGLETON.connect();
    }
    if (!BASE_LISTENERS_ATTACHED) {
      SOCKET_SINGLETON.on("connect", () => {
        console.log("✅ WebSocket connecté (entreprise)", new Date().toISOString());
        SOCKET_SINGLETON.emit("join_company");
      });
      SOCKET_SINGLETON.on("disconnect", (reason) => {
        console.log("🔌 WebSocket déconnecté:", reason);
      });
      SOCKET_SINGLETON.on("connect_error", (err) => {
        console.error("❌ Erreur WebSocket :", err?.message || err);
      });
      SOCKET_SINGLETON.on("error", (err) => {
        console.error("🚨 Erreur Socket.IO :", err);
      });
      BASE_LISTENERS_ATTACHED = true;
    }
    setSocket(SOCKET_SINGLETON);

    // écoute rafraîchissement token entre onglets
    const onStorage = (e) => {
      if (e.key === "access_token") {
        const t = getAccessToken();
        if (t) {
          SOCKET_SINGLETON.auth = { token: t };
          if (!SOCKET_SINGLETON.connected) SOCKET_SINGLETON.connect();
        }
      }
    };
    window.addEventListener("storage", onStorage);

    return () => {
      window.removeEventListener("storage", onStorage);
      // ne pas disconnect() le singleton ici
    };
  }, []);

  return socket;
}
