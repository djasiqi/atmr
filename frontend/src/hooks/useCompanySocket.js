// frontend/src/hooks/useCompanySocket.js
import { useEffect, useState } from "react";
import { io } from "socket.io-client";
import { getAccessToken } from "./useAuthToken";

// On dérive l'URL Socket depuis la base API.
// - Si REACT_APP_API_BASE_URL = "http://127.0.0.1:5000/api" -> "http://127.0.0.1:5000"
// - Si baseURL = "/api" (proxy CRA) -> window.location.origin (http://localhost:3000) et proxy /socket.io fera le reste.
const API_BASE = process.env.REACT_APP_API_BASE_URL || "/api";
const SOCKET_ORIGIN = API_BASE.startsWith("http")
  ? API_BASE.replace(/\/api\/?$/, "")
  : window.location.origin;

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
      SOCKET_SINGLETON = io(SOCKET_ORIGIN, {
        path: "/socket.io",
        transports: ["websocket", "polling"],   // tente WS, retombe sur polling si besoin
        timeout: 30000,                      
        reconnection: true,
        reconnectionAttempts: Infinity,
        reconnectionDelay: 500,
        reconnectionDelayMax: 5000,
        autoConnect: false,
        // withCredentials est inutile en JWT header/auth payload (cookies non utilisés)
        withCredentials: false,
        // Le token est passé dans le "auth" du handshake (côté serveur: socket.handshake.auth.token)
        auth: { token },
      });
    } else {
      // met à jour le token du handshake
      SOCKET_SINGLETON.auth = { token };
    }

    // connect si pas déjà en cours/établi
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
            // Gérer spécifiquement les erreurs d'authentification JWT
            if (err?.message?.includes("unauthorized") || err?.data?.includes("unauthorized") || err?.message?.includes("Subject must be a string")) {
              console.warn("🚨 Erreur d'authentification WebSocket - Token JWT invalide ou expiré");
              // Optionnel: déclencher un refresh token ou une reconnexion
              // window.dispatchEvent(new CustomEvent('websocket-auth-error'));
            }
      });
      SOCKET_SINGLETON.on("error", (err) => {
        console.error("🚨 Erreur Socket.IO :", err);
      });
      BASE_LISTENERS_ATTACHED = true;
    }

    setSocket(SOCKET_SINGLETON);

    // 🔁 Rafraîchissement token entre onglets — corrige la clé (authToken)
    const onStorage = (e) => {
      if (e.key === "authToken") {
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
