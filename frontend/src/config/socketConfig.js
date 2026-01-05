// Configuration centralisée Socket.IO pour le frontend
// ✅ Centralisation pour éviter duplication et incohérences

// Chemin Socket.IO depuis variable d'environnement ou défaut
export const SOCKET_PATH = process.env.REACT_APP_SOCKET_PATH || "/socket.io";

// Configuration de base Socket.IO
export const SOCKET_CONFIG = {
  path: SOCKET_PATH,
  reconnection: true,
  reconnectionAttempts: 10, // Max 10 tentatives
  reconnectionDelay: 1000, // Base delay (jitter sera ajouté)
  reconnectionDelayMax: 30000, // Base max delay (jitter sera ajouté)
  timeout: 30000, // 30s pour laisser plus de temps au proxy
  forceNew: false,
  withCredentials: true,
  // ✅ Transports : polling d'abord en dev (plus fiable avec proxy), websocket en prod
  transports:
    process.env.NODE_ENV === "production"
      ? ["websocket", "polling"]
      : ["polling", "websocket"],
};

// Helper pour détecter si on est en développement localhost
export const isDevelopmentLocalhost = () => {
  return (
    typeof window !== "undefined" &&
    window.location &&
    /localhost:3000$/i.test(window.location.host)
  );
};

// Configuration avec transports adaptés pour développement localhost
export const getSocketTransports = () => {
  if (isDevelopmentLocalhost()) {
    return ["polling", "websocket"]; // Polling d'abord en dev localhost
  }
  return SOCKET_CONFIG.transports; // Utiliser config par défaut
};

