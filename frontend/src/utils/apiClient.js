// frontend/src/utils/apiClient.js
import axios from "axios";

let apiBase =
  process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || "/api";

// En dev (CRA sur localhost:3000), on force le proxy '/api' pour éviter le CORS
try {
  if (
    typeof window !== "undefined" &&
    window.location &&
    /localhost:3000$/i.test(window.location.host)
  ) {
    // En dev, reste strictement sur /api pour s'aligner avec le backend (pas de /v1 implicite)
    apiBase = "/api";
  }
} catch (_) {
  // no-op
}

const apiClient = axios.create({
  baseURL: apiBase,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
  withCredentials: false, // ❌ pas de cookies, on est en JWT header
  timeout: 30000,
});

export const logoutUser = () => {
  localStorage.removeItem("authToken");
  localStorage.removeItem("user");
  localStorage.removeItem("public_id");
  window.location.href = "/login";
};

apiClient.interceptors.request.use((cfg) => {
  // normalise baseURL (évite //api si jamais apiBase finit par /)
  if (cfg.baseURL && cfg.baseURL.endsWith("/")) {
    cfg.baseURL = cfg.baseURL.slice(0, -1);
  }
  
  // NE PAS remplacer Authorization si déjà présent (cas du refresh token)
  const hasAuthHeader = cfg.headers && cfg.headers.Authorization;
  const token = localStorage.getItem("authToken");
  
  // N'ajoute pas Authorization si :
  // - Le header Authorization est déjà défini (refresh token)
  // - On a un opt-out explicite
  // - C'est une requête vers /auth/refresh-token
  if (
    !hasAuthHeader &&
    token &&
    !cfg.url?.includes('/auth/refresh-token') &&
    !(cfg.headers && (cfg.headers["X-Skip-Auth"] || cfg.headers["x-skip-auth"]))
  ) {
    cfg.headers.Authorization = `Bearer ${token}`;
  }
  return cfg;
});

apiClient.interceptors.response.use(
  (res) => res,
  (error) => {
    const status = error?.response?.status;
    const cfg = error?.config || {};
    
    // Message sympa pour 429 (limiter)
    if (status === 429) {
      console.warn(
        "Vous avez effectué trop de requêtes. Merci de patienter un peu."
      );
    }
    
    // Ne pas déconnecter automatiquement si c'est une requête de refresh token
    // ou si l'option skipAuthRedirect est définie
    if (status === 401 && !cfg.skipAuthRedirect && !cfg.url?.includes('/auth/refresh-token')) {
      logoutUser();
    }

    // Pas de fallback automatique vers /api/v1: on reste sur la vérité du backend
    return Promise.reject(error);
  }
);

export default apiClient;
