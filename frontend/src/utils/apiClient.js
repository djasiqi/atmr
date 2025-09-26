// frontend/src/utils/apiClient.js
import axios from "axios";

const apiBase = (process.env.REACT_APP_API_BASE_URL || "/api").replace(/\/+$/, "");

const apiClient = axios.create({
  baseURL: apiBase,
  headers: { "Content-Type": "application/json" },
  withCredentials: true,
  timeout: 15000,
});

// Endpoints d'auth où on ne veut pas d'Authorization
const SKIP_AUTH = new Set([
  "/auth/login",
  "/auth/refresh-token",
  "/auth/forgot-password",
  "/auth/reset-password",
]);

apiClient.interceptors.request.use((config) => {
  // normalise l’URL
  let url = config.url || "";
  if (typeof url === "string" && !/^https?:\/\//i.test(url)) {
    if (!url.startsWith("/")) url = "/" + url;
    if (apiClient.defaults.baseURL.endsWith("/api") && url.startsWith("/api/")) {
      url = url.replace(/^\/api\//, "/"); // évite /api/api
    }
    config.url = url;
  }

  // n'ajoute pas Authorization pour les routes d'auth ou si X-Skip-Auth
  const skip = SKIP_AUTH.has(config.url) || config.headers?.["X-Skip-Auth"];
  if (!skip) {
    const token = localStorage.getItem("authToken");
    if (token) config.headers.Authorization = `Bearer ${token}`;
  } else if (config.headers?.Authorization) {
    delete config.headers.Authorization;
  }
  if (config.headers && config.headers["X-Skip-Auth"]) {
    delete config.headers["X-Skip-Auth"];
  }

  return config;
});

// 👉 export nommé attendu par tes composants
export const logoutUser = () => {
  try {
    localStorage.removeItem("authToken");
    localStorage.removeItem("user");
    localStorage.removeItem("public_id");
  } finally {
    // recharge proprement l’app sur la page de login
    window.location.assign("/login");
  }
};

apiClient.interceptors.response.use(
  (res) => res,
  (error) => {
    const { response } = error || {};
    if (response?.status === 401) {
      const reason = response.data?.error || response.data?.message || "";
      // si token expiré/non valide -> déconnexion
      if (/token/i.test(reason) || reason === "token_expired") {
        console.warn("Token invalide/expiré → logout");
        logoutUser();
      }
    }
    return Promise.reject(error);
  }
);

export default apiClient;
