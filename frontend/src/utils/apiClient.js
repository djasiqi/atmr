// frontend/src/utils/apiClient.js
import axios from "axios";

const apiBase =
  process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || "/api";

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
  const token = localStorage.getItem("authToken");
  // N'ajoute pas Authorization si on a explicitement un opt-out
  if (
    token &&
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
    // Message sympa pour 429 (limiter)
    if (status === 429) {
      console.warn(
        "Vous avez effectué trop de requêtes. Merci de patienter un peu."
      );
    }
    if (status === 401) logoutUser();
    return Promise.reject(error);
  }
);

export default apiClient;
