import axios from "axios";

// Base URL unifiée → toujours sous /api (en dev: configure le proxy CRA)
const apiClient = axios.create({
  baseURL: "http://127.0.0.1:5000/api", // <-- MODIFICATION CRUCIALE
  headers: { "Content-Type": "application/json" },
  withCredentials: true,
});

// ✅ Fonction exportable pour gérer la déconnexion proprement
export const logoutUser = () => {
  localStorage.removeItem("authToken");
  localStorage.removeItem("user");
  localStorage.removeItem("public_id");
  window.location.href = "/login"; // ✅ Redirection forcée
};

// 🔹 Intercepteur pour ajouter automatiquement le token JWT
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem("authToken");
  if (process.env.NODE_ENV !== "production" && token) {
    const safe = token.length > 20 ? `${token.slice(0,10)}…${token.slice(-10)}` : token;
    console.log("🔍 Token pour la requête :", safe);
  }
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  } else {
    // on évite d'envoyer un header Authorization vide
    if (config.headers && "Authorization" in config.headers) {
      delete config.headers.Authorization;
    }
  }
  // 🛡️ Dé-doublonnage défensif du préfixe /api
  if (config.baseURL?.endsWith("/api") && config.url?.startsWith("/api/")) {
    config.url = config.url.replace(/^\/api\//, "/");
  }
  // Normalise les URLs relatives : ajoute un leading slash si manquant
  if (typeof config.url === "string" && !/^https?:\/\//i.test(config.url)) {
    if (!config.url.startsWith("/")) config.url = "/" + config.url;
  }
  if (config.params) {
    try { console.log("🧭 axios params:", JSON.parse(JSON.stringify(config.params))); }
    catch { console.log("🧭 axios params (raw):", config.params); }
  }  
  return config;
});


// 🔹 Intercepteur pour gérer les erreurs globales
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const cfg = error.config || {};
    const isNetworkError = !error.response;
    const method = (cfg.method || "get").toUpperCase();
    // Retry unique, seulement pour GET/HEAD, et par requête
    if (isNetworkError && !cfg._retried && (method === "GET" || method === "HEAD")) {
      cfg._retried = true;
      console.warn("🔄 Erreur réseau détectée, tentative unique de reconnexion (GET/HEAD)…");
      await new Promise((r) => setTimeout(r, 300));
      return apiClient(cfg);
    }

    if (error.response) {
      const { status, data } = error.response;

      if (status === 401 && data?.error === "token_expired") {
        console.warn("🔐 Token expiré. Déconnexion...");
        logoutUser();
      } else {
        switch (status) {
          case 403:
            alert("⛔ Vous n'avez pas les permissions nécessaires.");
            break;
          case 404:
            alert("❌ La ressource demandée est introuvable.");
            break;
          case 500:
            alert("⚠️ Erreur interne du serveur. Réessayez plus tard.");
            break;
          default:
            alert(data?.message || "⚠️ Une erreur inconnue est survenue.");        }
      }
    } else {
      alert("🌐 Erreur réseau. Vérifiez votre connexion internet.");
    }

    return Promise.reject(error);
  }
);

export default apiClient;
