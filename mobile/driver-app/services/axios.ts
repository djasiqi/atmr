// services/axios.ts
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";

const api = axios.create({
  baseURL: "http://192.168.1.216:5000", // Use same IP as socket
});

// 1️⃣ Intercepteur de debug : logge méthode, URL et payload
api.interceptors.request.use(config => {
  console.log(
    `🚀 [HTTP] ${config.method?.toUpperCase()} ${config.url}`,
    "– payload:", 
    config.data
  );
  return config;
});

// 2️⃣ Intercepteur d’injection du token d’accès
api.interceptors.request.use(
  async config => {
    const token = await AsyncStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  error => Promise.reject(error)
);

// 3️⃣ Intercepteur de réponse : rafraîchit le token sur 401
api.interceptors.response.use(
  response => response,
  async error => {
    const originalRequest = error.config;
    // Si 401 non déjà retry
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      const refreshToken = await AsyncStorage.getItem("refresh_token");
      if (refreshToken) {
        try {
          // Use the same base URL as the main api instance
          const res = await axios.post(
            `${api.defaults.baseURL}/refresh`, // 👈 FIX HERE
            null,
            { headers: { Authorization: `Bearer ${refreshToken}` } }
          );
          const newAccessToken = res.data.access_token;
          await AsyncStorage.setItem("token", newAccessToken);
          // Rejoue la requête initiale avec le nouveau token
          originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
          return api(originalRequest);
        } catch (refreshError) {
          console.error("Erreur de refresh token :", refreshError);
          await AsyncStorage.removeItem("token");
          await AsyncStorage.removeItem("refresh_token");
          // Ici tu peux forcer une déconnexion ou rediriger vers login
        }
      }
    }
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  response => response,
  error => {
    console.error("❌ API Error Details:", {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      data: error.response?.data,
      message: error.message,
      network: !error.response ? "NETWORK_ERROR" : "HTTP_ERROR"
    });
    return Promise.reject(error);
  }
);

export default api;
