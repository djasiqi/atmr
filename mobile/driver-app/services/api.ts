import Constants from "expo-constants";
import axios, { isAxiosError } from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";

// --- config / baseURL -------------------------------------------------
const expoExtra = Constants.expoConfig?.extra || {};
const PROD_API_URL = expoExtra.productionApiUrl;

const getDevHost = (): string => {
  const legacyHost = (Constants as any)?.manifest?.debuggerHost?.split(":")[0];
  const newHost = (Constants as any)?.expoConfig?.hostUri?.split(":")[0];
  const detectedHost = newHost || legacyHost;
  if (
    !detectedHost ||
    detectedHost === "localhost" ||
    detectedHost === "127.0.0.1"
  ) {
    return "192.168.1.216";
  }
  return detectedHost;
};

const PORT = expoExtra.backendPort || "5000";

export const baseURL = __DEV__
  ? `http://${getDevHost()}:${PORT}/api`
  : `${(PROD_API_URL || "").replace(/\/$/, "")}/api`;

const TOKEN_KEY = "token";
const REFRESH_KEY = "refresh_token";

// --- axios instance ---------------------------------------------------
const api = axios.create({
  baseURL,
  timeout: 10000,
  headers: { "Content-Type": "application/json" },
});

// Attach access token
api.interceptors.request.use(async (config) => {
  const token = await AsyncStorage.getItem(TOKEN_KEY);
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// --- Refresh (deduped) ------------------------------------------------
let refreshingPromise: Promise<string | null> | null = null;

const refreshPaths = ["/auth/refresh-token", "/auth/refresh", "/refresh"];

const doRefresh = async (): Promise<string | null> => {
  if (refreshingPromise) return refreshingPromise;
  refreshingPromise = (async () => {
    const refreshToken = await AsyncStorage.getItem(REFRESH_KEY);
    if (!refreshToken) return null;

    for (const path of refreshPaths) {
      try {
        const res = await axios.post(`${baseURL}${path}`, null, {
          headers: { Authorization: `Bearer ${refreshToken}` },
        });
        const newAccessToken =
          (res.data as any)?.access_token || (res.data as any)?.token;
        if (newAccessToken) {
          await AsyncStorage.setItem(TOKEN_KEY, newAccessToken);
          return newAccessToken;
        }
      } catch (e: any) {
        // try next path on 404/405 or other failures
        continue;
      }
    }
    return null;
  })();
  return refreshingPromise.finally(() => {
    refreshingPromise = null;
  });
};

// Skip refresh for auth endpoints to avoid loops/noise
const isAuthEndpoint = (url?: string) =>
  !!url && (url.includes("/auth/login") || url.includes("/auth/refresh"));

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config || {};
    const status = error?.response?.status;

    if (
      status === 401 &&
      !originalRequest._retry &&
      !isAuthEndpoint(originalRequest.url)
    ) {
      originalRequest._retry = true;
      const newToken = await doRefresh();
      if (newToken) {
        originalRequest.headers = originalRequest.headers || {};
        (originalRequest.headers as any).Authorization = `Bearer ${newToken}`;
        return api(originalRequest);
      } else {
        await AsyncStorage.removeItem(TOKEN_KEY);
        await AsyncStorage.removeItem(REFRESH_KEY);
      }
    }
    return Promise.reject(error);
  }
);

export type User = {
  id: number;
  public_id: string;
  first_name: string;
  last_name: string;
  email: string;
  phone?: string;
  // ajoute d’autres si nécessaire
};

// 5. Typescript types pour données structurées
export type Driver = {
  id: number;
  user_id: number;

  username: string;
  first_name: string;
  last_name: string;
  phone: string;
  photo: string;
  company_id: number;
  company_name: string;
  is_active: boolean;
  is_available: boolean;
  vehicle_assigned: string;
  brand: string;
  license_plate: string;
  driver_photo?: string;
  latitude: number | null;
  longitude: number | null;

  // Ces champs manquaient :
  user: {
    id: number;
    username: string;
    email: string;
    role: string;
    public_id: string;
  };
  company: {
    id: number;
    name: string;
  };
};

export const registerPushToken = async (payload: {
  token: string;
  driverId: number;
}) => {
  // Use the correct backend URL and send the required payload
  const res = await api.post("/driver/save-push-token", payload);
  return res.data;
};

export type AuthResponse = {
  message: string;
  token: string;
  refresh_token?: string;
  user: {
    id: number;
    public_id: string;
    username: string;
    email: string;
    role: string;
    force_password_change: boolean;
  };
};

// 6. Payload pour mise à jour du profil chauffeur
export interface DriverProfilePayload {
  vehicle_assigned?: string;
  brand?: string;
  license_plate?: string;
  phone?: string;
}

// 7. Response de mise à jour de photo
export type UpdatePhotoResponse = {
  profile: Driver;
  message: string;
};

export type Booking = {
  id: number;
  pickup_location: string;
  dropoff_location: string;
  scheduled_time: string;
  status: string;
  client_name: string;
  estimated_duration?: string; // <- Safe
  customer_name?: string; // <- Optionnel
  client?: {
    id: number;
    first_name: string;
    last_name: string;
    full_name: string;
  };
  client_phone: string;
  medical_destination?: string;
  wheelchair?: boolean;
  notes?: string;
  is_return: boolean;
  [key: string]: any;
};

export type BookingStatus =
  | "en_route"
  | "in_progress"
  | "completed"
  | "return_completed";

// 9. Authentification & profil
export const loginDriver = async (email: string, password: string) => {
  const response = await api.post("/auth/login", {
    email,
    username: email, // fallback if backend expects "username"
    password,
  });
  const data = response.data as {
    token: string;
    refresh_token?: string;
    user: any;
  };
  if (data?.token) await AsyncStorage.setItem(TOKEN_KEY, data.token);
  if (data?.refresh_token)
    await AsyncStorage.setItem(REFRESH_KEY, data.refresh_token);
  return data;
};

export const fetchUserInfo = async (): Promise<{
  id: number;
  public_id: string;
  username: string;
  email: string;
  role: string;
}> => {
  const res = await api.get("/auth/me");
  return res.data;
};

export const fetchDriverProfile = async (): Promise<Driver> => {
  const res = await api.get<{ profile: Driver }>("/driver/me/profile");
  return res.data.profile;
};

export const updateDriverProfile = async (
  payload: DriverProfilePayload
): Promise<Driver> => {
  const res = await api.put<{ profile: Driver; message: string }>(
    "/driver/me/profile",
    payload
  );
  return res.data.profile;
};

export const updateDriverPhoto = async (
  photo: string
): Promise<UpdatePhotoResponse> => {
  const response = await api.put<UpdatePhotoResponse>("/driver/me/photo", {
    photo,
  });
  return response.data;
};

// 10. Disponibilité & localisation
export const updateDriverAvailability = async (
  is_available: boolean
): Promise<{ message: string }> => {
  const response = await api.put<{ message: string }>(
    "/driver/me/availability",
    { is_available }
  );
  return response.data;
};

export interface DriverLocationPayload {
  latitude: number;
  longitude: number;
  speed?: number;
  heading?: number;
  accuracy?: number;
  timestamp?: number | string;
}

type UpdateLocationResp = { ok?: boolean; source?: string; message?: string };

/** Nouvelle signature: on passe un OBJET */
export const updateDriverLocation = async (
  payload: DriverLocationPayload
): Promise<UpdateLocationResp> => {
  if (__DEV__) {
    // trace utile en dev
    // eslint-disable-next-line no-console
    console.log("🚀 updateDriverLocation called with:", payload);
  }

  // ✅ Validation correcte (0 autorisé) + bornes géographiques
  const { latitude, longitude } = payload;
  if (typeof latitude !== "number" || typeof longitude !== "number") {
    throw new Error("Latitude et longitude doivent être numériques");
  }
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    throw new Error("Coordonnées invalides (NaN/Infinity)");
  }
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
    throw new Error("Coordonnées hors bornes");
  }

  try {
    // Normalise le timestamp en 'ts' (ISO) pour le backend
    const ts =
      typeof payload.timestamp === "number"
        ? new Date(payload.timestamp).toISOString()
        : payload.timestamp || new Date().toISOString();

    const body = {
      latitude: payload.latitude,
      longitude: payload.longitude,
      speed: payload.speed ?? 0,
      heading: payload.heading ?? 0,
      accuracy: payload.accuracy ?? 0,
      ts,
    };

    const response = await api.put<UpdateLocationResp>(
      "/driver/me/location",
      body
    );

    if (__DEV__) {
      // eslint-disable-next-line no-console
      console.log("✅ updateDriverLocation success:", response.data);
    }
    return response.data ?? {};
  } catch (error: unknown) {
    // ⛑️ Narrowing TS-safe
    if (isAxiosError(error)) {
      console.error("❌ updateDriverLocation axios error:", {
        status: error.response?.status,
        data: error.response?.data,
        url: error.config?.url,
        method: error.config?.method,
      });
      const msg =
        typeof error.response?.data === "string"
          ? error.response.data
          : ((error.response?.data as any)?.message ?? error.message);
      throw new Error(msg);
    }

    if (error instanceof Error) {
      console.error("❌ updateDriverLocation error:", error.message);
      throw error; // conserve la stack utile
    }

    console.error("❌ updateDriverLocation unknown error:", error);
    throw new Error("Erreur inconnue lors de la mise à jour de la position");
  }
};

/** (Optionnel) compat avec l’ancienne signature à 2 args */
export const updateDriverLocationLegacy = async (
  latitude: number,
  longitude: number
) => updateDriverLocation({ latitude, longitude });

// 11. Trajets & historique
export const getAssignedTrips = async (): Promise<Booking[]> => {
  const response = await api.get<Booking[]>("/driver/me/bookings");
  return response.data;
};

// 12. alias pour planning, même endpoint que assigned bookings
export const getDriverSchedule = getAssignedTrips;

export const getCompletedTrips = async (
  driverId: number
): Promise<Booking[]> => {
  const response = await api.get<Booking[]>(
    `/driver/${driverId}/completed-trips`
  );
  return response.data;
};

export const getTripDetails = async (bookingId: number): Promise<Booking> => {
  const response = await api.get<Booking>(`/bookings/${bookingId}`);
  return response.data;
};

// 1) TRACE + exécution PUT
export const updateTripStatus = async (
  bookingId: number,
  status: BookingStatus
): Promise<void> => {
  console.log(`[updateTripStatus] → bookingId=${bookingId}, status=${status}`);
  await api.put(`/driver/me/bookings/${bookingId}/status`, { status });
};

// 2) TRACE + décision du statut
export const completeTrip = async (
  bookingId: number,
  isReturn: boolean = false
): Promise<void> => {
  console.log(`[completeTrip] → bookingId=${bookingId}, isReturn=${isReturn}`);
  const status: BookingStatus = isReturn ? "return_completed" : "completed";
  console.log(`[completeTrip] → envoi status=${status}`);
  await updateTripStatus(bookingId, status);
};

// 14 Routes IA
export type OptimizedRoute = {
  route: any;
};

export const getOptimizedRoute = async (
  pickup: string,
  dropoff: string
): Promise<OptimizedRoute> => {
  const response = await api.post<OptimizedRoute>("/ai/optimized-route", {
    pickup,
    dropoff,
  });
  return response.data;
};

export const toggleDriverAvailability = updateDriverAvailability;

// 15. Chargement historique des messages (chat)
export const getCompanyMessages = async (
  companyId: number
): Promise<Message[]> => {
  const response = await api.get<Message[]>(`/messages/${companyId}`);
  return response.data;
};
export type Message = {
  id: number;
  sender: string;
  receiver?: string | null;
  content: string;
  timestamp: string;
  sender_role: string;
};

export default api;
