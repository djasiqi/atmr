// services/api.ts
import Constants from "expo-constants";
import axios, { isAxiosError } from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";

// --- Config base URL (inclut /api pour matcher le backend) ---
const expoExtra = Constants.expoConfig?.extra || {};
const PROD_API_URL = expoExtra.productionApiUrl;

const getDevHost = (): string => {
  const legacyHost = (Constants as any)?.manifest?.debuggerHost?.split(":")[0]; // Expo < 49
  const newHost = (Constants as any)?.expoConfig?.hostUri?.split(":")[0]; // Expo 49+
  const detectedHost = newHost || legacyHost;
  if (
    !detectedHost ||
    detectedHost === "localhost" ||
    detectedHost === "127.0.0.1"
  ) {
    return "172.20.10.2"; // ← IP locale mise à jour
  }
  return detectedHost;
};

const PORT = expoExtra.backendPort || "5000";
export const baseURL = __DEV__
  ? `http://${getDevHost()}:${PORT}/api`
  : `${(PROD_API_URL || "").replace(/\/$/, "")}/api`;

// --- clés stockage ---
const TOKEN_KEY = "token";

// --- instance axios ---
export const api = axios.create({
  baseURL,
  timeout: 10000,
  headers: { "Content-Type": "application/json" },
});

// --- Authorization bearer ---
api.interceptors.request.use(
  async (config) => {
    const token = await AsyncStorage.getItem(TOKEN_KEY);
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

// --- (Optionnel) log d’erreurs sans refresh automatique ---
api.interceptors.response.use(
  (res) => res,
  async (error) => {
    // simple log ; pas de refresh token car le backend n'en fournit pas
    if (isAxiosError(error)) {
      console.warn("API Error", {
        url: error.config?.url,
        method: error.config?.method,
        status: error.response?.status,
        data: error.response?.data,
      });
    }
    return Promise.reject(error);
  }
);

// ========= Types =========
export type User = {
  id: number;
  public_id: string;
  first_name: string;
  last_name: string;
  email: string;
  phone?: string;
};

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
  user: {
    id: number;
    username: string;
    email: string;
    role: string;
    public_id: string;
  };
  company: { id: number; name: string };
};

export const registerPushToken = async (payload: {
  token: string;
  driverId: number;
}) => {
  const res = await api.post("/driver/save-push-token", payload);
  return res.data;
};

export type AuthResponse = {
  message: string;
  token: string; // <-- ton backend renvoie "token"
  user: {
    id: number;
    public_id: string;
    username: string;
    email: string;
    role: string;
    force_password_change: boolean;
  };
};

// ========== Auth ==========
export const loginDriver = async (
  email: string,
  password: string
): Promise<AuthResponse> => {
  // Ton backend attend exactement { email, password }
  const response = await api.post<AuthResponse>("/auth/login", {
    email,
    password,
  });
  const data = response.data;
  if (data?.token) await AsyncStorage.setItem(TOKEN_KEY, data.token);
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

// ========== Driver ==========
export const fetchDriverProfile = async (): Promise<Driver> => {
  const res = await api.get<{ profile: Driver }>("/driver/me/profile");
  return res.data.profile;
};

export interface DriverProfilePayload {
  vehicle_assigned?: string;
  brand?: string;
  license_plate?: string;
  phone?: string;
}

export const updateDriverProfile = async (
  payload: DriverProfilePayload
): Promise<Driver> => {
  const res = await api.put<{ profile: Driver; message: string }>(
    "/driver/me/profile",
    payload
  );
  return res.data.profile;
};

export type UpdatePhotoResponse = { profile: Driver; message: string };

export const updateDriverPhoto = async (
  photo: string
): Promise<UpdatePhotoResponse> => {
  const response = await api.put<UpdatePhotoResponse>("/driver/me/photo", {
    photo,
  });
  return response.data;
};

export const updateDriverAvailability = async (
  is_available: boolean
): Promise<{ message: string }> => {
  const response = await api.put<{ message: string }>(
    "/driver/me/availability",
    { is_available }
  );
  return response.data;
};

// ========== Localisation ==========
export interface DriverLocationPayload {
  latitude: number;
  longitude: number;
  speed?: number;
  heading?: number;
  accuracy?: number;
  timestamp?: number | string;
}
type UpdateLocationResp = { ok?: boolean; source?: string; message?: string };

export const updateDriverLocation = async (
  payload: DriverLocationPayload
): Promise<UpdateLocationResp> => {
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

  try {
    const response = await api.put<UpdateLocationResp>(
      "/driver/me/location",
      body
    );
    return response.data ?? {};
  } catch (error: unknown) {
    if (isAxiosError(error)) {
      const msg =
        typeof error.response?.data === "string"
          ? error.response.data
          : ((error.response?.data as any)?.message ?? error.message);
      throw new Error(msg);
    }
    if (error instanceof Error) throw error;
    throw new Error("Erreur inconnue lors de la mise à jour de la position");
  }
};

export const updateDriverLocationLegacy = async (
  latitude: number,
  longitude: number
) => updateDriverLocation({ latitude, longitude });

// ========== Bookings ==========
export type Booking = {
  id: number;
  pickup_location: string;
  dropoff_location: string;
  scheduled_time: string;
  status: string;
  client_name: string;
  estimated_duration?: string;
  duration_seconds?: number; // Durée estimée du trajet en secondes
  distance_meters?: number; // Distance en mètres
  customer_name?: string;
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
  // Nouveaux champs pour les informations médicales
  medical_facility?: string;
  doctor_name?: string;
  hospital_service?: string;
  notes_medical?: string;
  // Nouveaux champs pour la chaise roulante
  wheelchair_client_has?: boolean;
  wheelchair_need?: boolean;
  [key: string]: any;
};

export const getAssignedTrips = async (): Promise<Booking[]> => {
  const response = await api.get<Booking[]>("/driver/me/bookings");
  return response.data;
};

// ✅ FIX: Ajouter la fonction manquante getCompletedTrips
export const getCompletedTrips = async (
  driverId: number
): Promise<Booking[]> => {
  const response = await api.get<Booking[]>("/driver/me/bookings/all");
  // Filtrer uniquement les courses complétées
  return response.data.filter(
    (booking) =>
      booking.status === "completed" || booking.status === "return_completed"
  );
};

// Détail d’une course : route conforme à backend driver.py
export const getTripDetails = async (bookingId: number): Promise<Booking> => {
  const response = await api.get<Booking>(`/driver/me/bookings/${bookingId}`);
  return response.data;
};

export type BookingStatus =
  | "en_route"
  | "in_progress"
  | "completed"
  | "return_completed";

export const updateTripStatus = async (
  bookingId: number,
  status: BookingStatus
): Promise<void> => {
  await api.put(`/driver/me/bookings/${bookingId}/status`, { status });
};

export const completeTrip = async (
  bookingId: number,
  isReturn = false
): Promise<void> => {
  const status: BookingStatus = isReturn ? "return_completed" : "completed";
  await updateTripStatus(bookingId, status);
};

export type OptimizedRoute = { route: any };
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

// Messages
export type Message = {
  id: number | string;
  company_id?: number;
  sender_id?: number | null;
  receiver_id?: number | null;
  content: string;
  timestamp: string;
  sender_role: "DRIVER" | "COMPANY" | string;
  sender_name?: string | null;
  receiver_name?: string | null;
  _localId?: string | null;
};
export const getCompanyMessages = async (
  companyId: number
): Promise<Message[]> => {
  const response = await api.get<Message[]>(`/messages/${companyId}`);
  return response.data;
};

export default api;
