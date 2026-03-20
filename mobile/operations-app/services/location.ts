// services/location.ts

import * as Location from "expo-location";
import { type DriverLocationPayload } from "@/services/api";
import { getLogger } from "@/utils/logger";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { enqueueLocation } from "./locationQueue";
import { resolveMissionContext } from "./locationMissionContext";

const log = getLogger("LocationSvc");


/** Distance en mètres via Haversine */
export const getDistanceInMeters = (
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number => {
  const R = 6371000; // m
  const toRad = (v: number) => (v * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};

/** Envoi simple (legacy): enqueue uniquement, plus d'appel HTTP direct. */
export const sendDriverLocation = async (payload: DriverLocationPayload) => {  try {
    log.info("send start", { payload });
    
    // Validation robuste (0 autorisé, NaN rejeté, bornes checkées)
    const { latitude, longitude } = payload as any;
    if (typeof latitude !== "number" || typeof longitude !== "number") {
      throw new Error("Coordonnées manquantes ou non numériques");
    }
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      throw new Error("Coordonnées invalides (NaN/Inf)");
    }
    if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
      throw new Error("Coordonnées hors bornes");
    }

    // Assurer que les coordonnées sont des nombres
    // ✅ P1: Normaliser timestamp en string ISO
    const ts = typeof payload.timestamp === 'number'
      ? new Date(payload.timestamp).toISOString()
      : payload.timestamp || new Date().toISOString();
    
    const cleanPayload = {
      ...payload,
      latitude: Number(payload.latitude),
      longitude: Number(payload.longitude),
      speed: Number(payload.speed || 0),
      heading: Number(payload.heading || 0),
      accuracy: Number(payload.accuracy || 10),
      ts,  // ✅ Toujours string ISO pour le backend
    };

    log.info("clean payload", cleanPayload);

    const driverIdStr = await AsyncStorage.getItem("driver_id");
    const driver_id = driverIdStr ? parseInt(driverIdStr, 10) : 0;
    if (!driver_id || !Number.isFinite(driver_id)) {
      return { ok: false, message: "driver_id manquant" };
    }
    const { missionId, mode } = resolveMissionContext();
    await enqueueLocation({
      latitude: Number(cleanPayload.latitude),
      longitude: Number(cleanPayload.longitude),
      speed: Number(cleanPayload.speed || 0),
      heading: Number(cleanPayload.heading || 0),
      accuracy: Number(cleanPayload.accuracy || 10),
      timestamp: Date.now(),
      driver_id,
      location_mode: mode,
      mission_id: missionId,
      recorded_at: cleanPayload.ts || new Date().toISOString(),
      sent_at: new Date().toISOString(),
      is_background: false,
    });
    log.success("position enqueued from legacy helper");
    return { ok: true };
    
  } catch (e: any) {
    // Supprimer les erreurs 401/403/404 car elles sont attendues si l'utilisateur n'est pas un chauffeur
    const status = e?.response?.status;
    if (status === 401 || status === 403 || status === 404) {
      log.debug("access not authorized", { status });
      // Ne pas lancer d'erreur, juste retourner
      return { ok: false, message: "Accès non autorisé" };
    }
    log.error("send location failed", { error: e });
    throw e;
  }
};

/* -------- Tracking en continu (foreground) -------- */

let locationSub: Location.LocationSubscription | null = null;
let lastSentAt = 0;
let lastLat: number | null = null;
let lastLon: number | null = null;

/**
 * Démarre l'envoi périodique de la position.
 * - intervalMs: délai mini entre 2 envois (default 7000ms ~ 7s)
 * - distanceMin: mouvement mini pour renvoyer (default 10m)
 */
export async function startLocationTracking(
  intervalMs = 5000,
  minDistanceM = 50
) {
  if (locationSub) {
    log.warn("tracking already active");
    return;
  }

  const { status } = await Location.requestForegroundPermissionsAsync();
  if (status !== "granted") {
    throw new Error("Permission de localisation refusée");
  }

  locationSub = await Location.watchPositionAsync(
    {
      accuracy: Location.Accuracy.High,
      timeInterval: Math.max(2000, intervalMs),
      distanceInterval: Math.max(10, minDistanceM),
    },
    async (location) => {
      const { latitude, longitude, speed, heading, accuracy } = location.coords;
      const now = Date.now();

      try {
        const movedEnough = !lastLat || !lastLon || 
          getDistanceInMeters(lastLat, lastLon, latitude, longitude) >= minDistanceM;
        
        const timeOk = now - lastSentAt >= Math.max(2000, intervalMs);

        if (movedEnough && timeOk) {
          const payload = {
            latitude: Number(latitude),
            longitude: Number(longitude),
            speed: Number(speed || 0),
            heading: Number(heading || 0),
            accuracy: Number(accuracy || 10),
            ts: new Date().toISOString(),
          };

          log.info("tracking payload", payload);

          await sendDriverLocation(payload);
          
          lastSentAt = now;
          lastLat = latitude;
          lastLon = longitude;
        }
      } catch (e: any) {
        log.warn("send failed ignored", { message: e?.message || e });
      }
    }
  );

  log.success("tracking started");
}

/** Arrête le tracking en continu */
export async function stopLocationTracking() {
  if (locationSub) {
    try {
      locationSub.remove();
    } catch {}
    locationSub = null;
    log.success("tracking stopped");
  }
}
