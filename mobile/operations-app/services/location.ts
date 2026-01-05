// services/location.ts

import * as Location from "expo-location";
import {
  updateDriverLocation,
  type DriverLocationPayload, // ← on réutilise le type exporté par api.ts
} from "@/services/api";


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

/** Envoi simple (utilisé par le watcher) */
export const sendDriverLocation = async (payload: DriverLocationPayload) => {  try {
    console.log("📍 sendDriverLocation START:", payload);
    
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
    const cleanPayload = {
      ...payload,
      latitude: Number(payload.latitude),
      longitude: Number(payload.longitude),
      speed: Number(payload.speed || 0),
      heading: Number(payload.heading || 0),
      accuracy: Number(payload.accuracy || 10),
      ts: payload.timestamp || new Date().toISOString()
    };

    console.log("📍 Clean payload:", cleanPayload);

    const res = await updateDriverLocation(cleanPayload);
    console.log("✅ Localisation envoyée:", res?.ok ?? true, res?.source ? `(${res.source})` : "");
    
    if (res?.message) console.log("ℹ️", res.message);
    return res;
    
  } catch (e: any) {
    // Supprimer les erreurs 401/403/404 car elles sont attendues si l'utilisateur n'est pas un chauffeur
    const status = e?.response?.status;
    if (status === 401 || status === 403 || status === 404) {
      console.debug(
        "[sendDriverLocation] Accès non autorisé (utilisateur n'est probablement pas un chauffeur):",
        status
      );
      // Ne pas lancer d'erreur, juste retourner
      return { ok: false, message: "Accès non autorisé" };
    }
    console.error("❌ Erreur envoi localisation:", e);
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
    console.log("⚠️ Tracking déjà actif");
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

          console.log("📍 Tracking payload:", payload);
          
          await sendDriverLocation(payload);
          
          lastSentAt = now;
          lastLat = latitude;
          lastLon = longitude;
        }
      } catch (e: any) {
        console.warn("⚠️ Envoi localisation échoué (ignoré):", e?.message || e);
      }
    }
  );

  console.log("▶️ Tracking localisation démarré");
}

/** Arrête le tracking en continu */
export async function stopLocationTracking() {
  if (locationSub) {
    try {
      locationSub.remove();
    } catch {}
    locationSub = null;
    console.log("⏹️ Tracking localisation arrêté");
  }
}
