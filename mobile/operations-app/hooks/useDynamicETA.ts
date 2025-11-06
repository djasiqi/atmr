import { useState, useEffect, useCallback } from "react";
import { api } from "@/services/api";

export interface BookingETA {
  id: number;
  eta_to_pickup_seconds: number | null;
  duration_seconds: number | null;
  distance_meters: number | null;
  estimated_arrival: string | null;
}

export interface ETAResponse {
  has_gps: boolean;
  driver_position?: { lat: number; lon: number };
  bookings: BookingETA[];
}

/**
 * Hook qui récupère les ETAs dynamiques basés sur la position GPS du chauffeur
 * Mise à jour automatique toutes les 30 secondes
 */
export function useDynamicETA(enabled: boolean = true) {
  const [etas, setEtas] = useState<Map<number, BookingETA>>(new Map());
  const [hasGPS, setHasGPS] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const fetchETAs = useCallback(async () => {
    if (!enabled) return;

    try {
      setIsLoading(true);

      // Utiliser l'instance api qui a déjà l'interceptor pour le token
      const response = await api.get<ETAResponse>("/driver/me/bookings/eta");
      const data = response.data;

      setHasGPS(data.has_gps);

      // Convertir en Map pour accès rapide par ID
      const etaMap = new Map<number, BookingETA>();
      data.bookings.forEach((booking) => {
        etaMap.set(booking.id, booking);
      });

      setEtas(etaMap);

      console.log("[useDynamicETA] ETAs mis à jour:", {
        has_gps: data.has_gps,
        count: data.bookings.length,
        driver_pos: data.driver_position,
      });
    } catch (error) {
      console.error(
        "[useDynamicETA] Erreur lors de la récupération des ETAs:",
        error
      );
    } finally {
      setIsLoading(false);
    }
  }, [enabled]);

  // Charger au montage
  useEffect(() => {
    if (enabled) {
      fetchETAs();
    }
  }, [enabled, fetchETAs]);

  // Recharger toutes les 30 secondes
  useEffect(() => {
    if (!enabled) return;

    const interval = setInterval(() => {
      fetchETAs();
    }, 30000); // 30 secondes

    return () => clearInterval(interval);
  }, [enabled, fetchETAs]);

  return {
    etas,
    hasGPS,
    isLoading,
    refresh: fetchETAs,
    getDuration: (bookingId: number) =>
      etas.get(bookingId)?.duration_seconds || null,
    getETAToPickup: (bookingId: number) =>
      etas.get(bookingId)?.eta_to_pickup_seconds || null,
  };
}
