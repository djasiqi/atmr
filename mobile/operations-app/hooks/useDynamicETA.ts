import { useState, useEffect, useCallback } from "react";
import { api } from "@/services/api";
import { useAuth } from "@/hooks/useAuth";

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
  const { driver, authMode } = useAuth();
  const [etas, setEtas] = useState<Map<number, BookingETA>>(new Map());
  const [hasGPS, setHasGPS] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Vérifier que l'utilisateur est bien un chauffeur avant d'appeler l'API
  const isDriverMode = authMode === "driver" && !!driver;

  const fetchETAs = useCallback(async () => {
    if (!enabled || !isDriverMode) return;

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
    } catch (error: any) {
      // Supprimer les erreurs 401/403/404 car elles sont attendues si l'utilisateur n'est pas un chauffeur
      const status = error?.response?.status;
      if (status === 401 || status === 403 || status === 404) {
        console.debug(
          "[useDynamicETA] Accès non autorisé (utilisateur n'est probablement pas un chauffeur):",
          status
        );
        // Désactiver le hook si l'utilisateur n'a pas les permissions
        return;
      }
      console.error(
        "[useDynamicETA] Erreur lors de la récupération des ETAs:",
        error
      );
    } finally {
      setIsLoading(false);
    }
  }, [enabled, isDriverMode]);

  // Charger au montage
  useEffect(() => {
    if (enabled && isDriverMode) {
      fetchETAs();
    }
  }, [enabled, isDriverMode, fetchETAs]);

  // Recharger toutes les 30 secondes
  useEffect(() => {
    if (!enabled || !isDriverMode) return;

    const interval = setInterval(() => {
      fetchETAs();
    }, 30000); // 30 secondes

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, isDriverMode]); // Ne pas inclure fetchETAs pour éviter les re-renders infinis

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
