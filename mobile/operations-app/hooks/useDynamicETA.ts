import { useState, useEffect, useCallback } from "react";
import { api } from "@/services/api";
import { useAuth } from "@/hooks/useAuth";

/**
 * ETA chauffeur : l'app chauffeur utilise UNIQUEMENT GET /driver/me/bookings/eta
 * pour les ETA et le retard estimé. GET company_dispatch/delays/live est réservé
 * au dashboard company (vue globale). Ne pas appeler company_dispatch depuis l'app chauffeur.
 */

export interface BookingETA {
  id: number;
  eta_to_pickup_seconds: number | null;
  eta_to_dropoff_seconds: number | null;
  duration_seconds: number | null;
  distance_meters: number | null;
  estimated_arrival: string | null;
  estimated_arrival_dropoff: string | null;
}

export interface ETAResponse {
  has_gps: boolean;
  driver_position?: { lat: number; lon: number };
  bookings: BookingETA[];
}

/**
 * Hook qui récupère les ETAs dynamiques basés sur la position GPS du chauffeur.
 * Mise à jour automatique toutes les 15 secondes.
 * Source : GET /driver/me/bookings/eta uniquement (pas company_dispatch/delays/live).
 */
export function useDynamicETA(enabled: boolean = true) {
  const { driver, mode } = useAuth();
  const [etas, setEtas] = useState<Map<number, BookingETA>>(new Map());
  const [hasGPS, setHasGPS] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Vérifier que l'utilisateur est bien un chauffeur avant d'appeler l'API
  const isDriverMode = mode === "driver" && !!driver;

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

  // Recharger toutes les 15 secondes
  useEffect(() => {
    if (!enabled || !isDriverMode) return;

    const interval = setInterval(() => {
      fetchETAs();
    }, 15000); // 15 secondes

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, isDriverMode]); // Ne pas inclure fetchETAs pour éviter les re-renders infinis

  /** Heure d'arrivée estimée au point de prise en charge (pickup). */
  const getEstimatedArrival = useCallback(
    (bookingId: number): Date | null => {
      const raw = etas.get(bookingId)?.estimated_arrival;
      if (!raw) return null;
      const d = new Date(raw);
      return isNaN(d.getTime()) ? null : d;
    },
    [etas]
  );

  /**
   * Retard estimé en minutes : ETA pickup vs scheduled_time.
   * Retourne null si pas d'ETA ou pas de scheduled_time.
   * Valeur >= 0 = retard en minutes ; on peut afficher "en avance" si < 0.
   */
  const getDelayMinutes = useCallback(
    (bookingId: number, scheduledTime: string): number | null => {
      const etaDate = getEstimatedArrival(bookingId);
      if (!etaDate) return null;
      const scheduled = new Date(scheduledTime);
      if (isNaN(scheduled.getTime())) return null;
      const diffMs = etaDate.getTime() - scheduled.getTime();
      return Math.round(diffMs / 60000);
    },
    [getEstimatedArrival]
  );

  /** Heure d'arrivée estimée à destination (après pickup, client à bord). */
  const getEstimatedArrivalDropoff = useCallback(
    (bookingId: number): Date | null => {
      const raw = etas.get(bookingId)?.estimated_arrival_dropoff;
      if (!raw) return null;
      const d = new Date(raw);
      return isNaN(d.getTime()) ? null : d;
    },
    [etas]
  );

  return {
    etas,
    hasGPS,
    isLoading,
    refresh: fetchETAs,
    getDuration: (bookingId: number) =>
      etas.get(bookingId)?.duration_seconds || null,
    getETAToPickup: (bookingId: number) =>
      etas.get(bookingId)?.eta_to_pickup_seconds ?? null,
    getETAToDropoff: (bookingId: number) =>
      etas.get(bookingId)?.eta_to_dropoff_seconds ?? null,
    getEstimatedArrival,
    getEstimatedArrivalDropoff,
    getDelayMinutes,
  };
}
