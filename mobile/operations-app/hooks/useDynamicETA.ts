import { useState, useEffect, useCallback, useRef } from "react";
import type { Socket } from "socket.io-client";
import { api } from "@/services/api";
import { useAuth } from "@/hooks/useAuth";
import { isAuthReadySync } from "@/services/authSync";
import { isAuthNotReadyError } from "@/services/authGuards";
import { getLogger } from "@/utils/logger";
import { getSocket, subscribeSocketStatus } from "@/services/socket";

const log = getLogger("ETA");

/** P1: secours HTTP long (socket `eta_changed` = source chaude). */
const HTTP_ETA_FALLBACK_MS = 90_000;

/**
 * ETA chauffeur : GET /driver/me/bookings/eta en secours + événement socket `eta_changed`
 * (même forme que le GET). Ne pas appeler company_dispatch depuis l'app chauffeur.
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

/** GET + socket `eta_changed` ; le backend enrichit souvent avec `timestamp` / `event_id` (SocketEvent). */
export interface ETAResponse {
  has_gps: boolean;
  driver_position?: { lat: number; lon: number };
  bookings: BookingETA[];
  timestamp?: string;
  event_id?: string;
  event_type?: string;
}

export function useDynamicETA(enabled: boolean = true) {
  const { driver, mode } = useAuth();
  const [etas, setEtas] = useState<Map<number, BookingETA>>(new Map());
  const [hasGPS, setHasGPS] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const isDriverMode = mode === "driver" && !!driver;

  const fetchInFlight = useRef(false);
  /** Dernier `timestamp` serveur (ISO) appliqué pour un `eta_changed` — évite d'appliquer un event plus ancien hors ordre. */
  const lastEtaSocketServerTsRef = useRef(0);

  useEffect(() => {
    if (!isDriverMode) {
      lastEtaSocketServerTsRef.current = 0;
    }
  }, [isDriverMode]);

  const applyEtaPayload = useCallback((data: ETAResponse, source: "http" | "socket") => {
    setHasGPS(data.has_gps);
    const etaMap = new Map<number, BookingETA>();
    data.bookings.forEach((booking) => {
      etaMap.set(booking.id, booking);
    });
    setEtas(etaMap);
    log.info("etas updated", {
      source,
      has_gps: data.has_gps,
      count: data.bookings.length,
      driver_pos: data.driver_position,
    });
  }, []);

  const fetchETAs = useCallback(async () => {
    if (!enabled || !isDriverMode) return;
    if (!isAuthReadySync()) return;
    if (fetchInFlight.current) return;

    fetchInFlight.current = true;
    try {
      setIsLoading(true);

      const response = await api.get<ETAResponse>("/driver/me/bookings/eta");
      applyEtaPayload(response.data, "http");
    } catch (error: any) {
      if (isAuthNotReadyError(error)) return;
      const status = error?.response?.status;
      if (status === 401 || status === 403 || status === 404) {
        log.debug("eta skipped", { status });
        return;
      }
      log.warn("fetch etas failed", { error });
    } finally {
      setIsLoading(false);
      fetchInFlight.current = false;
    }
  }, [enabled, isDriverMode, applyEtaPayload]);

  useEffect(() => {
    if (enabled && isDriverMode) {
      fetchETAs();
    }
  }, [enabled, isDriverMode, fetchETAs]);

  /** P1: `eta_changed` — même contrat que GET. */
  useEffect(() => {
    if (!enabled || !isDriverMode) return;

    const onEtaChanged = (data: ETAResponse) => {
      const tsRaw = data.timestamp;
      const tsMs = tsRaw ? Date.parse(tsRaw) : NaN;
      if (Number.isFinite(tsMs) && tsMs < lastEtaSocketServerTsRef.current) {
        log.debug("eta_changed ignored (stale socket order)", {
          tsMs,
          last: lastEtaSocketServerTsRef.current,
          event_id: data.event_id,
        });
        return;
      }
      if (Number.isFinite(tsMs)) {
        lastEtaSocketServerTsRef.current = Math.max(
          lastEtaSocketServerTsRef.current,
          tsMs
        );
      }
      applyEtaPayload(data, "socket");
    };

    const attach = (sock: Socket | null) => {
      if (!sock) return;
      sock.off("eta_changed", onEtaChanged);
      sock.on("eta_changed", onEtaChanged);
    };

    attach(getSocket());
    const unsub = subscribeSocketStatus(() => {
      attach(getSocket());
    });

    return () => {
      unsub();
      const s = getSocket();
      s?.off("eta_changed", onEtaChanged);
    };
  }, [enabled, isDriverMode, applyEtaPayload]);

  /** Secours HTTP long (pas de polling 15 s). */
  useEffect(() => {
    if (!enabled || !isDriverMode) return;

    const interval = setInterval(() => {
      fetchETAs();
    }, HTTP_ETA_FALLBACK_MS);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, isDriverMode]);

  const getEstimatedArrival = useCallback(
    (bookingId: number): Date | null => {
      const raw = etas.get(bookingId)?.estimated_arrival;
      if (!raw) return null;
      const d = new Date(raw);
      return isNaN(d.getTime()) ? null : d;
    },
    [etas]
  );

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
