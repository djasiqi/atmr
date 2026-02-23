import { useCallback, useEffect, useRef, useState } from "react";
import type { Socket } from "socket.io-client";

import { getLogger } from "@/utils/logger";
import { connectSocket } from "@/services/socket";
import { useAuth } from "@/hooks/useAuth";
import { enterpriseStandardApi } from "@/services/enterpriseStandardApi";
import { isAuthNotReadyError } from "@/services/authGuards";

const log = getLogger("EntTracking");

type DriverMarker = {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  updatedAt?: string;
};

type DriverLocationEvent = {
  driver_id?: number | string;
  first_name?: string | null;
  last_name?: string | null;
  // ✅ Accepter les deux formats pour compatibilité
  latitude?: number | string | null;
  lat?: number | string | null;
  longitude?: number | string | null;
  lon?: number | string | null;
  timestamp?: string | null;
  ts?: string | null;
};

const toNumber = (value: unknown): number | null => {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === "string") {
    const parsed = parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

export const useEnterpriseDriverTracking = () => {
  const { enterpriseSession } = useAuth();
  const [markers, setMarkers] = useState<DriverMarker[]>([]);
  const socketRef = useRef<Socket | null>(null);
  const httpPollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchLocationsViaHTTP = useCallback(async () => {
    const companyId = enterpriseSession?.company?.id;
    const token = enterpriseSession?.token;
    if (!companyId || !token) return;

    try {
      const url = `/driver/company/${companyId}/live-locations`;
      log.info("fetching live locations", { url });

      const response = await enterpriseStandardApi.get<{
        items: Array<{
          driver_id: number;
          first_name?: string | null;
          last_name?: string | null;
          // ✅ Accepter les deux formats pour compatibilité
          latitude?: number | null;
          lat?: number | null;
          longitude?: number | null;
          lon?: number | null;
          timestamp?: string | null;
          ts?: string | null;
        }>;
      }>(url);

      log.info("live locations response received", { status: response.status });
      const items = response.data?.items || [];
      log.info("live locations items count", { count: items.length });
      const newMarkers: DriverMarker[] = items
        .map((item) => {
          // ✅ FIX: Accepter les deux formats (lat/latitude, lon/longitude)
          // Backend retourne "lat"/"lon", pas "latitude"/"longitude"
          const latitude = toNumber(item.latitude ?? item.lat);
          const longitude = toNumber(item.longitude ?? item.lon);
          if (latitude === null || longitude === null) return null;

          const nameParts = [item.first_name, item.last_name]
            .filter(Boolean)
            .map((part) => String(part));
          const markerName =
            nameParts.length > 0
              ? nameParts.join(" ")
              : `Chauffeur ${item.driver_id}`;

          return {
            id: String(item.driver_id),
            name: markerName,
            latitude,
            longitude,
            updatedAt: item.timestamp ?? undefined,
          } as DriverMarker;
        })
        .filter((marker): marker is DriverMarker => marker !== null);

      log.info("new markers from http", { count: newMarkers.length });
      
      setMarkers((prev) => {
        // Fusionner avec les markers existants (priorité aux sockets si disponibles)
        const merged = new Map<string, DriverMarker>();
        // D'abord les markers HTTP
        newMarkers.forEach((marker) => {
          merged.set(marker.id, marker);
        });
        // Ensuite les markers socket (écrasent HTTP si plus récents)
        prev.forEach((marker) => {
          const existing = merged.get(marker.id);
          if (!existing || !existing.updatedAt || (marker.updatedAt && marker.updatedAt > existing.updatedAt)) {
            merged.set(marker.id, marker);
          }
        });
        const result = Array.from(merged.values());
        log.info("final markers count", { count: result.length });
        return result;
      });
    } catch (error) {
      if (isAuthNotReadyError(error)) {
        log.warn("http live locations skipped (auth not ready)");
      } else {
        log.error("http live locations fetch failed", { error });
      }
    }
  }, [enterpriseSession?.company?.id, enterpriseSession?.token]);

  const refreshLocations = useCallback(() => {
    const socket = socketRef.current;
    if (socket) {
      socket.emit("get_driver_locations");
    }
    // ✅ Toujours essayer HTTP aussi comme fallback
    fetchLocationsViaHTTP();
  }, [fetchLocationsViaHTTP]);

  useEffect(() => {
    const token = enterpriseSession?.token;
    if (!token) {
      setMarkers([]);
      return;
    }

    let isActive = true;
    let socketInstance: Socket | null = null;

    const handleDriverLocation = (payload: DriverLocationEvent) => {
      if (!isActive || !payload) return;
      const driverIdRaw = payload.driver_id;
      if (driverIdRaw === undefined || driverIdRaw === null) return;
      const driverId = String(driverIdRaw);

      // ✅ FIX: Accepter les deux formats (lat/latitude, lon/longitude)
      // Backend émet "lat"/"lon" via Socket.IO, pas "latitude"/"longitude"
      const latitude = toNumber(payload.latitude ?? payload.lat);
      const longitude = toNumber(payload.longitude ?? payload.lon);
      if (latitude === null || longitude === null) return;

      const nameParts = [payload.first_name, payload.last_name]
        .filter(Boolean)
        .map((part) => String(part));
      const markerName =
        nameParts.length > 0 ? nameParts.join(" ") : `Chauffeur ${driverId}`;

      setMarkers((prev) => {
        const others = prev.filter((marker) => marker.id !== driverId);
        return [
          ...others,
          {
            id: driverId,
            name: markerName,
            latitude,
            longitude,
            updatedAt: payload.timestamp ?? undefined,
          },
        ];
      });
    };

    (async () => {
      try {
        const s = await connectSocket(token, "enterprise");
        if (!isActive) {
          // Composant démonté pendant le connect — nettoyer immédiatement
          if (s) s.off("driver_location_update", handleDriverLocation);
          return;
        }
        if (!s) {
          log.warn("sockets unavailable, using http fallback", {});
          fetchLocationsViaHTTP();
          httpPollIntervalRef.current = setInterval(() => {
            if (isActive) fetchLocationsViaHTTP();
          }, 10000);
          return;
        }

        socketInstance = s;
        socketRef.current = s;

        s.off("driver_location_update", handleDriverLocation);
        s.on("driver_location_update", handleDriverLocation);
        s.emit("join_company");
        s.emit("get_driver_locations");

        httpPollIntervalRef.current = setInterval(() => {
          if (isActive) fetchLocationsViaHTTP();
        }, 30000);
      } catch (error) {
        log.warn("enterprise socket connection failed", { error });
        if (!isActive) return;
        fetchLocationsViaHTTP();
        httpPollIntervalRef.current = setInterval(() => {
          if (isActive) fetchLocationsViaHTTP();
        }, 10000);
      }
    })();

    return () => {
      isActive = false;
      if (socketInstance) {
        socketInstance.off("driver_location_update", handleDriverLocation);
      }
      if (socketRef.current) {
        socketRef.current.off("driver_location_update", handleDriverLocation);
        socketRef.current = null;
      }
      if (httpPollIntervalRef.current) {
        clearInterval(httpPollIntervalRef.current);
        httpPollIntervalRef.current = null;
      }
    };
  }, [enterpriseSession?.token, fetchLocationsViaHTTP]);

  return {
    markers,
    refreshLocations,
  };
};

export type { DriverMarker };


