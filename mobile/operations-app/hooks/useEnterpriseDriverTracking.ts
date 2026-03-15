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
  status?: string;
  updatedAt?: string;
};

type DriverLocationEvent = {
  driver_id?: number | string;
  first_name?: string | null;
  last_name?: string | null;
  latitude?: number | string | null;
  lat?: number | string | null;
  longitude?: number | string | null;
  lon?: number | string | null;
  timestamp?: string | null;
  ts?: string | null;
};

/** Payload canonique : driver_live_state_update (source principale) */
type DriverLiveStateEvent = DriverLocationEvent & {
  status?: string | null;
  mission_status?: string | null;
  client_short?: string | null;
  presence_status?: string | null;
  location_status?: string | null;
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
  const lastHttpFetchAtRef = useRef(0);
  const lastLiveStateRef = useRef<Record<string, { status?: string }>>({});

  const fetchLocationsViaHTTP = useCallback(async () => {
    const token = enterpriseSession?.token;
    if (!token) return;

    try {
      const url = "/companies/me/drivers/locations";
      log.info("fetching live locations", { url });
      lastHttpFetchAtRef.current = Date.now();

      const response = await enterpriseStandardApi.get<{
        locations: Array<{
          driver_id: number;
          first_name?: string | null;
          last_name?: string | null;
          latitude?: number | null;
          lat?: number | null;
          longitude?: number | null;
          lon?: number | null;
          timestamp?: string | null;
          ts?: string | null;
          status?: string | null;
        }>;
      }>(url);

      log.info("live locations response received", { status: response.status });
      const items = response.data?.locations || [];
      log.info("live locations items count", { count: items.length });
      const newMarkers: DriverMarker[] = items
        .map((item) => {
          // Compat lat/latitude, lon/longitude, timestamp/ts
          const latitude = toNumber(item.lat ?? item.latitude);
          const longitude = toNumber(item.lon ?? item.longitude);
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
            status: item.status ?? undefined,
            updatedAt: item.timestamp ?? item.ts ?? undefined,
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
  }, [enterpriseSession?.token]);

  const fetchLocationsViaHTTPWithThrottle = useCallback(
    (minDelayMs = 8000) => {
      const now = Date.now();
      if (now - lastHttpFetchAtRef.current < minDelayMs) {
        return;
      }
      fetchLocationsViaHTTP();
    },
    [fetchLocationsViaHTTP]
  );

  const refreshLocations = useCallback(() => {
    const socket = socketRef.current;
    if (socket) {
      socket.emit("get_driver_locations");
    }
    // HTTP seulement en fallback et avec throttling
    fetchLocationsViaHTTPWithThrottle();
  }, [fetchLocationsViaHTTPWithThrottle]);

  useEffect(() => {
    const token = enterpriseSession?.token;
    if (!token) {
      setMarkers([]);
      return;
    }

    let isActive = true;
    let socketInstance: Socket | null = null;

    const updateDriverFromLiveState = (payload: DriverLiveStateEvent) => {
      if (!isActive || !payload) return;
      const driverIdRaw = payload.driver_id;
      if (driverIdRaw === undefined || driverIdRaw === null) return;
      const driverId = String(driverIdRaw);

      const latitude = toNumber(payload.latitude ?? payload.lat);
      const longitude = toNumber(payload.longitude ?? payload.lon);
      if (latitude === null || longitude === null) return;

      const nameParts = [payload.first_name, payload.last_name]
        .filter(Boolean)
        .map((part) => String(part));
      const markerName =
        nameParts.length > 0 ? nameParts.join(" ") : `Chauffeur ${driverId}`;

      const status = payload.status ?? undefined;
      lastLiveStateRef.current[driverId] = { status };

      setMarkers((prev) => {
        const others = prev.filter((marker) => marker.id !== driverId);
        return [
          ...others,
          {
            id: driverId,
            name: markerName,
            latitude,
            longitude,
            status,
            updatedAt: payload.timestamp ?? payload.ts ?? undefined,
          },
        ];
      });
    };

    const updateDriverPosition = (payload: DriverLocationEvent) => {
      if (!isActive || !payload) return;
      const driverIdRaw = payload.driver_id;
      if (driverIdRaw === undefined || driverIdRaw === null) return;
      const driverId = String(driverIdRaw);

      const latitude = toNumber(payload.latitude ?? payload.lat);
      const longitude = toNumber(payload.longitude ?? payload.lon);
      if (latitude === null || longitude === null) return;

      const nameParts = [payload.first_name, payload.last_name]
        .filter(Boolean)
        .map((part) => String(part));
      const markerName =
        nameParts.length > 0 ? nameParts.join(" ") : `Chauffeur ${driverId}`;

      setMarkers((prev) => {
        const existing = prev.find((m) => m.id === driverId);
        const status = existing?.status ?? lastLiveStateRef.current[driverId]?.status;
        const others = prev.filter((marker) => marker.id !== driverId);
        return [
          ...others,
          {
            id: driverId,
            name: markerName,
            latitude,
            longitude,
            status,
            updatedAt: payload.timestamp ?? payload.ts ?? undefined,
          },
        ];
      });
    };

    const handleLiveStateUpdate = (payload: DriverLiveStateEvent) => {
      if (payload.status !== undefined && payload.status !== null) {
        updateDriverFromLiveState(payload);
      } else {
        updateDriverPosition(payload);
      }
    };

    const handleLocationUpdate = (payload: DriverLocationEvent) => {
      updateDriverPosition(payload);
    };

    (async () => {
      try {
        const s = await connectSocket(token, "enterprise");
        if (!isActive) {
          if (s) {
            s.off("driver_live_state_update", handleLiveStateUpdate);
            s.off("driver_location_update", handleLocationUpdate);
          }
          return;
        }
        if (!s) {
          log.warn("sockets unavailable, using http fallback", {});
          fetchLocationsViaHTTPWithThrottle(0);
          httpPollIntervalRef.current = setInterval(() => {
            if (isActive) fetchLocationsViaHTTPWithThrottle(0);
          }, 10000);
          return;
        }

        socketInstance = s;
        socketRef.current = s;

        s.off("driver_live_state_update", handleLiveStateUpdate);
        s.off("driver_location_update", handleLocationUpdate);
        s.on("driver_live_state_update", handleLiveStateUpdate);
        s.on("driver_location_update", handleLocationUpdate);
        s.emit("get_driver_locations");

        httpPollIntervalRef.current = setInterval(() => {
          if (isActive) fetchLocationsViaHTTPWithThrottle(25000);
        }, 30000);
      } catch (error) {
        log.warn("enterprise socket connection failed", { error });
        if (!isActive) return;
        fetchLocationsViaHTTPWithThrottle(0);
        httpPollIntervalRef.current = setInterval(() => {
          if (isActive) fetchLocationsViaHTTPWithThrottle(0);
        }, 10000);
      }
    })();

    return () => {
      isActive = false;
      if (socketInstance) {
        socketInstance.off("driver_live_state_update", handleLiveStateUpdate);
        socketInstance.off("driver_location_update", handleLocationUpdate);
      }
      if (socketRef.current) {
        socketRef.current.off("driver_live_state_update", handleLiveStateUpdate);
        socketRef.current.off("driver_location_update", handleLocationUpdate);
        socketRef.current = null;
      }
      if (httpPollIntervalRef.current) {
        clearInterval(httpPollIntervalRef.current);
        httpPollIntervalRef.current = null;
      }
    };
  }, [enterpriseSession?.token, fetchLocationsViaHTTPWithThrottle]);

  return {
    markers,
    refreshLocations,
  };
};

export type { DriverMarker };


