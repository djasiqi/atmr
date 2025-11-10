import { useCallback, useEffect, useRef, useState } from "react";
import type { Socket } from "socket.io-client";

import { connectSocket } from "@/services/socket";
import { useAuth } from "@/hooks/useAuth";

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
  latitude?: number | string | null;
  longitude?: number | string | null;
  timestamp?: string | null;
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

  const refreshLocations = useCallback(() => {
    const socket = socketRef.current;
    if (!socket) return;
    socket.emit("get_driver_locations");
  }, []);

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

      const latitude = toNumber(payload.latitude);
      const longitude = toNumber(payload.longitude);
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
        socketInstance = await connectSocket(token, "enterprise");
        if (!socketInstance || !isActive) return;

        socketRef.current = socketInstance;

        socketInstance.off("driver_location_update", handleDriverLocation);
        socketInstance.on("driver_location_update", handleDriverLocation);
        socketInstance.emit("join_company");
        socketInstance.emit("get_driver_locations");
      } catch (error) {
        console.warn("❗ Erreur connexion socket entreprise :", error);
      }
    })();

    return () => {
      isActive = false;
      if (socketInstance) {
        socketInstance.off("driver_location_update", handleDriverLocation);
      }
      if (socketRef.current === socketInstance) {
        socketRef.current = null;
      }
    };
  }, [enterpriseSession?.token]);

  return {
    markers,
    refreshLocations,
  };
};

export type { DriverMarker };


