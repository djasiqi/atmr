import { useCallback, useEffect, useRef, useState } from "react";
import type { Socket } from "socket.io-client";
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";

import { connectSocket } from "@/services/socket";
import { useAuth } from "@/hooks/useAuth";
import { enterpriseApi, ENTERPRISE_TOKEN_KEY } from "@/services/enterpriseAuth";

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

  // ✅ Fallback HTTP : récupérer les positions via API REST
  const fetchLocationsViaHTTP = useCallback(async () => {
    const companyId = enterpriseSession?.company?.id;
    if (!companyId) return;

    try {
      // ✅ Utiliser l'endpoint standard API (pas company_mobile)
      // L'endpoint est /api/v1/driver/company/<company_id>/live-locations
      // Construire l'URL complète en remplaçant le préfixe company_mobile
      const baseURL = enterpriseApi.defaults.baseURL || "";
      const standardApiURL = baseURL.replace("/api/v1/company_mobile", "/api/v1");
      
      // Récupérer le token pour l'authentification
      const token = await AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY);
      
      const url = `${standardApiURL}/driver/company/${companyId}/live-locations`;
      console.log('[useEnterpriseDriverTracking] 🔍 Fetching from:', url);
      
      const response = await axios.get<{
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
      }>(url, {
        headers: {
          Authorization: token ? `Bearer ${token}` : undefined,
        },
      });

      console.log('[useEnterpriseDriverTracking] ✅ Response received:', response.status);
      const items = response.data?.items || [];
      console.log('[useEnterpriseDriverTracking] 📍 Items count:', items.length);
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

      console.log('[useEnterpriseDriverTracking] 📌 New markers:', newMarkers.length);
      
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
        console.log('[useEnterpriseDriverTracking] 🗺️ Final markers count:', result.length);
        return result;
      });
    } catch (error) {
      console.error('[useEnterpriseDriverTracking] ❌ Erreur récupération positions HTTP:', error);
      if (axios.isAxiosError(error)) {
        console.error('[useEnterpriseDriverTracking] 📡 Status:', error.response?.status);
        console.error('[useEnterpriseDriverTracking] 📡 Data:', error.response?.data);
        console.error('[useEnterpriseDriverTracking] 📡 URL:', error.config?.url);
      }
    }
  }, [enterpriseSession?.company?.id]);

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

    // ✅ Essayer d'abord les sockets (temps réel)
    (async () => {
      try {
        socketInstance = await connectSocket(token, "enterprise");
        if (!socketInstance || !isActive) {
          // Si les sockets échouent, utiliser uniquement HTTP
          console.warn("⚠️ Sockets non disponibles, utilisation du fallback HTTP");
          fetchLocationsViaHTTP();
          // Poll HTTP toutes les 10 secondes si pas de sockets
          httpPollIntervalRef.current = setInterval(() => {
            if (isActive) {
              fetchLocationsViaHTTP();
            }
          }, 10000);
          return;
        }

        socketRef.current = socketInstance;

        socketInstance.off("driver_location_update", handleDriverLocation);
        socketInstance.on("driver_location_update", handleDriverLocation);
        socketInstance.emit("join_company");
        socketInstance.emit("get_driver_locations");

        // ✅ Fallback HTTP : poll toutes les 30 secondes même avec sockets (au cas où)
        httpPollIntervalRef.current = setInterval(() => {
          if (isActive) {
            fetchLocationsViaHTTP();
          }
        }, 30000);
      } catch (error) {
        console.warn("❗ Erreur connexion socket entreprise :", error);
        // ✅ Si les sockets échouent, utiliser uniquement HTTP
        fetchLocationsViaHTTP();
        // Poll HTTP toutes les 10 secondes
        httpPollIntervalRef.current = setInterval(() => {
          if (isActive) {
            fetchLocationsViaHTTP();
          }
        }, 10000);
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
      // ✅ Nettoyer l'intervalle HTTP
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


