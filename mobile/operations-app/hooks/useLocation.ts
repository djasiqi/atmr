// hooks/useLocation.ts

import { useEffect, useState, useRef } from "react";
import * as Location from "expo-location";
import { Alert, Platform } from "react-native";
import { sendDriverLocation, getDistanceInMeters } from "@/services/location";
import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";

// ✅ PERF: Configuration batching pour économiser batterie
const BATCH_SIZE = 3;  // Buffer de 3-5 positions avant envoi
const BATCH_INTERVAL_MS = 15000;  // Flush toutes les 15s (au lieu de 5s)

export const useLocation = () => {
  const { driver } = useAuth();
  const socket = useSocket();

  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const locationSubscription = useRef<Location.LocationSubscription | number | null>(null);
  const lastSentLocation = useRef<{ latitude: number; longitude: number } | null>(null);
  // ✅ PERF: Buffer pour batching des positions
  const positionBuffer = useRef<Location.LocationObject[]>([]);

  useEffect(() => {
    let isMounted = true;

    const requestLocationPermissions = async () => {
      if (Platform.OS === "web") {
        if (!navigator.geolocation) {
          Alert.alert("Erreur", "La géolocalisation n’est pas disponible sur ce navigateur.");
          return;
        }

        const watchId = navigator.geolocation.watchPosition(
          async (position) => {
            if (!isMounted) return;
            const loc: Location.LocationObject = {
              coords: {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                accuracy: position.coords.accuracy,
                altitude: position.coords.altitude ?? null,
                altitudeAccuracy: position.coords.altitudeAccuracy ?? null,
                heading: position.coords.heading ?? null,
                speed: position.coords.speed ?? null,
              },
              timestamp: position.timestamp,
            };
            setLocation(loc);
            await handleLocationUpdate(loc);
          },
          (error) => {
            console.error("Erreur géolocalisation navigateur:", error);
            Alert.alert("Erreur", "Erreur de géolocalisation navigateur.");
          },
          { enableHighAccuracy: true, timeout: 10000, maximumAge: 10000 }
        );

        locationSubscription.current = watchId;
      } else {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== "granted") {
          Alert.alert("Permission refusée", "Impossible d’accéder à votre localisation.");
          return;
        }

        try {
          const initial = await Location.getCurrentPositionAsync({
            accuracy: Location.Accuracy.Balanced,
          });
          if (isMounted) {
            setLocation(initial);
            await handleLocationUpdate(initial);
          }
        } catch (error) {
          console.error("Erreur récupération position initiale:", error);
        }

        try {
          locationSubscription.current = await Location.watchPositionAsync(
            {
              // ✅ PERF: Balanced au lieu de High (-40% batterie)
              accuracy: Location.Accuracy.Balanced,
              timeInterval: 10000,
              // ✅ PERF: Ne update que si déplacement >50m
              distanceInterval: 50,
            },
            async (loc) => {
          if (!isMounted) return;
          setLocation(loc);
          await handleLocationUpdate(loc);            }
          );
        } catch (error) {
          console.error("Erreur création subscription localisation mobile:", error);
        }
      }
    };

  // ✅ PERF: Flush batch de positions (réduit réseau et batterie)
  const flushPositionBatch = async () => {
    if (positionBuffer.current.length === 0 || !driver) return;
    
    const batch = [...positionBuffer.current];
    positionBuffer.current = [];  // Clear buffer
    
    try {
      // Envoyer batch via Socket.IO (plus efficient)
      socket?.emit("driver_location_batch", {
        positions: batch.map(loc => ({
          latitude: loc.coords.latitude,
          longitude: loc.coords.longitude,
          speed: loc.coords.speed ?? 0,
          heading: loc.coords.heading ?? 0,
          accuracy: loc.coords.accuracy ?? 0,
          timestamp: loc.timestamp ?? Date.now(),
        })),
        driver_id: driver.id,
      });
      
      console.log(`📍 Batch envoyé: ${batch.length} positions`);
      
      // Mettre à jour dernière position
      const lastPos = batch[batch.length - 1];
      lastSentLocation.current = {
        latitude: lastPos.coords.latitude,
        longitude: lastPos.coords.longitude
      };
    } catch (error) {
      console.error("Erreur envoi batch localisation:", error);
    }
  };

  const handleLocationUpdate = async (loc: Location.LocationObject) => {
      const { latitude, longitude } = loc.coords;
      if (!driver) return;

      const lastLoc = lastSentLocation.current;
      const movedDistance = lastLoc
        ? getDistanceInMeters(lastLoc.latitude, lastLoc.longitude, latitude, longitude)
        : Infinity;

      // ✅ PERF: Ajouter au buffer au lieu d'envoyer immédiatement
      if (movedDistance >= 50) {
        positionBuffer.current.push(loc);
        
        // Flush si buffer plein
        if (positionBuffer.current.length >= BATCH_SIZE) {
          await flushPositionBatch();
        }
      }
    };

    requestLocationPermissions();

    // ✅ PERF: Flush périodique du buffer (toutes les 15s)
    const flushInterval = setInterval(() => {
      flushPositionBatch();
    }, BATCH_INTERVAL_MS);

    return () => {
      isMounted = false;
      clearInterval(flushInterval);  // Cleanup interval
      
      // Flush final avant cleanup
      if (positionBuffer.current.length > 0) {
        flushPositionBatch();
      }
      
      const sub = locationSubscription.current;
      if (typeof sub === "number" && Platform.OS === "web") {
        navigator.geolocation.clearWatch(sub);
      } else if (sub && typeof sub !== "number") {
        sub.remove();
      }
    };
  }, [driver, socket]);

  return { location };
};