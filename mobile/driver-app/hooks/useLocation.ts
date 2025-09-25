// hooks/useLocation.ts

import { useEffect, useState, useRef } from "react";
import * as Location from "expo-location";
import { Alert, Platform } from "react-native";
import { sendDriverLocation, getDistanceInMeters } from "@/services/location";
import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";

export const useLocation = () => {
  const { driver } = useAuth();
  const socket = useSocket();

  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const locationSubscription = useRef<Location.LocationSubscription | number | null>(null);
  const lastSentLocation = useRef<{ latitude: number; longitude: number } | null>(null);

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
          locationSubscription.current = await Location.watchPositionAsync(
            {
              accuracy: Location.Accuracy.High,
              timeInterval: 10000,
              distanceInterval: 10,
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

  const handleLocationUpdate = async (loc: Location.LocationObject) => {
      const { latitude, longitude, speed, heading, accuracy } = loc.coords;      if (!driver) return;

      const lastLoc = lastSentLocation.current;
      const movedDistance = lastLoc
        ? getDistanceInMeters(lastLoc.latitude, lastLoc.longitude, latitude, longitude)        : Infinity; // envoyer immédiatement si pas de précédente position

      if (movedDistance >= 50) {
        try {
          // REST API : on envoie un objet conforme
          await sendDriverLocation({
            latitude,
            longitude,
            speed: typeof speed === "number" ? speed : 0,
            heading: typeof heading === "number" ? heading : 0,
            accuracy: typeof accuracy === "number" ? accuracy : 0,
            timestamp: loc.timestamp ?? Date.now(),
          });

          // WebSocket en temps réel
          socket?.emit("driver_location", {
            latitude,
            longitude,
            driverId: driver?.id,
            first_name: driver.first_name,
          });

          console.log("📍 Position envoyée (socket + REST):", latitude, longitude);

          lastSentLocation.current = { latitude, longitude };
        } catch (error) {
          console.error("Erreur d’envoi de la localisation:", error);
        }
      }
    };

    requestLocationPermissions();

    return () => {
      isMounted = false;
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