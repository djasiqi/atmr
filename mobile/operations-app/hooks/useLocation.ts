// hooks/useLocation.ts
// Plan 2G/3G Phase 4 : Consomme locationTracker.subscribeToPosition pour l'UI.
// Plus de watchPosition, buffer, enqueue, flush — locationTracker est la source unique.

import { useEffect, useRef, useState } from "react";
import * as Location from "expo-location";
import { Alert, Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { getLogger } from "@/utils/logger";
import { useAuth } from "@/hooks/useAuth";
import { getAdaptiveLocationTracker } from "@/services/locationTracker";

const log = getLogger("Location");

export const useLocation = () => {
  const { driver } = useAuth();
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const unsubRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    let isMounted = true;

    const setup = async (): Promise<(() => void) | void> => {
      if (Platform.OS === "web") {
        if (!navigator.geolocation) {
          Alert.alert("Erreur", "La géolocalisation n'est pas disponible sur ce navigateur.");
          return;
        }
        const initial = await new Promise<Location.LocationObject | null>((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(
            (p) =>
              resolve({
                coords: {
                  latitude: p.coords.latitude,
                  longitude: p.coords.longitude,
                  accuracy: p.coords.accuracy,
                  altitude: p.coords.altitude ?? null,
                  altitudeAccuracy: p.coords.altitudeAccuracy ?? null,
                  heading: p.coords.heading ?? null,
                  speed: p.coords.speed ?? null,
                },
                timestamp: p.timestamp,
              }),
            reject,
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 10000 }
          );
        });
        if (isMounted && initial) setLocation(initial);
        return undefined;
      }

      const { status: foregroundStatus } = await Location.requestForegroundPermissionsAsync();
      if (foregroundStatus !== "granted") {
        Alert.alert("Permission refusée", "Impossible d'accéder à votre localisation.");
        return;
      }

      if (driver) {
        await AsyncStorage.setItem("driver_id", driver.id.toString());
      }

      if ((Platform.OS as string) !== "web" && driver) {
        const tracker = getAdaptiveLocationTracker();
        return tracker.subscribeToPosition((loc) => {
          if (isMounted) setLocation(loc);
        });
      }
    };

    setup().then((fn) => {
      unsubRef.current = typeof fn === "function" ? fn : null;
    });
    return () => {
      isMounted = false;
      unsubRef.current?.();
      unsubRef.current = null;
    };
  }, [driver]);

  return { location };
};
