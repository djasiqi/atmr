// hooks/useLocation.ts

import { useEffect, useState, useRef } from "react";
import * as Location from "expo-location";
import { Alert, Platform, AppState } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { sendDriverLocation, getDistanceInMeters } from "@/services/location";
import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";

// ✅ Nom de la tâche en arrière-plan (doit correspondre à locationTask.ts)
const BACKGROUND_TASK_NAME = "background-location-task";

// ✅ Mutex module-level pour éviter les doubles inits (React StrictMode / HMR)
let backgroundInitDone = false;
let backgroundInitRunning = false;

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
  // ✅ Stocker la dernière position reçue pour forcer l'envoi périodique
  const lastReceivedLocation = useRef<Location.LocationObject | null>(null);
  // ✅ Suivre si le tracking en arrière-plan a été démarré (local au hook)
  const backgroundTrackingStarted = useRef<boolean>(false);

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
        // ✅ Demander d'abord les permissions en premier plan
        const { status: foregroundStatus } = await Location.requestForegroundPermissionsAsync();
        if (foregroundStatus !== "granted") {
          Alert.alert("Permission refusée", "Impossible d'accéder à votre localisation.");
          return;
        }

        // ✅ Demander les permissions en arrière-plan (nécessaire pour le tracking continu)
        const { status: backgroundStatus } = await Location.requestBackgroundPermissionsAsync();
        if (backgroundStatus !== "granted") {
          console.warn("⚠️ Permission de localisation en arrière-plan refusée. Le tracking ne fonctionnera qu'en premier plan.");
        } else {
          console.log("✅ Permission de localisation en arrière-plan accordée");
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

        // ✅ Démarrer le tracking en arrière-plan si les permissions sont accordées
        if (backgroundStatus === "granted" && driver) {
          try {
            // On stocke le driver_id pour la tâche en arrière-plan
            await AsyncStorage.setItem("driver_id", driver.id.toString());
            console.log("[useLocation] ✅ driver_id stocké pour la tâche background");

            // Mutex global pour éviter les doubles appels
            if (backgroundInitDone || backgroundInitRunning) {
              console.log("[useLocation] ⚠️ Background init déjà en cours / déjà fait → skip");
            } else {
              backgroundInitRunning = true;

              const startBackgroundTracking = async () => {
                try {
                  console.log("[useLocation] 🚀 Init background tracking…");

                  // Vérifier si déjà démarré côté natif
                  let hasStarted = false;
                  try {
                    hasStarted = await Location.hasStartedLocationUpdatesAsync(BACKGROUND_TASK_NAME);
                    console.log("[useLocation] ℹ️ hasStartedLocationUpdatesAsync =", hasStarted);
                  } catch (checkError) {
                    console.warn("[useLocation] ⚠️ Erreur hasStartedLocationUpdatesAsync:", checkError);
                  }

                  if (hasStarted) {
                    console.log("[useLocation] ✅ Updates déjà démarrés → on ne relance pas");
                    backgroundTrackingStarted.current = true;
                    backgroundInitDone = true;
                    return;
                  }

                  // Re-check permission background par sécurité
                  const { status } = await Location.requestBackgroundPermissionsAsync();
                  console.log("[useLocation] ℹ️ Background permission status =", status);
                  if (status !== "granted") {
                    console.warn("[useLocation] ⚠️ Permission background refusée au moment du start");
                    return;
                  }

                  // Petit délai pour laisser Android initialiser le contexte
                  console.log("[useLocation] ⏳ Attente initialisation contexte Android (3s)…");
                  await new Promise((resolve) => setTimeout(resolve, 3000));

                  console.log("[useLocation] 🚀 Appel Location.startLocationUpdatesAsync()");
                  await Location.startLocationUpdatesAsync(BACKGROUND_TASK_NAME, {
                    // ✅ PRODUCTION: Paramètres optimisés pour économiser la batterie
                    accuracy: Location.Accuracy.Balanced, // Bon compromis précision/batterie
                    timeInterval: 10000, // 10s = mises à jour toutes les 10 secondes
                    distanceInterval: 50, // 50m = mises à jour si déplacement > 50m
                    foregroundService: {
                      notificationTitle: "Liri Opérations",
                      notificationBody: "Suivi de localisation en cours",
                    },
                  });

                  backgroundTrackingStarted.current = true;
                  backgroundInitDone = true;
                  console.log("[useLocation] ✅ startLocationUpdatesAsync démarré avec succès");
                } catch (startError: any) {
                  console.warn("[useLocation] ⚠️ startLocationUpdatesAsync a échoué:", startError?.message || startError);
                  console.log("[useLocation] ℹ️ Le tracking continue au moins en foreground");
                  backgroundTrackingStarted.current = false;
                } finally {
                  backgroundInitRunning = false;
                }
              };

              // ✅ TEST : Appel immédiat sans timeout pour diagnostiquer
              console.log("[useLocation] 🚀 Appel immédiat de startBackgroundTracking (test)");
              startBackgroundTracking().catch((err) => {
                console.error("[useLocation] ❌ Erreur dans startBackgroundTracking:", err);
                backgroundInitRunning = false;
              });
            }
          } catch (error: any) {
            console.error("❌ Erreur démarrage tracking arrière-plan:", error);
            backgroundTrackingStarted.current = false;
          }
        }

        // ✅ Tracking en premier plan (pour l'UI)
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
              await handleLocationUpdate(loc);
            }
          );
        } catch (error) {
          console.error("Erreur création subscription localisation mobile:", error);
        }
      }
    };

  // ✅ PERF: Flush batch de positions (réduit réseau et batterie)
  const flushPositionBatch = async () => {
    if (positionBuffer.current.length === 0) {
      console.log("[useLocation] ⚠️ Buffer vide, pas d'envoi");
      return;
    }
    if (!driver) {
      console.log("[useLocation] ⚠️ Driver non défini, pas d'envoi");
      return;
    }
    if (!socket || !socket.connected) {
      console.log("[useLocation] ⚠️ Socket non connecté, pas d'envoi", { socket: !!socket, connected: socket?.connected });
      return;
    }
    
    const batch = [...positionBuffer.current];
    positionBuffer.current = [];  // Clear buffer
    
    try {
      const payload = {
        positions: batch.map(loc => ({
          latitude: loc.coords.latitude,
          longitude: loc.coords.longitude,
          speed: loc.coords.speed ?? 0,
          heading: loc.coords.heading ?? 0,
          accuracy: loc.coords.accuracy ?? 0,
          timestamp: loc.timestamp ?? Date.now(),
        })),
        driver_id: driver.id,
      };
      
      console.log(`📍 [useLocation] Envoi batch: ${batch.length} positions, driver_id=${driver.id}, socket_connected=${socket.connected}`);
      
      // Envoyer batch via Socket.IO (plus efficient)
      socket.emit("driver_location_batch", payload);
      
      console.log(`✅ [useLocation] Batch envoyé: ${batch.length} positions`);
      
      // Mettre à jour dernière position
      const lastPos = batch[batch.length - 1];
      lastSentLocation.current = {
        latitude: lastPos.coords.latitude,
        longitude: lastPos.coords.longitude
      };
    } catch (error) {
      console.error("❌ [useLocation] Erreur envoi batch localisation:", error);
    }
  };

  const handleLocationUpdate = async (loc: Location.LocationObject) => {
      const { latitude, longitude } = loc.coords;
      if (!driver) return;

      // ✅ Toujours stocker la dernière position reçue (pour forcer l'envoi périodique)
      lastReceivedLocation.current = loc;

      const lastLoc = lastSentLocation.current;
      const movedDistance = lastLoc
        ? getDistanceInMeters(lastLoc.latitude, lastLoc.longitude, latitude, longitude)
        : Infinity;

      // ✅ Toujours envoyer la première position (même si déplacement faible)
      // ✅ Réduire le seuil à 20m pour être plus réactif
      const DISTANCE_THRESHOLD = 20; // Réduit de 50m à 20m
      
      if (!lastLoc || movedDistance >= DISTANCE_THRESHOLD) {
        positionBuffer.current.push(loc);
        console.log(`📍 [useLocation] Position ajoutée au buffer: ${positionBuffer.current.length}/${BATCH_SIZE}, distance=${lastLoc ? movedDistance.toFixed(0) : 'première'}m`);
        
        // Flush si buffer plein
        if (positionBuffer.current.length >= BATCH_SIZE) {
          console.log(`📍 [useLocation] Buffer plein (${BATCH_SIZE}), flush immédiat`);
          await flushPositionBatch();
        }
      } else {
        console.log(`📍 [useLocation] Position ignorée (déplacement < ${DISTANCE_THRESHOLD}m): ${movedDistance.toFixed(0)}m`);
      }
    };

    requestLocationPermissions();

    // ✅ PERF: Flush périodique du buffer (toutes les 15s)
    // ✅ Si buffer vide mais position récente disponible, forcer l'envoi de la dernière position
    const flushInterval = setInterval(() => {
      console.log(`⏰ [useLocation] Flush périodique (buffer=${positionBuffer.current.length})`);
      
      // Si buffer vide mais on a une position récente, l'ajouter au buffer
      if (positionBuffer.current.length === 0 && lastReceivedLocation.current) {
        console.log(`📍 [useLocation] Buffer vide, ajout de la dernière position reçue pour flush périodique`);
        positionBuffer.current.push(lastReceivedLocation.current);
      }
      
      flushPositionBatch();
    }, BATCH_INTERVAL_MS);

    return () => {
      isMounted = false;
      clearInterval(flushInterval);  // Cleanup interval
      
      // Flush final avant cleanup
      if (positionBuffer.current.length > 0) {
        flushPositionBatch();
      }
      
      // ✅ Arrêter le tracking en arrière-plan (uniquement si démarré)
      if (Platform.OS !== "web" && backgroundTrackingStarted.current) {
        Location.stopLocationUpdatesAsync("background-location-task").catch((error: any) => {
          // Ignorer l'erreur si la tâche n'existe pas (normal si elle n'a jamais été démarrée)
          if (error?.message?.includes("TaskNotFoundException") || error?.message?.includes("not found")) {
            console.log("ℹ️ Tâche de tracking arrière-plan non trouvée (normal si non démarrée)");
          } else {
            console.error("Erreur arrêt tracking arrière-plan:", error);
          }
        });
        backgroundTrackingStarted.current = false;
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