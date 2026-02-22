// hooks/useLocation.ts

import { useEffect, useState, useRef } from "react";
import * as Location from "expo-location";
import { Alert, Platform, AppState } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { getLogger } from "@/utils/logger";
import { sendDriverLocation, getDistanceInMeters } from "@/services/location";
import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";

const log = getLogger("Location");
// ✅ P2-1: Mode Offline Mobile - Persister queue GPS
import {
  enqueueLocation,
  getLocationQueue,
  clearLocationQueue,
  type QueuedLocation,
} from "@/services/locationQueue";

// ✅ Nom de la tâche en arrière-plan (doit correspondre à locationTask.ts)
const BACKGROUND_TASK_NAME = "background-location-task";

// ✅ Mutex module-level pour éviter les doubles inits (React StrictMode / HMR)
let backgroundInitDone = false;
let backgroundInitRunning = false;

// ✅ PERF: Configuration batching pour économiser batterie
const BATCH_SIZE = 3;  // Buffer de 3-5 positions avant envoi
// Config via env: EXPO_PUBLIC_GPS_FAST_MS=5000, MEDIUM_MS=10000, SLOW_MS=20000
const BATCH_INTERVAL_MS =
  parseInt(process.env.EXPO_PUBLIC_GPS_MEDIUM_MS ?? "10000", 10) || 10000;
const HEARTBEAT_INTERVAL_MS =
  parseInt(process.env.EXPO_PUBLIC_GPS_SLOW_MS ?? "30000", 10) || 30000;

// ✅ P2: Bug #6 - Retry automatique avec backoff exponentiel
const MAX_RETRY_ATTEMPTS = 5;
let retryTimeout: ReturnType<typeof setTimeout> | null = null;
let retryAttempts = 0;

export const useLocation = () => {
  const { driver } = useAuth();
  const socket = useSocket();
  
  // ✅ Vérifier que l'utilisateur est bien un chauffeur avant d'envoyer la position
  // Note: authMode n'existe plus dans AuthContextType, on vérifie directement driver
  const isDriverMode = !!driver;

  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const locationSubscription = useRef<Location.LocationSubscription | number | null>(null);
  const lastSentLocation = useRef<{ latitude: number; longitude: number } | null>(null);
  // ✅ PERF: Buffer pour batching des positions
  const positionBuffer = useRef<Location.LocationObject[]>([]);
  // ✅ Stocker la dernière position reçue pour forcer l'envoi périodique
  const lastReceivedLocation = useRef<Location.LocationObject | null>(null);
  // ✅ Dédup: éviter de re-queue la même position (même timestamp) en boucle (flush/heartbeat)
  const lastEnqueuedTimestampRef = useRef<number | null>(null);
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
            log.error("browser geolocation error", { error });
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
          log.warn("background location permission denied, tracking foreground only");
        } else {
          log.success("background location permission granted");
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
          log.error("initial position fetch failed", { error });
        }

        // ✅ Démarrer le tracking en arrière-plan si les permissions sont accordées
        if (backgroundStatus === "granted" && driver) {
          try {
            // On stocke le driver_id pour la tâche en arrière-plan
            await AsyncStorage.setItem("driver_id", driver.id.toString());
            log.success("driver_id stored for background task");

            // Mutex global pour éviter les doubles appels
            if (backgroundInitDone || backgroundInitRunning) {
              log.info("background init already in progress or done, skip");
            } else {
              backgroundInitRunning = true;

              const startBackgroundTracking = async () => {
                try {
                  log.info("init background tracking");

                  // Vérifier si déjà démarré côté natif
                  let hasStarted = false;
                  try {
                    hasStarted = await Location.hasStartedLocationUpdatesAsync(BACKGROUND_TASK_NAME);
                    log.info("hasStartedLocationUpdatesAsync", { hasStarted });
                  } catch (checkError) {
                    log.warn("hasStartedLocationUpdatesAsync check failed", { checkError });
                  }

                  if (hasStarted) {
                    log.success("updates already started, not restarting");
                    backgroundTrackingStarted.current = true;
                    backgroundInitDone = true;
                    return;
                  }

                  // Re-check permission background par sécurité
                  const { status } = await Location.requestBackgroundPermissionsAsync();
                  log.info("background permission status", { status });
                  if (status !== "granted") {
                    log.warn("background permission denied at start time");
                    return;
                  }

                  // Petit délai pour laisser Android initialiser le contexte
                  log.info("waiting for android context init (3s)");
                  await new Promise((resolve) => setTimeout(resolve, 3000));

                  log.info("calling startLocationUpdatesAsync");
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
                  log.success("startLocationUpdatesAsync started");
                } catch (startError: any) {
                  log.warn("startLocationUpdatesAsync failed", { message: startError?.message || startError });
                  log.info("tracking continues in foreground at least");
                  backgroundTrackingStarted.current = false;
                } finally {
                  backgroundInitRunning = false;
                }
              };

              // ✅ TEST : Appel immédiat sans timeout pour diagnostiquer
              log.info("immediate startBackgroundTracking call (test)");
              startBackgroundTracking().catch((err) => {
                log.error("startBackgroundTracking error", { err });
                backgroundInitRunning = false;
              });
            }
          } catch (error: any) {
            log.error("background tracking startup failed", { error });
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
          log.error("mobile location subscription creation failed", { error });
        }
      }
    };

  // ✅ P2: Bug #6 - Retry automatique avec backoff exponentiel
  const retryFailedBatch = async () => {
    if (retryAttempts >= MAX_RETRY_ATTEMPTS) {
      log.warn("max retry attempts reached, waiting for reconnect");
      retryAttempts = 0;
      return;
    }
    
    const queueSize = await getLocationQueue().then(q => q.length).catch(() => 0);
    if (queueSize === 0) {
      retryAttempts = 0;
      return;
    }
    
    if (socket && socket.connected) {
      log.info("retry batch", { attempt: retryAttempts + 1, queueSize });
      try {
        const { syncLocationQueue } = await import("@/services/locationQueue");
        await syncLocationQueue(socket);
        retryAttempts = 0;  // Reset on success
        log.success("retry succeeded, queue cleared");
      } catch (error: any) {
        log.error("retry failed", { attempt: retryAttempts + 1, error });
        // ✅ Rate limit: respecter retry_after si fourni, sinon backoff plus long
        const retryAfterSeconds =
          typeof error?.retry_after === "number" && Number.isFinite(error.retry_after)
            ? error.retry_after
            : undefined;

        const isRateLimit =
          String(error?.message || error).toLowerCase().includes("rate limit");

        const delay = retryAfterSeconds
          ? (retryAfterSeconds + 1) * 1000
          : isRateLimit
            ? Math.min(10000 * Math.pow(2, retryAttempts), 120000) // 10s, 20s, 40s, ... max 2m
            : Math.min(2000 * Math.pow(2, retryAttempts), 32000); // 2s,4s,8s,... max 32s

        retryAttempts++;
        
        log.info("next retry scheduled", { delaySeconds: delay / 1000 });
        
        if (retryTimeout) {
          clearTimeout(retryTimeout);
        }
        retryTimeout = setTimeout(() => {
          retryFailedBatch();
        }, delay);
      }
    } else {
      // Exponential backoff si socket déconnecté
      const delay = Math.min(2000 * Math.pow(2, retryAttempts), 32000);
      retryAttempts++;
      
      log.info("socket disconnected, retry scheduled", { delaySeconds: delay / 1000 });
      
      if (retryTimeout) {
        clearTimeout(retryTimeout);
      }
      retryTimeout = setTimeout(() => {
        retryFailedBatch();
      }, delay);
    }
  };

  // ✅ PERF: Flush batch de positions (réduit réseau et batterie)
  // ✅ P2-1: Mode Offline Mobile - Persister queue GPS si offline
  // ✅ P0: Fix Race Condition - Attendre ACK avant de vider buffer
  const flushPositionBatch = async () => {
    if (positionBuffer.current.length === 0) {
      log.info("buffer empty, skip send");
      return;
    }
    if (!isDriverMode) {
      log.info("user not driver, skip send");
      positionBuffer.current = []; // Vider le buffer
      return;
    }
    
    const batch = [...positionBuffer.current];
    // ✅ P0: NE PAS VIDER LE BUFFER IMMÉDIATEMENT (fix race condition)
    
    // ✅ P2-1: Préparer les positions pour la queue
    const queuedLocations: QueuedLocation[] = batch.map(loc => ({
      latitude: loc.coords.latitude,
      longitude: loc.coords.longitude,
      speed: loc.coords.speed ?? 0,
      heading: loc.coords.heading ?? 0,
      accuracy: loc.coords.accuracy ?? 0,
      timestamp: loc.timestamp ?? Date.now(),
      driver_id: driver.id,
    }));

    // ✅ Stabilisation: UN seul émetteur de `driver_location_batch` (locationQueue)
    // On persiste d'abord, puis on déclenche un resync (singleflight + throttle côté locationQueue).
    for (const loc of queuedLocations) {
      await enqueueLocation(loc);
      lastEnqueuedTimestampRef.current = loc.timestamp;
    }

    // Vider le buffer après enqueue (évite inflation et doublons)
    positionBuffer.current = [];

    if (socket && socket.connected) {
      try {
        const { syncLocationQueue } = await import("@/services/locationQueue");
        await syncLocationQueue(socket);

        // Mettre à jour dernière position "envoyée" (approx) pour la logique de distance
        const lastPos = batch[batch.length - 1];
        lastSentLocation.current = {
          latitude: lastPos.coords.latitude,
          longitude: lastPos.coords.longitude,
        };
      } catch (error: any) {
        // ✅ Réduire le bruit: warn au lieu de error pour les timeouts ACK (non bloquants)
        const msg = error?.message || String(error);
        if (msg.includes("Timeout") || msg.includes("ACK")) {
          log.warn("resync gps timeout (non-blocking, auto retry)", { msg });
        } else {
          log.error("resync gps queue failed", { error });
        }
        retryFailedBatch();
      }
    }
  };

  const handleLocationUpdate = async (loc: Location.LocationObject) => {
      const { latitude, longitude } = loc.coords;
      if (!isDriverMode) {
        log.debug("user not driver, position ignored");
        return;
      }

      // ✅ Toujours stocker la dernière position reçue (pour forcer l'envoi périodique)
      lastReceivedLocation.current = loc;

      const lastLoc = lastSentLocation.current;
      const movedDistance = lastLoc
        ? getDistanceInMeters(lastLoc.latitude, lastLoc.longitude, latitude, longitude)
        : Infinity;

      // ✅ Toujours envoyer la première position (même si déplacement faible)
      // ✅ Réduire le seuil à 10m pour être plus réactif
      const DISTANCE_THRESHOLD = 10; // Réduit de 20m à 10m
      
      if (!lastLoc || movedDistance >= DISTANCE_THRESHOLD) {
        positionBuffer.current.push(loc);
        log.info("position buffered", { bufferLength: positionBuffer.current.length, batchSize: BATCH_SIZE, distanceM: lastLoc ? movedDistance : "first" });
        
        // Flush si buffer plein
        if (positionBuffer.current.length >= BATCH_SIZE) {
          log.info("buffer full, flush immediate", { batchSize: BATCH_SIZE });
          await flushPositionBatch();
        }
      } else {
        log.info("position ignored (move below threshold)", { distanceM: movedDistance.toFixed(0), threshold: DISTANCE_THRESHOLD });
      }
    };

    requestLocationPermissions();

    // ✅ PERF: Flush périodique du buffer (toutes les 10s)
    // ✅ Si buffer vide mais position récente disponible, forcer l'envoi de la dernière position
    const flushInterval = setInterval(() => {
      log.info("flush periodic", { bufferLength: positionBuffer.current.length });
      
      // Si buffer vide mais on a une position récente, l'ajouter au buffer
      if (positionBuffer.current.length === 0 && lastReceivedLocation.current) {
        const ts = lastReceivedLocation.current.timestamp ?? null;
        if (ts && lastEnqueuedTimestampRef.current === ts) {
          log.info("last position already in queue, skip");
        } else {
          log.info("buffer empty, add last position for periodic flush");
          positionBuffer.current.push(lastReceivedLocation.current);
        }
      }
      
      flushPositionBatch();
    }, BATCH_INTERVAL_MS);

    // ✅ Heartbeat GPS : forcer l'envoi de la position toutes les 30s même si immobile
    // Cela garantit que le serveur reçoit régulièrement des positions même sans mouvement
    const heartbeatInterval = setInterval(() => {
      if (lastReceivedLocation.current && isDriverMode && socket?.connected) {
        log.info("heartbeat gps, force send last position");
        // Ajouter la dernière position au buffer si elle n'y est pas déjà
        const lastPos = lastReceivedLocation.current;
        const ts = lastPos.timestamp ?? null;
        if (ts && lastEnqueuedTimestampRef.current === ts) {
          return;
        }
        const alreadyInBuffer = positionBuffer.current.some(
          (loc) =>
            loc.coords.latitude === lastPos.coords.latitude &&
            loc.coords.longitude === lastPos.coords.longitude &&
            Math.abs((loc.timestamp || 0) - (lastPos.timestamp || 0)) < 1000
        );
        if (!alreadyInBuffer) {
          positionBuffer.current.push(lastReceivedLocation.current);
        }
        // Forcer le flush immédiat
        flushPositionBatch();
      } else {
        log.info("heartbeat gps skip (no position or socket disconnected)");
      }
    }, HEARTBEAT_INTERVAL_MS);

    return () => {
      isMounted = false;
      clearInterval(flushInterval);  // Cleanup flush interval
      clearInterval(heartbeatInterval);  // Cleanup heartbeat interval
      
      // ✅ P2: Bug #6 - Nettoyer retry timeout
      if (retryTimeout) {
        clearTimeout(retryTimeout);
        retryTimeout = null;
      }
      retryAttempts = 0;
      
      // Flush final avant cleanup
      if (positionBuffer.current.length > 0) {
        flushPositionBatch();
      }
      
      // ✅ Arrêter le tracking en arrière-plan (uniquement si démarré)
      if (Platform.OS !== "web" && backgroundTrackingStarted.current) {
        Location.stopLocationUpdatesAsync("background-location-task").catch((error: any) => {
          // Ignorer l'erreur si la tâche n'existe pas (normal si elle n'a jamais été démarrée)
          if (error?.message?.includes("TaskNotFoundException") || error?.message?.includes("not found")) {
            log.info("background tracking task not found (normal if never started)");
          } else {
            log.error("stop background tracking failed", { error });
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