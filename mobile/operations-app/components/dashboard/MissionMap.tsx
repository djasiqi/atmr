import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, Alert, Platform } from 'react-native';
import MapView, { Marker, Polyline, LatLng, PROVIDER_GOOGLE } from 'react-native-maps';
import MapViewDirections from 'react-native-maps-directions';
import * as Location from 'expo-location';
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Ionicons } from '@expo/vector-icons';
import { GOOGLE_API_KEY } from '../../src/config/env';
import { styles, LIRIE_MAP_STYLE, MAP_BRAND } from '@/styles/missionMapStyles';
import { getLogger } from "@/utils/logger";
import { getDriverRoute } from "@/services/api";
import { decodePolyline } from "@/utils/polyline";

const log = getLogger("MissionMap");

/** STR-03 : cache polyline courte durée (retour onglet / cold start carte). */
const ROUTE_POLY_CACHE_PREFIX = "mission_route_poly_v1_";
const ROUTE_POLY_CACHE_TTL_MS = 10 * 60 * 1000;

type CachedRoutePayload = { exp: number; coords: LatLng[] };

async function readRoutePolyCache(routeKey: string): Promise<LatLng[] | null> {
  try {
    const storageKey =
      ROUTE_POLY_CACHE_PREFIX + encodeURIComponent(routeKey).slice(0, 220);
    const raw = await AsyncStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CachedRoutePayload;
    if (!parsed?.coords?.length || parsed.exp < Date.now()) return null;
    return parsed.coords;
  } catch {
    return null;
  }
}

async function writeRoutePolyCache(routeKey: string, coords: LatLng[]): Promise<void> {
  try {
    const storageKey =
      ROUTE_POLY_CACHE_PREFIX + encodeURIComponent(routeKey).slice(0, 220);
    const payload: CachedRoutePayload = {
      exp: Date.now() + ROUTE_POLY_CACHE_TTL_MS,
      coords,
    };
    await AsyncStorage.setItem(storageKey, JSON.stringify(payload));
  } catch {
    /* ignore */
  }
}

type Props = {
  location: { coords: { latitude: number; longitude: number } };
  /** Vide si carte « position seule » (STW-03). */
  destination?: string;
  /** Coords backend (pickup_lat/lon, dropoff_lat/lon). Si fournies, aucun géocodage. */
  destinationCoords?: { latitude: number; longitude: number } | null;
  /** false par défaut. true seulement pour compatibilité legacy ou données incomplètes. */
  allowGeocodeFallback?: boolean;
  contentWidth?: number;
  mapHeight?: number;
};

const DIRECTIONS_KEY = GOOGLE_API_KEY;

const mask = (val: string | undefined) =>
  val ? `${val.slice(0, 6)}...${val.slice(-4)}` : 'undefined';

const MissionMap: React.FC<Props> = ({
  location,
  destination = "",
  destinationCoords: destinationCoordsProp,
  allowGeocodeFallback = false,
  contentWidth,
  mapHeight,
}) => {
  const mapRef = useRef<MapView | null>(null);
  const mountTimeRef = useRef<number>(Date.now());
  const [destinationCoords, setDestinationCoords] = useState<LatLng | null>(null);
  const [routeCoords, setRouteCoords] = useState<LatLng[] | null>(null);
  const [useGoogleFallback, setUseGoogleFallback] = useState(false);
  const lastGeocodeAlertAtRef = useRef<number>(0);
  const lastRouteKeyRef = useRef<string | null>(null);

  useEffect(() => {
    log.info("MissionMap mounted", { hasDestCoordsProp: destinationCoordsProp != null });
    if (!DIRECTIONS_KEY) {
      log.warn("google api key missing", {});
    } else {
      log.info("directions key loaded", { key: mask(DIRECTIONS_KEY) });
    }
  }, []);

  // 1. Si destinationCoords fournies -> utiliser directement, aucun géocodage
  useEffect(() => {
    if (destinationCoordsProp != null) {
      const elapsed = Date.now() - mountTimeRef.current;
      log.info("MissionMap destinationCoords ready (from backend)", { elapsedMs: elapsed });
      setDestinationCoords({
        latitude: destinationCoordsProp.latitude,
        longitude: destinationCoordsProp.longitude,
      });
      return;
    }
    // 2. Sinon, si allowGeocodeFallback et destination string -> géocoder (legacy)
    if (!allowGeocodeFallback || !destination?.trim()) {
      setDestinationCoords(null);
      return;
    }
    const fetchDestinationCoords = async () => {
      try {
        const geocode = await Location.geocodeAsync(destination);
        if (geocode.length > 0) {
          setDestinationCoords({
            latitude: geocode[0].latitude,
            longitude: geocode[0].longitude,
          });
        } else {
          Alert.alert('Adresse non trouvée', "Impossible de localiser l'adresse de destination.");
          setDestinationCoords(null);
        }
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        const isTransient =
          msg.includes('UNAVAILABLE') ||
          msg.includes('java.io.IOException') ||
          msg.toLowerCase().includes('rejected');

        if (isTransient) {
          log.warn("geocode error transient", { error });
        } else {
          log.error("geocode error", { error });
          const now = Date.now();
          if (now - lastGeocodeAlertAtRef.current > 60_000) {
            lastGeocodeAlertAtRef.current = now;
            Alert.alert('Erreur', 'Le géocodage a échoué.');
          }
        }
        setDestinationCoords(null);
      }
    };

    fetchDestinationCoords();
  }, [destinationCoordsProp, allowGeocodeFallback, destination]);

  const region = useMemo(
    () => ({
      latitude: location.coords.latitude,
      longitude: location.coords.longitude,
      latitudeDelta: 0.02,
      longitudeDelta: 0.02,
    }),
    [location.coords.latitude, location.coords.longitude]
  );

  const canDrawRoute = Boolean(DIRECTIONS_KEY && destinationCoords);

  useEffect(() => {
    if (!destinationCoords) {
      lastRouteKeyRef.current = null;
      setRouteCoords(null);
      setUseGoogleFallback(false);
    }
  }, [destinationCoords]);

  const routeKey = destinationCoords
    ? `${location.coords.latitude},${location.coords.longitude}:${destinationCoords.latitude},${destinationCoords.longitude}`
    : null;

  const hasCachedRoute =
    routeKey != null &&
    lastRouteKeyRef.current === routeKey &&
    routeCoords != null &&
    routeCoords.length > 0;

  const needsDirectionsFetch =
    canDrawRoute && routeKey != null && lastRouteKeyRef.current !== routeKey;

  // Priorité OSRM (backend) puis fallback Google Directions
  useEffect(() => {
    if (!needsDirectionsFetch || !destinationCoords || !routeKey) return;

    setUseGoogleFallback(false);
    setRouteCoords(null);
    let cancelled = false;

    const fetchOsrmRoute = async () => {
      const routeFetchStart = Date.now();
      log.info("MissionMap OSRM route fetch start", { routeKey: routeKey?.slice(0, 40) });
      try {
        const res = await getDriverRoute(
          location.coords.latitude,
          location.coords.longitude,
          destinationCoords.latitude,
          destinationCoords.longitude
        );
        if (cancelled) return;

        let coords: LatLng[];
        if (res.polyline_encoded) {
          coords = decodePolyline(res.polyline_encoded);
        } else if (res.coordinates?.length) {
          coords = res.coordinates.map((c) => ({
            latitude: c.lat,
            longitude: c.lon,
          }));
        } else {
          setUseGoogleFallback(true);
          return;
        }

        if (coords.length > 0) {
          const elapsed = Date.now() - routeFetchStart;
          log.info("[PERF] mission_map_osrm_fetch_done", { elapsedMs: elapsed, points: coords.length });
          lastRouteKeyRef.current = routeKey;
          setRouteCoords(coords);
          void writeRoutePolyCache(routeKey, coords);
          if (mapRef.current) {
            mapRef.current.fitToCoordinates(coords, {
              edgePadding: { top: 50, right: 50, bottom: 50, left: 50 },
              animated: true,
            });
          }
        } else {
          setUseGoogleFallback(true);
        }
      } catch (e) {
        if (!cancelled) {
          const elapsed = Date.now() - routeFetchStart;
          log.warn("[PERF] mission_map_osrm_failed_fallback_google", {
            error: e,
            elapsedMs: elapsed,
          });
          setUseGoogleFallback(true);
        }
      }
    };

    fetchOsrmRoute();
    return () => {
      cancelled = true;
    };
  }, [
    needsDirectionsFetch,
    routeKey,
    destinationCoords,
    location.coords.latitude,
    location.coords.longitude,
  ]);

  const containerStyle = [
    styles.container,
    contentWidth != null && { width: contentWidth, alignSelf: 'center' as const, marginHorizontal: 0 },
    mapHeight != null && { height: mapHeight },
  ];

  return (
    <View style={containerStyle}>
      <MapView
        ref={mapRef}
        style={styles.map}
        provider={Platform.OS === 'android' ? PROVIDER_GOOGLE : undefined}
        initialRegion={region}
        showsUserLocation
        showsMyLocationButton={false}
        showsPointsOfInterest={false}
        showsBuildings={false}
        loadingEnabled
        loadingIndicatorColor={MAP_BRAND.primary}
        customMapStyle={LIRIE_MAP_STYLE}
      >
        <Marker
          coordinate={location.coords}
          title="Votre position"
          anchor={{ x: 0.5, y: 0.5 }}
          tracksViewChanges={false}
        >
          <View style={styles.markerPickup}>
            <Ionicons name="navigate" size={14} color="#fff" />
          </View>
        </Marker>

        {destinationCoords && (
          <Marker
            key="dest"
            coordinate={destinationCoords}
            title="Destination"
            anchor={{ x: 0.5, y: 0.5 }}
            tracksViewChanges={false}
          >
            <View style={styles.markerDropoff}>
              <Ionicons name="flag" size={14} color="#fff" />
            </View>
          </Marker>
        )}

        {hasCachedRoute && routeCoords && (
          <Polyline
            coordinates={routeCoords}
            strokeWidth={4}
            strokeColor={MAP_BRAND.primary}
          />
        )}
        {needsDirectionsFetch && useGoogleFallback && (
          <MapViewDirections
            key={`directions-${routeKey}`}
            origin={location.coords}
            destination={destinationCoords!}
            apikey={DIRECTIONS_KEY}
            mode="DRIVING"
            strokeWidth={4}
            strokeColor={MAP_BRAND.primary}
            optimizeWaypoints
            onReady={(result) => {
              if (result.coordinates?.length && routeKey) {
                lastRouteKeyRef.current = routeKey;
                setRouteCoords(result.coordinates);
                void writeRoutePolyCache(routeKey, result.coordinates);
                if (mapRef.current) {
                  mapRef.current.fitToCoordinates(result.coordinates, {
                    edgePadding: { top: 50, right: 50, bottom: 50, left: 50 },
                    animated: true,
                  });
                }
              }
            }}
            onError={(e) => {
              log.warn("directions error", { error: e });
            }}
          />
        )}
      </MapView>

      {/* Overlay badge distance/durée (se remplit via onReady) */}
    </View>
  );
};

export default MissionMap;
