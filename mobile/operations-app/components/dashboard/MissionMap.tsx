import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, Alert, Platform } from 'react-native';
import MapView, { Marker, LatLng, PROVIDER_GOOGLE } from 'react-native-maps';
import MapViewDirections from 'react-native-maps-directions';
import * as Location from 'expo-location';
import { Ionicons } from '@expo/vector-icons';
import { GOOGLE_API_KEY } from '../../src/config/env';
import { styles, LIRIE_MAP_STYLE, MAP_BRAND } from '@/styles/missionMapStyles';
import { getLogger } from "@/utils/logger";

const log = getLogger("MissionMap");

type Props = {
  location: { coords: { latitude: number; longitude: number } };
  destination: string;
  contentWidth?: number;
  mapHeight?: number;
};

const DIRECTIONS_KEY = GOOGLE_API_KEY;

const mask = (val: string | undefined) =>
  val ? `${val.slice(0, 6)}...${val.slice(-4)}` : 'undefined';

const MissionMap: React.FC<Props> = ({ location, destination, contentWidth, mapHeight }) => {
  const mapRef = useRef<MapView | null>(null);
  const [destinationCoords, setDestinationCoords] = useState<LatLng | null>(null);
  const lastGeocodeAlertAtRef = useRef<number>(0);

  useEffect(() => {
    if (!DIRECTIONS_KEY) {
      log.warn("google api key missing", {});
    } else {
      log.info("directions key loaded", { key: mask(DIRECTIONS_KEY) });
    }
  }, []);

  useEffect(() => {
    const fetchDestinationCoords = async () => {
      try {
        if (!destination?.trim()) {
          setDestinationCoords(null);
          return;
        }
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
  }, [destination]);

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

        {canDrawRoute && (
          <MapViewDirections
            key="directions"
            origin={location.coords}
            destination={destinationCoords!}
            apikey={DIRECTIONS_KEY}
            mode="DRIVING"
            strokeWidth={4}
            strokeColor={MAP_BRAND.primary}
            optimizeWaypoints
            onReady={(result) => {
              if (mapRef.current && result.coordinates?.length) {
                mapRef.current.fitToCoordinates(result.coordinates, {
                  edgePadding: { top: 50, right: 50, bottom: 50, left: 50 },
                  animated: true,
                });
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
