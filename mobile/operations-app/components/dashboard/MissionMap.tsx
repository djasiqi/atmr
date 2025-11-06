// components/dashboard/MissionMap.tsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, Alert } from 'react-native';
import MapView, { Marker, LatLng } from 'react-native-maps';
import MapViewDirections from 'react-native-maps-directions';
import * as Location from 'expo-location';
import { GOOGLE_API_KEY, ANDROID_MAPS_API_KEY } from '../../src/config/env';
import { styles } from '@/styles/missionMapStyles';

type Props = {
  location: { coords: { latitude: number; longitude: number } };
  destination: string; // ex: "10 rue ... ville"
};

// ✅ Directions API key (HTTP). Ne pas utiliser la clé Android Maps ici.
const DIRECTIONS_KEY = GOOGLE_API_KEY;

const mask = (val: string | undefined) =>
  val ? `${val.slice(0, 6)}...${val.slice(-4)}` : 'undefined';

const MissionMap: React.FC<Props> = ({ location, destination }) => {
  const mapRef = useRef<MapView | null>(null);
  const [destinationCoords, setDestinationCoords] = useState<LatLng | null>(null);

  useEffect(() => {
    if (!DIRECTIONS_KEY) {
      console.warn(
        '⚠️ [ENV] EXPO_PUBLIC_GOOGLE_API_KEY manquant — ajoute-le dans .env.local puis redémarre Metro.'
      );
    } else {
      console.log('[MissionMap] Directions key chargée:', mask(DIRECTIONS_KEY));
    }
  }, []);

  // Géocodage de l'adresse destination -> LatLng
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
        console.error('Erreur de géocodage :', error);
        Alert.alert('Erreur', 'Le géocodage a échoué.');
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

  return (
    <View style={styles.container}>
      <MapView
        ref={mapRef}
        style={styles.map}
        initialRegion={region}
        showsUserLocation
        loadingEnabled
      >
        <Marker coordinate={location.coords} title="Vous êtes ici" pinColor="blue" />

        {destinationCoords && (
          <Marker key="marker" coordinate={destinationCoords} title="Destination" pinColor="green" />
        )}

        {canDrawRoute && (
          <MapViewDirections
            key="directions"
            origin={location.coords}
            destination={destinationCoords!}
            apikey={DIRECTIONS_KEY}
            mode="DRIVING"
            strokeWidth={4}
            strokeColor="#00BFA6"
            optimizeWaypoints
            onReady={(result) => {
              // Fit map to route
              if (mapRef.current && result.coordinates?.length) {
                mapRef.current.fitToCoordinates(result.coordinates, {
                  edgePadding: { top: 60, right: 60, bottom: 60, left: 60 },
                  animated: true,
                });
              }
            }}
            onError={(e) => {
              console.warn('Directions error:', e);
              // Message fréquent si mauvaise clé / restrictions
              // => activer "Directions API" et vérifier que la clé HTTP n’est pas restreinte à Android apps.
            }}
          />
        )}
      </MapView>
    </View>
  );
};

export default MissionMap;
