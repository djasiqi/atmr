import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { View, Text, StyleSheet, ViewStyle } from "react-native";
import MapView, { Marker, PROVIDER_GOOGLE, Region } from "react-native-maps";

type DriverMarker = {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  status?: string;
  eta?: string;
  updatedAt?: string;
};

type EnterpriseDriversMapProps = {
  markers: DriverMarker[];
  style?: ViewStyle;
  fallbackMessage?: string;
};

export const EnterpriseDriversMap: React.FC<EnterpriseDriversMapProps> = ({
  markers,
  style,
  fallbackMessage = "Position des chauffeurs indisponible pour le moment",
}) => {
  const getInitials = useCallback((label: string) => {
    return label
      .trim()
      .split(/\s+/)
      .map((part) => part.charAt(0).toUpperCase())
      .join("")
      .slice(0, 3);
  }, []);

  const formatTimestamp = useCallback((value?: string) => {
    if (!value) return "Tracking actif";
    const parsed = new Date(value);
    // Vérifie date valide
    if (Number.isNaN(parsed.getTime())) {
      return "MAJ récente";
    }
    return `MAJ ${parsed.toLocaleTimeString("fr-FR", {
      hour: "2-digit",
      minute: "2-digit",
    })}`;
  }, []);

  const getStatusColor = useCallback((status?: string) => {
    switch ((status || "").toLowerCase()) {
      case "available":
      case "libre":
        return "#34d399";
      case "busy":
      case "occupied":
      case "assigned":
        return "#f59e0b";
      case "offline":
      case "unavailable":
        return "#f87171";
      default:
        return "#60a5fa";
    }
  }, []);

  const mapRef = useRef<MapView | null>(null);
  const region = useMemo(() => {
    if (!markers.length) {
      return {
        latitude: 46.2044,
        longitude: 6.1432,
        latitudeDelta: 0.25,
        longitudeDelta: 0.25,
      } as Region;
    }

    if (markers.length === 1) {
      const marker = markers[0];
      return {
        latitude: marker.latitude,
        longitude: marker.longitude,
        latitudeDelta: 0.05,
        longitudeDelta: 0.05,
      } as Region;
    }

    return null;
  }, [markers]);

  useEffect(() => {
    if (mapRef.current && markers.length > 1) {
      mapRef.current.fitToCoordinates(
        markers.map((marker) => ({
          latitude: marker.latitude,
          longitude: marker.longitude,
        })),
        {
          edgePadding: { top: 60, right: 40, bottom: 60, left: 40 },
          animated: true,
        }
      );
    }
  }, [markers]);

  const [tracksViewChanges, setTracksViewChanges] = useState(true);

  useEffect(() => {
    setTracksViewChanges(true);
    const timer = setTimeout(() => setTracksViewChanges(false), 600);
    return () => clearTimeout(timer);
  }, [markers]);

  return (
    <View style={[styles.container, style]}>
      <MapView
        ref={mapRef}
        style={StyleSheet.absoluteFill}
        provider={PROVIDER_GOOGLE}
        region={region ?? undefined}
        customMapStyle={MAP_STYLE}
      >
        {markers.map((marker) => (
          <Marker
            key={marker.id}
            coordinate={{
              latitude: marker.latitude,
              longitude: marker.longitude,
            }}
            title={marker.name}
            description={formatTimestamp(marker.updatedAt)}
            tracksViewChanges={tracksViewChanges}
          >
            <View style={styles.markerWrapper}>
              <View
                style={[
                  styles.markerHalo,
                  { backgroundColor: getStatusColor(marker.status) },
                ]}
              />
              <View
                style={[
                  styles.markerBadge,
                  { borderColor: getStatusColor(marker.status) },
                ]}
              >
                <Text style={styles.markerInitials}>
                  {getInitials(marker.name)}
                </Text>
              </View>
              <View
                style={[
                  styles.markerTail,
                  { borderTopColor: getStatusColor(marker.status) },
                ]}
              />
            </View>
          </Marker>
        ))}
      </MapView>

      {!markers.length ? (
        <View style={styles.overlay}>
          <Text style={styles.overlayTitle}>Carte chauffeurs</Text>
          <Text style={styles.overlayText}>{fallbackMessage}</Text>
          <Text style={styles.overlayMeta}>
            Activez le tracking temps réel pour voir les positions.
          </Text>
        </View>
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    height: 240,
    borderRadius: 22,
    overflow: "hidden",
    marginTop: 10,
    backgroundColor: "#0B1736",
  },
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(5,11,28,0.72)",
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 24,
  },
  overlayTitle: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 16,
    marginBottom: 6,
  },
  overlayText: {
    color: "rgba(214,224,255,0.85)",
    textAlign: "center",
    fontSize: 13,
  },
  overlayMeta: {
    color: "rgba(148,163,255,0.7)",
    marginTop: 10,
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  markerWrapper: {
    alignItems: "center",
    position: "relative",
  },
  markerHalo: {
    position: "absolute",
    width: 36,
    height: 36,
    borderRadius: 18,
    opacity: 0.28,
  },
  markerBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: "rgba(15,22,38,0.95)",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  markerInitials: {
    color: "#f8faff",
    fontWeight: "700",
    fontSize: 12,
    letterSpacing: 0.6,
  },
  markerTail: {
    width: 0,
    height: 0,
    borderLeftWidth: 4,
    borderRightWidth: 4,
    borderTopWidth: 6,
    borderLeftColor: "transparent",
    borderRightColor: "transparent",
    borderTopColor: "rgba(46,71,140,0.95)",
    marginTop: 1,
  },
});

const MAP_STYLE = [
  {
    elementType: "geometry",
    stylers: [{ color: "#f2f3f7" }],
  },
  {
    elementType: "labels.text.fill",
    stylers: [{ color: "#4a4f63" }],
  },
  {
    elementType: "labels.text.stroke",
    stylers: [{ color: "#ffffff" }],
  },
  {
    featureType: "administrative",
    elementType: "geometry",
    stylers: [{ color: "#d9dce8" }],
  },
  {
    featureType: "administrative.land_parcel",
    elementType: "labels.text.fill",
    stylers: [{ color: "#6b7280" }],
  },
  {
    featureType: "poi",
    elementType: "geometry",
    stylers: [{ color: "#e8ebf4" }],
  },
  {
    featureType: "poi",
    elementType: "labels.text.fill",
    stylers: [{ color: "#5f6476" }],
  },
  {
    featureType: "road",
    elementType: "geometry",
    stylers: [{ color: "#ffffff" }],
  },
  {
    featureType: "road",
    elementType: "geometry.stroke",
    stylers: [{ color: "#cfd4e6" }],
  },
  {
    featureType: "road.highway",
    elementType: "geometry",
    stylers: [{ color: "#ffd37f" }],
  },
  {
    featureType: "road.highway",
    elementType: "geometry.stroke",
    stylers: [{ color: "#f1b85c" }],
  },
  {
    featureType: "transit",
    elementType: "geometry",
    stylers: [{ color: "#d7dbeb" }],
  },
  {
    featureType: "water",
    elementType: "geometry",
    stylers: [{ color: "#a5d5f8" }],
  },
  {
    featureType: "water",
    elementType: "labels.text.fill",
    stylers: [{ color: "#4f6c8c" }],
  },
];

export default EnterpriseDriversMap;
