import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Platform, Text, StyleSheet, View, ViewStyle } from "react-native";
import MapView, { Marker, PROVIDER_GOOGLE, Region } from "react-native-maps";
import { Ionicons } from "@expo/vector-icons";
import { getLogger } from "@/utils/logger";

const log = getLogger("DriversMap");

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

const BRAND = {
  primary: "#00796B",
  primaryDark: "#00695C",
  primaryLight: "#26a69a",
  success: "#22c55e",
  warning: "#f59e0b",
  danger: "#ef4444",
  muted: "#94A3B8",
  text: "#1E293B",
  textSec: "#64748B",
  border: "rgba(15,54,43,0.06)",
  bg: "#f0f2f4",
  card: "#ffffff",
} as const;

const STATUS_COLORS: Record<string, string> = {
  available: BRAND.success,
  libre: BRAND.success,
  busy: BRAND.primary,
  occupied: BRAND.primary,
  assigned: BRAND.warning,
  offline: BRAND.muted,
  unavailable: BRAND.muted,
  emergency: BRAND.danger,
};

export const EnterpriseDriversMap: React.FC<EnterpriseDriversMapProps> = ({
  markers,
  style,
  fallbackMessage = "Position des chauffeurs indisponible pour le moment",
}) => {
  useEffect(() => {
    log.debug("markers received", { count: markers.length });
  }, [markers]);

  const getInitials = useCallback((label: string) => {
    return label
      .trim()
      .split(/\s+/)
      .map((part) => part.charAt(0).toUpperCase())
      .join("")
      .slice(0, 2);
  }, []);

  const formatTimestamp = useCallback((value?: string) => {
    if (!value) return "Tracking actif";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return "MAJ récente";
    return `MAJ ${parsed.toLocaleTimeString("fr-FR", {
      hour: "2-digit",
      minute: "2-digit",
    })}`;
  }, []);

  const getStatusColor = useCallback((status?: string) => {
    const key = (status || "").toLowerCase();
    return STATUS_COLORS[key] || BRAND.primary;
  }, []);

  const mapRef = useRef<MapView | null>(null);
  const previousMarkersCountRef = useRef(0);

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
      return {
        latitude: markers[0].latitude,
        longitude: markers[0].longitude,
        latitudeDelta: 0.05,
        longitudeDelta: 0.05,
      } as Region;
    }
    return null;
  }, [markers]);

  useEffect(() => {
    const previousCount = previousMarkersCountRef.current;
    previousMarkersCountRef.current = markers.length;

    // Evite de réanimer la carte à chaque update de position.
    if (mapRef.current && markers.length > 1 && previousCount !== markers.length) {
      mapRef.current.fitToCoordinates(
        markers.map((m) => ({ latitude: m.latitude, longitude: m.longitude })),
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
        provider={Platform.OS === "android" ? PROVIDER_GOOGLE : undefined}
        region={region ?? undefined}
        customMapStyle={Platform.OS === "android" ? LIRIE_MAP_STYLE : undefined}
        showsPointsOfInterest={false}
        showsBuildings={false}
        showsMyLocationButton={false}
        loadingIndicatorColor={BRAND.primary}
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
            anchor={{ x: 0.5, y: 0.5 }}
          >
            <View style={styles.markerOuter}>
              <View
                style={[
                  styles.markerPulse,
                  { backgroundColor: getStatusColor(marker.status) },
                ]}
              />
              <View
                style={[
                  styles.markerCircle,
                  { backgroundColor: getStatusColor(marker.status) },
                ]}
              >
                <Text style={styles.markerInitials}>
                  {getInitials(marker.name)}
                </Text>
              </View>
            </View>
          </Marker>
        ))}
      </MapView>

      {markers.length > 0 && (
        <View style={styles.countBadge}>
          <Ionicons name="people" size={12} color={BRAND.card} />
          <Text style={styles.countText}>{markers.length}</Text>
        </View>
      )}

      {!markers.length && (
        <View style={styles.overlay}>
          <View style={styles.overlayIcon}>
            <Ionicons name="map-outline" size={28} color={BRAND.primary} />
          </View>
          <Text style={styles.overlayTitle}>Carte chauffeurs</Text>
          <Text style={styles.overlayText}>{fallbackMessage}</Text>
          <Text style={styles.overlayMeta}>
            Activez le tracking temps réel pour voir les positions
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    height: 200,
    borderRadius: 16,
    overflow: "hidden",
    marginHorizontal: 0,
    marginTop: 4,
    backgroundColor: BRAND.bg,
    borderWidth: 1,
    borderColor: BRAND.border,
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.08,
        shadowRadius: 16,
      },
      android: { elevation: 4 },
      default: {},
    }),
  },

  countBadge: {
    position: "absolute",
    top: 12,
    right: 12,
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    backgroundColor: BRAND.primary,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    ...Platform.select({
      ios: {
        shadowColor: BRAND.primaryDark,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.3,
        shadowRadius: 4,
      },
      android: { elevation: 4 },
      default: {},
    }),
  },
  countText: {
    color: BRAND.card,
    fontSize: 12,
    fontWeight: "700",
  },

  markerOuter: {
    alignItems: "center",
    justifyContent: "center",
    width: 40,
    height: 40,
  },
  markerPulse: {
    position: "absolute",
    width: 38,
    height: 38,
    borderRadius: 19,
    opacity: 0.18,
  },
  markerCircle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 3,
    borderColor: "#fff",
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
      },
      android: { elevation: 5 },
      default: {},
    }),
  },
  markerInitials: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 11,
    letterSpacing: 0.4,
  },

  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(244,247,252,0.92)",
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 28,
  },
  overlayIcon: {
    width: 52,
    height: 52,
    borderRadius: 14,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 12,
  },
  overlayTitle: {
    color: BRAND.text,
    fontWeight: "700",
    fontSize: 16,
    letterSpacing: -0.2,
    marginBottom: 4,
  },
  overlayText: {
    color: BRAND.textSec,
    textAlign: "center",
    fontSize: 13,
    lineHeight: 19,
  },
  overlayMeta: {
    color: BRAND.muted,
    marginTop: 10,
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.4,
    fontWeight: "500",
  },
});

const LIRIE_MAP_STYLE = [
  { featureType: "poi", stylers: [{ visibility: "off" }] },
  { featureType: "poi.medical", stylers: [{ visibility: "on" }] },
  {
    featureType: "poi.medical",
    elementType: "labels.icon",
    stylers: [{ saturation: -60 }],
  },
  { featureType: "transit", stylers: [{ visibility: "simplified" }] },
  {
    featureType: "water",
    elementType: "geometry",
    stylers: [{ color: "#c8dce8" }],
  },
  {
    featureType: "water",
    elementType: "labels.text.fill",
    stylers: [{ color: "#94A3B8" }],
  },
  {
    featureType: "landscape.man_made",
    elementType: "geometry",
    stylers: [{ color: "#f0f2f4" }],
  },
  {
    featureType: "landscape.natural",
    elementType: "geometry",
    stylers: [{ color: "#e4ebe7" }],
  },
  {
    featureType: "landscape.natural.terrain",
    elementType: "geometry",
    stylers: [{ color: "#dde5e0" }],
  },
  {
    featureType: "road.highway",
    elementType: "geometry",
    stylers: [{ color: "#d5dbe0" }],
  },
  {
    featureType: "road.highway",
    elementType: "geometry.stroke",
    stylers: [{ color: "#c3cad0" }],
  },
  {
    featureType: "road.highway",
    elementType: "labels.text.fill",
    stylers: [{ color: "#64748B" }],
  },
  {
    featureType: "road.arterial",
    elementType: "geometry",
    stylers: [{ color: "#e0e5e9" }],
  },
  {
    featureType: "road.local",
    elementType: "geometry",
    stylers: [{ color: "#ebeef1" }],
  },
  {
    featureType: "road",
    elementType: "labels.text.fill",
    stylers: [{ color: "#94A3B8" }],
  },
  {
    featureType: "administrative",
    elementType: "labels.text.fill",
    stylers: [{ color: "#64748B" }],
  },
  {
    featureType: "administrative.locality",
    elementType: "labels.text.fill",
    stylers: [{ color: "#1E293B" }],
  },
  {
    featureType: "administrative.locality",
    elementType: "labels.text.stroke",
    stylers: [{ color: "#ffffff" }, { weight: 3 }],
  },
  {
    featureType: "administrative.neighborhood",
    elementType: "labels.text.fill",
    stylers: [{ color: "#94A3B8" }],
  },
];
