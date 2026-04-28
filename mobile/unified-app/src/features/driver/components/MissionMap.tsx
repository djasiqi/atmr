import { Text, View } from "react-native";
import MapView, { Marker, Region } from "react-native-maps";
import MapViewDirections from "react-native-maps-directions";

type MissionMapProps = {
  pickupLat?: number | null;
  pickupLng?: number | null;
  dropoffLat?: number | null;
  dropoffLng?: number | null;
  height?: number;
};

function toFinite(input: unknown): number | null {
  return typeof input === "number" && Number.isFinite(input) ? input : null;
}

export function MissionMap(props: MissionMapProps) {
  const pickupLat = toFinite(props.pickupLat);
  const pickupLng = toFinite(props.pickupLng);
  const dropoffLat = toFinite(props.dropoffLat);
  const dropoffLng = toFinite(props.dropoffLng);

  const firstLat = pickupLat ?? dropoffLat;
  const firstLng = pickupLng ?? dropoffLng;

  if (firstLat == null || firstLng == null) {
    return (
      <View style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12 }}>
        <Text>Carte indisponible (coordonnees manquantes).</Text>
      </View>
    );
  }

  const region: Region = {
    latitude: firstLat,
    longitude: firstLng,
    latitudeDelta: 0.04,
    longitudeDelta: 0.04,
  };
  const canRenderRoute =
    pickupLat != null &&
    pickupLng != null &&
    dropoffLat != null &&
    dropoffLng != null &&
    typeof process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY === "string" &&
    process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.length > 0;

  return (
    <MapView
      style={{ height: props.height ?? 220, borderRadius: 10 }}
      initialRegion={region}
      showsUserLocation
      loadingEnabled
    >
      {pickupLat != null && pickupLng != null ? (
        <Marker coordinate={{ latitude: pickupLat, longitude: pickupLng }} title="Pickup" />
      ) : null}
      {dropoffLat != null && dropoffLng != null ? (
        <Marker coordinate={{ latitude: dropoffLat, longitude: dropoffLng }} title="Dropoff" />
      ) : null}
      {canRenderRoute ? (
        <MapViewDirections
          origin={{ latitude: pickupLat as number, longitude: pickupLng as number }}
          destination={{ latitude: dropoffLat as number, longitude: dropoffLng as number }}
          apikey={process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY as string}
          strokeWidth={4}
          strokeColor="#0a7ea4"
        />
      ) : null}
    </MapView>
  );
}

