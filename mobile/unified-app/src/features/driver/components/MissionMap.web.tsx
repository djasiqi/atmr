import { View } from "react-native";
import { AppText } from "../../../design/ui/AppText";

type MissionMapProps = {
  pickupLat?: number | null;
  pickupLng?: number | null;
  dropoffLat?: number | null;
  dropoffLng?: number | null;
  height?: number;
};

function formatCoord(lat?: number | null, lng?: number | null): string {
  if (typeof lat !== "number" || typeof lng !== "number") return "N/A";
  return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
}

export function MissionMap(props: MissionMapProps) {
  return (
    <View
      style={{
        borderWidth: 1,
        borderColor: "#ddd",
        borderRadius: 10,
        padding: 12,
        minHeight: props.height ?? 220,
        justifyContent: "center",
        gap: 6,
      }}
    >
      <AppText variant="sectionTitle">Carte mission (web)</AppText>
      <AppText variant="body">Le rendu cartographique natif n&apos;est pas disponible sur web.</AppText>
      <AppText variant="body">Pickup: {formatCoord(props.pickupLat, props.pickupLng)}</AppText>
      <AppText variant="body">Dropoff: {formatCoord(props.dropoffLat, props.dropoffLng)}</AppText>
    </View>
  );
}

