import { StyleSheet, View, type StyleProp, type ViewStyle } from "react-native";
import { GoogleMapsFleetCanvas } from "./maps/GoogleMapsFleetCanvas.web";
import type { CompanyDriverLiveLocation } from "../api/contracts";

type Props = {
  drivers: CompanyDriverLiveLocation[];
  mapHeight?: number;
  containerStyle?: StyleProp<ViewStyle>;
};

const BORDER = "rgba(145, 165, 157, 0.45)";

const styles = StyleSheet.create({
  root: {
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 12,
    overflow: "hidden",
    boxShadow: "0 2px 10px rgba(22, 58, 52, 0.06)",
  },
});

/** Carte entreprise sur le web : Google Maps JS (pas react-native-maps). Même coque visuelle que le natif. */
export function EnterpriseDriversMap({
  drivers,
  mapHeight = 200,
  containerStyle,
}: Props) {
  return (
    <View style={[styles.root, containerStyle]} accessibilityLabel="Carte des chauffeurs en direct">
      <GoogleMapsFleetCanvas drivers={drivers} height={mapHeight} />
    </View>
  );
}
