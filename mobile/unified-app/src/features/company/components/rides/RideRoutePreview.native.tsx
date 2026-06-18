import { StyleSheet, View } from "react-native";
import { RideRouteStats } from "./RideRouteStats";

export type RideRoutePreviewProps = {
  pickupLat: number | null;
  pickupLng: number | null;
  dropoffLat: number | null;
  dropoffLng: number | null;
  routePoints: readonly { lat: number; lng: number }[];
  distanceMeters: number | null;
  durationSeconds: number | null;
  /** Libellé du type d'itinéraire (ex. "Le plus rapide"). */
  routeKind?: string;
};

/**
 * Version native : la carte a été retirée pour gagner de la hauteur sur mobile.
 * On garde uniquement les indicateurs (distance, temps estimé, itinéraire).
 */
export function RideRoutePreview({
  distanceMeters,
  durationSeconds,
  routeKind = "Le plus rapide",
}: RideRoutePreviewProps) {
  return (
    <View style={s.wrap}>
      <RideRouteStats
        distanceMeters={distanceMeters}
        durationSeconds={durationSeconds}
        routeKind={routeKind}
      />
    </View>
  );
}

const s = StyleSheet.create({
  wrap: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.22)",
    backgroundColor: "#FFFFFF",
    overflow: "hidden" as const,
  },
});
