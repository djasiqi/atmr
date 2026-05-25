import { StyleSheet, View } from "react-native";
import { RideRouteStats } from "./RideRouteStats";

export type RideRoutePreviewProps = {
  pickupLat: number | null;
  pickupLng: number | null;
  dropoffLat: number | null;
  dropoffLng: number | null;
  routePoints: ReadonlyArray<{ lat: number; lng: number }>;
  distanceMeters: number | null;
  durationSeconds: number | null;
  /** Libellé du type d'itinéraire (ex. "Le plus rapide"). */
  routeKind?: string;
};

/**
 * Indicateurs de trajet (distance, temps estimé, itinéraire) sans carte visuelle.
 * Les props géographiques sont conservées pour rétro-compatibilité d'API mais ne sont pas utilisées
 * dans le rendu : la carte a été retirée (peu de valeur ajoutée tant qu'on n'a pas un vrai aperçu).
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
