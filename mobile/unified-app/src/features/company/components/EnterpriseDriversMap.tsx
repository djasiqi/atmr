import { useMemo } from "react";
import { Platform, StyleSheet, View, type StyleProp, type ViewStyle } from "react-native";
import MapView, { Marker, PROVIDER_GOOGLE } from "react-native-maps";
import ClusteredMapView from "react-native-map-clustering";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import type { CompanyDriverLiveLocation } from "../api/contracts";
import { isDriverPositionStale } from "../utils/companyDriverMapStatus";

type Props = {
  drivers: CompanyDriverLiveLocation[];
  /** Hauteur de la zone carte (dp). Défaut : 200. */
  mapHeight?: number;
  /** Fusion avec le conteneur racine (coins, bordure…). */
  containerStyle?: StyleProp<ViewStyle>;
};

const BORDER = "rgba(145, 165, 157, 0.45)";

const mapCardShadow = Platform.select({
  web: { boxShadow: "0 2px 10px rgba(22, 58, 52, 0.06)" },
  default: {
    shadowColor: "#163A34",
    shadowOpacity: 0.06,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
});

const styles = StyleSheet.create({
  root: {
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 12,
    overflow: "hidden",
    ...mapCardShadow,
  },
  map: { width: "100%" as const },
});

function computeRegion(drivers: CompanyDriverLiveLocation[]) {
  if (drivers.length === 0) {
    return {
      latitude: 48.8566,
      longitude: 2.3522,
      latitudeDelta: 0.2,
      longitudeDelta: 0.2,
    };
  }
  const latitudes = drivers.map((driver) => driver.latitude);
  const longitudes = drivers.map((driver) => driver.longitude);
  const minLat = Math.min(...latitudes);
  const maxLat = Math.max(...latitudes);
  const minLng = Math.min(...longitudes);
  const maxLng = Math.max(...longitudes);
  const centerLat = (minLat + maxLat) / 2;
  const centerLng = (minLng + maxLng) / 2;
  return {
    latitude: centerLat,
    longitude: centerLng,
    latitudeDelta: Math.max(0.03, (maxLat - minLat) * 1.8),
    longitudeDelta: Math.max(0.03, (maxLng - minLng) * 1.8),
  };
}

export function EnterpriseDriversMap({
  drivers,
  mapHeight = 200,
  containerStyle,
}: Props) {
  const clusteringEnabled = isFeatureEnabled("company_mobile_map_clustering_enabled");
  const region = useMemo(() => computeRegion(drivers), [drivers]);
  const mapDims = { height: mapHeight };
  return (
    <View
      style={[styles.root, containerStyle]}
      accessibilityLabel="Carte des chauffeurs en direct"
    >
      {clusteringEnabled ? (
        <ClusteredMapView
          provider={PROVIDER_GOOGLE}
          style={[styles.map, mapDims]}
          initialRegion={region}
          region={region}
          radius={45}
        >
          {drivers.map((driver) => {
            const isStale = isDriverPositionStale(driver);
            return (
              <Marker
                key={driver.driver_id}
                coordinate={{ latitude: driver.latitude, longitude: driver.longitude }}
                title={`Driver #${driver.driver_id}`}
                description={driver.mission_id ? `Mission #${driver.mission_id}` : "Aucune mission"}
                pinColor={isStale ? "#9e9e9e" : "#2e7d32"}
                opacity={isStale ? 0.7 : 1}
              />
            );
          })}
        </ClusteredMapView>
      ) : (
        <MapView provider={PROVIDER_GOOGLE} style={[styles.map, mapDims]} initialRegion={region} region={region}>
          {drivers.map((driver) => {
            const isStale = isDriverPositionStale(driver);
            return (
              <Marker
                key={driver.driver_id}
                coordinate={{ latitude: driver.latitude, longitude: driver.longitude }}
                title={`Driver #${driver.driver_id}`}
                description={driver.mission_id ? `Mission #${driver.mission_id}` : "Aucune mission"}
                pinColor={isStale ? "#9e9e9e" : "#2e7d32"}
                opacity={isStale ? 0.7 : 1}
              />
            );
          })}
        </MapView>
      )}
    </View>
  );
}
