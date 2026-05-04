import { useMemo, useState } from "react";
import { Ionicons } from "@expo/vector-icons";
import { Platform, Pressable, StyleSheet, View } from "react-native";
import { brandPrimary, brandText } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import MapView, { Marker, PROVIDER_GOOGLE } from "react-native-maps";
import ClusteredMapView from "react-native-map-clustering";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import type { CompanyDriverLiveLocation } from "../api/contracts";
import { isDriverPositionStale, resolveDriverStatus } from "../utils/companyDriverMapStatus";

type Props = {
  drivers: CompanyDriverLiveLocation[];
  showTitleRow?: boolean;
};
const BRAND = "#0A8F7A";
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
  header: {
    paddingHorizontal: 10,
    paddingVertical: 9,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
    backgroundColor: "#F8FBFA",
  },
  headerRow: { flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 2 },
  subtitleSpacing: { marginTop: 1 },
  subtitleCompactSpacing: { marginBottom: 2 },
  chipRow: { flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 6 },
  chip: {
    borderWidth: 1.5,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  chipOn: { borderColor: BRAND, backgroundColor: "rgba(10, 143, 122, 0.1)" },
  chipOff: { borderColor: BORDER, backgroundColor: "#FFFFFF" },
  map: { height: 200, width: "100%" as const },
});

type DriverMapFilter = "all" | "available" | "en_mission" | "offline";

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

export function EnterpriseDriversMap({ drivers, showTitleRow = true }: Props) {
  const [filter, setFilter] = useState<DriverMapFilter>("all");
  const clusteringEnabled = isFeatureEnabled("company_mobile_map_clustering_enabled");
  const filteredDrivers = useMemo(
    () => drivers.filter((driver) => filter === "all" || resolveDriverStatus(driver) === filter),
    [drivers, filter]
  );
  const region = useMemo(() => computeRegion(filteredDrivers), [filteredDrivers]);
  return (
    <View
      style={[
        styles.root,
        !showTitleRow && { borderTopLeftRadius: 0, borderTopRightRadius: 0, borderTopWidth: 0 },
      ]}
      accessibilityLabel="Carte des chauffeurs en direct"
    >
      <View style={styles.header}>
        {showTitleRow ? (
          <>
            <View style={styles.headerRow}>
              <Ionicons name="map-outline" size={15} color={BRAND} />
              <AppText variant="sectionTitle">Carte live · chauffeurs</AppText>
            </View>
            <AppText variant="caption" style={styles.subtitleSpacing}>
              {filteredDrivers.length} sur {drivers.length} après filtre
            </AppText>
          </>
        ) : (
          <AppText variant="label" style={styles.subtitleCompactSpacing}>
            {filteredDrivers.length} / {drivers.length} après filtre
          </AppText>
        )}
        <View style={[styles.chipRow, !showTitleRow && { marginTop: 2 }]}>
          {[
            { key: "all", label: "Tous" },
            { key: "available", label: "Disponibles" },
            { key: "en_mission", label: "En mission" },
            { key: "offline", label: "Hors ligne" },
          ].map((option) => {
            const on = filter === option.key;
            return (
              <Pressable
                key={option.key}
                onPress={() => setFilter(option.key as DriverMapFilter)}
                style={({ pressed }) => [
                  styles.chip,
                  on ? styles.chipOn : styles.chipOff,
                  pressed && { opacity: 0.88 },
                ]}
                accessibilityState={{ selected: on }}
              >
                <AppText variant={on ? "label" : "caption"} style={{ color: on ? brandPrimary : brandText }}>
                  {option.label}
                </AppText>
              </Pressable>
            );
          })}
        </View>
      </View>
      {clusteringEnabled ? (
        <ClusteredMapView
          provider={PROVIDER_GOOGLE}
          style={styles.map}
          initialRegion={region}
          region={region}
          radius={45}
        >
          {filteredDrivers.map((driver) => {
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
        <MapView provider={PROVIDER_GOOGLE} style={styles.map} initialRegion={region} region={region}>
          {filteredDrivers.map((driver) => {
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
