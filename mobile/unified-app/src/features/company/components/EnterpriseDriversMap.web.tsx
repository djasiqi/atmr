import { useMemo, useState } from "react";
import { Ionicons } from "@expo/vector-icons";
import { Pressable, StyleSheet, View, type StyleProp, type ViewStyle } from "react-native";
import { brandPrimary, brandText } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { GoogleMapsFleetCanvas } from "./maps/GoogleMapsFleetCanvas.web";
import type { CompanyDriverLiveLocation } from "../api/contracts";
import { resolveDriverStatus } from "../utils/companyDriverMapStatus";

type Props = {
  drivers: CompanyDriverLiveLocation[];
  showTitleRow?: boolean;
  /** Aligné sur la variante native (dashboard). */
  mapHeight?: number;
  containerStyle?: StyleProp<ViewStyle>;
};
const BRAND = "#0A8F7A";
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

function readGoogleMapsApiKey(): string {
  return typeof process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY === "string"
    ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.trim()
    : "";
}

/** Carte entreprise sur le web : Google Maps JS (pas react-native-maps). */
export function EnterpriseDriversMap({
  drivers,
  showTitleRow = true,
  mapHeight = 200,
  containerStyle,
}: Props) {
  const apiKey = useMemo(() => readGoogleMapsApiKey(), []);
  const [filter, setFilter] = useState<DriverMapFilter>("all");
  const filteredDrivers = useMemo(
    () => drivers.filter((driver) => filter === "all" || resolveDriverStatus(driver) === filter),
    [drivers, filter]
  );

  /**
   * Sans clé API : uniquement le fallback `GoogleMapsFleetCanvas`.
   * - Dashboard : `containerStyle` (shell `mapHeroShell` à l’extérieur).
   * - Autres écrans : cadre `styles.root` si pas de `containerStyle`.
   */
  if (!apiKey) {
    return (
      <View style={containerStyle ?? styles.root} accessibilityLabel="Carte des chauffeurs en direct">
        <GoogleMapsFleetCanvas drivers={drivers} height={mapHeight} />
      </View>
    );
  }

  return (
    <View
      style={[
        styles.root,
        !showTitleRow && { borderTopLeftRadius: 0, borderTopRightRadius: 0, borderTopWidth: 0 },
        containerStyle,
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
      <GoogleMapsFleetCanvas drivers={filteredDrivers} height={mapHeight} />
    </View>
  );
}
