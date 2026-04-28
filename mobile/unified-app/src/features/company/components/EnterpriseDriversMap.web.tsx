import { useMemo, useState } from "react";
import { Ionicons } from "@expo/vector-icons";
import { Platform, Pressable, StyleSheet, Text, View } from "react-native";
import type { CompanyDriverLiveLocation } from "../api/contracts";

type Props = {
  drivers: CompanyDriverLiveLocation[];
  showTitleRow?: boolean;
};

const STALE_SECONDS_THRESHOLD = 120;
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

const s = StyleSheet.create({
  root: {
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 12,
    overflow: "hidden",
    ...mapCardShadow,
  },
  header: { paddingHorizontal: 10, paddingVertical: 9, borderBottomWidth: 1, borderBottomColor: BORDER, backgroundColor: "#F8FBFA" },
  headerRow: { flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 2 },
  title: { fontSize: 14, fontWeight: "800", color: "#163A34" },
  subtitle: { color: "#5F7369", fontSize: 12, marginTop: 1 },
  subtitleCompact: { color: "#5F7369", fontSize: 11, fontWeight: "600", marginBottom: 2 },
  chipRow: { flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 6 },
  chip: { borderWidth: 1.5, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 },
  chipOn: { borderColor: BRAND, backgroundColor: "rgba(10, 143, 122, 0.1)" },
  chipOff: { borderColor: BORDER, backgroundColor: "#FFFFFF" },
  chipTextOn: { color: BRAND, fontSize: 11, fontWeight: "700" },
  chipTextOff: { color: "#3D4F47", fontSize: 11, fontWeight: "600" },
  listWrap: { padding: 10, backgroundColor: "#EAF3F1", gap: 8 },
  info: { color: "#5F7369", fontSize: 12, lineHeight: 17 },
  driverCard: {
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.4)",
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
    backgroundColor: "#FFFFFF",
  },
  driverName: { fontSize: 14, fontWeight: "800", color: "#163A34" },
  driverLine: { color: "#5F7369", fontSize: 11, marginTop: 2, lineHeight: 16 },
  more: { color: "#0A7A6A", fontSize: 12, fontWeight: "600" },
});

type DriverMapFilter = "all" | "available" | "en_mission" | "offline";

function resolveDriverStatus(driver: CompanyDriverLiveLocation): Exclude<DriverMapFilter, "all"> {
  const lastSeen = Number(driver.last_seen_seconds);
  const byAge = Number.isFinite(lastSeen) && lastSeen > STALE_SECONDS_THRESHOLD;
  const byStatus = driver.location_status === "stale" || driver.location_status === "offline";
  if (byAge || byStatus) return "offline";
  if (driver.mission_id != null) return "en_mission";
  return "available";
}

function statusLabel(s: ReturnType<typeof resolveDriverStatus>): string {
  if (s === "available") return "Disponible";
  if (s === "en_mission") return "En mission";
  return "Hors ligne";
}

export function EnterpriseDriversMap({ drivers, showTitleRow = true }: Props) {
  const [filter, setFilter] = useState<DriverMapFilter>("all");
  const filteredDrivers = useMemo(
    () => drivers.filter((driver) => filter === "all" || resolveDriverStatus(driver) === filter),
    [drivers, filter]
  );

  return (
    <View
      style={[s.root, !showTitleRow && { borderTopLeftRadius: 0, borderTopRightRadius: 0, borderTopWidth: 0 }]}
      accessibilityLabel="Liste des chauffeurs (aperçu web)"
    >
      <View style={s.header}>
        {showTitleRow ? (
          <>
            <View style={s.headerRow}>
              <Ionicons name="map-outline" size={15} color={BRAND} />
              <Text style={s.title}>Carte live · chauffeurs</Text>
            </View>
            <Text style={s.subtitle}>
              {filteredDrivers.length} sur {drivers.length} après filtre
            </Text>
          </>
        ) : (
          <Text style={s.subtitleCompact}>
            {filteredDrivers.length} / {drivers.length} après filtre
          </Text>
        )}
        <View style={[s.chipRow, !showTitleRow && { marginTop: 2 }]}>
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
                style={({ pressed }) => [s.chip, on ? s.chipOn : s.chipOff, pressed && { opacity: 0.9 }]}
                accessibilityState={{ selected: on }}
              >
                <Text style={on ? s.chipTextOn : s.chipTextOff}>{option.label}</Text>
              </Pressable>
            );
          })}
        </View>
      </View>

      <View style={s.listWrap}>
        {showTitleRow ? (
          <Text style={s.info}>
            La carte n’est pas disponible sur le web. Aperçu des positions ci-dessous — pour la vue cartographique, utilisez
            l’application mobile.
          </Text>
        ) : null}
        {filteredDrivers.slice(0, 6).map((driver) => {
          const status = resolveDriverStatus(driver);
          return (
            <View key={driver.driver_id} style={s.driverCard}>
              <Text style={s.driverName}>Chauffeur #{driver.driver_id}</Text>
              <Text style={s.driverLine}>
                {statusLabel(status)}
                {driver.mission_id ? ` · mission #${driver.mission_id}` : " · aucune mission"}
              </Text>
              <Text style={s.driverLine}>
                {driver.latitude.toFixed(5)}, {driver.longitude.toFixed(5)}
              </Text>
            </View>
          );
        })}
        {filteredDrivers.length > 6 ? (
          <Text style={s.more}>+ {filteredDrivers.length - 6} autre(s) chauffeur(s)</Text>
        ) : null}
      </View>
    </View>
  );
}
