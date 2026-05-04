import { useMemo, useState } from "react";
import { Ionicons } from "@expo/vector-icons";
import { Platform, Pressable, StyleSheet, View } from "react-native";
import { brandPrimary, brandText } from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
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
  subtitleSpacing: { marginTop: 1 },
  subtitleCompactSpacing: { marginBottom: 2 },
  chipRow: { flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 6 },
  chip: { borderWidth: 1.5, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 },
  chipOn: { borderColor: BRAND, backgroundColor: "rgba(10, 143, 122, 0.1)" },
  chipOff: { borderColor: BORDER, backgroundColor: "#FFFFFF" },
  listWrap: { padding: 10, backgroundColor: "#EAF3F1", gap: 8 },
  infoLineHeight: { lineHeight: 17 },
  driverCard: {
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.4)",
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
    backgroundColor: "#FFFFFF",
  },
  driverLineSpacing: { marginTop: 2, lineHeight: 16 },
  moreSpacing: { marginTop: 4 },
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
              <AppText variant="sectionTitle">Carte live · chauffeurs</AppText>
            </View>
            <AppText variant="caption" style={s.subtitleSpacing}>
              {filteredDrivers.length} sur {drivers.length} après filtre
            </AppText>
          </>
        ) : (
          <AppText variant="label" style={s.subtitleCompactSpacing}>
            {filteredDrivers.length} / {drivers.length} après filtre
          </AppText>
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
                <AppText variant={on ? "label" : "caption"} style={{ color: on ? brandPrimary : brandText }}>
                  {option.label}
                </AppText>
              </Pressable>
            );
          })}
        </View>
      </View>

      <View style={s.listWrap}>
        {showTitleRow ? (
          <AppText variant="bodyMuted" style={s.infoLineHeight}>
            La carte n’est pas disponible sur le web. Aperçu des positions ci-dessous — pour la vue cartographique, utilisez
            l’application mobile.
          </AppText>
        ) : null}
        {filteredDrivers.slice(0, 6).map((driver) => {
          const status = resolveDriverStatus(driver);
          return (
            <View key={driver.driver_id} style={s.driverCard}>
              <AppText variant="sectionTitle">Chauffeur #{driver.driver_id}</AppText>
              <AppText variant="caption" style={s.driverLineSpacing}>
                {statusLabel(status)}
                {driver.mission_id ? ` · mission #${driver.mission_id}` : " · aucune mission"}
              </AppText>
              <AppText variant="caption" style={s.driverLineSpacing}>
                {driver.latitude.toFixed(5)}, {driver.longitude.toFixed(5)}
              </AppText>
            </View>
          );
        })}
        {filteredDrivers.length > 6 ? (
          <AppText variant="label" style={[s.moreSpacing, { color: brandPrimary }]}>
            + {filteredDrivers.length - 6} autre(s) chauffeur(s)
          </AppText>
        ) : null}
      </View>
    </View>
  );
}
