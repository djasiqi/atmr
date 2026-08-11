import { useEffect, useMemo, useState } from "react";
import { FlatList, Platform, Pressable, ScrollView, StyleSheet, TextInput, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { EnterpriseBottomSheet } from "../EnterpriseBottomSheet";
import { formatMissionTime } from "../../dashboard/companyDashboardMissionUi";
import { resolveGoogleMapsNativeApiKey } from "../../../../config/googleMapsKeys";
import {
  driverFleetMarkerInitials,
  resolveDriverDisplayName,
} from "../../utils/companyDriverMapStatus";
import type { FleetDriverMapItem } from "./fleetMapTypes";
import type { FleetOperationalStatus } from "./mapStatusTheme";
import { FLEET_MAP_COLORS, FLEET_STATUS_THEME } from "./mapStatusTheme";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import {
  formatDriverLocationPresenceLabel,
  matchesFleetGpsFilter,
  resolveDriverLocationPresence,
  type FleetGpsFilter,
} from "./driverLocationPresence";

type Props = {
  visible: boolean;
  drivers: FleetDriverMapItem[];
  query: string;
  onChangeQuery: (query: string) => void;
  onSelectDriver: (driver: FleetDriverMapItem) => void;
  onClose: () => void;
};

type SearchStatusFilter = "all" | "available" | "busy" | "assigned" | "break" | "delayed" | "offline";

const STATUS_CHIPS: SearchStatusFilter[] = [
  "all",
  "available",
  "busy",
  "assigned",
  "break",
  "delayed",
  "offline",
];

const GPS_CHIPS: { id: FleetGpsFilter; label: string }[] = [
  { id: "all", label: "GPS : Tous" },
  { id: "live", label: "En direct" },
  { id: "not_recent", label: "Non récent" },
];

const ETA_CACHE_TTL_MS = 2 * 60 * 1000;
const etaRouteCache = new Map<string, { minutes: number; distanceKm: number; atMs: number }>();

function resolveSearchFilterLabel(filter: SearchStatusFilter): string {
  if (filter === "all") return "Tous";
  if (filter === "delayed") return "Retard";
  return FLEET_STATUS_THEME[filter].label;
}

/** Statut métier uniquement — ne mélange pas la présence GPS. */
function normalizeOperationalStatus(status: FleetOperationalStatus): SearchStatusFilter {
  if (status === "incident" || status === "emergency") return "delayed";
  if (status === "delayed") return "delayed";
  if (status === "busy" || status === "assigned" || status === "available" || status === "break" || status === "offline") {
    return status;
  }
  // constrained / last_known legacy : pas un filtre métier « offline »
  if (status === "constrained" || status === "last_known") return "available";
  return "available";
}

function isDriverInStatusFilter(driver: FleetDriverMapItem, filter: SearchStatusFilter): boolean {
  if (filter === "all") return true;
  if (filter === "offline") {
    return driver.enrichment.operationalStatus === "offline";
  }
  return normalizeOperationalStatus(driver.enrichment.operationalStatus) === filter;
}

function resolveDistanceLabel(driver: FleetDriverMapItem, liveDistanceKm?: number): string {
  if (Number.isFinite(liveDistanceKm) && (liveDistanceKm ?? 0) > 0) {
    const km = liveDistanceKm ?? 0;
    return km >= 10 ? `${Math.round(km)} km` : `${km.toFixed(1).replace(".", ",")} km`;
  }
  const liveDistance = driver.enrichment.distanceLabel?.trim();
  if (liveDistance) return liveDistance;
  const km = Number(driver.enrichment.linkedMission?.route_distance_km);
  if (Number.isFinite(km) && km > 0) return `${km.toFixed(1).replace(".", ",")} km`;
  return "—";
}

function resolveEtaTargetPoint(driver: FleetDriverMapItem): { lat: number; lon: number } | null {
  const mission = driver.enrichment.linkedMission;
  if (!mission) return null;
  const towardDropoff = mission.status === "in_progress" || mission.status === "arrived";
  const lat = towardDropoff ? mission.dropoff_lat : mission.pickup_lat;
  const lon = towardDropoff ? mission.dropoff_lon : mission.pickup_lon;
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return { lat: Number(lat), lon: Number(lon) };
}

function buildEtaRouteKey(driver: FleetDriverMapItem, target: { lat: number; lon: number }): string {
  return [
    driver.latitude.toFixed(4),
    driver.longitude.toFixed(4),
    target.lat.toFixed(4),
    target.lon.toFixed(4),
  ].join(">");
}

async function fetchRouteEtaMinutes(
  origin: { lat: number; lon: number },
  destination: { lat: number; lon: number },
  apiKey: string
): Promise<{ minutes: number; distanceKm: number } | null> {
  const params = new URLSearchParams({
    origin: `${origin.lat},${origin.lon}`,
    destination: `${destination.lat},${destination.lon}`,
    mode: "driving",
    region: "ch",
    departure_time: "now",
    traffic_model: "best_guess",
    key: apiKey,
  });
  try {
    const response = await fetch(`https://maps.googleapis.com/maps/api/directions/json?${params.toString()}`);
    const data = (await response.json()) as {
      status?: string;
      routes?: {
        legs?: {
          duration?: { value?: number };
          duration_in_traffic?: { value?: number };
          distance?: { value?: number };
        }[];
      }[];
    };
    if (data.status !== "OK") return null;
    const legs = data.routes?.[0]?.legs ?? [];
    if (legs.length === 0) return null;
    const routeStats = legs.reduce((acc, leg) => {
      const traffic = Number(leg.duration_in_traffic?.value);
      const normal = Number(leg.duration?.value);
      const value = Number.isFinite(traffic) && traffic > 0 ? traffic : normal;
      const distanceMeters = Number(leg.distance?.value);
      return {
        seconds: acc.seconds + (Number.isFinite(value) ? value : 0),
        meters: acc.meters + (Number.isFinite(distanceMeters) ? distanceMeters : 0),
      };
    }, { seconds: 0, meters: 0 });
    if (!Number.isFinite(routeStats.seconds) || routeStats.seconds <= 0) return null;
    return {
      minutes: Math.max(1, Math.round(routeStats.seconds / 60)),
      distanceKm:
        Number.isFinite(routeStats.meters) && routeStats.meters > 0
          ? routeStats.meters / 1000
          : 0,
    };
  } catch {
    return null;
  }
}

function resolveSecondLine(driver: FleetDriverMapItem): string {
  const address =
    driver.enrichment.currentAddress?.trim() ??
    driver.enrichment.linkedMission?.pickup_label?.trim() ??
    driver.enrichment.linkedMission?.dropoff_label?.trim() ??
    null;
  const gpsLabel = formatDriverLocationPresenceLabel(resolveDriverLocationPresence(driver));
  if (address) return `${address}\nGPS : ${gpsLabel}`;
  return `GPS : ${gpsLabel}`;
}

function extractMinutes(value: string | null | undefined): number | null {
  if (!value) return null;
  const trimmed = value.trim().toLowerCase();
  if (trimmed.length === 0) return null;
  if (trimmed.includes("imminent") || trimmed.includes("en route")) return null;
  if (trimmed.startsWith("+")) return null;
  const match = trimmed.match(/(\d+)\s*min/);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}

function resolveEtaLine(driver: FleetDriverMapItem, liveRouteEtaMinutes?: number): string | null {
  const mission = driver.enrichment.linkedMission;
  if (!mission) return null;
  const towardDropoff = mission.status === "in_progress" || mission.status === "arrived";
  const targetLabel = towardDropoff ? "au dropoff" : "au pickup";

  if (Number.isFinite(liveRouteEtaMinutes) && (liveRouteEtaMinutes ?? 0) > 0) {
    return `Arrivée ${targetLabel} : ${Math.round(liveRouteEtaMinutes ?? 0)} min`;
  }

  const minutes = Number(mission.route_duration_min);
  if (Number.isFinite(minutes) && minutes > 0) {
    return `Arrivée ${targetLabel} : ${Math.round(minutes)} min`;
  }
  const etaMinutes = extractMinutes(driver.enrichment.etaLabel);
  if (etaMinutes != null) return `Arrivée ${targetLabel} : ${etaMinutes} min`;
  const scheduled = formatMissionTime(mission.scheduled_at);
  if (scheduled) return `Arrivée ${targetLabel} : ${scheduled}`;
  return null;
}

function toRgba(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  const parsed =
    value.length === 3
      ? value
          .split("")
          .map((ch) => ch + ch)
          .join("")
      : value;
  const int = Number.parseInt(parsed, 16);
  if (!Number.isFinite(int)) return `rgba(15,23,42,${alpha})`;
  const r = (int >> 16) & 255;
  const g = (int >> 8) & 255;
  const b = int & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function MapDriverSearchSheet({
  visible,
  drivers,
  query,
  onChangeQuery,
  onSelectDriver,
  onClose,
}: Props) {
  const [statusFilter, setStatusFilter] = useState<SearchStatusFilter>("all");
  const [gpsFilter, setGpsFilter] = useState<FleetGpsFilter>("all");
  const [liveEtaByDriverId, setLiveEtaByDriverId] = useState<Record<number, number>>({});
  const [liveDistanceByDriverId, setLiveDistanceByDriverId] = useState<Record<number, number>>({});
  const nativeMapsApiKey = useMemo(() => resolveGoogleMapsNativeApiKey(), []);
  const matches = useMemo(() => {
    const search = query.trim().toLowerCase();
    const sorted = [...drivers].sort((a, b) =>
      resolveDriverDisplayName(a).localeCompare(resolveDriverDisplayName(b), "fr")
    );
    if (!search) return sorted.slice(0, 40);
    return sorted
      .filter((driver) => {
        const name = resolveDriverDisplayName(driver).toLowerCase();
        const id = String(driver.driver_id);
        const missionId = driver.enrichment.linkedMission?.mission_id ?? driver.mission_id;
        return (
          name.includes(search) ||
          id.includes(search) ||
          (missionId != null && String(missionId).includes(search))
        );
      })
      .slice(0, 40);
  }, [drivers, query]);
  const statusCounts = useMemo(() => {
    const counts: Record<SearchStatusFilter, number> = {
      all: drivers.length,
      available: 0,
      busy: 0,
      assigned: 0,
      break: 0,
      delayed: 0,
      offline: 0,
    };
    for (const driver of drivers) {
      if (driver.enrichment.operationalStatus === "offline") {
        counts.offline += 1;
        continue;
      }
      const normalized = normalizeOperationalStatus(driver.enrichment.operationalStatus);
      if (normalized !== "offline") counts[normalized] += 1;
    }
    return counts;
  }, [drivers]);
  const results = useMemo(
    () =>
      matches.filter((driver) => {
        if (!isDriverInStatusFilter(driver, statusFilter)) return false;
        const presence = resolveDriverLocationPresence(driver).presence;
        return matchesFleetGpsFilter(presence, gpsFilter);
      }),
    [matches, statusFilter, gpsFilter]
  );
  const trimmed = query.trim();

  useEffect(() => {
    if (!visible || Platform.OS === "web" || !nativeMapsApiKey) return;
    const candidates = results
      .slice(0, 14)
      .map((driver) => {
        const target = resolveEtaTargetPoint(driver);
        if (!target) return null;
        const key = buildEtaRouteKey(driver, target);
        return { driver, target, key };
      })
      .filter((v): v is { driver: FleetDriverMapItem; target: { lat: number; lon: number }; key: string } => Boolean(v));
    if (candidates.length === 0) return;

    let cancelled = false;
    const now = Date.now();
    const warmEta: Record<number, number> = {};
    const warmDistance: Record<number, number> = {};
    const toFetch = candidates.filter((entry) => {
      const cached = etaRouteCache.get(entry.key);
      if (cached && now - cached.atMs < ETA_CACHE_TTL_MS) {
        warmEta[entry.driver.driver_id] = cached.minutes;
        if (cached.distanceKm > 0) {
          warmDistance[entry.driver.driver_id] = cached.distanceKm;
        }
        return false;
      }
      return true;
    });
    if (Object.keys(warmEta).length > 0) {
      setLiveEtaByDriverId((prev) => ({ ...prev, ...warmEta }));
    }
    if (Object.keys(warmDistance).length > 0) {
      setLiveDistanceByDriverId((prev) => ({ ...prev, ...warmDistance }));
    }

    if (toFetch.length === 0) return;
    void (async () => {
      const fetchedEta: Record<number, number> = {};
      const fetchedDistance: Record<number, number> = {};
      await Promise.all(
        toFetch.map(async ({ driver, target, key }) => {
          const metrics = await fetchRouteEtaMinutes(
            { lat: driver.latitude, lon: driver.longitude },
            { lat: target.lat, lon: target.lon },
            nativeMapsApiKey
          );
          if (metrics == null) return;
          etaRouteCache.set(key, {
            minutes: metrics.minutes,
            distanceKm: metrics.distanceKm,
            atMs: Date.now(),
          });
          fetchedEta[driver.driver_id] = metrics.minutes;
          if (metrics.distanceKm > 0) {
            fetchedDistance[driver.driver_id] = metrics.distanceKm;
          }
        })
      );
      if (cancelled || Object.keys(fetchedEta).length === 0) return;
      setLiveEtaByDriverId((prev) => ({ ...prev, ...fetchedEta }));
      if (Object.keys(fetchedDistance).length > 0) {
        setLiveDistanceByDriverId((prev) => ({ ...prev, ...fetchedDistance }));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [nativeMapsApiKey, results, visible]);

  return (
    <EnterpriseBottomSheet
      visible={visible}
      onClose={onClose}
      title="Rechercher un chauffeur"
      subtitle="Nom chauffeur ou mission"
      scrollable={false}
    >
      <View style={s.headerRow}>
        <View style={s.searchWrap}>
          <Ionicons name="search-outline" size={18} color={FLEET_MAP_COLORS.textMuted} />
          <TextInput
            value={query}
            onChangeText={onChangeQuery}
            placeholder="Nom chauffeur ou mission..."
            placeholderTextColor={FLEET_MAP_COLORS.textMuted}
            style={s.searchInput}
            autoCapitalize="words"
            autoCorrect={false}
            clearButtonMode="while-editing"
            accessibilityLabel="Rechercher un chauffeur sur la carte"
            returnKeyType="search"
          />
          {trimmed.length > 0 ? (
            <Pressable
              onPress={() => onChangeQuery("")}
              hitSlop={8}
              accessibilityRole="button"
              accessibilityLabel="Effacer la recherche"
            >
              <Ionicons name="close-circle" size={20} color={FLEET_MAP_COLORS.textMuted} />
            </Pressable>
          ) : null}
        </View>
        <Pressable
          onPress={onClose}
          style={({ pressed }) => [s.closeBtn, pressed && s.closeBtnPressed]}
          accessibilityRole="button"
          accessibilityLabel="Fermer la recherche"
        >
          <Ionicons name="close" size={18} color={FLEET_MAP_COLORS.text} />
        </Pressable>
      </View>

      <ScrollView
        horizontal
        style={s.chipsWrap}
        contentContainerStyle={s.chipsContent}
        keyboardShouldPersistTaps="handled"
        showsHorizontalScrollIndicator={false}
      >
        {STATUS_CHIPS.map((filter) => {
          const active = statusFilter === filter;
          const count = statusCounts[filter];
          const dotColor =
            filter === "all"
              ? FLEET_MAP_COLORS.text
              : filter === "delayed"
                ? FLEET_MAP_COLORS.delayed
                : FLEET_STATUS_THEME[filter].fill;
          return (
            <Pressable
              key={filter}
              onPress={() => setStatusFilter(filter)}
              style={({ pressed }) => [s.chip, active && s.chipActive, pressed && s.chipPressed]}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
            >
              <View style={[s.chipDot, { backgroundColor: dotColor }]} />
              <AppText variant="caption" style={[s.chipLabel, active && s.chipLabelActive]}>
                {resolveSearchFilterLabel(filter)}
              </AppText>
              <View style={[s.chipCount, active && s.chipCountActive]}>
                <AppText variant="caption" style={[s.chipCountText, active && s.chipCountTextActive]}>
                  {count}
                </AppText>
              </View>
            </Pressable>
          );
        })}
      </ScrollView>

      <ScrollView
        horizontal
        style={s.chipsWrap}
        contentContainerStyle={s.chipsContent}
        keyboardShouldPersistTaps="handled"
        showsHorizontalScrollIndicator={false}
      >
        {GPS_CHIPS.map((chip) => {
          const active = gpsFilter === chip.id;
          return (
            <Pressable
              key={chip.id}
              onPress={() => setGpsFilter(chip.id)}
              style={({ pressed }) => [s.chip, active && s.chipActive, pressed && s.chipPressed]}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
            >
              <AppText variant="caption" style={[s.chipLabel, active && s.chipLabelActive]}>
                {chip.label}
              </AppText>
            </Pressable>
          );
        })}
      </ScrollView>

      <View style={s.metaRow}>
        <AppText variant="body" style={s.metaTitle}>
          {results.length} chauffeur{results.length > 1 ? "s" : ""} actifs
        </AppText>
        <View style={s.metaRealtime}>
          <Ionicons name="sync-outline" size={14} color={FLEET_MAP_COLORS.brand} />
          <AppText variant="caption" style={s.metaRealtimeText}>
            Mise a jour en temps reel
          </AppText>
        </View>
      </View>

      <FlatList
        data={results}
        keyExtractor={(item) => String(item.driver_id)}
        keyboardShouldPersistTaps="handled"
        style={s.list}
        contentContainerStyle={results.length === 0 ? s.listEmpty : undefined}
        ListEmptyComponent={
          <AppText variant="body" style={s.empty}>
            {trimmed.length > 0
              ? "Aucun chauffeur ne correspond a la recherche."
              : "Saisissez un nom ou un numero de mission."}
          </AppText>
        }
        renderItem={({ item }) => {
          const etaLine = resolveEtaLine(item, liveEtaByDriverId[item.driver_id]);
          return (
            <Pressable
              onPress={() => onSelectDriver(item)}
              style={({ pressed }) => [
                s.card,
                pressed && s.rowPressed,
                (item.enrichment.operationalStatus === "busy" ||
                  item.enrichment.operationalStatus === "assigned") &&
                  s.cardMission,
              ]}
              accessibilityRole="button"
              accessibilityLabel={`Afficher ${resolveDriverDisplayName(item)}`}
            >
            <View
              style={[
                s.avatar,
                { backgroundColor: toRgba(FLEET_STATUS_THEME[item.enrichment.operationalStatus].fill, 0.14) },
              ]}
            >
              <AppText variant="label" style={[s.avatarText, { color: FLEET_STATUS_THEME[item.enrichment.operationalStatus].fill }]}>
                {driverFleetMarkerInitials(item).slice(0, 1)}
              </AppText>
            </View>
            <View style={s.cardMain}>
              <View style={s.cardTitleRow}>
                <AppText variant="body" style={s.rowName} numberOfLines={1}>
                  {resolveDriverDisplayName(item)}
                </AppText>
                <AppText variant="body" style={s.distanceText}>
                  {resolveDistanceLabel(item, liveDistanceByDriverId[item.driver_id])}
                </AppText>
              </View>
              <View style={s.cardStatusRow}>
                <View style={[s.statusDot, { backgroundColor: FLEET_STATUS_THEME[item.enrichment.operationalStatus].fill }]} />
                <AppText
                  variant="caption"
                  style={[s.statusLabel, { color: FLEET_STATUS_THEME[item.enrichment.operationalStatus].fill }]}
                  numberOfLines={1}
                >
                  {FLEET_STATUS_THEME[item.enrichment.operationalStatus].label}
                </AppText>
                {item.enrichment.linkedMission ? (
                  <AppText variant="caption" style={s.rowMeta} numberOfLines={1}>
                    · Mission {item.enrichment.linkedMission.mission_id}
                  </AppText>
                ) : null}
              </View>
              {etaLine ? (
                <AppText variant="caption" style={s.etaLine} numberOfLines={1}>
                  {etaLine}
                </AppText>
              ) : null}
              <AppText variant="caption" style={s.secondaryLine} numberOfLines={1}>
                {resolveSecondLine(item)}
              </AppText>
            </View>
            <Ionicons name="chevron-forward" size={18} color={FLEET_MAP_COLORS.textMuted} />
            </Pressable>
          );
        }}
      />
    </EnterpriseBottomSheet>
  );
}

const s = StyleSheet.create({
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  searchWrap: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    flex: 1,
    borderWidth: 1,
    borderColor: FLEET_MAP_COLORS.fabBorder,
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 9,
    backgroundColor: "#fff",
  },
  closeBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    borderWidth: 1,
    borderColor: FLEET_MAP_COLORS.fabBorder,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  closeBtnPressed: { opacity: 0.85 },
  searchInput: {
    flex: 1,
    fontSize: FONT_SIZE.px14,
    color: FLEET_MAP_COLORS.text,
    paddingVertical: 0,
  },
  chipsWrap: {
    marginTop: 8,
    marginHorizontal: -2,
  },
  chipsContent: {
    paddingRight: 4,
    gap: 8,
  },
  chip: {
    height: 30,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: FLEET_MAP_COLORS.fabBorder,
    paddingHorizontal: 9,
    backgroundColor: "#fff",
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  chipActive: {
    backgroundColor: "#0F172A",
    borderColor: "#0F172A",
  },
  chipPressed: { opacity: 0.9 },
  chipDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  chipLabel: {
    color: FLEET_MAP_COLORS.text,
    fontSize: FONT_SIZE.px11,
    fontWeight: "600",
  },
  chipLabelActive: {
    color: "#fff",
  },
  chipCount: {
    minWidth: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: "rgba(148, 163, 184, 0.22)",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 5,
  },
  chipCountActive: {
    backgroundColor: "rgba(255,255,255,0.2)",
  },
  chipCountText: {
    color: FLEET_MAP_COLORS.text,
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
  },
  chipCountTextActive: {
    color: "#fff",
  },
  metaRow: {
    marginTop: 8,
    marginBottom: 8,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  metaTitle: {
    color: FLEET_MAP_COLORS.text,
    fontWeight: "700",
  },
  metaRealtime: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
  },
  metaRealtimeText: {
    color: FLEET_MAP_COLORS.textMuted,
    fontWeight: "600",
  },
  list: {
    maxHeight: 430,
  },
  listEmpty: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 24,
  },
  empty: {
    color: FLEET_MAP_COLORS.textMuted,
    textAlign: "center",
    paddingHorizontal: 12,
  },
  card: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    borderWidth: 1,
    borderColor: "rgba(226, 232, 240, 0.9)",
    borderRadius: 14,
    backgroundColor: "#fff",
    paddingHorizontal: 10,
    paddingVertical: 9,
    marginBottom: 8,
  },
  cardMission: {
    backgroundColor: "rgba(59, 130, 246, 0.06)",
    borderColor: "rgba(59, 130, 246, 0.2)",
  },
  rowPressed: { opacity: 0.85 },
  avatar: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: {
    fontWeight: "800",
    fontSize: FONT_SIZE.px16,
  },
  cardMain: { flex: 1, minWidth: 0 },
  cardTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  rowName: { color: FLEET_MAP_COLORS.text, fontWeight: "600" },
  distanceText: {
    color: FLEET_MAP_COLORS.text,
    fontWeight: "700",
    fontSize: FONT_SIZE.px13,
  },
  cardStatusRow: {
    marginTop: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
  },
  rowMeta: { color: FLEET_MAP_COLORS.textMuted, marginTop: 0 },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    marginLeft: 2,
  },
  statusLabel: {
    fontWeight: "700",
  },
  etaLine: {
    marginTop: 2,
    color: "#0F172A",
    fontWeight: "600",
  },
  secondaryLine: {
    marginTop: 1,
    color: FLEET_MAP_COLORS.textMuted,
  },
});
