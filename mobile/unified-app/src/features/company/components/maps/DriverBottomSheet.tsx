import { useEffect, useMemo, useState } from "react";
import { Linking, Modal, Platform, Pressable, StyleSheet, View } from "react-native";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

import { useSafeAreaInsets } from "react-native-safe-area-context";

import { Ionicons } from "@expo/vector-icons";

import { AppText } from "../../../../design/ui/AppText";

import type { CompanyDispatchMission } from "../../api/contracts";

import type { FleetDriverMapItem } from "./fleetMapTypes";

import { FLEET_STATUS_THEME, FLEET_MAP_COLORS } from "./mapStatusTheme";

import { driverFleetMarkerInitials, resolveDriverDisplayName } from "../../utils/companyDriverMapStatus";
import { formatFleetConstraintReason } from "./fleetMapLogic";

import {
  conciseRouteSegment,
  formatMissionScheduleTimeLabel,
  resolveMissionUiStatus,
} from "../../dashboard/companyDashboardMissionUi";
import { hasScheduledPickupTime } from "../../utils/pickupSentinel";
import { resolveGoogleMapsNativeApiKey } from "../../../../config/googleMapsKeys";

import { FLEET_UI, fleetGlassPanel } from "./fleetMapUiTokens";
import { DriverFocusPanelSections } from "./driverFocusPanelSections";



type Props = {

  visible: boolean;

  driver: FleetDriverMapItem | null;
  selectedMission?: CompanyDispatchMission | null;

  peekDriver?: FleetDriverMapItem | null;
  upcomingMissions?: CompanyDispatchMission[];
  availableDrivers?: FleetDriverMapItem[];

  onClose: () => void;

  onOpenFromPeek?: () => void;

  onViewMission?: (missionId: number) => void;

  /** Focus carte + sheet sur une mission (cockpit « Prochaines courses »). */
  onMissionFocus?: (missionId: number) => void;

  onMessage?: () => void;

  /** `overlay` : carte cockpit (compact). `modal` : plein écran. */

  presentation?: "overlay" | "modal";

  bottomOffset?: number;

  snapLevel?: "collapsed" | "medium" | "expanded";

  onSnapChange?: (level: "collapsed" | "medium" | "expanded") => void;

  dismissible?: boolean;

  onRecenter?: () => void;

  /** Cockpit : toujours le tableau « Prochaines courses », jamais la fiche modale. */
  inlineTableOnly?: boolean;

  /** Cockpit : tableau « Prochaines courses » développé (défaut) ou réduit. */
  upcomingTableExpanded?: boolean;

  onUpcomingTableExpandedChange?: (expanded: boolean) => void;

  /** Regroupement carte : liste inline (même design que le tableau courses). */
  clusterDrivers?: FleetDriverMapItem[] | null;

  onCloseCluster?: () => void;

  onSelectClusterDriver?: (driver: FleetDriverMapItem) => void;

};



export function DriverBottomSheet({

  visible,

  driver,
  selectedMission = null,

  peekDriver = null,
  upcomingMissions,
  availableDrivers = [],

  onClose,

  onOpenFromPeek,

  onViewMission,

  onMissionFocus,

  onMessage,

  presentation = "overlay",

  bottomOffset = 0,

  snapLevel = "collapsed",

  onSnapChange,

  dismissible = true,

  onRecenter,

  inlineTableOnly = false,

  upcomingTableExpanded = true,

  onUpcomingTableExpandedChange,

  clusterDrivers = null,

  onCloseCluster,

  onSelectClusterDriver,

}: Props) {

  const insets = useSafeAreaInsets();

  if (!visible) return null;



  const compact = presentation === "overlay";
  const resolvedDriver = driver ?? peekDriver;
  const showPeek = compact && !driver && !!peekDriver;
  const hasUpcomingSource = upcomingMissions != null;
  const peekBottomOffset =
    bottomOffset > 0 ? Math.max(6, Math.min(14, bottomOffset)) : bottomOffset;

  if (compact && inlineTableOnly && hasUpcomingSource) {
    const selected = driver ?? null;
    const clusterActive = clusterDrivers != null && clusterDrivers.length > 1;
    const showUpcomingDefault = !clusterActive && !selected;
    const upcoming = selectUpcomingMissions(upcomingMissions ?? [], 3);

    if (showUpcomingDefault && !upcomingTableExpanded) {
      return (
        <View style={s.peekRoot} pointerEvents="box-none">
          <UpcomingMissionsExpandFab
            bottomOffset={peekBottomOffset}
            missionCount={upcoming.length}
            onPress={() => onUpcomingTableExpandedChange?.(true)}
          />
        </View>
      );
    }

    return (
      <View style={s.peekRoot} pointerEvents="box-none">
        <View
          style={[s.overlayAnchor, peekBottomOffset > 0 ? { paddingBottom: peekBottomOffset } : null]}
          pointerEvents="box-none"
        >
          {clusterActive ? (
            <CompactClusterDriversPeek
              drivers={clusterDrivers}
              onClose={onCloseCluster ?? (() => {})}
              onSelectDriver={onSelectClusterDriver ?? (() => {})}
            />
          ) : selected ? (
            <CompactSelectedDriverPeek
              driver={selected}
              selectedMission={selectedMission}
              onClose={onClose}
              onViewMission={onViewMission}
            />
          ) : (
            <CompactUpcomingMissionsPeek
              missions={upcoming}
              drivers={availableDrivers}
              onMissionPress={onMissionFocus}
              onMinimize={() => onUpcomingTableExpandedChange?.(false)}
            />
          )}
        </View>
      </View>
    );
  }

  if (!resolvedDriver) return null;

  if (showPeek) {
    const upcoming = selectUpcomingMissions(upcomingMissions ?? [], 3);
    return (
      <View style={s.peekRoot} pointerEvents="box-none">
        <View
          style={[s.overlayAnchor, peekBottomOffset > 0 ? { paddingBottom: peekBottomOffset } : null]}
          pointerEvents="box-none"
        >
          {hasUpcomingSource ? (
            <CompactUpcomingMissionsPeek
              missions={upcoming}
              drivers={availableDrivers}
              onPress={onOpenFromPeek}
              onMissionPress={onMissionFocus}
            />
          ) : (
            <CompactDriverPeekBar driver={resolvedDriver} onPress={onOpenFromPeek} />
          )}
        </View>
      </View>
    );
  }

  /** Web : Modal plein écran (z-index élevé) — la carte Google recouvre un overlay local. */
  const useRootModal = Platform.OS === "web";



  const sheet = compact ? (

    <CompactDriverSheet

      driver={resolvedDriver}

      onClose={onClose}

      onViewMission={onViewMission}

      onMessage={onMessage}

      bottomOffset={bottomOffset}

      snapLevel={snapLevel}

      onSnapChange={onSnapChange}

      dismissible={dismissible}

    />

  ) : (

    <FullDriverSheet

      driver={resolvedDriver}

      onClose={onClose}

      onViewMission={onViewMission}

      onMessage={onMessage}

      paddingBottom={Math.max(16, insets.bottom + 10)}

    />

  );



  if (useRootModal) {

    return (

      <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>

        <View style={s.modalRoot}>

          <Pressable
            style={s.modalBackdrop}
            onPress={dismissible ? onClose : undefined}
            accessibilityLabel="Fermer"
            pointerEvents={dismissible ? "auto" : "none"}
          />

          <View style={s.modalAnchor}>{sheet}</View>

        </View>

      </Modal>

    );

  }



  if (compact) {

    return (

      <View style={s.overlayRoot} pointerEvents="box-none">

        <Pressable
          style={s.overlayBackdrop}
          onPress={dismissible ? onClose : undefined}
          accessibilityLabel="Fermer"
          pointerEvents={dismissible ? "auto" : "none"}
        />

        <View
          style={[s.overlayAnchor, bottomOffset > 0 ? { paddingBottom: bottomOffset } : null]}
          pointerEvents="box-none"
        >

          {sheet}

        </View>

      </View>

    );

  }



  return (

    <Modal visible transparent animationType="slide" onRequestClose={onClose}>

      <View style={s.modalRoot}>

        <Pressable
          style={s.modalBackdrop}
          onPress={dismissible ? onClose : undefined}
          pointerEvents={dismissible ? "auto" : "none"}
        />

        <View style={s.modalAnchor}>{sheet}</View>

      </View>

    </Modal>

  );

}



function statusPillBg(fill: string): string {
  return `${fill}1A`;
}

function cycleSnap(level: "collapsed" | "medium" | "expanded"): "collapsed" | "medium" | "expanded" {
  if (level === "collapsed") return "medium";
  if (level === "medium") return "expanded";
  return "collapsed";
}

/**
 * Horodatage de tri d'une course.
 * Les courses sans heure confirmée (legacy « À définir », valeur vide ou
 * invalide) sont renvoyées en fin de liste : on privilégie les courses réellement
 * planifiées — y compris celles déjà passées mais non effectuées / en cours —
 * et on n'affiche les « À définir » que pour compléter s'il reste de la place.
 */
function missionScheduleSortTs(mission: CompanyDispatchMission): number {
  if (!hasScheduledPickupTime(mission)) return Number.MAX_SAFE_INTEGER;
  const ts = Date.parse(mission.scheduled_at as string);
  return Number.isNaN(ts) ? Number.MAX_SAFE_INTEGER : ts;
}

function selectUpcomingMissions(missions: CompanyDispatchMission[], limit = 3): CompanyDispatchMission[] {
  const ranked = missions
    .filter((mission) => mission.status !== "completed" && mission.status !== "cancelled")
    .sort((a, b) => missionScheduleSortTs(a) - missionScheduleSortTs(b));
  return ranked.slice(0, limit);
}

function FleetDriverPeekTableColumnsHeader() {
  return (
    <View style={s.peekMissionsTableHeader} accessibilityElementsHidden importantForAccessibility="no-hide-descendants">
      <View style={s.peekRowLeft}>
        <View style={[s.peekCol, s.peekColTime]}>
          <AppText variant="caption" style={s.peekHeaderCell}>
            Chauffeur
          </AppText>
        </View>
        <View style={[s.peekCol, s.peekColRoute]}>
          <AppText variant="caption" style={s.peekHeaderCell}>
            Trajet
          </AppText>
        </View>
      </View>
      <View style={s.peekRowRight}>
        <AppText variant="caption" style={[s.peekHeaderCell, s.peekHeaderCellRight]}>
          Statut
        </AppText>
      </View>
    </View>
  );
}

function FleetDriverPeekTableRow({
  driver,
  missionOverride,
  onPress,
  showBorder,
}: {
  driver: FleetDriverMapItem;
  missionOverride?: CompanyDispatchMission | null;
  onPress?: () => void;
  showBorder?: boolean;
}) {
  const { enrichment } = driver;
  const [liveRoute, setLiveRoute] = useState<{ minutes: number; distanceKm: number } | null>(null);
  const mapsApiKey = useMemo(() => resolveGoogleMapsNativeApiKey(), []);
  const theme = FLEET_STATUS_THEME[enrichment.operationalStatus];
  const name = conciseRouteSegment(resolveDriverDisplayName(driver), 14);
  const mission = missionOverride ?? enrichment.linkedMission;
  const pickup = mission ? formatPeekTableRouteLabel(mission.pickup_label) : "—";
  const dropoff = mission ? formatPeekTableRouteLabel(mission.dropoff_label) : "—";
  useEffect(() => {
    if (!mission || Platform.OS === "web" || !mapsApiKey) {
      setLiveRoute(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      const metrics = await fetchPeekRouteEtaMinutes({
        mission,
        driver,
        apiKey: mapsApiKey,
      });
      if (cancelled) return;
      setLiveRoute(metrics);
    })();
    return () => {
      cancelled = true;
    };
  }, [driver, mapsApiKey, mission]);

  const travelMetrics = mission
    ? formatPeekTableTravelMetrics(mission, liveRoute?.minutes, liveRoute?.distanceKm)
    : {
        durationLabel: enrichment.etaLabel?.trim() || "— min",
        distanceLabel: enrichment.distanceLabel?.trim()
          ? enrichment.distanceLabel.replace(/^→\s*/, "")
          : "— km",
      };

  return (
    <Pressable
      onPress={onPress}
      disabled={!onPress}
      style={({ pressed }) => [
        s.peekMissionRow,
        showBorder ? s.peekMissionRowBorder : null,
        pressed && onPress ? s.peekCardPressed : null,
      ]}
      accessibilityRole={onPress ? "button" : "text"}
      accessibilityLabel={`${name}, ${theme.label}, ${pickup} vers ${dropoff}`}
    >
      <View style={s.peekRowLeft}>
        <View style={[s.peekCol, s.peekColTime]}>
          <AppText variant="caption" style={s.peekMainCell} numberOfLines={2} ellipsizeMode="tail">
            {name}
          </AppText>
        </View>
        <View style={[s.peekCol, s.peekColRoute]}>
          <View style={s.peekRouteLine}>
            <View style={s.peekRouteIconWrap}>
              <Ionicons name="navigate-outline" size={11} color="rgba(15, 23, 42, 0.72)" />
            </View>
            <View style={s.peekRouteTextWrap}>
              <AppText variant="caption" style={s.peekMainCell} numberOfLines={1} ellipsizeMode="tail">
                {pickup}
              </AppText>
            </View>
          </View>
          <View style={s.peekRouteLine}>
            <View style={s.peekRouteIconWrap}>
              <Ionicons name="location-outline" size={11} color="rgba(71, 85, 105, 0.78)" />
            </View>
            <View style={s.peekRouteTextWrap}>
              <AppText variant="caption" style={s.peekSubCell} numberOfLines={1} ellipsizeMode="tail">
                {dropoff}
              </AppText>
            </View>
          </View>
        </View>
      </View>
      <View style={s.peekRowRight}>
        <View style={s.peekTravelStack}>
          <AppText variant="caption" style={s.peekEtaCompact} numberOfLines={1}>
            {travelMetrics.durationLabel}
          </AppText>
          <AppText variant="caption" style={s.peekDistanceCompact} numberOfLines={1}>
            {travelMetrics.distanceLabel}
          </AppText>
        </View>
        <View style={[s.peekStatusPill, s.peekStatusPillRight, { backgroundColor: `${theme.fill}16` }]}>
          <View style={[s.peekStatusDotMini, { backgroundColor: theme.fill }]} />
          <AppText variant="caption" style={[s.peekStatusPillText, { color: theme.fill }]} numberOfLines={1}>
            {theme.label}
          </AppText>
        </View>
      </View>
    </Pressable>
  );
}

/** Même carte / tableau que « Prochaines courses », colonnes Chauffeur · Trajet · Statut. */
function CompactClusterDriversPeek({
  drivers,
  onClose,
  onSelectDriver,
}: {
  drivers: FleetDriverMapItem[];
  onClose: () => void;
  onSelectDriver: (driver: FleetDriverMapItem) => void;
}) {
  return (
    <View style={s.peekMissionsCard} accessibilityLabel={`${drivers.length} chauffeurs à proximité`}>
      <View style={s.peekMissionsHeaderRow}>
        <AppText variant="caption" style={s.peekMissionsTitle}>
          {`${drivers.length} chauffeurs à proximité`}
        </AppText>
        <Pressable
          onPress={onClose}
          hitSlop={10}
          style={s.peekHeaderClose}
          accessibilityRole="button"
          accessibilityLabel="Fermer la liste des chauffeurs"
        >
          <Ionicons name="close" size={16} color={FLEET_MAP_COLORS.textMuted} />
        </Pressable>
      </View>
      <FleetDriverPeekTableColumnsHeader />
      {drivers.map((d, index) => (
        <FleetDriverPeekTableRow
          key={d.driver_id}
          driver={d}
          showBorder={index > 0}
          onPress={() => onSelectDriver(d)}
        />
      ))}
    </View>
  );
}

/** Même carte / tableau que « Prochaines courses », colonnes Chauffeur · Trajet · Statut. */
function CompactSelectedDriverPeek({
  driver,
  selectedMission,
  onClose,
  onViewMission,
}: {
  driver: FleetDriverMapItem;
  selectedMission?: CompanyDispatchMission | null;
  onClose: () => void;
  onViewMission?: (missionId: number) => void;
}) {
  const name = conciseRouteSegment(resolveDriverDisplayName(driver), 14);
  const mission =
    selectedMission?.driver_id === driver.driver_id ? selectedMission : driver.enrichment.linkedMission;
  const missionId = mission?.mission_id ?? driver.mission_id;

  return (
    <View style={s.peekMissionsCard} accessibilityLabel={`Chauffeur sélectionné ${name}`}>
      <View style={s.peekMissionsHeaderRow}>
        <AppText variant="caption" style={s.peekMissionsTitle}>
          Chauffeur sélectionné
        </AppText>
        <Pressable
          onPress={onClose}
          hitSlop={10}
          style={s.peekHeaderClose}
          accessibilityRole="button"
          accessibilityLabel="Désélectionner le chauffeur"
        >
          <Ionicons name="close" size={16} color={FLEET_MAP_COLORS.textMuted} />
        </Pressable>
      </View>
      <FleetDriverPeekTableColumnsHeader />
      <FleetDriverPeekTableRow
        driver={driver}
        missionOverride={mission}
        onPress={missionId != null && onViewMission ? () => onViewMission(missionId) : undefined}
      />
    </View>
  );
}

function estimateDistanceKmFromCoords(
  pickupLat: number | null | undefined,
  pickupLon: number | null | undefined,
  dropoffLat: number | null | undefined,
  dropoffLon: number | null | undefined
): number | null {
  const pickupLatNum = Number(pickupLat);
  const pickupLonNum = Number(pickupLon);
  const dropoffLatNum = Number(dropoffLat);
  const dropoffLonNum = Number(dropoffLon);

  if (
    !Number.isFinite(pickupLatNum) ||
    !Number.isFinite(pickupLonNum) ||
    !Number.isFinite(dropoffLatNum) ||
    !Number.isFinite(dropoffLonNum)
  ) {
    return null;
  }

  const toRad = (value: number) => (value * Math.PI) / 180;
  const earthRadiusKm = 6371;
  const dLat = toRad(dropoffLatNum - pickupLatNum);
  const dLon = toRad(dropoffLonNum - pickupLonNum);
  const lat1 = toRad(pickupLatNum);
  const lat2 = toRad(dropoffLatNum);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  const straightLineKm = earthRadiusKm * c;

  if (!Number.isFinite(straightLineKm) || straightLineKm <= 0) return null;
  // Urban route fallback: road path is usually longer than straight line.
  return straightLineKm * 1.22;
}

function abbreviateRouteAddressLabel(value: string | null | undefined): string | null | undefined {
  if (typeof value !== "string") return value;
  return value
    .replace(/\bAvenue\b/gi, "Av.")
    .replace(/\bChemin\b/gi, "Ch.")
    .replace(/\bRoute\b/gi, "Rte")
    .replace(/\bpour personnes âgées\b/gi, "p. âgées")
    .replace(/\bpersonnes âgées\b/gi, "p. âgées")
    .replace(/\bFoyer de jour\b/gi, "FdJ");
}

/** Gabarit mobile « Prochaines courses » — évite que le texte déborde sur Assigné. */
const PEEK_TABLE_ROUTE_MAX_CHARS = 22;
const PEEK_TABLE_DRIVER_MAX_CHARS = 10;
const PEEK_ROUTE_ETA_CACHE_TTL_MS = 2 * 60 * 1000;
const peekRouteEtaCache = new Map<string, { minutes: number; distanceKm: number; atMs: number }>();

function formatPeekTableRouteLabel(value: string | null | undefined): string {
  return conciseRouteSegment(abbreviateRouteAddressLabel(value), PEEK_TABLE_ROUTE_MAX_CHARS);
}

function formatPeekTableDriverLabel(value: string | null | undefined): string {
  const raw = value?.trim() ?? "";
  if (!raw) return "—";
  if (raw.toLowerCase() === "non assigné") return "N/A";
  return conciseRouteSegment(raw, PEEK_TABLE_DRIVER_MAX_CHARS);
}

function resolvePeekTableDistanceKm(mission: CompanyDispatchMission): number | null {
  const declaredDistance = Number(mission.route_distance_km);
  if (Number.isFinite(declaredDistance) && declaredDistance > 0) return declaredDistance;
  return estimateDistanceKmFromCoords(
    mission.pickup_lat,
    mission.pickup_lon,
    mission.dropoff_lat,
    mission.dropoff_lon
  );
}

function isInProgressMission(status: CompanyDispatchMission["status"]): boolean {
  return status === "in_progress" || status === "arrived";
}

function buildPeekEtaKey(
  driver: FleetDriverMapItem,
  destination: { latitude: number; longitude: number }
): string {
  return [
    driver.latitude.toFixed(4),
    driver.longitude.toFixed(4),
    destination.latitude.toFixed(4),
    destination.longitude.toFixed(4),
  ].join(">");
}

async function fetchPeekRouteEtaMinutes(options: {
  driver: FleetDriverMapItem;
  mission: CompanyDispatchMission;
  apiKey: string;
}): Promise<{ minutes: number; distanceKm: number } | null> {
  const { driver, mission, apiKey } = options;
  const toDropoff = isInProgressMission(mission.status);
  const latitude = toDropoff ? mission.dropoff_lat : mission.pickup_lat;
  const longitude = toDropoff ? mission.dropoff_lon : mission.pickup_lon;
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  const destination = { latitude: Number(latitude), longitude: Number(longitude) };
  const cacheKey = buildPeekEtaKey(driver, destination);
  const cached = peekRouteEtaCache.get(cacheKey);
  const now = Date.now();
  if (cached && now - cached.atMs < PEEK_ROUTE_ETA_CACHE_TTL_MS) {
    return { minutes: cached.minutes, distanceKm: cached.distanceKm };
  }

  const params = new URLSearchParams({
    origin: `${driver.latitude},${driver.longitude}`,
    destination: `${destination.latitude},${destination.longitude}`,
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
        }[];
      }[];
    };
    if (data.status !== "OK") return null;
    const legs = data.routes?.[0]?.legs ?? [];
    if (legs.length === 0) return null;
    const routeStats = legs.reduce(
      (acc, leg) => {
      const traffic = Number(leg.duration_in_traffic?.value);
      const normal = Number(leg.duration?.value);
      const value = Number.isFinite(traffic) && traffic > 0 ? traffic : normal;
        const distanceMeters = Number((leg as { distance?: { value?: number } }).distance?.value);
        return {
          seconds: acc.seconds + (Number.isFinite(value) ? value : 0),
          meters: acc.meters + (Number.isFinite(distanceMeters) ? distanceMeters : 0),
        };
      },
      { seconds: 0, meters: 0 }
    );
    if (!Number.isFinite(routeStats.seconds) || routeStats.seconds <= 0) return null;
    const minutes = Math.max(1, Math.round(routeStats.seconds / 60));
    const distanceKm = Number.isFinite(routeStats.meters) && routeStats.meters > 0
      ? routeStats.meters / 1000
      : 0;
    peekRouteEtaCache.set(cacheKey, { minutes, distanceKm, atMs: now });
    return { minutes, distanceKm };
  } catch {
    return null;
  }
}

/** Durée + distance compacts pour la colonne Suivi (2 lignes, pas de débordement). */
function formatPeekTableTravelMetrics(
  mission: CompanyDispatchMission,
  liveEtaMinutes?: number | null,
  liveDistanceKm?: number | null
): {
  durationLabel: string;
  distanceLabel: string;
} {
  const distanceKm = resolvePeekTableDistanceKm(mission);
  const declaredDurationMin = Number(mission.route_duration_min);
  const durationMin =
    Number.isFinite(declaredDurationMin) && declaredDurationMin > 0
      ? Math.round(declaredDurationMin)
      : distanceKm != null && Number.isFinite(distanceKm) && distanceKm > 0
        ? Math.max(4, Math.round((distanceKm / 35) * 60))
        : null;

  const durationLabel =
    Number.isFinite(liveEtaMinutes) && (liveEtaMinutes ?? 0) > 0
      ? `~${Math.round(liveEtaMinutes ?? 0)} min`
      : durationMin != null
        ? `~${durationMin} min`
        : "— min";
  const distanceLabel =
    Number.isFinite(liveDistanceKm) && (liveDistanceKm ?? 0) > 0
      ? (liveDistanceKm ?? 0) >= 10
        ? `${Math.round(liveDistanceKm ?? 0)} km`
        : `${(liveDistanceKm ?? 0).toFixed(1)} km`
      : distanceKm != null && Number.isFinite(distanceKm) && distanceKm > 0
        ? distanceKm >= 10
          ? `${Math.round(distanceKm)} km`
          : `${distanceKm.toFixed(1)} km`
      : "— km";

  return { durationLabel, distanceLabel };
}

function UpcomingMissionsExpandFab({
  bottomOffset,
  missionCount,
  onPress,
}: {
  bottomOffset: number;
  missionCount: number;
  onPress: () => void;
}) {
  return (
    <View
      style={[s.upcomingFabAnchor, bottomOffset > 0 ? { paddingBottom: bottomOffset } : null]}
      pointerEvents="box-none"
    >
      <Pressable
        onPress={onPress}
        style={({ pressed }) => [fleetGlassPanel(s.upcomingFab), pressed ? s.peekCardPressed : null]}
        accessibilityRole="button"
        accessibilityLabel={
          missionCount > 0
            ? `Afficher les prochaines courses, ${missionCount} course${missionCount > 1 ? "s" : ""}`
            : "Afficher les prochaines courses"
        }
      >
        <Ionicons name="list-outline" size={18} color={FLEET_MAP_COLORS.text} />
        {missionCount > 0 ? (
          <View style={s.upcomingFabBadge}>
            <AppText variant="caption" style={s.upcomingFabBadgeText}>
              {missionCount > 9 ? "9+" : missionCount}
            </AppText>
          </View>
        ) : null}
      </Pressable>
    </View>
  );
}

function CompactUpcomingMissionsPeek({
  missions,
  drivers,
  onPress,
  onMissionPress,
  onMinimize,
}: {
  missions: CompanyDispatchMission[];
  drivers: FleetDriverMapItem[];
  onPress?: () => void;
  onMissionPress?: (missionId: number) => void;
  onMinimize?: () => void;
}) {
  const [liveEtaByMissionId, setLiveEtaByMissionId] = useState<Record<number, number>>({});
  const [liveDistanceByMissionId, setLiveDistanceByMissionId] = useState<Record<number, number>>({});
  const mapsApiKey = useMemo(() => resolveGoogleMapsNativeApiKey(), []);
  const driversById = useMemo(() => {
    const map = new Map<number, FleetDriverMapItem>();
    for (const driver of drivers) map.set(driver.driver_id, driver);
    return map;
  }, [drivers]);

  useEffect(() => {
    if (Platform.OS === "web" || !mapsApiKey || missions.length === 0) return;
    let cancelled = false;
    const eligible = missions
      .map((mission) => {
        const driverId = mission.driver_id;
        if (driverId == null) return null;
        const driver = driversById.get(driverId);
        if (!driver) return null;
        return { mission, driver };
      })
      .filter((entry): entry is { mission: CompanyDispatchMission; driver: FleetDriverMapItem } => Boolean(entry));
    if (eligible.length === 0) return;

    void (async () => {
      const updates: Record<number, number> = {};
      const distanceUpdates: Record<number, number> = {};
      await Promise.all(
        eligible.map(async ({ mission, driver }) => {
          const metrics = await fetchPeekRouteEtaMinutes({
            mission,
            driver,
            apiKey: mapsApiKey,
          });
          if (metrics == null) return;
          updates[mission.mission_id] = metrics.minutes;
          if (metrics.distanceKm > 0) {
            distanceUpdates[mission.mission_id] = metrics.distanceKm;
          }
        })
      );
      if (cancelled || Object.keys(updates).length === 0) return;
      setLiveEtaByMissionId((prev) => ({ ...prev, ...updates }));
      if (Object.keys(distanceUpdates).length > 0) {
        setLiveDistanceByMissionId((prev) => ({ ...prev, ...distanceUpdates }));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [driversById, mapsApiKey, missions]);

  const hasMissions = missions.length > 0;
  return (
    <Pressable
      onPress={onPress}
      disabled={!onPress}
      style={({ pressed }) => [s.peekMissionsCard, pressed && onPress ? s.peekCardPressed : null]}
      accessibilityRole={onPress ? "button" : "text"}
      accessibilityLabel="Voir les prochaines courses"
    >
      <View style={onMinimize ? s.peekMissionsHeaderRow : s.peekMissionsHeader}>
        <AppText variant="caption" style={s.peekMissionsTitle}>
          Prochaines courses
        </AppText>
        {onMinimize ? (
          <Pressable
            onPress={onMinimize}
            hitSlop={10}
            style={s.peekHeaderClose}
            accessibilityRole="button"
            accessibilityLabel="Réduire le tableau des prochaines courses"
          >
            <Ionicons name="chevron-down" size={16} color={FLEET_MAP_COLORS.textMuted} />
          </Pressable>
        ) : null}
      </View>
      {hasMissions ? (
        <View style={s.peekMissionsTableHeader} accessibilityElementsHidden importantForAccessibility="no-hide-descendants">
          <View style={s.peekRowLeft}>
            <View style={[s.peekCol, s.peekColTime]}>
              <AppText variant="caption" style={s.peekHeaderCell}>
                Heure
              </AppText>
            </View>
            <View style={[s.peekCol, s.peekColRoute]}>
              <AppText variant="caption" style={s.peekHeaderCell}>
                Départ → Arrivée
              </AppText>
            </View>
          </View>
          <View style={s.peekRowRight}>
            <AppText variant="caption" style={[s.peekHeaderCell, s.peekHeaderCellRight]}>
              Suivi
            </AppText>
          </View>
        </View>
      ) : null}
      {hasMissions ? missions.map((mission, index) => {
        const ui = resolveMissionUiStatus(mission);
        const pickup = formatPeekTableRouteLabel(mission.pickup_label);
        const dropoff = formatPeekTableRouteLabel(mission.dropoff_label);
        const travelMetrics = formatPeekTableTravelMetrics(
          mission,
          liveEtaByMissionId[mission.mission_id],
          liveDistanceByMissionId[mission.mission_id]
        );
        const driverName = formatPeekTableDriverLabel(
          mission.driver_name?.trim() || mission.partner_company_name?.trim() || "Non assigné"
        );
        const scheduleUndefined = !hasScheduledPickupTime(mission);
        const plannedDeparture = formatMissionScheduleTimeLabel(mission);
        return (
          <Pressable
            key={mission.mission_id}
            onPress={() => onMissionPress?.(mission.mission_id)}
            disabled={!onMissionPress}
            style={({ pressed }) => [
              s.peekMissionRow,
              index > 0 ? s.peekMissionRowBorder : null,
              pressed && onMissionPress ? s.peekCardPressed : null,
            ]}
            accessibilityRole={onMissionPress ? "button" : "text"}
            accessibilityLabel={
              scheduleUndefined
                ? `Transport à heure à définir, ${pickup} vers ${dropoff}`
                : `Mission ${plannedDeparture}, ${pickup} vers ${dropoff}`
            }
          >
            <View style={s.peekRowLeft}>
              <View style={[s.peekCol, s.peekColTime]}>
                {scheduleUndefined ? (
                  <View style={s.peekTimeUndefined}>
                    <Ionicons name="time-outline" size={11} color="#F59E0B" />
                    <AppText variant="caption" style={s.peekTimeTbd} numberOfLines={2}>
                      {plannedDeparture}
                    </AppText>
                  </View>
                ) : (
                  <AppText variant="caption" style={s.peekMainCell}>
                    {plannedDeparture}
                  </AppText>
                )}
              </View>
              <View style={[s.peekCol, s.peekColRoute]}>
                <View style={s.peekRouteLine}>
                  <View style={s.peekRouteIconWrap}>
                    <Ionicons name="navigate-outline" size={11} color="rgba(15, 23, 42, 0.72)" />
                  </View>
                  <View style={s.peekRouteTextWrap}>
                    <AppText variant="caption" style={s.peekMainCell} numberOfLines={1} ellipsizeMode="tail">
                      {pickup}
                    </AppText>
                  </View>
                </View>
                <View style={s.peekRouteLine}>
                  <View style={s.peekRouteIconWrap}>
                    <Ionicons name="location-outline" size={11} color="rgba(71, 85, 105, 0.78)" />
                  </View>
                  <View style={s.peekRouteTextWrap}>
                    <AppText variant="caption" style={s.peekSubCell} numberOfLines={1} ellipsizeMode="tail">
                      {dropoff}
                    </AppText>
                  </View>
                </View>
              </View>
            </View>
            <View style={s.peekRowRight}>
              <View style={s.peekTravelStack}>
                <AppText variant="caption" style={s.peekEtaCompact} numberOfLines={1}>
                  {travelMetrics.durationLabel}
                </AppText>
                <AppText variant="caption" style={s.peekDistanceCompact} numberOfLines={1}>
                  {travelMetrics.distanceLabel}
                </AppText>
              </View>
              <AppText
                variant="caption"
                style={s.peekDriverCompact}
                numberOfLines={1}
                ellipsizeMode="tail"
              >
                {driverName}
              </AppText>
              <View style={[s.peekStatusPill, s.peekStatusPillRight, { backgroundColor: `${ui.barColor}16` }]}>
                <View style={[s.peekStatusDotMini, { backgroundColor: ui.barColor }]} />
                <AppText variant="caption" style={[s.peekStatusPillText, { color: ui.barColor }]} numberOfLines={1}>
                  {ui.label}
                </AppText>
              </View>
            </View>
          </Pressable>
        );
      }) : (
        <View style={s.peekMissionEmpty}>
          <AppText variant="caption" style={s.peekMissionEmptyText} numberOfLines={1}>
            Aucune course a venir pour le moment
          </AppText>
        </View>
      )}
    </Pressable>
  );
}

function CompactDriverPeekBar({
  driver,
  onPress,
}: {
  driver: FleetDriverMapItem;
  onPress?: () => void;
}) {
  const { enrichment } = driver;
  const theme = FLEET_STATUS_THEME[enrichment.operationalStatus];
  const mission = enrichment.linkedMission;
  const name = resolveDriverDisplayName(driver);
  const secondary =
    enrichment.etaLabel ??
    (mission ? "Trajet actif" : enrichment.operationalStatus === "available" ? "Prêt à partir" : "Sans mission");
  const routeLabel = mission
    ? `${conciseRouteSegment(mission.pickup_label)} → ${conciseRouteSegment(mission.dropoff_label)}`
    : null;

  return (
    <Pressable
      onPress={onPress}
      disabled={!onPress}
      style={({ pressed }) => [s.peekCard, pressed && onPress ? s.peekCardPressed : null]}
      accessibilityRole={onPress ? "button" : "text"}
      accessibilityLabel={`Ouvrir la fiche chauffeur ${name}`}
    >
      <View style={[s.peekStatusDot, { backgroundColor: theme.fill }]} />
      <View style={s.peekBody}>
        <AppText variant="caption" style={s.peekLine} numberOfLines={1}>
          {`${name} • ${theme.label} • ${secondary}`}
        </AppText>
        {routeLabel ? (
          <AppText variant="caption" style={s.peekRoute} numberOfLines={1}>
            {routeLabel}
          </AppText>
        ) : null}
      </View>
      <Ionicons name="chevron-up" size={18} color={FLEET_MAP_COLORS.textMuted} />
    </Pressable>
  );
}

function CompactDriverSheet({
  driver,
  onClose,
  onViewMission,
  onMessage,
  bottomOffset = 0,
  snapLevel = "collapsed",
  onSnapChange,
  dismissible = true,
  upcomingMissions,
  onRecenter,
}: {
  driver: FleetDriverMapItem;
  onClose: () => void;
  onViewMission?: (missionId: number) => void;
  onMessage?: () => void;
  bottomOffset?: number;
  snapLevel?: "collapsed" | "medium" | "expanded";
  onSnapChange?: (level: "collapsed" | "medium" | "expanded") => void;
  dismissible?: boolean;
  upcomingMissions?: CompanyDispatchMission[];
  onRecenter?: () => void;
}) {
  const { enrichment } = driver;
  const theme = FLEET_STATUS_THEME[enrichment.operationalStatus];
  const name = resolveDriverDisplayName(driver);
  const initials = driverFleetMarkerInitials(driver);
  const mission = enrichment.linkedMission;
  const missionId = mission?.mission_id ?? driver.mission_id;
  const hasPhone = Boolean(enrichment.phone?.trim());
  const inProgress =
    mission?.status === "in_progress" || mission?.status === "en_route" || mission?.status === "arrived";

  const subtitle = mission
    ? [
        inProgress ? "Passager à bord" : null,
        enrichment.etaLabel,
        enrichment.distanceLabel ? `→ ${enrichment.distanceLabel}` : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : enrichment.operationalStatus === "constrained"
      ? formatFleetConstraintReason(driver)
      : enrichment.operationalStatus === "available"
        ? "Prêt pour une nouvelle course"
        : enrichment.operationalStatus === "break"
          ? "Pause en cours"
          : enrichment.operationalStatus === "last_known"
            ? "Dernière position connue (non active)"
            : null;

  const handleCall = () => {
    const phone = enrichment.phone?.trim();
    if (phone) void Linking.openURL(`tel:${phone}`);
  };

  const expanded = snapLevel === "expanded";
  const medium = snapLevel === "medium" || expanded;

  return (
    <View
      style={[s.compactCard, medium && s.compactCardMedium, expanded && s.compactCardExpanded]}
      accessibilityLabel="Fiche chauffeur"
    >
      {dismissible ? (
        <Pressable
          onPress={onClose}
          hitSlop={10}
          style={s.compactClose}
          accessibilityRole="button"
          accessibilityLabel="Fermer"
        >
          <Ionicons name="close" size={18} color={FLEET_MAP_COLORS.textMuted} />
        </Pressable>
      ) : null}

      <Pressable
        onPress={() => onSnapChange?.(cycleSnap(snapLevel))}
        style={s.compactHandleHit}
        accessibilityRole="button"
        accessibilityLabel="Agrandir ou réduire la fiche"
      >
        <View style={s.compactHandle} />
      </Pressable>

      <View style={s.compactRow}>
        <View style={s.avatarRing}>
          <View style={[s.avatar, { backgroundColor: theme.fill }]}>
            <AppText variant="label" style={s.avatarText}>
              {initials}
            </AppText>
          </View>
        </View>

        <View style={s.compactBody}>
          <AppText variant="body" style={s.compactName} numberOfLines={1}>
            {name}
          </AppText>

          <View style={[s.statusPill, { backgroundColor: statusPillBg(theme.fill) }]}>
            <View style={[s.statusDot, { backgroundColor: theme.fill }]} />
            <AppText variant="caption" style={[s.statusPillText, { color: theme.fill }]}>
              {theme.label}
            </AppText>
            {inProgress ? (
              <View style={s.liveChip}>
                <View style={[s.livePulse, { backgroundColor: theme.fill }]} />
                <AppText variant="caption" style={[s.liveText, { color: theme.fill }]}>
                  LIVE
                </AppText>
              </View>
            ) : null}
          </View>

          {subtitle ? (
            <AppText variant="caption" style={s.subtitle} numberOfLines={2}>
              {subtitle}
            </AppText>
          ) : null}

          {mission && !enrichment.etaLabel ? (
            <AppText variant="caption" style={s.routeHint} numberOfLines={1}>
              {conciseRouteSegment(mission.pickup_label)} → {conciseRouteSegment(mission.dropoff_label)}
            </AppText>
          ) : null}
        </View>

        <View style={s.roundActions}>
          {hasPhone ? (
            <Pressable
              onPress={handleCall}
              style={({ pressed }) => [s.roundBtn, s.roundBtnPhone, pressed && s.roundBtnPressed]}
              accessibilityRole="button"
              accessibilityLabel="Appeler"
            >
              <Ionicons name="call" size={20} color="#3498DB" />
            </Pressable>
          ) : null}

          {missionId != null ? (
            <Pressable
              onPress={() => onViewMission?.(missionId)}
              style={({ pressed }) => [s.roundBtn, s.roundBtnMission, pressed && s.roundBtnPressed]}
              accessibilityRole="button"
              accessibilityLabel="Voir la mission"
            >
              <Ionicons name="arrow-up" size={22} color="#fff" />
            </Pressable>
          ) : onMessage ? (
            <Pressable
              onPress={onMessage}
              style={({ pressed }) => [s.roundBtn, s.roundBtnMessage, pressed && s.roundBtnPressed]}
              accessibilityRole="button"
              accessibilityLabel="Message"
            >
              <Ionicons name="chatbubble" size={18} color="#fff" />
            </Pressable>
          ) : null}
        </View>
      </View>

      {expanded ? (
        <DriverFocusPanelSections
          driver={driver}
          upcomingMissions={upcomingMissions}
          onMessage={onMessage}
          onRecenter={onRecenter}
          onViewMission={onViewMission}
        />
      ) : null}
    </View>
  );
}



function FullDriverSheet({

  driver,

  onClose,

  onViewMission,

  onMessage,

  paddingBottom,

}: {

  driver: FleetDriverMapItem;

  onClose: () => void;

  onViewMission?: (missionId: number) => void;

  onMessage?: () => void;

  paddingBottom: number;

}) {

  const { enrichment } = driver;

  const theme = FLEET_STATUS_THEME[enrichment.operationalStatus];

  const name = resolveDriverDisplayName(driver);

  const initials = driverFleetMarkerInitials(driver);

  const mission = enrichment.linkedMission;

  const missionId = mission?.mission_id ?? driver.mission_id;

  const inProgress =

    mission?.status === "in_progress" || mission?.status === "en_route" || mission?.status === "arrived";



  const handleCall = () => {

    const phone = enrichment.phone?.trim();

    if (phone) void Linking.openURL(`tel:${phone}`);

  };



  return (

    <View style={[s.fullCard, { paddingBottom }]}>

      <View style={s.handle} />

      <View style={s.headerRow}>

        <View style={[s.avatar, { backgroundColor: theme.fill }]}>

          <AppText variant="label" style={s.avatarText}>

            {initials}

          </AppText>

        </View>

        <View style={s.headerText}>

          <AppText variant="body" style={s.name} numberOfLines={1}>

            {name}

          </AppText>

          <View style={s.statusRow}>

            <View style={[s.statusDot, { backgroundColor: theme.fill }]} />

            <AppText variant="caption" style={s.statusLabel}>

              {theme.label}

            </AppText>

          </View>

        </View>

        <Pressable onPress={onClose} hitSlop={10} accessibilityLabel="Fermer">

          <Ionicons name="close" size={22} color={FLEET_MAP_COLORS.textMuted} />

        </Pressable>

      </View>



      {mission ? (

        <View style={s.missionBlock}>

          {inProgress && mission.client_name ? (

            <AppText variant="caption" style={s.clientLine}>

              Passager à bord · {mission.client_name}

            </AppText>

          ) : null}

          <AppText variant="caption" style={s.routeLine} numberOfLines={2}>

            {conciseRouteSegment(mission.pickup_label)} → {conciseRouteSegment(mission.dropoff_label)}

          </AppText>

          {enrichment.etaLabel ? (

            <AppText variant="caption" style={s.metaStrong}>

              {enrichment.etaLabel}

              {enrichment.distanceLabel ? ` → ${enrichment.distanceLabel}` : ""}

            </AppText>

          ) : null}

        </View>

      ) : (

        <AppText variant="caption" style={s.metaMuted}>

          Aucune mission active

        </AppText>

      )}



      <View style={s.actions}>

        <ActionChip icon="call-outline" label="Appeler" disabled={!enrichment.phone} onPress={handleCall} />

        <ActionChip icon="chatbubble-outline" label="Message" onPress={onMessage} />

      </View>



      {missionId != null ? (

        <Pressable

          onPress={() => onViewMission?.(missionId)}

          style={({ pressed }) => [s.primaryCta, pressed && { opacity: 0.92 }]}

          accessibilityRole="button"

          accessibilityLabel="Voir la mission"

        >

          <AppText variant="label" style={s.primaryCtaText}>

            Voir la mission

          </AppText>

          <Ionicons name="arrow-forward" size={18} color="#fff" />

        </Pressable>

      ) : null}

    </View>

  );

}



function ActionChip({

  icon,

  label,

  onPress,

  disabled,

}: {

  icon: keyof typeof Ionicons.glyphMap;

  label: string;

  onPress?: () => void;

  disabled?: boolean;

}) {

  return (

    <Pressable

      onPress={onPress}

      disabled={disabled}

      style={({ pressed }) => [s.chip, (pressed || disabled) && { opacity: 0.85 }]}

      accessibilityRole="button"

      accessibilityLabel={label}

    >

      <Ionicons name={icon} size={16} color={disabled ? FLEET_MAP_COLORS.textMuted : FLEET_MAP_COLORS.text} />

      <AppText variant="caption" style={s.chipLabel}>

        {label}

      </AppText>

    </Pressable>

  );

}



const s = StyleSheet.create({

  peekRoot: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 58,
    justifyContent: "flex-end",
    ...Platform.select({
      web: { position: "absolute" as const, zIndex: 58 },
      default: {},
    }),
  },

  overlayRoot: {

    ...StyleSheet.absoluteFillObject,

    zIndex: 60,

    justifyContent: "flex-end",

    ...Platform.select({

      web: { position: "absolute" as const, zIndex: 60 },

      default: {},

    }),

  },

  overlayBackdrop: {

    ...StyleSheet.absoluteFillObject,

    backgroundColor: "rgba(15, 23, 42, 0.18)",

  },

  overlayAnchor: {

    paddingHorizontal: 10,

    paddingBottom: 10,

    width: "100%",

    alignItems: "stretch",

  },
  peekCard: {
    ...fleetGlassPanel({
      borderRadius: 18,
      paddingHorizontal: 12,
      paddingVertical: 9,
      backgroundColor: "rgba(255, 255, 255, 0.95)",
      borderColor: "rgba(255, 255, 255, 0.9)",
    }),
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    minHeight: 52,
    width: "100%",
    pointerEvents: "auto",
  },
  peekMissionsCard: {
    ...fleetGlassPanel({
      borderRadius: 18,
      paddingHorizontal: 12,
      paddingVertical: 8,
      backgroundColor: "rgba(255, 255, 255, 0.95)",
      borderColor: "rgba(255, 255, 255, 0.9)",
    }),
    width: "100%",
    maxWidth: "100%",
    overflow: "hidden",
    pointerEvents: "auto",
  },
  peekMissionsHeader: {
    paddingBottom: 4,
  },
  peekMissionsHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingBottom: 4,
    gap: 8,
  },
  peekHeaderClose: {
    padding: 2,
    flexShrink: 0,
  },
  upcomingFabAnchor: {
    position: "absolute",
    right: FLEET_UI.fabRight,
    bottom: FLEET_UI.overlayBottom,
    zIndex: 1,
  },
  upcomingFab: {
    width: FLEET_UI.fabSize,
    height: FLEET_UI.fabSize,
    borderRadius: FLEET_UI.fabSize / 2,
    alignItems: "center",
    justifyContent: "center",
  },
  upcomingFabBadge: {
    position: "absolute",
    top: 0,
    right: 0,
    minWidth: 15,
    height: 15,
    borderRadius: 8,
    paddingHorizontal: 3,
    backgroundColor: FLEET_MAP_COLORS.brand,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1.5,
    borderColor: "#fff",
  },
  upcomingFabBadgeText: {
    color: "#fff",
    fontSize: FONT_SIZE.px9,
    fontWeight: "800",
    lineHeight: 11,
  },
  peekMissionsTitle: {
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
    letterSpacing: 0.6,
    textTransform: "uppercase",
  },
  peekMissionsTableHeader: {
    flexDirection: "row",
    alignItems: "center",
    width: "100%",
    maxWidth: "100%",
    overflow: "hidden",
    paddingBottom: 4,
    marginBottom: 2,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "rgba(148, 163, 184, 0.28)",
  },
  peekMissionRow: {
    flexDirection: "row",
    alignItems: "center",
    width: "100%",
    maxWidth: "100%",
    overflow: "hidden",
    minHeight: 44,
    paddingVertical: 4,
  },
  peekMissionRowBorder: {
    borderTopWidth: 1,
    borderTopColor: "rgba(148, 163, 184, 0.2)",
  },
  peekRowLeft: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    minWidth: 0,
    maxWidth: "72%",
    paddingRight: 6,
    overflow: "hidden",
  },
  peekRowRight: {
    width: 82,
    maxWidth: 82,
    flexShrink: 0,
    flexGrow: 0,
    flexDirection: "column",
    alignItems: "flex-end",
    justifyContent: "center",
    gap: 2,
    paddingLeft: 8,
    marginLeft: 6,
    overflow: "hidden",
    borderLeftWidth: StyleSheet.hairlineWidth,
    borderLeftColor: "rgba(148, 163, 184, 0.32)",
  },
  peekCol: {
    minWidth: 0,
  },
  peekColTime: {
    width: 44,
    flexShrink: 0,
  },
  peekTimeUndefined: {
    alignItems: "flex-start",
    gap: 2,
    maxWidth: 44,
  },
  peekTimeTbd: {
    color: "#F59E0B",
    fontSize: FONT_SIZE.px9,
    fontWeight: "700",
    lineHeight: 11,
  },
  peekColRoute: {
    flex: 1,
    minWidth: 0,
    alignItems: "flex-start",
    alignSelf: "stretch",
  },

  peekEtaCompact: {
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
    lineHeight: 12,
    maxWidth: 72,
    textAlign: "right",
  },
  peekTravelStack: {
    alignItems: "flex-end",
    justifyContent: "center",
    gap: 1,
    maxWidth: 78,
  },
  peekDistanceCompact: {
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px9,
    fontWeight: "600",
    lineHeight: 11,
    maxWidth: 72,
    textAlign: "right",
  },
  peekDriverCompact: {
    color: FLEET_MAP_COLORS.text,
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
    lineHeight: 12,
    maxWidth: 78,
    textAlign: "right",
  },
  peekHeaderCell: {
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px9,
    fontWeight: "700",
    letterSpacing: 0.35,
    textTransform: "uppercase",
    lineHeight: 12,
  },
  peekHeaderCellRight: {
    textAlign: "right",
    alignSelf: "flex-end",
  },
  peekMainCellRight: {
    textAlign: "right",
    alignSelf: "flex-end",
  },
  peekMainCell: {
    color: FLEET_MAP_COLORS.text,
    fontSize: FONT_SIZE.px11,
    fontWeight: "700",
    lineHeight: 14,
  },
  peekSubCell: {
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px10,
    lineHeight: 13,
  },
  peekRouteLine: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    alignSelf: "stretch",
    minWidth: 0,
    maxWidth: "100%",
  },
  peekRouteTextWrap: {
    flex: 1,
    minWidth: 0,
  },
  peekRouteIconWrap: {
    width: 13,
    flexShrink: 0,
    alignItems: "center",
    justifyContent: "center",
  },
  peekStatusPill: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    borderRadius: 999,
    paddingHorizontal: 5,
    paddingVertical: 1,
    marginTop: 2,
    gap: 4,
  },
  peekStatusPillRight: {
    alignSelf: "flex-end",
    maxWidth: "100%",
  },
  peekStatusDotMini: {
    width: 4,
    height: 4,
    borderRadius: 999,
  },
  peekStatusPillText: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
    lineHeight: 13,
  },
  peekDriverName: {
    minWidth: 0,
    color: FLEET_MAP_COLORS.text,
    fontSize: FONT_SIZE.px11,
    fontWeight: "700",
    lineHeight: 14,
  },
  peekMissionEmpty: {
    minHeight: 36,
    justifyContent: "center",
    paddingVertical: 6,
  },
  peekMissionEmptyText: {
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px11,
    lineHeight: 15,
  },
  peekCardPressed: {
    opacity: 0.92,
    transform: [{ scale: 0.99 }],
  },
  peekStatusDot: {
    width: 9,
    height: 9,
    borderRadius: 999,
    flexShrink: 0,
  },
  peekBody: {
    flex: 1,
    minWidth: 0,
  },
  peekLine: {
    color: FLEET_MAP_COLORS.text,
    fontSize: FONT_SIZE.px12,
    fontWeight: "700",
    lineHeight: 16,
  },
  peekRoute: {
    marginTop: 1,
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px10,
    lineHeight: 14,
  },

  modalRoot: {
    flex: 1,
    justifyContent: "flex-end",
    backgroundColor: "rgba(15, 23, 42, 0.2)",
    ...Platform.select({
      web: {
        position: "fixed" as const,
        top: 0,
        right: 0,
        bottom: 0,
        left: 0,
        zIndex: 10050,
      },
      default: {},
    }),
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
  },
  modalAnchor: {
    paddingHorizontal: 10,
    paddingBottom: 12,
    width: "100%",
    maxWidth: 480,
    alignSelf: "center",
    pointerEvents: "box-none",
  },
  compactCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 22,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.22)",
    paddingHorizontal: 14,
    paddingTop: 10,
    paddingBottom: 14,
    width: "100%",
    position: "relative",
    pointerEvents: "auto",
    ...Platform.select({
      ios: {
        shadowColor: "#0f172a",
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.12,
        shadowRadius: 20,
      },
      android: { elevation: 8 },
      web: {
        boxShadow: "0 16px 40px rgba(15, 23, 42, 0.14), 0 0 0 1px rgba(15, 23, 42, 0.04)",
        backdropFilter: "blur(12px)",
      } as object,
      default: {},
    }),
  },
  compactClose: {
    position: "absolute",
    top: 8,
    right: 8,
    zIndex: 2,
    padding: 4,
  },
  compactHandleHit: {
    alignSelf: "center",
    paddingVertical: 6,
    paddingHorizontal: 24,
    marginBottom: 4,
  },
  compactHandle: {
    alignSelf: "center",
    width: 32,
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(148, 163, 184, 0.4)",
  },
  compactCardMedium: {
    minHeight: 120,
  },
  compactCardExpanded: {
    minHeight: 168,
  },
  metaMuted: {
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px11,
  },
  compactRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
    paddingRight: 20,
  },
  compactBody: {
    flex: 1,
    minWidth: 0,
    gap: 5,
    paddingTop: 2,
  },
  compactName: {
    color: FLEET_MAP_COLORS.text,
    fontWeight: "700",
    fontSize: FONT_SIZE.px16,
    lineHeight: 20,
    paddingRight: 4,
  },
  statusPill: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: 4,
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 20,
  },
  statusPillText: {
    fontWeight: "700",
    fontSize: FONT_SIZE.px11,
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },
  subtitle: {
    color: FLEET_MAP_COLORS.text,
    fontWeight: "600",
    fontSize: FONT_SIZE.px12,
    lineHeight: 17,
  },
  routeHint: {
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px11,
    lineHeight: 15,
  },
  liveChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    marginLeft: 2,
    paddingLeft: 5,
    borderLeftWidth: 1,
    borderLeftColor: "rgba(148, 163, 184, 0.32)",
  },
  livePulse: {
    width: 6,
    height: 6,
    borderRadius: 999,
    opacity: 0.85,
  },
  liveText: {
    fontWeight: "800",
    fontSize: FONT_SIZE.px10,
    letterSpacing: 0.6,
  },
  roundActions: {
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    flexShrink: 0,
    paddingTop: 2,
  },
  roundBtn: {
    width: 46,
    height: 46,
    borderRadius: 23,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1.5,
  },
  roundBtnPressed: {
    opacity: 0.9,
    transform: [{ scale: 0.96 }],
  },
  roundBtnPhone: {
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(59, 130, 246, 0.45)",
    ...Platform.select({
      web: { boxShadow: "0 2px 8px rgba(59, 130, 246, 0.15)" } as object,
      default: {},
    }),
  },
  roundBtnMission: {
    backgroundColor: FLEET_MAP_COLORS.available,
    borderColor: "rgba(255, 255, 255, 0.5)",
    ...Platform.select({
      ios: {
        shadowColor: "#15803d",
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.28,
        shadowRadius: 6,
      },
      android: { elevation: 5 },
      web: { boxShadow: "0 4px 14px rgba(0, 121, 107, 0.4)" } as object,
      default: {},
    }),
  },
  roundBtnMessage: {
    backgroundColor: FLEET_MAP_COLORS.brand,
    borderColor: "transparent",
    ...Platform.select({
      web: { boxShadow: "0 4px 12px rgba(0, 121, 107, 0.35)" } as object,
      default: {},
    }),
  },

  fullCard: {

    ...fleetGlassPanel({

      borderTopLeftRadius: FLEET_MAP_COLORS.sheetRadius,

      borderTopRightRadius: FLEET_MAP_COLORS.sheetRadius,

      borderBottomLeftRadius: 20,

      borderBottomRightRadius: 20,

      paddingHorizontal: 16,

      paddingTop: 8,

      gap: 10,

    }),

    backgroundColor: "#FFFFFF",

    width: "100%",

  },

  handle: {

    alignSelf: "center",

    width: 36,

    height: 4,

    borderRadius: 2,

    backgroundColor: "rgba(148, 163, 184, 0.45)",

    marginBottom: 4,

  },

  headerRow: { flexDirection: "row", alignItems: "center", gap: 10 },

  avatarRing: {
    padding: 3,
    borderRadius: 30,
    backgroundColor: "#FFFFFF",
    flexShrink: 0,
    ...Platform.select({
      web: { boxShadow: "0 2px 8px rgba(15, 23, 42, 0.1)" } as object,
      default: {},
    }),
  },

  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: "center",
    justifyContent: "center",
  },

  avatarText: { color: "#fff", fontWeight: "800", fontSize: FONT_SIZE.px16 },

  headerText: { flex: 1, minWidth: 0 },

  name: { color: FLEET_MAP_COLORS.text, fontWeight: "700", fontSize: FONT_SIZE.px17, flex: 1 },

  statusRow: { flexDirection: "row", alignItems: "center", gap: 6 },

  statusDot: { width: 7, height: 7, borderRadius: 4 },

  statusLabel: { color: FLEET_MAP_COLORS.textMuted, fontWeight: "600", fontSize: FONT_SIZE.px12 },

  missionBlock: { gap: 4 },

  clientLine: { color: FLEET_MAP_COLORS.busy, fontWeight: "700", fontSize: FONT_SIZE.px12 },

  routeLine: { color: FLEET_MAP_COLORS.text, fontWeight: "600", lineHeight: 18, fontSize: FONT_SIZE.px13 },

  metaStrong: { color: FLEET_MAP_COLORS.text, fontWeight: "700", fontSize: FONT_SIZE.px12 },

  actions: { flexDirection: "row", flexWrap: "wrap", gap: 8 },

  chip: {

    flexDirection: "row",

    alignItems: "center",

    gap: 6,

    paddingHorizontal: 12,

    paddingVertical: 8,

    borderRadius: 14,

    borderWidth: 1,

    borderColor: FLEET_MAP_COLORS.fabBorder,

    backgroundColor: "#fff",

  },

  chipLabel: { color: FLEET_MAP_COLORS.text, fontWeight: "600", fontSize: FONT_SIZE.px12 },

  primaryCta: {

    flexDirection: "row",

    alignItems: "center",

    justifyContent: "center",

    gap: 8,

    minHeight: 48,

    borderRadius: 14,

    backgroundColor: FLEET_MAP_COLORS.brand,

  },

  primaryCtaText: { color: "#fff", fontWeight: "700", fontSize: FONT_SIZE.px15 },

});


