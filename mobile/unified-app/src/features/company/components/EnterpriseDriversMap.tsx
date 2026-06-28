import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement, type MutableRefObject } from "react";
import { Platform, StyleSheet, View, type StyleProp, type ViewStyle } from "react-native";
import MapView, { Circle, Polyline, PROVIDER_GOOGLE, type Region } from "react-native-maps";
import { LirieMapLogoClip } from "../../maps/LirieMapLogoClip";
import {
  LIRIE_GOOGLE_ATTRIBUTION_ANDROID_MAP_PADDING_LEFT_PX,
  lirieMapLogoHidePadding,
  resolveNativeGoogleLogoClipPx,
} from "../../maps/lirieMapChrome";
import { getNativeGoogleMapViewStyleProps } from "../../maps/nativeGoogleMapStyle";
import type { CompanyDriverLiveLocation } from "../api/contracts";
import { DriverMarker } from "./maps/DriverMarker";
import { ClusterMarker } from "./maps/ClusterMarker";
import { clusterSharesDriverIds, computeFleetRegion } from "./maps/fleetMapLogic";
import type { FleetHeatmapPoint } from "./maps/fleetMapGeo";
import { resolveGoogleMapsNativeApiKey } from "../../../config/googleMapsKeys";
import {
  resolveFleetMissionEtaAnchor,
  resolveFleetMissionEtaBadgeOverlay,
} from "./maps/fleetMapDirections";
import { FleetMapEtaBadgeMarker } from "./maps/FleetMapEtaBadgeMarker";
import { FleetMapLiriePointMarker } from "./maps/FleetMapLiriePointMarker";
import { MissionRoutePolylines } from "./maps/MissionRoutePolyline";
import { useFleetMissionRoutedPaths } from "./maps/useFleetMissionRoutedPaths";
import { FLEET_MISSION_MAP_POLICY } from "./maps/fleetMapMissionPolicies";
import type { FleetMissionOverlay } from "./maps/fleetMapMissionVisual";
import type {
  FleetActiveRoute,
  FleetDriverMapItem,
  FleetMapLayersState,
  FleetMapMarker,
  FleetMapRecenterMode,
  FleetPinnedClusterFocus,
} from "./maps/fleetMapTypes";
import {
  collectFleetMarkerPositions,
  collectValidFleetPositions,
  expandFleetBoundsSpan,
} from "./maps/fleetMapWebCamera";
import { DEFAULT_FLEET_MAP_LAYERS } from "./maps/fleetMapTypes";
import type { ImminentDeparture } from "../dashboard/cockpit/imminentDepartures";
import { ImminentDepartureMarkers } from "./maps/ImminentDepartureMarkers";
import {
  applyFleetFitVerticalBias,
  computeFleetMapFitEdgePadding,
  computeFleetMaxRegionDelta,
  type MapEdgePadding,
} from "./maps/fleetMapFitPadding";

export type FleetMapCameraControl = {
  clearUserCameraLock: () => void;
};

export type EnterpriseDriversMapProps = {
  drivers: CompanyDriverLiveLocation[];
  markers?: FleetMapMarker[];
  mapHeight?: number;
  containerStyle?: StyleProp<ViewStyle>;
  selectedDriverId?: number | null;
  layers?: FleetMapLayersState;
  activeRoute?: FleetActiveRoute | null;
  missionOverlays?: FleetMissionOverlay[];
  missionsById?: Map<number, import("../api/contracts").CompanyDispatchMission>;
  stableEtaAnchors?: Map<number, { latitude: number; longitude: number; atMs: number }>;
  selectedMissionId?: number | null;
  heatmapPoints?: FleetHeatmapPoint[];
  recenterRegion?: Region | null;
  recenterMode?: FleetMapRecenterMode;
  recenterToken?: number;
  cameraInsets?: { top: number; right: number; bottom: number; left: number };
  onDriverPress?: (driver: FleetDriverMapItem) => void;
  onClusterPress?: (drivers: FleetDriverMapItem[]) => void;
  imminentDepartures?: ImminentDeparture[];
  globalVectorMode?: boolean;
  pinnedClusterFocus?: FleetPinnedClusterFocus | null;
  /** Cockpit carte plein écran : rognage logo Google en flex. */
  logoClipFill?: boolean;
  /** Padding dédié au `fitToCoordinates` (sans les insets caméra complets). */
  fitEdgePadding?: MapEdgePadding;
  /** Décale le cadre flotte vers le haut (panneau bas cockpit). */
  cameraVerticalBias?: number;
  /** Limite le dézoom manuel au cadre des chauffeurs. */
  constrainFleetZoom?: boolean;
  /**
   * iOS : false tant que socket instable / carte non prête — évite montage markers+routes
   * pendant reconnexion (crash natif New Arch).
   */
  nativeOverlaysEnabled?: boolean;
  /** Pan / zoom manuel : bloque le recentrage auto sur tick GPS. */
  onUserCameraGesture?: () => void;
  /** Référence pour réautoriser le recentrage (bouton centrer). */
  cameraControlRef?: MutableRefObject<FleetMapCameraControl | null>;
};

const BORDER = "rgba(145, 165, 157, 0.45)";
const STANDALONE_ROUTE_STROKES = Platform.select({
  android: { glow: 4.5, main: 2.2 },
  default: { glow: 8, main: 4 },
});

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

const DEFAULT_FLEET_MAP_EDGE_INSETS: MapEdgePadding = { top: 52, right: 52, bottom: 52, left: 52 };

const styles = StyleSheet.create({
  root: {
    position: "relative",
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 12,
    overflow: "hidden",
    ...mapCardShadow,
  },
  map: { width: "100%" as const },
});

function resolveMapType(layers: FleetMapLayersState): "standard" | "satellite" | "hybrid" | "terrain" {
  if (layers.mapType === "satellite") return "satellite";
  if (layers.mapType === "terrain") return "terrain";
  return "standard";
}

export function EnterpriseDriversMap({
  drivers,
  markers: markersProp,
  mapHeight = 200,
  containerStyle,
  selectedDriverId = null,
  layers = DEFAULT_FLEET_MAP_LAYERS,
  activeRoute = null,
  missionOverlays = [],
  missionsById,
  stableEtaAnchors,
  selectedMissionId = null,
  heatmapPoints = [],
  recenterRegion,
  recenterMode = "all",
  recenterToken = 0,
  cameraInsets,
  onDriverPress,
  onClusterPress,
  imminentDepartures = [],
  globalVectorMode = false,
  pinnedClusterFocus = null,
  logoClipFill = false,
  fitEdgePadding,
  cameraVerticalBias = 0,
  constrainFleetZoom = false,
  nativeOverlaysEnabled = true,
  onUserCameraGesture,
  cameraControlRef,
}: EnterpriseDriversMapProps) {
  const mapRef = useRef<MapView | null>(null);
  const mapReadyRef = useRef(false);
  const [mapReady, setMapReady] = useState(false);
  const pendingFitRef = useRef(false);
  const userCameraLockedRef = useRef(false);
  const programmaticCameraRef = useRef(false);
  const driversRef = useRef(drivers);
  const markersRef = useRef<FleetMapMarker[]>(markersProp ?? []);
  const driversByIdRef = useRef<Map<number, FleetDriverMapItem>>(new Map());

  driversRef.current = drivers;
  const markers = useMemo(() => markersProp ?? [], [markersProp]);
  markersRef.current = markers;

  useEffect(() => {
    const next = new Map<number, FleetDriverMapItem>();
    for (const m of markers) {
      if (m.kind === "driver") next.set(m.driver.driver_id, m.driver);
    }
    driversByIdRef.current = next;
  }, [markers]);

  const initialRegion = useMemo(
    () => computeFleetRegion(drivers),
    [drivers]
  );

  const resolvedFitPadding = useMemo(() => {
    const base =
      fitEdgePadding ??
      computeFleetMapFitEdgePadding(cameraInsets ?? DEFAULT_FLEET_MAP_EDGE_INSETS, { immersive: logoClipFill });
    return applyFleetFitVerticalBias(base, cameraVerticalBias);
  }, [cameraInsets, cameraVerticalBias, fitEdgePadding, logoClipFill]);

  const driversStructuralKey = useMemo(
    () =>
      drivers
        .map((d) => d.driver_id)
        .sort((a, b) => a - b)
        .join("|"),
    [drivers]
  );
  const maxDeltaCacheRef = useRef<{ key: string; value: number | undefined }>({
    key: "",
    value: undefined,
  });
  if (maxDeltaCacheRef.current.key !== driversStructuralKey) {
    maxDeltaCacheRef.current = {
      key: driversStructuralKey,
      value: constrainFleetZoom ? computeFleetMaxRegionDelta(drivers) : undefined,
    };
  }
  const fleetMaxDelta = maxDeltaCacheRef.current.value;

  const mapStyleProps = useMemo(() => getNativeGoogleMapViewStyleProps(), []);
  const mapDims = { height: mapHeight, width: "100%" as const };
  const mapType = resolveMapType(layers);
  const mapsApiKey = resolveGoogleMapsNativeApiKey();
  const mountNativeOverlays = nativeOverlaysEnabled && mapReady;
  const { routedPathsByMissionId, routedStateByMissionId } = useFleetMissionRoutedPaths(
    mountNativeOverlays ? missionOverlays : [],
    mapsApiKey,
    { enabled: mountNativeOverlays }
  );

  const fitAllDrivers = useCallback(() => {
    const map = mapRef.current;
    if (!map) {
      pendingFitRef.current = true;
      return;
    }
    pendingFitRef.current = false;
    programmaticCameraRef.current = true;

    const currentDrivers = driversRef.current;
    const currentMarkers = markersRef.current;
    const fromDrivers = collectValidFleetPositions(
      currentDrivers.map((d) => ({ latitude: d.latitude, longitude: d.longitude }))
    );
    const fromMarkers = collectValidFleetPositions(collectFleetMarkerPositions(currentMarkers));
    const positions =
      fromDrivers.length > 0 ? expandFleetBoundsSpan(fromDrivers) : expandFleetBoundsSpan(fromMarkers);
    if (positions.length === 0) return;

    if (positions.length === 1) {
      void map.animateToRegion(
        {
          latitude: positions[0].latitude,
          longitude: positions[0].longitude,
          latitudeDelta: 0.024,
          longitudeDelta: 0.024,
        },
        FLEET_MISSION_MAP_POLICY.cameraAnimationMs
      );
      return;
    }

    void map.fitToCoordinates(positions, {
      edgePadding: resolvedFitPadding,
      animated: true,
    });
  }, [resolvedFitPadding]);

  const fitAllDriversRef = useRef(fitAllDrivers);
  fitAllDriversRef.current = fitAllDrivers;

  const recenterRegionRef = useRef(recenterRegion);
  recenterRegionRef.current = recenterRegion;

  const clearUserCameraLock = useCallback(() => {
    userCameraLockedRef.current = false;
  }, []);

  useEffect(() => {
    if (!cameraControlRef) return;
    cameraControlRef.current = { clearUserCameraLock };
    return () => {
      cameraControlRef.current = null;
    };
  }, [cameraControlRef, clearUserCameraLock]);

  const applyExplicitRecenter = useCallback(() => {
    programmaticCameraRef.current = true;
    if (recenterMode === "all") {
      fitAllDrivers();
      return;
    }
    const map = mapRef.current;
    const region = recenterRegionRef.current;
    if (!map || !region) return;
    void map.animateToRegion(region, FLEET_MISSION_MAP_POLICY.cameraAnimationMs);
  }, [fitAllDrivers, recenterMode]);

  const handleRegionChangeComplete = useCallback(
    (region: Region, details?: { isGesture?: boolean }) => {
      void region;
      if (programmaticCameraRef.current) {
        programmaticCameraRef.current = false;
        return;
      }
      if (details?.isGesture !== false) {
        userCameraLockedRef.current = true;
        onUserCameraGesture?.();
      }
    },
    [onUserCameraGesture]
  );

  const onMapReady = useCallback(() => {
    mapReadyRef.current = true;
    setMapReady(true);
    if (pendingFitRef.current) {
      pendingFitRef.current = false;
      applyExplicitRecenter();
      return;
    }
    if (recenterMode === "all") {
      fitAllDrivers();
    }
  }, [applyExplicitRecenter, fitAllDrivers, recenterMode]);

  useEffect(() => {
    void recenterToken;
    if (!mapReadyRef.current) {
      pendingFitRef.current = true;
      return;
    }
    pendingFitRef.current = false;
    if (userCameraLockedRef.current) return;
    programmaticCameraRef.current = true;
    if (recenterMode === "all") {
      fitAllDriversRef.current();
      return;
    }
    const map = mapRef.current;
    const region = recenterRegionRef.current;
    if (!map || !region) return;
    void map.animateToRegion(region, FLEET_MISSION_MAP_POLICY.cameraAnimationMs);
  }, [recenterToken, recenterMode]);

  const handleDriverMarkerPress = useCallback(
    (driverId: number) => {
      const driver = driversByIdRef.current.get(driverId);
      if (driver) onDriverPress?.(driver);
    },
    [onDriverPress]
  );

  const handleClusterPress = useCallback(
    (clusterDrivers: FleetDriverMapItem[]) => {
      onClusterPress?.(clusterDrivers);
    },
    [onClusterPress]
  );

  const hasMissionRoutes = missionOverlays.length > 0;

  const routeCoords =
    !hasMissionRoutes && activeRoute && activeRoute.points.length >= 2
      ? activeRoute.points.map((p) => ({ latitude: p.latitude, longitude: p.longitude }))
      : [];

  const routeMid =
    routeCoords.length >= 2
      ? {
          latitude: (routeCoords[0].latitude + routeCoords[routeCoords.length - 1].latitude) / 2,
          longitude: (routeCoords[0].longitude + routeCoords[routeCoords.length - 1].longitude) / 2,
        }
      : null;

  const etaBadgeOverlay = useMemo(
    () => resolveFleetMissionEtaBadgeOverlay(missionOverlays, selectedMissionId),
    [missionOverlays, selectedMissionId]
  );

  const etaMid = useMemo(() => {
    if (!etaBadgeOverlay?.etaBadgeLabel) return null;
    return resolveFleetMissionEtaAnchor(
      etaBadgeOverlay,
      routedPathsByMissionId,
      routedStateByMissionId,
      stableEtaAnchors
    );
  }, [etaBadgeOverlay, routedPathsByMissionId, routedStateByMissionId, stableEtaAnchors]);

  const etaLabel = etaBadgeOverlay?.etaBadgeLabel ?? null;

  const heatmapNodes = useMemo(
    () =>
      heatmapPoints.map((p, index) => (
        <Circle
          key={`heatmap-${index}-${p.latitude}-${p.longitude}`}
          center={{ latitude: p.latitude, longitude: p.longitude }}
          radius={220 + p.weight * 35}
          fillColor="rgba(239, 68, 68, 0.2)"
          strokeWidth={0}
        />
      )),
    [heatmapPoints]
  );

  const missionAnchorNodes = useMemo(
    () =>
      missionOverlays.flatMap((overlay) => {
        const nodes = [];
        if (overlay.pickup && overlay.pickupAnchor) {
          nodes.push(
            <FleetMapLiriePointMarker
              key={`pickup-${overlay.missionId}`}
              coordinate={overlay.pickup}
              anchor={overlay.pickupAnchor}
              selected={overlay.isSelected}
            />
          );
        }
        if (overlay.dropoff && overlay.dropoffAnchor) {
          nodes.push(
            <FleetMapLiriePointMarker
              key={`dropoff-${overlay.missionId}`}
              coordinate={overlay.dropoff}
              anchor={overlay.dropoffAnchor}
              selected={overlay.isSelected}
            />
          );
        }
        return nodes;
      }),
    [missionOverlays]
  );

  const markerNodes = useMemo(() => {
    const pinnedClusterActive =
      pinnedClusterFocus != null &&
      selectedDriverId != null &&
      pinnedClusterFocus.drivers.some((driver) => driver.driver_id === selectedDriverId);
    const pinnedDriverIds = pinnedClusterActive
      ? new Set(pinnedClusterFocus.drivers.map((driver) => driver.driver_id))
      : null;

    const nodes = markers
      .map((m) => {
        if (m.kind === "cluster") {
          if (pinnedDriverIds && clusterSharesDriverIds(m.drivers, pinnedDriverIds)) {
            return null;
          }
          return (
            <ClusterMarker
              key={`cluster-${m.clusterKey}-${m.count}`}
              markerKey={m.clusterKey}
              latitude={m.latitude}
              longitude={m.longitude}
              count={m.count}
              drivers={m.drivers}
              onPress={() => handleClusterPress(m.drivers)}
            />
          );
        }
        if (
          pinnedDriverIds?.has(m.driver.driver_id) &&
          m.driver.driver_id !== selectedDriverId
        ) {
          return null;
        }
        const selected = m.driver.driver_id === selectedDriverId;
        const linkedMissionId = m.driver.enrichment.linkedMission?.mission_id ?? null;
        const missionSelected =
          selectedMissionId != null && linkedMissionId === selectedMissionId;
        const dimmed =
          (selectedDriverId != null && !selected) ||
          (globalVectorMode && selectedDriverId != null && !selected) ||
          (selectedMissionId != null && !missionSelected && selectedDriverId != null && !selected);
        return (
          <DriverMarker
            key={m.driver.driver_id}
            item={m.driver}
            selected={selected || missionSelected}
            dimmed={dimmed}
            vectorMode={globalVectorMode}
            onPress={handleDriverMarkerPress}
          />
        );
      })
      .filter((node): node is ReactElement => node != null);

    if (pinnedClusterActive && pinnedClusterFocus && selectedDriverId != null) {
      const focusedDriverInMarkers = markers.some(
        (marker) => marker.kind === "driver" && marker.driver.driver_id === selectedDriverId
      );
      nodes.push(
        <ClusterMarker
          key={`pinned-cluster-${pinnedClusterFocus.count}`}
          markerKey={`pinned-${pinnedClusterFocus.drivers
            .map((driver) => driver.driver_id)
            .sort((a, b) => a - b)
            .join(",")}`}
          latitude={pinnedClusterFocus.latitude}
          longitude={pinnedClusterFocus.longitude}
          count={pinnedClusterFocus.count}
          drivers={pinnedClusterFocus.drivers}
          dimmed
          onPress={() => handleClusterPress(pinnedClusterFocus.drivers)}
        />
      );
      if (!focusedDriverInMarkers) {
        const focusedDriver = pinnedClusterFocus.drivers.find(
          (driver) => driver.driver_id === selectedDriverId
        );
        if (focusedDriver) {
          nodes.push(
            <DriverMarker
              key={`pinned-focus-driver-${focusedDriver.driver_id}`}
              item={focusedDriver}
              selected
              dimmed={false}
              vectorMode={globalVectorMode}
              onPress={handleDriverMarkerPress}
            />
          );
        }
      }
    }

    return nodes;
  }, [
    globalVectorMode,
    handleClusterPress,
    handleDriverMarkerPress,
    markers,
    pinnedClusterFocus,
    selectedDriverId,
    selectedMissionId,
  ]);

  const mapPadding = logoClipFill
    ? Platform.select({
        android: {
          top: 0,
          right: 0,
          left: LIRIE_GOOGLE_ATTRIBUTION_ANDROID_MAP_PADDING_LEFT_PX,
          bottom: 0,
        },
        ios: {
          top: 0,
          right: 0,
          left: 0,
          bottom: resolveNativeGoogleLogoClipPx(),
        },
        default: undefined,
      })
    : lirieMapLogoHidePadding();

  const mapProps = {
    ref: mapRef,
    provider: PROVIDER_GOOGLE as const,
    style: [styles.map, logoClipFill ? { flex: 1, height: "100%" as const } : mapDims],
    initialRegion,
    mapType,
    showsTraffic: layers.traffic,
    showsUserLocation: false,
    mapPadding,
    onMapReady,
    onRegionChangeComplete: handleRegionChangeComplete,
    ...(fleetMaxDelta != null ? { maxDelta: fleetMaxDelta } : {}),
    ...Platform.select({
      ios: {
        legalLabelInsets: { top: 0, left: 0, right: 0, bottom: -160 },
        paddingAdjustmentBehavior: "never" as const,
      },
      default: {},
    }),
    ...mapStyleProps,
    ...(mapsApiKey ? { googleMapApiKey: mapsApiKey } : {}),
  };

  const etaBubble =
    !hasMissionRoutes && routeMid && activeRoute?.etaLabel ? (
      <FleetMapEtaBadgeMarker coordinate={routeMid} label={activeRoute.etaLabel} zIndex={180} />
    ) : null;

  const missionEtaBubble =
    hasMissionRoutes && etaMid && etaLabel ? (
      <FleetMapEtaBadgeMarker coordinate={etaMid} label={etaLabel} zIndex={200} />
    ) : null;

  const mapNode = (
    <MapView {...mapProps}>
      {mountNativeOverlays ? (
        <>
          {heatmapNodes}
          {imminentDepartures.length > 0 ? (
            <ImminentDepartureMarkers departures={imminentDepartures} />
          ) : null}
          {hasMissionRoutes ? (
            <MissionRoutePolylines
              overlays={missionOverlays}
              routedPathsByMissionId={routedPathsByMissionId}
              routedStateByMissionId={routedStateByMissionId}
            />
          ) : null}
          {!hasMissionRoutes && routeCoords.length >= 2 ? (
            <Polyline
              coordinates={routeCoords}
              strokeColor="rgba(239, 68, 68, 0.10)"
              strokeWidth={STANDALONE_ROUTE_STROKES.glow}
              lineCap="round"
              lineJoin="round"
            />
          ) : null}
          {!hasMissionRoutes && routeCoords.length >= 2 ? (
            <Polyline
              coordinates={routeCoords}
              strokeColor={activeRoute?.color ?? "#EF4444"}
              strokeWidth={STANDALONE_ROUTE_STROKES.main}
              lineCap="round"
              lineJoin="round"
            />
          ) : null}
          {missionAnchorNodes}
          {markerNodes}
          {missionEtaBubble}
          {etaBubble}
        </>
      ) : null}
    </MapView>
  );

  return (
    <View
      style={[
        styles.root,
        logoClipFill && { backgroundColor: "transparent", borderWidth: 0 },
        containerStyle,
      ]}
      accessibilityLabel="Carte des chauffeurs en direct"
    >
      <LirieMapLogoClip height={mapHeight} fill={logoClipFill}>
        {mapNode}
      </LirieMapLogoClip>
    </View>
  );
}
