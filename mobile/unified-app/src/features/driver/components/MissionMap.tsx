import { useCallback, useEffect, useMemo, useRef } from "react";
import { Platform, Pressable, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import MapView, { PROVIDER_GOOGLE, type LatLng, type Region } from "react-native-maps";
import MapViewDirections from "react-native-maps-directions";
import { FleetMapLiriePointMarker } from "../../company/components/maps/FleetMapLiriePointMarker";
import type { FleetMissionAnchorStyle } from "../../company/components/maps/fleetMapMissionVisual";
import { compactMissionMapLayerStyle } from "../../maps/lirieMapChrome";
import { getNativeGoogleMapViewStyleProps } from "../../maps/nativeGoogleMapStyle";
import { extractMissionMapCoordInput } from "../domain/missionMapCoordUtils";
import {
  isMissionMapLiveRouteStatus,
  resolveMissionMapRoute,
} from "../domain/resolveMissionMapRoute";
import {
  applyMissionMapFitCenterBias,
  computeMissionMapFitEdgePadding,
  computeMissionMapFitRegion,
  resolveMissionMapFitPoints,
} from "../domain/missionMapFit";
import { useDriverLiveMapPosition } from "../hooks/useDriverLiveMapPosition";
import { useMissionMapResolvedCoords } from "../hooks/useMissionMapResolvedCoords";
import { useMissionRouteMetrics } from "../hooks/useMissionRouteMetrics";
import type { DriverMission } from "../types";
import type { DriverEtaSnapshot } from "../api";
import {
  computeMissionRegion,
  MISSION_MAP_FALLBACK_HEIGHT,
  resolveGoogleMapsNativeApiKey,
} from "./maps/missionMapShared";
import { MissionMapLiveBadge } from "./MissionMapLiveBadge";
import { MissionMapUnavailable } from "./MissionMapUnavailable";

type MissionMapProps = {
  mission: DriverMission;
  height?: number;
  /** Bouton flottant recentrage (dashboard driver). */
  showRecenterControl?: boolean;
  etaMinutes?: number | null;
  etaSnapshot?: DriverEtaSnapshot | null;
};

const LIVE_REFIT_MIN_MS = 12_000;

const PICKUP_ANCHOR: FleetMissionAnchorStyle = {
  role: "pickup",
  fill: "#00796B",
  stroke: "#ffffff",
  radius: 7,
  opacity: 1,
  zIndex: 10,
};

const DROPOFF_ANCHOR: FleetMissionAnchorStyle = {
  role: "dropoff",
  fill: "#3B82F6",
  stroke: "#ffffff",
  radius: 7,
  opacity: 1,
  zIndex: 10,
};

const DRIVER_ANCHOR: FleetMissionAnchorStyle = {
  role: "driver",
  fill: "#0D9488",
  stroke: "#ffffff",
  radius: 8,
  opacity: 1,
  zIndex: 20,
};

function coordsEqual(a: LatLng | null, b: LatLng | null): boolean {
  if (!a || !b) return false;
  return (
    Math.abs(a.latitude - b.latitude) < 0.00005 && Math.abs(a.longitude - b.longitude) < 0.00005
  );
}

/**
 * Carte mission iOS / Android : position live, itinéraire selon statut, badge dynamique.
 */
export function MissionMap(props: MissionMapProps) {
  const mapRef = useRef<MapView | null>(null);
  const routeCoordsRef = useRef<LatLng[]>([]);
  const lastAutoFitAtRef = useRef(0);
  const lastDriverRef = useRef<LatLng | null>(null);
  const height = props.height ?? MISSION_MAP_FALLBACK_HEIGHT;
  const mapInput = useMemo(() => extractMissionMapCoordInput(props.mission), [props.mission]);

  const apiKey = resolveGoogleMapsNativeApiKey();
  const mapStyleProps = useMemo(() => getNativeGoogleMapViewStyleProps(), []);

  const { pickupCoord, dropoffCoord, fallbackCoord, resolving } =
    useMissionMapResolvedCoords(mapInput);

  const liveDriverCoord = useDriverLiveMapPosition(mapInput.driverLat, mapInput.driverLng, true);
  const driverCoord = liveDriverCoord ?? null;

  const routePlan = useMemo(
    () =>
      resolveMissionMapRoute({
        status: props.mission.status,
        driverCoord,
        pickupCoord,
        dropoffCoord,
      }),
    [props.mission.status, driverCoord, pickupCoord, dropoffCoord]
  );

  const routeMetrics = useMissionRouteMetrics(props.mission, {
    etaMinutes: props.etaMinutes,
    etaSnapshot: props.etaSnapshot ?? null,
  });

  const fitEdgePadding = useMemo(
    () =>
      computeMissionMapFitEdgePadding(height, {
        showRecenterControl: props.showRecenterControl,
        reserveBadge:
          routeMetrics.distanceLabel !== "—" || routeMetrics.durationLabel !== "—",
      }),
    [
      height,
      props.showRecenterControl,
      routeMetrics.distanceLabel,
      routeMetrics.durationLabel,
    ]
  );

  const fitMissionViewport = useCallback(
    (options?: { useRoute?: boolean; animated?: boolean }) => {
      if (!mapRef.current) return;

      const routeCoordinates =
        options?.useRoute === false ? [] : routeCoordsRef.current;
      const points = resolveMissionMapFitPoints({
        mode: routePlan.mode,
        driverCoord,
        pickupCoord,
        dropoffCoord,
        routeCoordinates,
        useFullRoute: true,
      });

      const region = computeMissionMapFitRegion(points);
      if (region) {
        mapRef.current.animateToRegion(
          applyMissionMapFitCenterBias(region, fitEdgePadding, height),
          options?.animated === false ? 0 : 320
        );
        return;
      }

      if (points.length === 1) {
        mapRef.current.animateToRegion(
          {
            latitude: points[0].latitude,
            longitude: points[0].longitude,
            latitudeDelta: 0.045,
            longitudeDelta: 0.045,
          },
          options?.animated === false ? 0 : 320
        );
      }
    },
    [routePlan.mode, driverCoord, pickupCoord, dropoffCoord, fitEdgePadding, height]
  );

  const isLiveRoute = isMissionMapLiveRouteStatus(props.mission.status);
  const canRenderRoute = Boolean(
    routePlan.origin &&
      routePlan.destination &&
      routePlan.mode !== "single_point" &&
      apiKey &&
      apiKey.length > 0 &&
      !coordsEqual(routePlan.origin, routePlan.destination)
  );

  const mapFallbackCoord = driverCoord ?? pickupCoord ?? dropoffCoord ?? fallbackCoord;
  const directionsKey = `${routePlan.mode}-${routePlan.origin?.latitude}-${routePlan.destination?.latitude}`;

  const region = useMemo<Region>(() => {
    if (!mapFallbackCoord) {
      return { latitude: 46.2044, longitude: 6.1432, latitudeDelta: 0.09, longitudeDelta: 0.09 };
    }
    const r = computeMissionRegion(pickupCoord, dropoffCoord, mapFallbackCoord);
    return {
      latitude: r.latitude,
      longitude: r.longitude,
      latitudeDelta: r.latitudeDelta,
      longitudeDelta: r.longitudeDelta,
    };
  }, [mapFallbackCoord, pickupCoord, dropoffCoord]);

  const fitKeyPoints = useCallback(() => {
    fitMissionViewport({ useRoute: true });
  }, [fitMissionViewport]);

  const onDirectionsReady = useCallback(
    (result: { coordinates: LatLng[] }) => {
      routeCoordsRef.current = result.coordinates ?? [];
      fitMissionViewport({ useRoute: true });
    },
    [fitMissionViewport]
  );

  const onMapReady = useCallback(() => {
    fitMissionViewport({ useRoute: routeCoordsRef.current.length > 0, animated: false });
  }, [fitMissionViewport]);

  useEffect(() => {
    routeCoordsRef.current = [];
  }, [directionsKey]);

  useEffect(() => {
    if (!mapFallbackCoord || resolving) return;
    fitMissionViewport({ useRoute: routeCoordsRef.current.length > 0 });
  }, [mapFallbackCoord, pickupCoord, dropoffCoord, resolving, routePlan.mode, fitMissionViewport]);

  useEffect(() => {
    if (!isLiveRoute || !driverCoord) return;
    const moved =
      lastDriverRef.current != null && !coordsEqual(lastDriverRef.current, driverCoord);
    lastDriverRef.current = driverCoord;
    if (!moved) return;

    const now = Date.now();
    if (now - lastAutoFitAtRef.current < LIVE_REFIT_MIN_MS) return;
    lastAutoFitAtRef.current = now;
    fitKeyPoints();
  }, [driverCoord, isLiveRoute, fitKeyPoints]);

  const mapLayerStyle = useMemo(() => compactMissionMapLayerStyle(height), [height]);

  if (!mapFallbackCoord) {
    return (
      <MissionMapUnavailable
        height={height}
        style={{ height }}
        pickupLocation={mapInput.pickupLocation}
        dropoffLocation={mapInput.dropoffLocation}
        loading={resolving}
      />
    );
  }

  return (
    <View style={{ width: "100%", height, overflow: "hidden", position: "relative" }}>
      <MapView
        ref={mapRef}
        provider={PROVIDER_GOOGLE}
        style={mapLayerStyle}
        initialRegion={region}
        showsUserLocation={false}
        showsMyLocationButton={false}
        loadingEnabled={false}
        onMapReady={onMapReady}
        {...mapStyleProps}
        {...Platform.select({
          ios: {
            legalLabelInsets: { top: 0, left: 0, right: 0, bottom: -120 },
            paddingAdjustmentBehavior: "never" as const,
          },
          default: {},
        })}
      >
        {pickupCoord ? (
          <FleetMapLiriePointMarker coordinate={pickupCoord} anchor={PICKUP_ANCHOR} />
        ) : null}
        {dropoffCoord ? (
          <FleetMapLiriePointMarker coordinate={dropoffCoord} anchor={DROPOFF_ANCHOR} />
        ) : null}
        {driverCoord && routePlan.mode !== "single_point" ? (
          <FleetMapLiriePointMarker coordinate={driverCoord} anchor={DRIVER_ANCHOR} />
        ) : null}
        {canRenderRoute && routePlan.origin && routePlan.destination && apiKey ? (
          <MapViewDirections
            key={directionsKey}
            origin={routePlan.origin}
            destination={routePlan.destination}
            apikey={apiKey}
            strokeWidth={routePlan.mode === "planned_full" ? 4 : 5}
            strokeColor={
              routePlan.mode === "planned_full"
                ? "rgba(0, 121, 107, 0.85)"
                : "rgba(13, 148, 136, 0.92)"
            }
            mode="DRIVING"
            onReady={onDirectionsReady}
          />
        ) : null}
      </MapView>

      <MissionMapLiveBadge
        prefix={routePlan.badgePrefix}
        distanceLabel={routeMetrics.distanceLabel}
        durationLabel={routeMetrics.durationLabel}
        live={isLiveRoute && Boolean(driverCoord)}
      />

      {props.showRecenterControl ? (
        <Pressable
          onPress={fitKeyPoints}
          accessibilityRole="button"
          accessibilityLabel="Recentrer la carte"
          style={({ pressed }) => [recenterStyles.btn, pressed && recenterStyles.btnPressed]}
        >
          <Ionicons name="locate-outline" size={18} color="#00796B" />
        </Pressable>
      ) : null}
    </View>
  );
}

const recenterStyles = {
  btn: {
    position: "absolute" as const,
    right: 10,
    bottom: 10,
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "#FFFFFF",
    alignItems: "center" as const,
    justifyContent: "center" as const,
    borderWidth: 1,
    borderColor: "rgba(0,121,107,0.12)",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.12,
    shadowRadius: 4,
    elevation: 3,
  },
  btnPressed: { opacity: 0.88 },
};
