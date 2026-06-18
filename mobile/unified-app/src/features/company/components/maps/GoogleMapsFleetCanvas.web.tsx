import { useEffect, useId, useMemo, useRef, useState } from "react";
import { StyleSheet, type Region } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import {
  conciseRouteSegment,
  resolveMissionUiStatus,
} from "../../dashboard/companyDashboardMissionUi";
import { loadGoogleMapsScriptWithKey } from "../../../../shared/google-maps/bootstrap";
import {
  resolveFleetMapsLibraryList,
  resolveGoogleMapsWebApiKey,
  diagnoseGoogleMapsWebKeyIssue,
  formatGoogleMapsWebKeyHelpMessage,
} from "../../../../config/googleMapsKeys";
import { attachGoogleMapAttributionHide } from "../../../maps/googleMapAttributionHide";
import { LiriWebMapFrame, LiriWebMapFramePlaceholder } from "../../../maps/LiriWebMapFrame.web";
import {
  buildLirieGoogleMapOptions,
  resolveExpoLirieGoogleMapLayer,
  triggerGoogleMapResize,
} from "../../../maps/expoLirieGoogleMapLayer";
import { enrichFleetDriver, computeFleetRegion , clusterSharesDriverIds } from "./fleetMapLogic";
import {
  buildMissionDirectionsPlanSignature,
  fleetRoutePathKey,
  type FleetRoutedPathState,
} from "./fleetMapDirections";
import { resolveFleetMissionRouteLegRenders } from "./fleetMapMissionVisual";
import { fetchFleetDirectionsPathsForOverlays } from "./fleetMapDirectionsWebApi";
import {
  applyFleetFitAllDrivers,
  applyFleetWebCamera,
  regionToZoom,
  type FleetWebMapCamera,
} from "./fleetMapWebCamera";
import { pickClusterRepresentativeStatus } from "./fleetLirieClusterMarker";
import { shouldFleetMarkerLivePulse } from "./fleetMapLiveMarker";
import { makeFleetVehicleMarkerDataUrl , createMissionDriverContextElement } from "./fleetMarkerIcons";

import { buildLirieDriverMarkerImageSource } from "./fleetLirieDriverMarkerAssets";
import { buildFleetClusterCountBadgeImageSource } from "./fleetNativeMarkerImage";
import type { FleetMissionOverlay } from "./fleetMapMissionVisual";
import type { FleetHeatmapPoint } from "./fleetMapGeo";
import type {
  FleetActiveRoute,
  FleetDriverMapItem,
  FleetMapLayersState,
  FleetMapMarker,
  FleetMapRecenterMode,
  FleetPinnedClusterFocus,
} from "./fleetMapTypes";
import { FLEET_MAP_MARKER_DIMMED_OPACITY , DEFAULT_FLEET_MAP_LAYERS } from "./fleetMapTypes";


import { FLEET_STATUS_THEME, type FleetOperationalStatus } from "./mapStatusTheme";
import { driverFleetMarkerTitle } from "../../utils/companyDriverMapStatus";
import type { FleetMarkerIconOptions } from "./fleetMarkerIcons";

const MAPS_ERROR_HELP_URL =
  "https://developers.google.com/maps/documentation/javascript/error-messages#api-target-blocked-map-error";

function getGoogleMaps(): Record<string, unknown> | undefined {
  if (typeof window === "undefined") return undefined;
  return (window as unknown as { google?: { maps: Record<string, unknown> } }).google?.maps;
}

function resolveAdaptiveMarkerSize(viewportWidth: number): number {
  if (viewportWidth <= 0) return 36;
  if (viewportWidth < 360) return 38;
  if (viewportWidth < 480) return 36;
  if (viewportWidth < 768) return 34;
  return 32;
}

function resolveWebMapType(layers: FleetMapLayersState): string {
  if (layers.mapType === "satellite") return "satellite";
  if (layers.mapType === "terrain") return "terrain";
  return "roadmap";
}

type ClassicFleetMarker = {
  setMap: (m: unknown) => void;
  setIcon?: (icon: { url: string; scaledSize: unknown; anchor: unknown }) => void;
  setZIndex?: (z: number) => void;
  addListener?: (event: string, handler: () => void) => void;
};

type DriverMarkerSelectionHandle = {
  driverId: number;
  setSelected: (selected: boolean) => void;
};

function buildDriverMarkerVisual(
  driver: FleetDriverMapItem,
  markerSize: number,
  selected: boolean
): {
  fill: string;
  markerOpts: FleetMarkerIconOptions;
  markerTitle: string;
  iconUrl: string;
  zIndex: number;
} {
  const status: FleetOperationalStatus = driver.enrichment.operationalStatus;
  const theme = FLEET_STATUS_THEME[status];
  const livePulse = shouldFleetMarkerLivePulse(status, driver);
  const markerOpts: FleetMarkerIconOptions = {
    sizePx: markerSize,
    selected,
    variant: theme.markerVariant,
    pulse: livePulse && Boolean(theme.pulse),
  };
  const markerTitle = driverFleetMarkerTitle(driver);
  const iconUrl =
    livePulse && markerOpts.pulse
      ? makeFleetVehicleMarkerDataUrl(theme.fill, markerOpts)
      : buildLirieDriverMarkerImageSource(status, markerSize).uri;
  return {
    fill: theme.fill,
    markerOpts,
    markerTitle,
    iconUrl,
    zIndex: selected ? 12 : 4,
  };
}

export type GoogleMapsFleetCanvasProps = {
  drivers: CompanyDriverLiveLocation[];
  markers?: FleetMapMarker[];
  height: number;
  selectedDriverId?: number | null;
  layers?: FleetMapLayersState;
  activeRoute?: FleetActiveRoute | null;
  missionOverlays?: FleetMissionOverlay[];
  missionsById?: Map<number, CompanyDispatchMission>;
  stableEtaAnchors?: Map<number, { latitude: number; longitude: number; atMs: number }>;
  selectedMissionId?: number | null;
  heatmapPoints?: FleetHeatmapPoint[];
  recenterRegion?: Region | null;
  recenterMode?: FleetMapRecenterMode;
  recenterToken?: number;
  cameraInsets?: { top: number; right: number; bottom: number; left: number };
  onDriverPress?: (driver: FleetDriverMapItem) => void;
  onClusterPress?: (drivers: FleetDriverMapItem[]) => void;
  pinnedClusterFocus?: FleetPinnedClusterFocus | null;
};

const MAP_ERR_AUTH_FR = [
  "Google Maps a refusé la clé (souvent ApiTargetBlockedMapError sur le web).",
  `Documentation : ${MAPS_ERROR_HELP_URL}`,
].join("\n");

const MAP_ERR_GENERIC_FR =
  "Impossible d'afficher la carte. Vérifiez EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.";

export function GoogleMapsFleetCanvas({
  drivers,
  markers: markersProp,
  height,
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
  pinnedClusterFocus = null,
}: GoogleMapsFleetCanvasProps) {
  const mapDomId = useId().replace(/:/g, "");
  const mapInstanceRef = useRef<(FleetWebMapCamera & { setMapTypeId?: (t: string) => void }) | null>(
    null
  );
  const overlaysRef = useRef<{ clear: () => void }[]>([]);
  const driverMarkerHandlesRef = useRef<DriverMarkerSelectionHandle[]>([]);
  const attributionHideCleanupRef = useRef<(() => void) | null>(null);
  const selectedDriverIdRef = useRef(selectedDriverId);
  const selectedMissionIdRef = useRef(selectedMissionId);
  const polylineRef = useRef<{ setMap: (m: unknown) => void } | null>(null);
  const polylineGlowRef = useRef<{ setMap: (m: unknown) => void } | null>(null);
  const routedPathsRef = useRef<Map<string, { latitude: number; longitude: number }[]>>(new Map());
  const routedStateRef = useRef<Map<string, FleetRoutedPathState>>(new Map());
  const routedPlanSignatureRef = useRef("");
  const [mapReady, setMapReady] = useState(false);
  const [mapLoadError, setMapLoadError] = useState<string | null>(null);
  const [viewportWidth, setViewportWidth] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth : 0
  );

  const libraryList = useMemo(() => resolveFleetMapsLibraryList(), []);

  const webKeyIssue = useMemo(() => diagnoseGoogleMapsWebKeyIssue(), []);
  const apiKey = useMemo(() => resolveGoogleMapsWebApiKey() ?? "", []);
  const webKeyHelpMessage = useMemo(() => formatGoogleMapsWebKeyHelpMessage(webKeyIssue), [webKeyIssue]);

  const fallbackMarkers = useMemo<FleetMapMarker[]>(
    () => drivers.map((d) => ({ kind: "driver", driver: enrichFleetDriver(d, []) })),
    [drivers]
  );
  const markers = markersProp ?? fallbackMarkers;
  const adaptiveMarkerSize = useMemo(
    () => resolveAdaptiveMarkerSize(viewportWidth),
    [viewportWidth]
  );
  const defaultRegion = useMemo(() => computeFleetRegion(drivers), [drivers]);

  const onDriverPressRef = useRef(onDriverPress);
  const onClusterPressRef = useRef(onClusterPress);
  useEffect(() => {
    onDriverPressRef.current = onDriverPress;
  }, [onDriverPress]);
  useEffect(() => {
    onClusterPressRef.current = onClusterPress;
  }, [onClusterPress]);
  useEffect(() => {
    selectedDriverIdRef.current = selectedDriverId;
  }, [selectedDriverId]);
  useEffect(() => {
    selectedMissionIdRef.current = selectedMissionId;
  }, [selectedMissionId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const onResize = () => setViewportWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    if (!apiKey) return;
    setMapLoadError(null);
    let cancelled = false;

    const run = async () => {
      try {
        await loadGoogleMapsScriptWithKey(apiKey, { libraryList });
        if (cancelled) return;
        const gmaps = getGoogleMaps();
        const el = document.getElementById(mapDomId);
        if (!el) {
          if (!cancelled) {
            setMapLoadError(
              `${MAP_ERR_GENERIC_FR}\n\nConteneur carte introuvable (id: ${mapDomId}). Rechargez la page.`
            );
          }
          return;
        }
        if (!gmaps) return;

        const MapCtor = gmaps.Map as new (
          el: HTMLElement,
          opts: Record<string, unknown>
        ) => typeof mapInstanceRef.current;

        const layer = resolveExpoLirieGoogleMapLayer();
        void layer;

        const mapTypeId = resolveWebMapType(layers);
        if (!mapInstanceRef.current) {
          mapInstanceRef.current = new MapCtor(
            el as HTMLElement,
            buildLirieGoogleMapOptions(mapTypeId, {
              center: { lat: defaultRegion.latitude, lng: defaultRegion.longitude },
              zoom: regionToZoom(defaultRegion.latitudeDelta ?? 0.12),
            })
          );
          if (!el.querySelector("[data-fleet-hide-gmp-pin]")) {
            const hideDefaultPinStyle = document.createElement("style");
            hideDefaultPinStyle.setAttribute("data-fleet-hide-gmp-pin", "1");
            hideDefaultPinStyle.textContent =
              "gmp-pin,.GMAMP-maps-pin-view,.IPAZAH-content-container gmp-pin{display:none!important;}";
            el.appendChild(hideDefaultPinStyle);
          }
          if (el instanceof HTMLElement) {
            attributionHideCleanupRef.current?.();
            attributionHideCleanupRef.current = attachGoogleMapAttributionHide(el);
          }
        }

        const map = mapInstanceRef.current;
        if (!map) return;

        const mapWithOptions = map as {
          setMapTypeId?: (t: string) => void;
          setOptions?: (o: Record<string, unknown>) => void;
        };
        mapWithOptions.setOptions?.(buildLirieGoogleMapOptions(mapTypeId));
        mapWithOptions.setMapTypeId?.(mapTypeId);

        for (const o of overlaysRef.current) o.clear();
        overlaysRef.current = [];
        driverMarkerHandlesRef.current = [];

        const TrafficLayerCtor = (
          gmaps as { TrafficLayer?: new () => { setMap: (m: unknown) => void } }
        ).TrafficLayer;
        if (layers.traffic && TrafficLayerCtor) {
          const traffic = new TrafficLayerCtor();
          traffic.setMap(map);
          overlaysRef.current.push({ clear: () => traffic.setMap(null) });
        }

        const CircleCtor = (
          gmaps as {
            Circle?: new (opts: Record<string, unknown>) => { setMap: (m: unknown) => void };
          }
        ).Circle;
        if (heatmapPoints.length > 0 && CircleCtor) {
          for (const p of heatmapPoints) {
            const circle = new CircleCtor({
              map,
              center: { lat: p.latitude, lng: p.longitude },
              radius: 220 + p.weight * 35,
              fillColor: "#EF4444",
              fillOpacity: 0.2,
              strokeWeight: 0,
              clickable: false,
            });
            overlaysRef.current.push({ clear: () => circle.setMap(null) });
          }
        }
        if (polylineRef.current) {
          polylineRef.current.setMap(null);
          polylineRef.current = null;
        }
        if (polylineGlowRef.current) {
          polylineGlowRef.current.setMap(null);
          polylineGlowRef.current = null;
        }

        const placeMarker = (
          position: { lat: number; lng: number },
          payload: {
            iconUrl: string;
            title: string;
            size: number;
            height?: number;
            anchor?: { x: number; y: number };
            zIndex?: number;
            opacity?: number;
          },
          onClick?: () => void
        ): ClassicFleetMarker | null => {
          const { iconUrl, title, size, zIndex } = payload;
          const height = payload.height ?? size;
          const anchor = payload.anchor ?? { x: 0.5, y: 0.5 };
          const MarkerCtor = gmaps.Marker as new (opts: Record<string, unknown>) => ClassicFleetMarker;
          const SizeCtor = gmaps.Size as new (w: number, h: number) => unknown;
          const PointCtor = gmaps.Point as new (x: number, y: number) => unknown;
          const marker = new MarkerCtor({
            map,
            position,
            title,
            icon: {
              url: iconUrl,
              scaledSize: new SizeCtor(size, height),
              anchor: new PointCtor(size * anchor.x, height * anchor.y),
            },
            optimized: true,
            zIndex: zIndex ?? 2,
          });
          if (onClick && marker.addListener) marker.addListener("click", onClick);
          overlaysRef.current.push({ clear: () => marker.setMap(null) });
          return marker;
        };

        const placeHtmlOverlay = (
          position: { lat: number; lng: number },
          htmlElement: HTMLElement,
          zIndex = 2
        ): void => {
          const OverlayViewCtor = gmaps.OverlayView as new () => {
            setMap: (m: unknown) => void;
            getPanes: () => { floatPane: HTMLElement } | null;
            getProjection: () => {
              fromLatLngToDivPixel: (p: unknown) => { x: number; y: number } | null;
            } | null;
          };
          const LatLngCtor = gmaps.LatLng as new (lat: number, lng: number) => unknown;
          const overlay = new OverlayViewCtor();
          const div = htmlElement;
          div.style.position = "absolute";
          div.style.pointerEvents = "none";
          div.style.zIndex = String(zIndex);
          overlay.onAdd = function onAdd() {
            overlay.getPanes()?.floatPane.appendChild(div);
          };
          overlay.draw = function draw() {
            const point = overlay.getProjection()?.fromLatLngToDivPixel(
              new LatLngCtor(position.lat, position.lng)
            );
            if (!point) return;
            div.style.left = `${point.x}px`;
            div.style.top = `${point.y}px`;
            div.style.transform = "translate(-50%, -100%)";
          };
          overlay.onRemove = function onRemove() {
            div.remove();
          };
          overlay.setMap(map);
          overlaysRef.current.push({ clear: () => overlay.setMap(null) });
        };

        const placeMissionAnchorMarker = (
          position: { lat: number; lng: number },
          payload: {
            fill: string;
            radius: number;
            stroke?: string;
            selected?: boolean;
            title: string;
            zIndex?: number;
          }
        ): void => {
          const size = payload.radius * 2 + (payload.selected ? 8 : 6);
          const stroke = payload.stroke ?? "#ffffff";
          const border = payload.selected ? 3 : 2;
          const iconUrl = `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(
            `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
              <circle cx="${size / 2}" cy="${size / 2}" r="${payload.radius}" fill="${payload.fill}" stroke="${stroke}" stroke-width="${border}"/>
            </svg>`
          )}`;
          placeMarker(position, {
            iconUrl,
            title: payload.title,
            size,
            zIndex: payload.zIndex,
          });
        };

        const showDriverMarkers = layers.heatmapMode === "off";
        const pinnedClusterActive =
          pinnedClusterFocus != null &&
          selectedDriverIdRef.current != null &&
          pinnedClusterFocus.drivers.some((driver) => driver.driver_id === selectedDriverIdRef.current);
        const pinnedDriverIds = pinnedClusterActive
          ? new Set(pinnedClusterFocus.drivers.map((driver) => driver.driver_id))
          : null;
        const dimmedOpacity = FLEET_MAP_MARKER_DIMMED_OPACITY;

        for (const m of markers) {
          if (!showDriverMarkers) continue;
          if (m.kind === "cluster") {
            if (pinnedDriverIds && clusterSharesDriverIds(m.drivers, pinnedDriverIds)) {
              continue;
            }
            const size = Math.min(48, adaptiveMarkerSize + 8);
            const clusterTitle = `${m.count} chauffeurs`;
            const clusterStatus = pickClusterRepresentativeStatus(m.drivers);
            const icon = buildLirieDriverMarkerImageSource(clusterStatus, size);
            const badge = buildFleetClusterCountBadgeImageSource(m.count);
            const onClusterTap = () => onClusterPressRef.current?.(m.drivers);
            placeMarker(
              { lat: m.latitude, lng: m.longitude },
              {
                iconUrl: icon.uri,
                title: clusterTitle,
                size: icon.width,
                height: icon.height,
                anchor: { x: 0.5, y: 1 },
                zIndex: 3,
              },
              onClusterTap
            );
            placeMarker(
              { lat: m.latitude, lng: m.longitude },
              {
                iconUrl: badge.uri,
                title: clusterTitle,
                size: badge.width,
                height: badge.height,
                anchor: FLEET_CLUSTER_COUNT_BADGE_ANCHOR,
                zIndex: 4,
              },
              onClusterTap
            );
            continue;
          }

          if (
            pinnedDriverIds?.has(m.driver.driver_id) &&
            m.driver.driver_id !== selectedDriverIdRef.current
          ) {
            continue;
          }

          const d = m.driver;
          const linkedMissionId = d.enrichment.linkedMission?.mission_id ?? null;
          const missionSelected =
            selectedMissionIdRef.current != null &&
            linkedMissionId === selectedMissionIdRef.current;
          const selected = d.driver_id === selectedDriverIdRef.current || missionSelected;
          const visual = buildDriverMarkerVisual(d, adaptiveMarkerSize, selected);
          const dimmed =
            (selectedDriverIdRef.current != null && !selected) ||
            (selectedMissionIdRef.current != null &&
              !missionSelected &&
              d.driver_id !== selectedDriverIdRef.current);
          const markerInstance = placeMarker(
            { lat: d.latitude, lng: d.longitude },
            {
              iconUrl: visual.iconUrl,
              title: visual.markerTitle,
              size: adaptiveMarkerSize,
              zIndex: visual.zIndex,
            },
            () => onDriverPressRef.current?.(d)
          );
          if (markerInstance) {
            const driverId = d.driver_id;
            if (dimmed) {
              (markerInstance as ClassicFleetMarker & { setOpacity?: (o: number) => void }).setOpacity?.(
                dimmedOpacity
              );
            }
            driverMarkerHandlesRef.current.push({
              driverId,
              setSelected: (nextSelected) => {
                const next = buildDriverMarkerVisual(d, adaptiveMarkerSize, nextSelected);
                if (markerInstance.setIcon) {
                  const SizeCtor = gmaps.Size as new (w: number, h: number) => unknown;
                  const PointCtor = gmaps.Point as new (x: number, y: number) => unknown;
                  const r = adaptiveMarkerSize / 2;
                  markerInstance.setIcon({
                    url: next.iconUrl,
                    scaledSize: new SizeCtor(adaptiveMarkerSize, adaptiveMarkerSize),
                    anchor: new PointCtor(r, r),
                  });
                }
                markerInstance.setZIndex?.(next.zIndex);
                (markerInstance as ClassicFleetMarker & { setOpacity?: (o: number) => void }).setOpacity?.(
                  nextSelected ? 1 : dimmed ? dimmedOpacity : 1
                );
              },
            });
          }
        }

        if (pinnedClusterActive && pinnedClusterFocus && selectedDriverIdRef.current != null) {
          const focusedDriverInMarkers = markers.some(
            (marker) =>
              marker.kind === "driver" && marker.driver.driver_id === selectedDriverIdRef.current
          );
          const size = Math.min(48, adaptiveMarkerSize + 8);
          const clusterTitle = `${pinnedClusterFocus.count} chauffeurs`;
          const clusterStatus = pickClusterRepresentativeStatus(pinnedClusterFocus.drivers);
          const icon = buildLirieDriverMarkerImageSource(clusterStatus, size);
          const badge = buildFleetClusterCountBadgeImageSource(pinnedClusterFocus.count);
          const onPinnedClusterTap = () => onClusterPressRef.current?.(pinnedClusterFocus.drivers);
          const position = {
            lat: pinnedClusterFocus.latitude,
            lng: pinnedClusterFocus.longitude,
          };
          placeMarker(
            position,
            {
              iconUrl: icon.uri,
              title: clusterTitle,
              size: icon.width,
              height: icon.height,
              anchor: { x: 0.5, y: 1 },
              zIndex: 3,
              opacity: dimmedOpacity,
            },
            onPinnedClusterTap
          );
          placeMarker(
            position,
            {
              iconUrl: badge.uri,
              title: clusterTitle,
              size: badge.width,
              height: badge.height,
              anchor: FLEET_CLUSTER_COUNT_BADGE_ANCHOR,
              zIndex: 4,
              opacity: dimmedOpacity,
            },
            onPinnedClusterTap
          );

          if (!focusedDriverInMarkers) {
            const focusedDriver = pinnedClusterFocus.drivers.find(
              (driver) => driver.driver_id === selectedDriverIdRef.current
            );
            if (focusedDriver) {
              const visual = buildDriverMarkerVisual(focusedDriver, adaptiveMarkerSize, true);
              placeMarker(
                { lat: focusedDriver.latitude, lng: focusedDriver.longitude },
                {
                  iconUrl: visual.iconUrl,
                  title: visual.markerTitle,
                  size: adaptiveMarkerSize,
                  zIndex: visual.zIndex,
                  opacity: 1,
                },
                () => onDriverPressRef.current?.(focusedDriver)
              );
            }
          }
        }

        const PolylineCtor = gmaps.Polyline as new (opts: Record<string, unknown>) => {
          setMap: (m: unknown) => void;
        };

        const drawMissionRoutes = missionOverlays.length > 0;
        const planSignature = drawMissionRoutes
          ? buildMissionDirectionsPlanSignature(missionOverlays)
          : "";
        if (drawMissionRoutes && planSignature !== routedPlanSignatureRef.current) {
          for (const overlay of missionOverlays) {
            if (overlay.legDirectionsPlans) {
              for (const leg of ["to_pickup", "to_dropoff"] as const) {
                if (overlay.legDirectionsPlans[leg]) {
                  routedStateRef.current.set(fleetRoutePathKey(overlay.missionId, leg), "loading");
                }
              }
            } else if (overlay.directionsPlan) {
              routedStateRef.current.set(fleetRoutePathKey(overlay.missionId), "loading");
            }
          }
          const fetched = await fetchFleetDirectionsPathsForOverlays(gmaps, missionOverlays);
          if (cancelled) return;
          routedPlanSignatureRef.current = planSignature;
          routedPathsRef.current = new Map();
          routedStateRef.current = new Map();
          for (const overlay of missionOverlays) {
            if (overlay.legDirectionsPlans) {
              for (const leg of ["to_pickup", "to_dropoff"] as const) {
                if (!overlay.legDirectionsPlans[leg]) continue;
                const key = fleetRoutePathKey(overlay.missionId, leg);
                const path = fetched.get(key);
                if (path && path.length >= 2) {
                  routedPathsRef.current.set(key, path);
                  routedStateRef.current.set(key, "ready");
                } else {
                  routedStateRef.current.set(key, "failed");
                }
              }
              continue;
            }
            if (!overlay.directionsPlan) continue;
            const key = fleetRoutePathKey(overlay.missionId);
            const path = fetched.get(key);
            if (path && path.length >= 2) {
              routedPathsRef.current.set(key, path);
              routedStateRef.current.set(key, "ready");
            } else {
              routedStateRef.current.set(key, "failed");
            }
          }
        }
        if (cancelled) return;
        const routedPathsByMissionId = drawMissionRoutes ? routedPathsRef.current : new Map();
        const routedStateByMissionId = drawMissionRoutes ? routedStateRef.current : new Map();

        if (drawMissionRoutes) {
          for (const overlay of missionOverlays) {
            const legRenders = resolveFleetMissionRouteLegRenders(
              overlay,
              routedPathsByMissionId,
              routedStateByMissionId
            );
            for (const legRender of legRenders) {
              if (legRender.coordinates.length < 2) continue;
              const path = legRender.coordinates.map((p) => ({
                lat: p.latitude,
                lng: p.longitude,
              }));
              const strokeOpacity = overlay.displayOpacity * legRender.style.opacity;
              const glow = new PolylineCtor({
                path,
                geodesic: false,
                strokeColor: legRender.style.glowColor,
                strokeOpacity: strokeOpacity * 0.85,
                strokeWeight: legRender.style.glowWidth,
                map,
                zIndex: legRender.zIndex,
              });
              overlaysRef.current.push({ clear: () => glow.setMap(null) });
              const line = new PolylineCtor({
                path,
                geodesic: false,
                strokeColor: legRender.style.color,
                strokeOpacity,
                strokeWeight: legRender.style.strokeWidth,
                map,
                zIndex: legRender.zIndex + 1,
              });
              overlaysRef.current.push({ clear: () => line.setMap(null) });
            }

            if (overlay.pickup && overlay.pickupAnchor) {
              const anchor = overlay.pickupAnchor;
              placeMissionAnchorMarker(
                { lat: overlay.pickup.latitude, lng: overlay.pickup.longitude },
                {
                  fill: anchor.fill,
                  radius: anchor.radius,
                  stroke: anchor.stroke,
                  selected: overlay.isSelected,
                  title: "Prise en charge",
                  zIndex: anchor.zIndex,
                }
              );
            }
            if (overlay.dropoff && overlay.dropoffAnchor) {
              const anchor = overlay.dropoffAnchor;
              placeMissionAnchorMarker(
                { lat: overlay.dropoff.latitude, lng: overlay.dropoff.longitude },
                {
                  fill: anchor.fill,
                  radius: anchor.radius,
                  stroke: anchor.stroke,
                  selected: overlay.isSelected,
                  title: "Destination",
                  zIndex: anchor.zIndex,
                }
              );
            }
          }

          const badgeOverlay =
            missionOverlays.find(
              (o) => o.showEtaBadge && !o.isSelected && o.missionId === selectedMissionIdRef.current
            ) ?? missionOverlays.find((o) => o.showEtaBadge && !o.isSelected);
          if (badgeOverlay?.etaBadgeLabel) {
            const legRenders = resolveFleetMissionRouteLegRenders(
              badgeOverlay,
              routedPathsByMissionId,
              routedStateByMissionId
            );
            const routePoints = legRenders[0]?.coordinates ?? [];
            if (routePoints.length >= 2) {
              const stable = stableEtaAnchors?.get(badgeOverlay.missionId);
              const mid = stable
                ? { lat: stable.latitude, lng: stable.longitude }
                : (() => {
                    const p =
                      routePoints[Math.floor(routePoints.length / 2)] ?? routePoints[0];
                    return { lat: p.latitude, lng: p.longitude };
                  })();
              const pillDom = document.createElement("div");
              pillDom.textContent = badgeOverlay.etaBadgeLabel;
              pillDom.style.cssText = [
                "background:rgba(0,121,107,0.96)",
                "color:#fff",
                "font-size:11px",
                "font-weight:700",
                "padding:5px 10px",
                "min-width:58px",
                "max-width:112px",
                "text-align:center",
                "border-radius:999px",
                "border:1px solid rgba(255,255,255,0.66)",
                "white-space:nowrap",
                "overflow:hidden",
                "text-overflow:ellipsis",
                "box-shadow:0 8px 18px rgba(0,121,107,0.25)",
              ].join(";");
              placeHtmlOverlay(mid, pillDom, 200);
            }
          }
        } else if (activeRoute && activeRoute.points.length >= 2) {
          const path = activeRoute.points.map((p) => ({ lat: p.latitude, lng: p.longitude }));
          polylineGlowRef.current = new PolylineCtor({
            path,
            geodesic: true,
            strokeColor: "#00796B",
            strokeOpacity: 0.26,
            strokeWeight: 11,
            map,
          });
          polylineRef.current = new PolylineCtor({
            path,
            geodesic: true,
            strokeColor: activeRoute.color,
            strokeOpacity: 0.94,
            strokeWeight: 5,
            map,
          });
        }

        const selectedOverlayForDriver =
          selectedMissionIdRef.current != null
            ? missionOverlays.find((o) => o.missionId === selectedMissionIdRef.current)
            : null;
        if (selectedDriverIdRef.current != null) {
          const driverEntry = markers.find(
            (m): m is Extract<typeof m, { kind: "driver" }> =>
              m.kind === "driver" && m.driver.driver_id === selectedDriverIdRef.current
          );
          const mission =
            (selectedOverlayForDriver
              ? missionsById?.get(selectedOverlayForDriver.missionId)
              : null) ??
            driverEntry?.driver.enrichment.linkedMission ??
            null;
          if (mission && driverEntry) {
            const status = resolveMissionUiStatus(mission);
            const eta =
              selectedOverlayForDriver?.etaBadgeLabel ??
              selectedOverlayForDriver?.etaLabel ??
              null;
            const pickup = conciseRouteSegment(mission.pickup_label, 22) || "Prise en charge";
            const dropoff = conciseRouteSegment(mission.dropoff_label, 22) || "Destination";
            const routeLine = `${pickup} → ${dropoff}`;
            const metaLine = `${status.label}${eta ? ` · ${eta}` : ""}`;
            const ctxDom = createMissionDriverContextElement({
              routeLine,
              metaLine,
              selected: true,
            });
            if (ctxDom) {
              placeHtmlOverlay(
                {
                  lat: driverEntry.driver.latitude + 0.00055,
                  lng: driverEntry.driver.longitude,
                },
                ctxDom,
                220
              );
            }
          }
        }

        setMapReady(true);
        triggerGoogleMapResize(map, gmaps);
      } catch (e: unknown) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : "";
        if (msg.includes("Authentification") || msg.includes("ApiTargetBlocked")) {
          setMapLoadError(MAP_ERR_AUTH_FR);
        } else {
          setMapLoadError(MAP_ERR_GENERIC_FR);
        }
      }
    };

    const t = window.setTimeout(() => void run(), 0);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [
    activeRoute,
    missionOverlays,
    missionsById,
    stableEtaAnchors,
    selectedMissionId,
    adaptiveMarkerSize,
    apiKey,
    defaultRegion.latitude,
    defaultRegion.longitude,
    defaultRegion.latitudeDelta,
    drivers,
    heatmapPoints,
    layers,
    libraryList,
    mapDomId,
    markers,
    pinnedClusterFocus,
    selectedDriverId,
  ]);

  useEffect(() => {
    const selectedId = selectedDriverId ?? null;
    for (const handle of driverMarkerHandlesRef.current) {
      handle.setSelected(handle.driverId === selectedId);
    }
  }, [selectedDriverId, adaptiveMarkerSize]);

  useEffect(() => {
    if (!apiKey) return;
    const map = mapInstanceRef.current;
    const gmaps = getGoogleMaps();
    if (!map || !gmaps) return;
    const t = window.requestAnimationFrame(() => triggerGoogleMapResize(map, gmaps));
    return () => window.cancelAnimationFrame(t);
  }, [apiKey, height, mapDomId]);

  useEffect(() => {
    if (!apiKey || !mapReady) return;
    let cancelled = false;

    const run = () => {
      if (cancelled) return;
      const map = mapInstanceRef.current;
      const gmaps = getGoogleMaps();
      if (!map || !gmaps) return;

      const LatLng = gmaps.LatLng as new (lat: number, lng: number) => unknown;
      const LatLngBounds = gmaps.LatLngBounds as new () => { extend: (p: unknown) => void };
      const defaultCenter = {
        latitude: defaultRegion.latitude,
        longitude: defaultRegion.longitude,
      };

      if (recenterMode === "all") {
        applyFleetFitAllDrivers(
          map,
          LatLng,
          LatLngBounds,
          drivers,
          markers,
          defaultCenter,
          cameraInsets
        );
      } else {
        applyFleetWebCamera(map, LatLng, LatLngBounds, {
          recenterRegion,
          markers,
          defaultCenter,
          fitPadding: cameraInsets,
        });
      }
      triggerGoogleMapResize(map, gmaps);
    };

    const frame = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(run);
    });

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
    };
  }, [
    apiKey,
    defaultRegion.latitude,
    defaultRegion.longitude,
    drivers,
    mapReady,
    markers,
    recenterMode,
    recenterRegion,
    recenterToken,
    cameraInsets,
  ]);

  useEffect(
    () => () => {
      attributionHideCleanupRef.current?.();
      attributionHideCleanupRef.current = null;
      for (const o of overlaysRef.current) o.clear();
      overlaysRef.current = [];
      if (polylineRef.current) polylineRef.current.setMap(null);
      if (polylineGlowRef.current) polylineGlowRef.current.setMap(null);
      mapInstanceRef.current = null;
      setMapReady(false);
    },
    []
  );

  if (!apiKey) {
    return (
      <LiriWebMapFramePlaceholder height={height} accessibilityLabel="Carte flotte indisponible">
        <AppText variant="caption" style={styles.fallbackText}>
          {webKeyHelpMessage || MAP_ERR_GENERIC_FR}
        </AppText>
      </LiriWebMapFramePlaceholder>
    );
  }

  if (mapLoadError) {
    return (
      <LiriWebMapFramePlaceholder height={height} accessibilityLabel="Erreur carte">
        <AppText variant="caption" style={styles.fallbackText}>
          {mapLoadError}
        </AppText>
      </LiriWebMapFramePlaceholder>
    );
  }

  return <LiriWebMapFrame height={height} mapDomId={mapDomId} accessibilityLabel="Carte opérationnelle" />;
}

const styles = StyleSheet.create({
  fallbackText: { color: "#64748B", lineHeight: 18 },
});
