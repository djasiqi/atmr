import { useEffect, useId, useMemo, useRef, useState } from "react";
import { StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import {
  loadGoogleMapsScriptWithKey,
  parseGoogleMapsLibraryList,
} from "../../../../shared/google-maps/bootstrap";
import {
  diagnoseGoogleMapsWebKeyIssue,
  formatGoogleMapsWebKeyHelpMessage,
  resolveGoogleMapsWebApiKey,
} from "../../../../config/googleMapsKeys";
import { MISSION_MAP_FALLBACK_HEIGHT } from "./missionMapShared";
import { LiriWebMapFrame, LiriWebMapFramePlaceholder } from "../../../maps/LiriWebMapFrame.web";
import { MissionMapUnavailable } from "../MissionMapUnavailable";
import { MissionMapLiveBadge } from "../MissionMapLiveBadge";
import { useMissionMapResolvedCoords } from "../../hooks/useMissionMapResolvedCoords";
import { useDriverLiveMapPosition } from "../../hooks/useDriverLiveMapPosition";
import { useMissionRouteMetrics } from "../../hooks/useMissionRouteMetrics";
import {
  isMissionMapLiveRouteStatus,
  resolveMissionMapRoute,
} from "../../domain/resolveMissionMapRoute";
import type { DriverMission } from "../../types";
import {
  getLirieBaseMapUiOptions,
  makeCircleMarkerDataUrl,
  resolveExpoLirieGoogleMapLayer,
} from "../../../maps/expoLirieGoogleMapLayer";

const MAPS_ERROR_HELP_URL =
  "https://developers.google.com/maps/documentation/javascript/error-messages#api-target-blocked-map-error";

function isPlausibleGoogleMapsBrowserKey(k: string): boolean {
  const t = k.trim();
  if (t.length < 20) return false;
  const lower = t.toLowerCase();
  if (lower.includes("ta_clef")) return false;
  if (lower.includes("google_maps_js")) return false;
  if (lower.includes("your_api")) return false;
  if (lower.includes("replace_me")) return false;
  if (lower.includes("changeme")) return false;
  if (lower.includes("example_key")) return false;
  if (lower.includes("placeholder")) return false;
  return true;
}

function getGoogleMaps(): Record<string, unknown> | undefined {
  if (typeof window === "undefined") return undefined;
  const g = (window as unknown as { google?: { maps: Record<string, unknown> } }).google?.maps;
  return g;
}

const MAP_ERR_AUTH_FR = [
  "Google Maps a refuse la cle (souvent ApiTargetBlockedMapError sur le web). Verifiez dans Google Cloud :",
  "• l'API « Maps JavaScript API » est activee ;",
  "• la cle autorise l'origine exacte (ex. http://localhost:8081) ;",
  "• la facturation est activee.",
  `Documentation : ${MAPS_ERROR_HELP_URL}`,
].join("\n");

const MAP_ERR_GENERIC_FR =
  "Impossible d'afficher la carte. Rechargez la page ou verifiez EXPO_PUBLIC_GOOGLE_MAPS_API_KEY (cle web Maps JavaScript API).";

function formatMapInitError(e: unknown): string {
  const raw = e instanceof Error ? e.message : typeof e === "string" ? e : "";
  const t = raw.trim();
  if (t.length === 0) return "";
  if (t.length > 220) return `${t.slice(0, 217)}…`;
  return t;
}

type Props = {
  pickupLat?: number | null;
  pickupLng?: number | null;
  dropoffLat?: number | null;
  dropoffLng?: number | null;
  pickupLocation?: string | null;
  dropoffLocation?: string | null;
  driverLat?: unknown;
  driverLng?: unknown;
  missionStatus?: string | null;
  etaMinutes?: number | null;
  height?: number;
};

type MapLike = {
  setCenter: (x: unknown) => void;
  setZoom: (z: number) => void;
};

type DirectionsRendererLike = { setMap: (m: unknown) => void; setDirections: (r: unknown) => void };

type LegacyMarkerLike = { setMap: (m: unknown) => void };

/**
 * Carte mission web : Google Maps JS (itineraire si pickup + dropoff, sinon marqueur).
 * Meme bootstrap / cle que `GoogleMapsFleetCanvas.web`.
 */
export function GoogleMapsMissionRoute(props: Props) {
  const mapDomId = useId().replace(/:/g, "");
  const height = props.height ?? MISSION_MAP_FALLBACK_HEIGHT;
  const { pickupCoord, dropoffCoord, fallbackCoord, resolving } = useMissionMapResolvedCoords({
    pickupLat: props.pickupLat,
    pickupLng: props.pickupLng,
    dropoffLat: props.dropoffLat,
    dropoffLng: props.dropoffLng,
    pickupLocation: props.pickupLocation,
    dropoffLocation: props.dropoffLocation,
    driverLat: props.driverLat,
    driverLng: props.driverLng,
  });

  const liveDriverCoord = useDriverLiveMapPosition(props.driverLat, props.driverLng, true);

  const pickupLat = pickupCoord?.latitude ?? null;
  const pickupLng = pickupCoord?.longitude ?? null;
  const dropoffLat = dropoffCoord?.latitude ?? null;
  const dropoffLng = dropoffCoord?.longitude ?? null;
  const driverLat = liveDriverCoord?.latitude ?? null;
  const driverLng = liveDriverCoord?.longitude ?? null;
  const mapFallbackCoord = liveDriverCoord ?? pickupCoord ?? dropoffCoord ?? fallbackCoord;
  const firstLat = mapFallbackCoord?.latitude ?? null;
  const firstLng = mapFallbackCoord?.longitude ?? null;

  const routePlan = useMemo(
    () =>
      resolveMissionMapRoute({
        status: props.missionStatus,
        driverCoord: liveDriverCoord,
        pickupCoord,
        dropoffCoord,
      }),
    [props.missionStatus, liveDriverCoord, pickupCoord, dropoffCoord]
  );

  const missionStub = useMemo(
    () =>
      ({
        id: 0,
        status: props.missionStatus ?? "",
        pickup_location: props.pickupLocation,
        dropoff_location: props.dropoffLocation,
        pickup_lat: pickupLat,
        pickup_lng: pickupLng,
        dropoff_lat: dropoffLat,
        dropoff_lng: dropoffLng,
      }) as DriverMission,
    [
      props.missionStatus,
      props.pickupLocation,
      props.dropoffLocation,
      pickupLat,
      pickupLng,
      dropoffLat,
      dropoffLng,
    ]
  );

  const routeMetrics = useMissionRouteMetrics(missionStub, { etaMinutes: props.etaMinutes });
  const isLiveRoute = isMissionMapLiveRouteStatus(props.missionStatus);

  const mapRef = useRef<MapLike | null>(null);
  const directionsRendererRef = useRef<DirectionsRendererLike | null>(null);
  const markerDisposersRef = useRef<Array<() => void>>([]);
  const [mapLoadError, setMapLoadError] = useState<string | null>(null);

  const libraryList = useMemo(
    () => parseGoogleMapsLibraryList(process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES || ""),
    []
  );

  const webKeyIssue = useMemo(() => diagnoseGoogleMapsWebKeyIssue(), []);
  const apiKey = useMemo(() => resolveGoogleMapsWebApiKey() ?? "", [webKeyIssue]);
  const webKeyHelpMessage = useMemo(() => formatGoogleMapsWebKeyHelpMessage(webKeyIssue), [webKeyIssue]);

  useEffect(() => {
    if (firstLat == null || firstLng == null) return;
    if (!apiKey) return;

    setMapLoadError(null);
    let cancelled = false;

    const clearDirections = () => {
      if (directionsRendererRef.current) {
        directionsRendererRef.current.setMap(null);
        directionsRendererRef.current = null;
      }
    };

    const clearMarkers = () => {
      for (const dispose of markerDisposersRef.current) {
        try {
          dispose();
        } catch {
          /* ignore */
        }
      }
      markerDisposersRef.current = [];
    };

    const disposeMapDom = () => {
      const el = document.getElementById(mapDomId);
      if (el) el.innerHTML = "";
      mapRef.current = null;
    };

    /** Marqueur LIRIE minimal (disque P/D) — jamais de pin Google / PinElement. */
    function placeMissionMarker(
      gmaps: Record<string, unknown>,
      map: MapLike,
      lat: number,
      lng: number,
      title: string,
      role: "pickup" | "dropoff" | "mission" | "driver",
      fillOverride?: string
    ) {
      const isPickup = role === "pickup" || role === "mission";
      const fill =
        fillOverride ??
        (role === "driver" ? "#0D9488" : isPickup ? "#00796B" : "#1E293B");
      const MarkerCtor = gmaps.Marker as new (opts: Record<string, unknown>) => LegacyMarkerLike;
      const SizeCtor = gmaps.Size as new (w: number, h: number) => unknown;
      const PointCtor = gmaps.Point as new (x: number, y: number) => unknown;
      const sizePx = 28;
      const half = sizePx / 2;
      const marker = new MarkerCtor({
        map,
        position: { lat, lng },
        title,
        icon: {
          url: makeCircleMarkerDataUrl(fill, 1, sizePx, {
            label: role === "driver" ? "C" : isPickup ? "P" : "D",
          }),
          scaledSize: new SizeCtor(sizePx, sizePx),
          anchor: new PointCtor(half, half),
        },
        optimized: true,
      });
      markerDisposersRef.current.push(() => {
        marker.setMap(null);
      });
    }

    const run = async () => {
      try {
        await loadGoogleMapsScriptWithKey(apiKey, { libraryList });
        const gmaps = getGoogleMaps();
        if (cancelled || !gmaps) return;

        if (typeof gmaps.importLibrary === "function") {
          await (gmaps.importLibrary as (name: string) => Promise<unknown>)("routes");
        }

        const el = document.getElementById(mapDomId);
        if (!el) {
          if (!cancelled) {
            setMapLoadError(
              `${MAP_ERR_GENERIC_FR}\n\nConteneur carte introuvable (id: ${mapDomId}). Rechargez la page.`
            );
          }
          return;
        }
        if (cancelled) return;

        const layer = resolveExpoLirieGoogleMapLayer();

        clearDirections();
        clearMarkers();
        disposeMapDom();

        const MapCtor = gmaps.Map as new (el: HTMLElement, opts: Record<string, unknown>) => MapLike;
        const baseOpts: Record<string, unknown> = {
          ...getLirieBaseMapUiOptions(),
          center: { lat: firstLat, lng: firstLng },
          zoom: 15,
        };
        if (layer.kind === "cloud") {
          baseOpts.mapId = layer.mapId;
        } else {
          baseOpts.styles = [...layer.styles];
        }
        mapRef.current = new MapCtor(el as HTMLElement, baseOpts);
        const map = mapRef.current;

        const origin = routePlan.origin;
        const destination = routePlan.destination;
        const canRoute =
          origin != null &&
          destination != null &&
          routePlan.mode !== "single_point" &&
          (origin.latitude !== destination.latitude ||
            origin.longitude !== destination.longitude);

        const placeAllMarkers = () => {
          if (pickupLat != null && pickupLng != null) {
            placeMissionMarker(gmaps, map, pickupLat, pickupLng, "Depart", "pickup");
          }
          if (dropoffLat != null && dropoffLng != null) {
            placeMissionMarker(gmaps, map, dropoffLat, dropoffLng, "Arrivee", "dropoff");
          }
          if (driverLat != null && driverLng != null) {
            placeMissionMarker(gmaps, map, driverLat, driverLng, "Chauffeur", "driver");
          }
        };

        if (canRoute && origin && destination) {
          const DirectionsService = gmaps.DirectionsService as new () => {
            route: (
              req: Record<string, unknown>,
              cb: (result: unknown | null, status: string) => void
            ) => void;
          };
          const DirectionsRenderer = gmaps.DirectionsRenderer as new (opts: Record<string, unknown>) => DirectionsRendererLike;

          const directionsService = new DirectionsService();
          const directionsRenderer = new DirectionsRenderer({
            map,
            suppressMarkers: true,
            preserveViewport: false,
          });
          directionsRendererRef.current = directionsRenderer;

          directionsService.route(
            {
              origin: { lat: origin.latitude, lng: origin.longitude },
              destination: { lat: destination.latitude, lng: destination.longitude },
              travelMode: "DRIVING",
            },
            (result, status) => {
              if (cancelled) return;
              if (status === "OK" && result) {
                directionsRenderer.setDirections(result);
              } else {
                map.setCenter({ lat: firstLat, lng: firstLng });
                map.setZoom(13);
              }
              placeAllMarkers();
            }
          );
        } else {
          map.setCenter({ lat: firstLat, lng: firstLng });
          map.setZoom(15);
          placeAllMarkers();
          if (
            pickupLat == null &&
            pickupLng == null &&
            dropoffLat == null &&
            dropoffLng == null &&
            driverLat == null
          ) {
            placeMissionMarker(gmaps, map, firstLat, firstLng, "Mission", "mission");
          }
        }
      } catch (e: unknown) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : "";
        if (
          msg.includes("Authentification Google Maps refusée") ||
          msg.includes("GOOGLE_MAPS_AUTH_FAILURE") ||
          msg.includes("ApiTargetBlocked")
        ) {
          setMapLoadError(MAP_ERR_AUTH_FR);
        } else {
          const detail = formatMapInitError(e);
          setMapLoadError(detail ? `${MAP_ERR_GENERIC_FR}\n\n${detail}` : MAP_ERR_GENERIC_FR);
        }
      }
    };

    const t = window.setTimeout(() => {
      void run();
    }, 0);

    return () => {
      cancelled = true;
      window.clearTimeout(t);
      clearDirections();
      clearMarkers();
      disposeMapDom();
    };
  }, [
    apiKey,
    mapDomId,
    libraryList,
    firstLat,
    firstLng,
    pickupLat,
    pickupLng,
    dropoffLat,
    dropoffLng,
    driverLat,
    driverLng,
    routePlan.mode,
    routePlan.origin?.latitude,
    routePlan.destination?.latitude,
  ]);

  if (firstLat == null || firstLng == null) {
    return (
      <LiriWebMapFramePlaceholder
        height={height}
        accessibilityLabel="Carte mission indisponible"
        showcaseStyle={styles.fallbackShowcase}
      >
        <MissionMapUnavailable
          height={height}
          pickupLocation={props.pickupLocation}
          dropoffLocation={props.dropoffLocation}
          loading={resolving}
        />
      </LiriWebMapFramePlaceholder>
    );
  }

  if (!apiKey) {
    return (
      <LiriWebMapFramePlaceholder
        height={height}
        accessibilityLabel="Cle Google Maps manquante"
        showcaseStyle={styles.fallbackShowcase}
      >
        <AppText variant="caption" style={styles.fallbackText}>
          {webKeyHelpMessage || MAP_ERR_GENERIC_FR}
        </AppText>
      </LiriWebMapFramePlaceholder>
    );
  }

  if (mapLoadError) {
    return (
      <LiriWebMapFramePlaceholder
        height={height}
        accessibilityLabel="Erreur carte Google"
        showcaseStyle={styles.fallbackShowcase}
      >
        <AppText variant="caption" style={styles.fallbackText}>
          {mapLoadError}
        </AppText>
      </LiriWebMapFramePlaceholder>
    );
  }

  return (
    <View style={{ height, width: "100%", position: "relative" }}>
      <LiriWebMapFrame
        height={height}
        mapDomId={mapDomId}
        compact
        accessibilityLabel="Carte trajet mission"
      />
      <MissionMapLiveBadge
        prefix={routePlan.badgePrefix}
        distanceLabel={routeMetrics.distanceLabel}
        durationLabel={routeMetrics.durationLabel}
        live={isLiveRoute && driverLat != null}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  fallbackShowcase: {
    backgroundColor: "transparent",
    borderWidth: 0,
    padding: 0,
  },
  fallbackText: {
    color: "#64748B",
    lineHeight: 18,
  },
});
