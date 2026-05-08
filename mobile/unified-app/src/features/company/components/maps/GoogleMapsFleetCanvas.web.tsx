import { useEffect, useId, useMemo, useRef, useState } from "react";
import { StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { CompanyDriverLiveLocation } from "../../api/contracts";
import { isDriverPositionStale } from "../../utils/companyDriverMapStatus";
import {
  loadGoogleMapsScriptWithKey,
  parseGoogleMapsLibraryList,
} from "../../../../../../../frontend/src/shared/google-maps/bootstrap.js";

/** Guide officiel erreurs (dont ApiTargetBlockedMapError). */
const MAPS_ERROR_HELP_URL =
  "https://developers.google.com/maps/documentation/javascript/error-messages#api-target-blocked-map-error";

/**
 * Refuse les placeholders courants (ex. `ta_clef_google_maps_js` copié depuis un exemple de doc)
 * pour ne pas charger le SDK avec une clé invalide → InvalidKey / InvalidKeyMapError en console.
 */
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

function computeRegion(drivers: CompanyDriverLiveLocation[]) {
  if (drivers.length === 0) {
    return {
      latitude: 48.8566,
      longitude: 2.3522,
      latitudeDelta: 0.2,
      longitudeDelta: 0.2,
    };
  }
  const latitudes = drivers.map((d) => d.latitude);
  const longitudes = drivers.map((d) => d.longitude);
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

type Props = {
  drivers: CompanyDriverLiveLocation[];
  height: number;
};

const MAP_ERR_AUTH_FR = [
  "Google Maps a refusé la clé (souvent ApiTargetBlockedMapError sur le web). Vérifiez dans Google Cloud :",
  "• l’API « Maps JavaScript API » est activée pour le projet ;",
  "• la clé n’est pas limitée aux seules applis Android/iOS : ajoutez des restrictions « sites web » avec l’origine exacte (ex. http://localhost:8081, https://votre-domaine.com) ;",
  "• la facturation est active.",
  `Documentation : ${MAPS_ERROR_HELP_URL}`,
].join("\n");

const MAP_ERR_GENERIC_FR =
  "Impossible d’afficher la carte (réseau, timeout ou script bloqué). Rechargez la page ou vérifiez EXPO_PUBLIC_GOOGLE_MAPS_API_KEY (ou REACT_APP_GOOGLE_MAPS_API_KEY en build web).";

/**
 * Carte flotte web (API JavaScript Google Maps).
 * Clé : `EXPO_PUBLIC_GOOGLE_MAPS_API_KEY` ; secours `REACT_APP_GOOGLE_MAPS_API_KEY` si présent au build.
 * Charge le SDK via le bootstrap monorepo partagé avec le frontend entreprise (un seul SCRIPT_ID).
 */
export function GoogleMapsFleetCanvas({ drivers, height }: Props) {
  const mapDomId = useId().replace(/:/g, "");
  const mapInstanceRef = useRef<{ setCenter: (x: unknown) => void; setZoom: (z: number) => void; fitBounds: (b: unknown, pad?: number) => void } | null>(
    null
  );
  const markersRef = useRef<{ setMap: (v: null) => void }[]>([]);
  const [mapLoadError, setMapLoadError] = useState<string | null>(null);

  const libraryList = useMemo(
    () =>
      parseGoogleMapsLibraryList(
        process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES || process.env.REACT_APP_GOOGLE_MAPS_LIBRARIES
      ),
    []
  );

  const rawMapsKey = useMemo(() => {
    const expo = typeof process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY === "string" ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.trim() : "";
    const react =
      typeof process.env.REACT_APP_GOOGLE_MAPS_API_KEY === "string" ? process.env.REACT_APP_GOOGLE_MAPS_API_KEY.trim() : "";
    return expo || react;
  }, []);

  const apiKey = useMemo(
    () => (rawMapsKey && isPlausibleGoogleMapsBrowserKey(rawMapsKey) ? rawMapsKey : ""),
    [rawMapsKey]
  );
  const mapsKeyRejected = rawMapsKey.length > 0 && !apiKey;

  const region = useMemo(() => computeRegion(drivers), [drivers]);

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
        if (!el || !gmaps) return;

        const LatLng = gmaps.LatLng as new (lat: number, lng: number) => unknown;
        const LatLngBounds = gmaps.LatLngBounds as new () => { extend: (p: unknown) => void };
        const MapCtor = gmaps.Map as new (el: HTMLElement, opts: Record<string, unknown>) => {
          setCenter: (x: unknown) => void;
          setZoom: (z: number) => void;
          fitBounds: (b: unknown, padding?: number) => void;
        };
        const MarkerCtor = gmaps.Marker as new (opts: Record<string, unknown>) => { setMap: (v: null) => void };
        const SymbolPath = gmaps.SymbolPath as { CIRCLE: unknown };

        if (!mapInstanceRef.current) {
          mapInstanceRef.current = new MapCtor(el as HTMLElement, {
            mapTypeControl: false,
            streetViewControl: false,
            fullscreenControl: false,
          });
        }
        const map = mapInstanceRef.current;

        for (const m of markersRef.current) {
          m.setMap(null);
        }
        markersRef.current = [];

        const bounds = new LatLngBounds();
        for (const d of drivers) {
          bounds.extend(new LatLng(d.latitude, d.longitude));
        }

        for (const d of drivers) {
          const stale = isDriverPositionStale(d);
          const marker = new MarkerCtor({
            map,
            position: { lat: d.latitude, lng: d.longitude },
            title: `Chauffeur #${d.driver_id}`,
            opacity: stale ? 0.7 : 1,
            icon: {
              path: SymbolPath.CIRCLE,
              scale: 8,
              fillColor: stale ? "#9e9e9e" : "#2e7d32",
              fillOpacity: 1,
              strokeColor: "#ffffff",
              strokeWeight: 2,
            },
          });
          markersRef.current.push(marker);
        }

        if (drivers.length === 0) {
          map.setCenter(new LatLng(region.latitude, region.longitude));
          map.setZoom(11);
        } else if (drivers.length === 1) {
          map.setCenter(new LatLng(drivers[0].latitude, drivers[0].longitude));
          map.setZoom(13);
        } else {
          map.fitBounds(bounds, 48);
        }
      } catch (e: unknown) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : "";
        if (msg.includes("Authentification Google Maps refusée") || msg.includes("GOOGLE_MAPS_AUTH_FAILURE")) {
          setMapLoadError(MAP_ERR_AUTH_FR);
        } else {
          setMapLoadError(MAP_ERR_GENERIC_FR);
        }
      }
    };

    const t = window.setTimeout(() => {
      void run();
    }, 0);

    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [apiKey, drivers, libraryList, mapDomId, region.latitude, region.longitude]);

  useEffect(
    () => () => {
      for (const m of markersRef.current) {
        m.setMap(null);
      }
      markersRef.current = [];
      mapInstanceRef.current = null;
    },
    []
  );

  if (!apiKey) {
    return (
      <View style={[styles.fallback, { height }]} accessibilityLabel="Carte flotte indisponible">
        <AppText variant="caption" style={styles.fallbackText}>
          {mapsKeyRejected
            ? "Clé Google Maps invalide ou texte d’exemple — créez une clé Maps JavaScript API dans Google Cloud et mettez-la dans EXPO_PUBLIC_GOOGLE_MAPS_API_KEY (ou REACT_APP_GOOGLE_MAPS_API_KEY), puis redémarrez le bundler."
            : "Définissez EXPO_PUBLIC_GOOGLE_MAPS_API_KEY (ou REACT_APP_GOOGLE_MAPS_API_KEY) pour afficher la carte sur le web."}
        </AppText>
      </View>
    );
  }

  if (mapLoadError) {
    return (
      <View style={[styles.fallback, { height }]} accessibilityLabel="Erreur carte Google">
        <AppText variant="caption" style={styles.fallbackText}>
          {mapLoadError}
        </AppText>
      </View>
    );
  }

  return <View nativeID={mapDomId} style={[styles.map, { height }]} accessibilityLabel="Carte des chauffeurs" />;
}

const styles = StyleSheet.create({
  map: {
    width: "100%",
    borderRadius: 8,
    overflow: "hidden",
    backgroundColor: "#E2E8F0",
  },
  fallback: {
    width: "100%",
    borderRadius: 8,
    backgroundColor: "rgba(148, 163, 184, 0.15)",
    padding: 12,
    justifyContent: "center",
  },
  fallbackText: {
    color: "#64748B",
    textAlign: "left",
    lineHeight: 18,
  },
});
