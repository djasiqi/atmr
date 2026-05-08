import { useEffect, useId, useMemo, useRef, useState } from "react";
import { StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { CompanyDriverLiveLocation } from "../../api/contracts";
import { isDriverPositionStale } from "../../utils/companyDriverMapStatus";

const SCRIPT_ID = "google-maps-js-sdk-lirie-fleet";
const NAMESPACE_WAIT_MS = 15_000;

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

function isMapsCoreReady(): boolean {
  return typeof getGoogleMaps()?.Map === "function";
}

function waitForGoogleMapsNamespace(): Promise<void> {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (typeof window !== "undefined" && window.google?.maps) {
        resolve();
        return;
      }
      if (Date.now() - start > NAMESPACE_WAIT_MS) {
        reject(new Error("SDK Google Maps indisponible (timeout)"));
        return;
      }
      window.setTimeout(tick, 50);
    };
    tick();
  });
}

/** Avec `loading=async`, appeler `importLibrary('maps')` avant d'utiliser `google.maps.Map`. */
async function ensureMapsLibraryImported(): Promise<void> {
  if (isMapsCoreReady()) return;
  const gm = window.google?.maps as { importLibrary?: (name: string) => Promise<unknown> } | undefined;
  if (!gm || typeof gm.importLibrary !== "function") {
    throw new Error("Google Maps : importLibrary indisponible (script incomplet ?)");
  }
  await gm.importLibrary("maps");
  if (!isMapsCoreReady()) {
    throw new Error("Google Maps : échec après importLibrary(maps)");
  }
}

let mapsScriptLoadInflight: Promise<void> | null = null;

/**
 * Charge le bootstrap Maps JS (recommandation Google : `v=weekly` + `loading=async`).
 * @see https://developers.google.com/maps/documentation/javascript/load-maps-js-api
 */
function loadGoogleMapsScript(apiKey: string): Promise<void> {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return Promise.resolve();
  }
  if (isMapsCoreReady()) return Promise.resolve();
  if (mapsScriptLoadInflight) return mapsScriptLoadInflight;

  mapsScriptLoadInflight = new Promise((resolve, reject) => {
    const previousGmf = window.gm_authFailure;

    const cleanupAuthHook = (hook: () => void) => {
      if (window.gm_authFailure === hook) {
        window.gm_authFailure = previousGmf;
      }
    };

    const authHook = () => {
      cleanupAuthHook(authHook);
      mapsScriptLoadInflight = null;
      if (typeof previousGmf === "function") previousGmf();
      reject(new Error("GOOGLE_MAPS_AUTH_FAILURE"));
    };
    window.gm_authFailure = authHook;

    const finishAfterScriptLoaded = () => {
      waitForGoogleMapsNamespace()
        .then(() => ensureMapsLibraryImported())
        .then(() => {
          cleanupAuthHook(authHook);
          mapsScriptLoadInflight = null;
          resolve();
        })
        .catch((e: unknown) => {
          cleanupAuthHook(authHook);
          mapsScriptLoadInflight = null;
          reject(e instanceof Error ? e : new Error(String(e)));
        });
    };

    const existing = document.getElementById(SCRIPT_ID) as HTMLScriptElement | null;
    if (existing) {
      if (isMapsCoreReady() || window.google?.maps) {
        finishAfterScriptLoaded();
      } else {
        existing.addEventListener("load", finishAfterScriptLoaded, { once: true });
        existing.addEventListener(
          "error",
          () => {
            cleanupAuthHook(authHook);
            mapsScriptLoadInflight = null;
            reject(new Error("Google Maps script error"));
          },
          { once: true }
        );
      }
      return;
    }

    const script = document.createElement("script");
    script.id = SCRIPT_ID;
    script.async = true;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&v=weekly&loading=async`;
    script.onload = () => finishAfterScriptLoaded();
    script.onerror = () => {
      cleanupAuthHook(authHook);
      mapsScriptLoadInflight = null;
      reject(new Error("Impossible de charger Google Maps"));
    };
    document.head.appendChild(script);
  });

  return mapsScriptLoadInflight;
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
  "Impossible d’afficher la carte (réseau, timeout ou script bloqué). Rechargez la page ou vérifiez la clé EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.";

/**
 * Carte flotte web (API JavaScript Google Maps). Nécessite `EXPO_PUBLIC_GOOGLE_MAPS_API_KEY`.
 */
export function GoogleMapsFleetCanvas({ drivers, height }: Props) {
  const mapDomId = useId().replace(/:/g, "");
  const mapInstanceRef = useRef<{ setCenter: (x: unknown) => void; setZoom: (z: number) => void; fitBounds: (b: unknown, pad?: number) => void } | null>(
    null
  );
  const markersRef = useRef<{ setMap: (v: null) => void }[]>([]);
  const [mapLoadError, setMapLoadError] = useState<string | null>(null);
  const rawMapsKey = useMemo(
    () => (typeof process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY === "string" ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.trim() : ""),
    []
  );
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
        await loadGoogleMapsScript(apiKey);
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
        if (msg === "GOOGLE_MAPS_AUTH_FAILURE") {
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
  }, [apiKey, drivers, mapDomId, region.latitude, region.longitude]);

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
            ? "Clé Google Maps invalide ou texte d’exemple — créez une clé Maps JavaScript API dans Google Cloud et mettez-la dans EXPO_PUBLIC_GOOGLE_MAPS_API_KEY, puis redémarrez le bundler."
            : "Définissez EXPO_PUBLIC_GOOGLE_MAPS_API_KEY pour afficher la carte sur le web."}
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
