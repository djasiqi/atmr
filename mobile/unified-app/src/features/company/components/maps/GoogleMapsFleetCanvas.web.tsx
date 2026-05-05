import { useEffect, useId, useMemo, useRef } from "react";
import { StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import type { CompanyDriverLiveLocation } from "../../api/contracts";
import { isDriverPositionStale } from "../../utils/companyDriverMapStatus";

const SCRIPT_ID = "google-maps-js-sdk-lirie-fleet";

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

function loadGoogleMapsScript(apiKey: string): Promise<void> {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return Promise.resolve();
  }
  if (getGoogleMaps()) {
    return Promise.resolve();
  }
  const existing = document.getElementById(SCRIPT_ID);
  if (existing) {
    return new Promise((resolve, reject) => {
      const check = () => {
        if (getGoogleMaps()) resolve();
      };
      if (getGoogleMaps()) {
        resolve();
        return;
      }
      existing.addEventListener("load", () => {
        check();
        resolve();
      });
      existing.addEventListener("error", () => reject(new Error("Google Maps script error")));
    });
  }
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.id = SCRIPT_ID;
    s.async = true;
    s.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}`;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Impossible de charger Google Maps"));
    document.head.appendChild(s);
  });
}

type Props = {
  drivers: CompanyDriverLiveLocation[];
  height: number;
};

/**
 * Carte flotte web (API JavaScript Google Maps). Nécessite `EXPO_PUBLIC_GOOGLE_MAPS_API_KEY`.
 */
export function GoogleMapsFleetCanvas({ drivers, height }: Props) {
  const mapDomId = useId().replace(/:/g, "");
  const mapInstanceRef = useRef<{ setCenter: (x: unknown) => void; setZoom: (z: number) => void; fitBounds: (b: unknown, pad?: number) => void } | null>(
    null
  );
  const markersRef = useRef<{ setMap: (v: null) => void }[]>([]);
  const apiKey = useMemo(
    () => (typeof process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY === "string" ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.trim() : ""),
    []
  );

  const region = useMemo(() => computeRegion(drivers), [drivers]);

  useEffect(() => {
    if (!apiKey) return;
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
      } catch {
        /* message si pas de réseau / script bloqué */
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
          Définissez EXPO_PUBLIC_GOOGLE_MAPS_API_KEY pour afficher la carte sur le web.
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
    textAlign: "center",
    lineHeight: 18,
  },
});
