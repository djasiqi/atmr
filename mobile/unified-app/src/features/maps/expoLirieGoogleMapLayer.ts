/**
 * Style carte Lirie — partagé entre `react-native-maps` (iOS / Android) et la carte Google Maps JavaScript
 * (même module, variante navigateur). Variables `EXPO_PUBLIC_GOOGLE_MAPS_*`.
 *
 * - `EXPO_PUBLIC_GOOGLE_MAPS_LIRIE_STYLE=json` ou absent : `customMapStyle` / `styles` = `LIRIE_MAP_STYLES`.
 * - `EXPO_PUBLIC_GOOGLE_MAPS_LIRIE_STYLE=cloud` + **Map ID** carte Google Cloud : `googleMapId` / `mapId`.
 */

export const LIRIE_MAP_STYLES = [
  { elementType: "geometry", stylers: [{ saturation: -16 }, { lightness: 4 }] },
  { elementType: "labels.icon", stylers: [{ visibility: "off" }] },
  { featureType: "poi.business", stylers: [{ visibility: "off" }] },
  { featureType: "poi.attraction", stylers: [{ visibility: "off" }] },
  { featureType: "poi.government", stylers: [{ visibility: "off" }] },
  { featureType: "poi", stylers: [{ visibility: "off" }] },
  { featureType: "poi.medical", stylers: [{ visibility: "off" }] },
  { featureType: "poi.place", stylers: [{ visibility: "off" }] },
  { featureType: "poi.school", stylers: [{ visibility: "off" }] },
  { featureType: "poi.park", elementType: "labels.icon", stylers: [{ visibility: "off" }] },
  { featureType: "transit", stylers: [{ visibility: "simplified" }, { saturation: -40 }] },
  { featureType: "transit.station", elementType: "labels.text.fill", stylers: [{ visibility: "off" }] },
  { featureType: "water", elementType: "geometry", stylers: [{ color: "#c8dce8" }] },
  { featureType: "water", elementType: "labels.text.fill", stylers: [{ color: "#94A3B8" }] },
  { featureType: "landscape.man_made", elementType: "geometry", stylers: [{ color: "#f0f2f4" }] },
  { featureType: "landscape.natural", elementType: "geometry", stylers: [{ color: "#e4ebe7" }] },
  { featureType: "landscape.natural.terrain", elementType: "geometry", stylers: [{ color: "#dde5e0" }] },
  { featureType: "road.highway", elementType: "geometry", stylers: [{ color: "#ccd5dc" }] },
  { featureType: "road.highway", elementType: "geometry.stroke", stylers: [{ color: "#b7c2cb" }] },
  { featureType: "road.highway", elementType: "labels.text.fill", stylers: [{ color: "#64748B" }] },
  { featureType: "road.arterial", elementType: "geometry", stylers: [{ color: "#d8e0e6" }] },
  { featureType: "road.local", elementType: "geometry", stylers: [{ color: "#e8edf0" }] },
  { featureType: "road", elementType: "labels.text.fill", stylers: [{ color: "#94A3B8" }] },
  { featureType: "road.local", elementType: "labels.text.fill", stylers: [{ visibility: "off" }] },
  { featureType: "administrative", elementType: "labels.text.fill", stylers: [{ color: "#64748B" }] },
  { featureType: "administrative.locality", elementType: "labels.text.fill", stylers: [{ color: "#1E293B" }] },
  {
    featureType: "administrative.locality",
    elementType: "labels.text.stroke",
    stylers: [{ color: "#ffffff" }, { weight: 3 }],
  },
  { featureType: "administrative.neighborhood", elementType: "labels.text.fill", stylers: [{ color: "#94A3B8" }] },
] as const;

/** Sous-ensemble POI off — utile en mode Cloud Map ID où `LIRIE_MAP_STYLES` complet n’est pas appliqué. */
export const LIRIE_POI_SUPPRESSION_STYLES = [
  { elementType: "labels.icon", stylers: [{ visibility: "off" }] },
  { featureType: "poi", stylers: [{ visibility: "off" }] },
  { featureType: "poi.business", stylers: [{ visibility: "off" }] },
  { featureType: "poi.attraction", stylers: [{ visibility: "off" }] },
  { featureType: "poi.government", stylers: [{ visibility: "off" }] },
  { featureType: "poi.medical", stylers: [{ visibility: "off" }] },
  { featureType: "poi.place", stylers: [{ visibility: "off" }] },
  { featureType: "poi.school", stylers: [{ visibility: "off" }] },
] as const;

export type ExpoLirieMapLayer =
  | { kind: "cloud"; mapId: string }
  | { kind: "js"; styles: readonly (typeof LIRIE_MAP_STYLES)[number][] };

function normaliseLirieStyleMode(): string {
  const raw =
    typeof process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIRIE_STYLE === "string"
      ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIRIE_STYLE.trim().toLowerCase()
      : "";
  return raw;
}

export function resolveExpoLirieGoogleMapLayer(): ExpoLirieMapLayer {
  const mode = normaliseLirieStyleMode();
  const mapId =
    typeof process.env.EXPO_PUBLIC_GOOGLE_MAPS_MAP_ID === "string"
      ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_MAP_ID.trim()
      : "";
  if (mode === "cloud" && mapId.length > 0) {
    return { kind: "cloud", mapId };
  }
  return { kind: "js", styles: LIRIE_MAP_STYLES };
}

/**
 * UI Google Maps JS — cartes embarquées Lirie (flotte, mission).
 * Pas de zoom/échelle/caméra natifs Google : nos FAB (recentrer, filtres, couches).
 */
export function getLirieBaseMapUiOptions(): Record<string, unknown> {
  return {
    disableDefaultUI: true,
    mapTypeControl: false,
    streetViewControl: false,
    fullscreenControl: false,
    zoomControl: false,
    rotateControl: false,
    scaleControl: false,
    /** Bouton « commandes caméra » (tilt / rotation) — masqué. */
    cameraControl: false,
    keyboardShortcuts: false,
    clickableIcons: false,
    gestureHandling: "greedy",
  };
}

/** Options complètes (UI + charte Lirie) pour `new Map` et `setOptions`. */
export function buildLirieGoogleMapOptions(
  mapTypeId: string,
  extra?: Record<string, unknown>
): Record<string, unknown> {
  const layer = resolveExpoLirieGoogleMapLayer();
  const opts: Record<string, unknown> = {
    ...getLirieBaseMapUiOptions(),
    mapTypeId,
    ...extra,
  };
  if (layer.kind === "cloud") {
    opts.mapId = layer.mapId;
  } else {
    opts.styles = [...layer.styles];
  }
  return opts;
}

/** Recalcule la taille des tuiles après layout (conteneur RN, rognage logo, etc.). */
export function triggerGoogleMapResize(
  map: unknown,
  gmaps?: Record<string, unknown>
): void {
  const resolved =
    gmaps ??
    (typeof window !== "undefined"
      ? (window as { google?: { maps?: Record<string, unknown> } }).google?.maps
      : undefined);
  const event = resolved?.event as { trigger?: (target: unknown, name: string) => void } | undefined;
  event?.trigger?.(map, "resize");
}

function escapeSvgText(input: string): string {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/**
 * Marqueur disque (carte raster + Marker classique), même principe que `makeCircleMarkerIcon` côté CRA.
 * Supporte un libellé court (ex. identifiant chauffeur) au centre.
 */
export function makeCircleMarkerDataUrl(
  color: string,
  opacity = 1,
  sizePx = 24,
  options?: { label?: string; textColor?: string; ringColor?: string }
): string {
  const cx = sizePx / 2;
  const cy = sizePx / 2;
  const r = (8.2 * sizePx) / 24;
  const sw = (2.1 * sizePx) / 24;
  const labelRaw = typeof options?.label === "string" ? options.label.trim().toUpperCase() : "";
  const label = escapeSvgText(labelRaw.slice(0, 2));
  const textColor = options?.textColor || "#ffffff";
  const ringColor = options?.ringColor || "#ffffff";
  const textNode = label
    ? `<text x="${cx}" y="${cy + sizePx * 0.016}" text-anchor="middle" dominant-baseline="central" fill="${textColor}" font-size="${(7.4 * sizePx) / 24}" font-weight="700" letter-spacing="${(0.2 * sizePx) / 24}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">${label}</text>`
    : "";
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${sizePx}" height="${sizePx}">
    <defs>
      <filter id="ds" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000" flood-opacity="0.2"/>
      </filter>
    </defs>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="${color}" fill-opacity="${opacity}" stroke="${ringColor}" stroke-width="${sw}" filter="url(#ds)"/>
    ${textNode}
  </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}
