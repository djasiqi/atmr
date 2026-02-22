/**
 * Utilitaires partagés Google Maps — Charte Lirie.
 * Style carte, marqueurs SVG professionnels, validation coordonnées,
 * résolution position chauffeur, options par défaut.
 */

// Centre par défaut : Suisse
export const SWITZERLAND_CENTER = { lat: 46.8182, lng: 8.2275 };

// ─── Palette Lirie pour les cartes ───
export const MAP_COLORS = {
  brand: '#00796B',
  brandDark: '#00695C',
  brandLight: '#26a69a',
  success: '#22c55e',
  warning: '#f59e0b',
  danger: '#ef4444',
  muted: '#91A3A0',
  textPrimary: '#1E293B',
  textSecondary: '#64748B',
  textMuted: '#94A3B8',
  border: '#E2E8F0',
  bg: '#ffffff',
  routeDefault: '#00796B',
  routeActive: '#00695C',
  routePending: '#f59e0b',
  routeCompleted: '#22c55e',
  routeCanceled: '#ef4444',
};

// Couleurs des marqueurs chauffeurs par statut
export const STATUS_COLORS = {
  available: MAP_COLORS.success,
  busy: MAP_COLORS.brand,
  offline: MAP_COLORS.muted,
  emergency: MAP_COLORS.danger,
};

// ─── Style carte Lirie — épuré, professionnel, calme ───
const LIRIE_MAP_STYLES = [
  // Désactiver les POI et simplifier le transit
  { featureType: 'poi', stylers: [{ visibility: 'off' }] },
  { featureType: 'poi.medical', stylers: [{ visibility: 'on' }] },
  { featureType: 'poi.medical', elementType: 'labels.icon', stylers: [{ saturation: -60 }] },
  { featureType: 'transit', stylers: [{ visibility: 'simplified' }] },
  // Eau — bleu doux
  { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#c8dce8' }] },
  { featureType: 'water', elementType: 'labels.text.fill', stylers: [{ color: '#94A3B8' }] },
  // Paysage — tons neutres
  { featureType: 'landscape.man_made', elementType: 'geometry', stylers: [{ color: '#f0f2f4' }] },
  { featureType: 'landscape.natural', elementType: 'geometry', stylers: [{ color: '#e4ebe7' }] },
  { featureType: 'landscape.natural.terrain', elementType: 'geometry', stylers: [{ color: '#dde5e0' }] },
  // Routes — hiérarchie visuelle claire
  { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#d5dbe0' }] },
  { featureType: 'road.highway', elementType: 'geometry.stroke', stylers: [{ color: '#c3cad0' }] },
  { featureType: 'road.highway', elementType: 'labels.text.fill', stylers: [{ color: '#64748B' }] },
  { featureType: 'road.arterial', elementType: 'geometry', stylers: [{ color: '#e0e5e9' }] },
  { featureType: 'road.local', elementType: 'geometry', stylers: [{ color: '#ebeef1' }] },
  { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#94A3B8' }] },
  // Labels administratifs — texte Lirie
  { featureType: 'administrative', elementType: 'labels.text.fill', stylers: [{ color: '#64748B' }] },
  { featureType: 'administrative.locality', elementType: 'labels.text.fill', stylers: [{ color: '#1E293B' }] },
  { featureType: 'administrative.locality', elementType: 'labels.text.stroke', stylers: [{ color: '#ffffff' }, { weight: 3 }] },
  { featureType: 'administrative.neighborhood', elementType: 'labels.text.fill', stylers: [{ color: '#94A3B8' }] },
];

// Options par défaut pour <GoogleMap>
export const DEFAULT_MAP_OPTIONS = {
  disableDefaultUI: true,
  zoomControl: false,
  streetViewControl: false,
  mapTypeControl: false,
  fullscreenControl: false,
  scaleControl: true,
  clickableIcons: false,
  gestureHandling: 'greedy',
  styles: LIRIE_MAP_STYLES,
};

// Options carte pour pages publiques (Home, Client) — avec UI par défaut
export const PUBLIC_MAP_OPTIONS = {
  ...DEFAULT_MAP_OPTIONS,
};

// ─── Validation coordonnées ───

const toNumOrNull = (v) => {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

/**
 * Normalise lat/lon et rejette les valeurs invalides.
 * @returns {{ center: {lat, lng} | null, reason: string | null }}
 */
export const normaliseCoords = (lat, lon) => {
  const la = toNumOrNull(lat);
  const lo = toNumOrNull(lon);
  if (la === null || lo === null) return { center: null, reason: 'missing_or_nan' };
  if (Math.abs(la) > 90) return { center: null, reason: 'lat_out_of_range' };
  if (Math.abs(lo) > 180) return { center: null, reason: 'lon_out_of_range' };
  if (la === 0 && lo === 0) return { center: null, reason: 'zero_coords' };
  return { center: { lat: la, lng: lo }, reason: null };
};

/**
 * Résout les coordonnées d'un chauffeur (priorité GPS > DB > entreprise).
 */
export const resolveDriverCoords = (d, companyFallback) => {
  const candidates = [
    [d.current_lat, d.current_lon],
    [d.latitude, d.longitude],
    [d.lat, d.lon],
    [d.lat, d.lng],
    [d.last_latitude, d.last_longitude],
  ];
  if (d.last_position) candidates.push([d.last_position.lat, d.last_position.lon]);
  for (const [la, lo] of candidates) {
    const r = normaliseCoords(la, lo);
    if (r.center) return { coords: r.center, isFallback: false };
  }
  if (companyFallback) return { coords: companyFallback, isFallback: true };
  return null;
};

/**
 * Détermine le statut d'un chauffeur.
 */
export const getDriverStatus = (driver) => {
  if (!driver.is_active) return 'offline';
  if (driver.current_booking_id || driver.status === 'busy') return 'busy';
  if (driver.emergency_mode) return 'emergency';
  return 'available';
};

/**
 * Formate "il y a X s/min" pour les tooltips.
 */
export const formatLastSeen = (lastSeenSeconds) => {
  if (lastSeenSeconds == null || lastSeenSeconds < 0) return 'Dernier signal inconnu';
  if (lastSeenSeconds < 60) return `il y a ${lastSeenSeconds}s`;
  if (lastSeenSeconds < 3600) {
    const m = Math.floor(lastSeenSeconds / 60);
    return m === 1 ? 'il y a 1 min' : `il y a ${m} min`;
  }
  return '> 1h';
};

// ─── Marqueurs SVG professionnels Lirie ───

/**
 * Marqueur cercle chauffeur avec ombre portée.
 */
export const makeCircleMarkerIcon = (color, opacity = 1) => {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24">
    <defs>
      <filter id="ds" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000" flood-opacity="0.2"/>
      </filter>
    </defs>
    <circle cx="12" cy="12" r="8" fill="${color}" fill-opacity="${opacity}" stroke="#fff" stroke-width="2.5" filter="url(#ds)"/>
  </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
};

/**
 * Marqueur pin professionnel (pickup / destination).
 * Cache intégré pour ne pas ré-encoder le SVG à chaque render.
 * @param {'pickup'|'dropoff'|'default'} type
 */
const _pinCache = {};
export const makePinMarkerIcon = (type = 'default') => {
  if (_pinCache[type]) return _pinCache[type];

  const isPickup = type === 'pickup' || type === 'default';
  const bg = isPickup ? MAP_COLORS.brand : '#1E293B';
  const accent = isPickup ? MAP_COLORS.brandDark : '#0f172a';

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="40" height="52" viewBox="0 0 40 52" fill="none">
<defs>
  <filter id="m${type[0]}" x="0" y="0" width="40" height="52" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
    <feFlood flood-opacity="0" result="bg"/>
    <feColorMatrix in="SourceAlpha" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="a"/>
    <feOffset dy="2"/>
    <feGaussianBlur stdDeviation="3"/>
    <feColorMatrix values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.2 0"/>
    <feBlend in2="bg"/>
    <feBlend in="SourceGraphic"/>
  </filter>
</defs>
<g filter="url(#m${type[0]})">
  <path d="M20 4C12.268 4 6 10.268 6 18c0 9.6 14 28 14 28s14-18.4 14-28c0-7.732-6.268-14-14-14z" fill="${bg}"/>
  <path d="M20 5C12.82 5 7 10.82 7 18c0 8.8 13 26.5 13 26.5S33 26.8 33 18c0-7.18-5.82-13-13-13z" fill="none" stroke="${accent}" stroke-opacity=".15"/>
  <circle cx="20" cy="18" r="7" fill="#fff"/>
  <circle cx="20" cy="18" r="3" fill="${bg}"/>
</g>
</svg>`;

  const url = `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
  _pinCache[type] = url;
  return url;
};

/**
 * Marqueur point d'intérêt (POI) pour chauffeur — cercle avec lettre.
 * @param {string} label - 'P' ou 'D'
 * @param {string} bgColor - couleur de fond
 */
export const makePoiMarkerIcon = (label, bgColor) => {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28">
    <defs>
      <filter id="pf" x="-15%" y="-15%" width="130%" height="130%">
        <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000" flood-opacity="0.2"/>
      </filter>
    </defs>
    <circle cx="14" cy="14" r="12" fill="${bgColor}" stroke="#fff" stroke-width="2" filter="url(#pf)"/>
    <text x="14" y="14" text-anchor="middle" dominant-baseline="central" fill="#fff" font-size="12" font-weight="700" font-family="-apple-system,system-ui,sans-serif">${label}</text>
  </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
};

/**
 * Marqueur cluster personnalisé Lirie.
 */
export const makeClusterIcon = (count) => {
  const size = count < 10 ? 40 : count < 50 ? 46 : 52;
  const r = size / 2 - 2;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}">
    <defs>
      <filter id="cs" x="-15%" y="-15%" width="130%" height="130%">
        <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#000" flood-opacity="0.2"/>
      </filter>
    </defs>
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="${MAP_COLORS.brandDark}" stroke="#fff" stroke-width="2.5" filter="url(#cs)"/>
    <text x="${size / 2}" y="${size / 2}" text-anchor="middle" dominant-baseline="central" fill="#fff" font-size="13" font-weight="600" font-family="-apple-system,system-ui,sans-serif">${count}</text>
  </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
};

/**
 * Couleur de route selon le statut de la réservation.
 */
export const getRouteColor = (status) => {
  switch (status) {
    case 'pending': return MAP_COLORS.routePending;
    case 'accepted': case 'assigned': return MAP_COLORS.brand;
    case 'in_progress': return MAP_COLORS.routeActive;
    case 'completed': return MAP_COLORS.routeCompleted;
    case 'canceled': case 'cancelled': return MAP_COLORS.routeCanceled;
    default: return MAP_COLORS.routeDefault;
  }
};

// ─── Options de route partagées ───

export const ROUTE_OPTIONS = {
  strokeColor: MAP_COLORS.brand,
  strokeWeight: 3.5,
  strokeOpacity: 0.9,
  geodesic: true,
  zIndex: 1,
};

export const ROUTE_OUTLINE_OPTIONS = {
  strokeColor: MAP_COLORS.brand,
  strokeWeight: 5,
  strokeOpacity: 0.12,
  geodesic: true,
  zIndex: 0,
};

// ─── Style InfoWindow partagé (HTML string) ───

export const INFOWINDOW_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

/**
 * Enveloppe HTML InfoWindow avec style Lirie.
 */
export const wrapInfoWindowHtml = (content) => `
  <div style="
    font-family: ${INFOWINDOW_FONT};
    padding: 4px 2px;
    min-width: 140px;
    max-width: 260px;
    line-height: 1.4;
  ">
    ${content}
  </div>
`;
