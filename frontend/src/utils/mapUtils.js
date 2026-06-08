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
  assigned: MAP_COLORS.warning,
  busy: MAP_COLORS.brand,
  offline: MAP_COLORS.muted,
  emergency: MAP_COLORS.danger,
};

// ─── Style carte Lirie — épuré, professionnel, calme ───
// Avec Advanced Markers, `mapId` est requis : l’API n’autorise pas `styles` en JS en même temps.
// Cloud (prod) : style console « Carte Lirie – site », ID style a699eeb51ddb599964e58e8f ;
// import JSON `public/maps/lirie-google-map-style.json` — voir `frontend/maps.env.example` (Map ID ≠ ID style).
export const LIRIE_MAP_STYLES = [
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

/**
 * Map ID requis pour AdvancedMarkerElement (Google Cloud Console → Map Management).
 * En dev sans ID dédié, `DEMO_MAP_ID` permet de valider l’intégration.
 * En production avec `REACT_APP_GOOGLE_MAPS_LIRIE_STYLE=cloud`, un Map ID réel est obligatoire
 * (pas de fallback silencieux vers DEMO_MAP_ID).
 */
const _rawMapId = process.env.REACT_APP_GOOGLE_MAPS_MAP_ID;
const _useCloudStyle = process.env.REACT_APP_GOOGLE_MAPS_LIRIE_STYLE === 'cloud';
const _isProd = process.env.NODE_ENV === 'production';

function resolveGoogleMapsMapId() {
  const trimmed = typeof _rawMapId === 'string' ? _rawMapId.trim() : '';
  if (_useCloudStyle && _isProd) {
    if (!trimmed || trimmed === 'DEMO_MAP_ID') {
      // eslint-disable-next-line no-console
      console.error(
        '[Maps] Production + style cloud : définissez REACT_APP_GOOGLE_MAPS_MAP_ID avec un Map ID valide (Google Cloud). ' +
          'Le fallback DEMO_MAP_ID est désactivé pour éviter InvalidMapId silencieux.'
      );
      return '';
    }
    return trimmed;
  }
  return trimmed || 'DEMO_MAP_ID';
}

export const GOOGLE_MAPS_MAP_ID = resolveGoogleMapsMapId();

/**
 * `cloud` : mapId + marqueurs avancés ; le style Lirie doit être importé dans la console Google (pas de `styles` en JS).
 * `json` ou absent : `styles: LIRIE_MAP_STYLES` en JS + marqueurs classiques (recommandé en dev).
 */
export const GOOGLE_MAPS_USE_JS_STYLES =
  process.env.REACT_APP_GOOGLE_MAPS_LIRIE_STYLE !== 'cloud';

const BASE_MAP_UI_OPTIONS = {
  disableDefaultUI: true,
  zoomControl: false,
  streetViewControl: false,
  mapTypeControl: false,
  fullscreenControl: false,
  scaleControl: true,
  clickableIcons: false,
  gestureHandling: 'greedy',
};

// Options par défaut pour <GoogleMap>
export const DEFAULT_MAP_OPTIONS = GOOGLE_MAPS_USE_JS_STYLES
  ? {
      ...BASE_MAP_UI_OPTIONS,
      styles: LIRIE_MAP_STYLES,
    }
  : {
      ...BASE_MAP_UI_OPTIONS,
      mapId: GOOGLE_MAPS_MAP_ID,
    };

// Options carte pour pages publiques (Home, Client)
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
    [d.latitude, d.longitude],
    [d.lat, d.lon],
    [d.lat, d.lng],
    [d.current_lat, d.current_lon],
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
 * P1: Le backend (REST / driver_live_state_update) est la source de vérité.
 * Cette fonction sert uniquement de fallback quand status n'est pas fourni par le backend.
 */
export const getDriverStatus = (driver) => {
  const backendStatus = String(driver?.status || '').toLowerCase();
  if (backendStatus) return backendStatus;
  if (!driver.is_active) return 'offline';
  if (driver.mission_status === 'ASSIGNED') return 'assigned';
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

export const getFreshnessStatus = (driver) => {
  const trackingDisplay = String(driver?.tracking_display_status || '').toLowerCase();
  if (
    trackingDisplay === 'live' ||
    trackingDisplay === 'stale' ||
    trackingDisplay === 'degraded_constrained' ||
    trackingDisplay === 'silent_wake_pending' ||
    trackingDisplay === 'offline_unknown'
  ) {
    return trackingDisplay;
  }
  const backendStatus = String(driver?.location_status || '').toLowerCase();
  if (backendStatus === 'live' || backendStatus === 'recent' || backendStatus === 'stale' || backendStatus === 'offline') {
    return backendStatus;
  }
  const v = Number(driver?.last_seen_seconds);
  if (!Number.isFinite(v)) return 'offline';
  if (v <= 20) return 'live';
  if (v <= 90) return 'recent';
  if (v <= 300) return 'stale';
  return 'offline';
};

// ─── Marqueurs SVG professionnels Lirie ───

/**
 * Ancrage pour `google.maps.marker.AdvancedMarkerElement` : `anchorLeft` / `anchorTop` sont des
 * chaînes CSS (doc Google), p.ex. "-50%" / "-100%", pas des nombres 0–1.
 * Reproduit l’effet de `google.maps.Marker` avec `icon.anchor` (px) + `scaledSize`.
 *
 * @param {number} ax - pixels depuis le bord gauche de l’icône
 * @param {number} ay - pixels depuis le haut
 * @param {number} w - largeur
 * @param {number} h - hauteur
 * @returns {{ anchorLeft: string, anchorTop: string }}
 */
export function iconAnchorToAdvancedMarkerCss(ax, ay, w, h) {
  if (!w || !h || !Number.isFinite(ax) || !Number.isFinite(ay)) {
    return { anchorLeft: '-50%', anchorTop: '-50%' };
  }
  return {
    anchorLeft: `${-(ax / w) * 100}%`,
    anchorTop: `${-(ay / h) * 100}%`,
  };
}

/**
 * Marqueur cercle chauffeur avec ombre portée et libellé optionnel (initiales).
 * @param {string} color
 * @param {number} opacity
 * @param {{ label?: string, textColor?: string, ringColor?: string }} options
 */
export const makeCircleMarkerIcon = (color, opacity = 1, options = {}) => {
  const labelRaw = typeof options.label === 'string' ? options.label.trim().toUpperCase() : '';
  const label = labelRaw.slice(0, 2);
  const textColor = options.textColor || '#ffffff';
  const ringColor = options.ringColor || '#ffffff';
  const textNode = label
    ? `<text x="12" y="12.4" text-anchor="middle" dominant-baseline="central" fill="${textColor}" font-size="7.4" font-weight="700" letter-spacing="0.2" font-family="-apple-system,Segoe UI,Roboto,sans-serif">${label}</text>`
    : '';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24">
    <defs>
      <filter id="ds" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000" flood-opacity="0.2"/>
      </filter>
    </defs>
    <circle cx="12" cy="12" r="8.2" fill="${color}" fill-opacity="${opacity}" stroke="${ringColor}" stroke-width="2.1" filter="url(#ds)"/>
    ${textNode}
  </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
};

/** Taille des points LIRIE (pickup / destination) — ancrage centré. */
export const LIRIE_POINT_MARKER_SIZE_PX = 28;

const _liriePointCache = {};

/**
 * Marqueur LIRIE minimal (disque + lettre P/D) — remplace les pins goutte Google.
 * @param {'pickup'|'dropoff'|'default'} type
 */
export const makeLiriePointMarkerIcon = (type = 'default') => {
  if (_liriePointCache[type]) return _liriePointCache[type];
  const isPickup = type === 'pickup' || type === 'default';
  const url = makePoiMarkerIcon(isPickup ? 'P' : 'D', isPickup ? MAP_COLORS.brand : MAP_COLORS.textPrimary);
  _liriePointCache[type] = url;
  return url;
};

/** @deprecated Alias — préférer makeLiriePointMarkerIcon */
export const makePinMarkerIcon = makeLiriePointMarkerIcon;

/**
 * Icône + ancrage centré pour google.maps.Marker / AdvancedMarker (img).
 * @param {typeof google.maps | undefined} gmaps
 * @param {'pickup'|'dropoff'|'default'} type
 */
export function resolveLiriePointMarkerIcon(gmaps, type = 'default') {
  const size = LIRIE_POINT_MARKER_SIZE_PX;
  const half = size / 2;
  return {
    url: makeLiriePointMarkerIcon(type),
    scaledSize: gmaps?.Size ? new gmaps.Size(size, size) : undefined,
    anchor: gmaps?.Point ? new gmaps.Point(half, half) : undefined,
  };
}

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
