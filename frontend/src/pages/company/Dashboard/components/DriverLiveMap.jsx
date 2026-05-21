// src/pages/company/Dashboard/components/DriverLiveMap.jsx
import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { GoogleMap } from '@react-google-maps/api';
import { MarkerClusterer } from '@googlemaps/markerclusterer';
import { useLirieCompany } from '../../../../hooks/useLirieCompany';
import { useCompanySocketConnected } from '../../../../hooks/enterprise/useCompanySocketConnected';
import { useGoogleMapsLoaded } from '../../../../components/common/GoogleMapsProvider';
import MapPlaceholder from '../../../../components/common/MapPlaceholder';
import {
  SWITZERLAND_CENTER,
  STATUS_COLORS,
  DEFAULT_MAP_OPTIONS,
  normaliseCoords,
  resolveDriverCoords,
  getDriverStatus,
  getFreshnessStatus,
  formatLastSeen,
  makeCircleMarkerIcon,
  makeClusterIcon,
  iconAnchorToAdvancedMarkerCss,
  GOOGLE_MAPS_USE_JS_STYLES,
} from '../../../../utils/mapUtils';

/** Cercle chauffeur 24×24, ancrage au centre du disque (12, 12). */
const DRIVER_MARKER_ANCHOR = iconAnchorToAdvancedMarkerCss(12, 12, 24, 24);

const ENABLE_CLUSTERING = process.env.REACT_APP_ENABLE_DRIVER_CLUSTERING === 'true';
const MAP_DEBUG =
  typeof window !== 'undefined' &&
  (window.__MAP_DEBUG === true || sessionStorage.getItem('MAP_DEBUG') === '1');
const AVAILABLE_LIGHT_GREEN = '#4ade80';
const STALE_SECONDS_THRESHOLD = 120;
const STATUS_TITLE_LABELS = {
  available: 'Disponible',
  assigned: 'Assigné',
  busy: 'En course',
  offline: 'Hors-ligne',
  emergency: 'Urgence',
};

const CONTAINER_STYLE = { width: '100%', height: '100%', minHeight: '280px' };

function getDriverDisplayName(driver) {
  return driver.full_name ||
    (driver.first_name || driver.last_name
      ? `${driver.first_name || ''} ${driver.last_name || ''}`.trim()
      : driver.username || `#${driver.id}`);
}

function getDriverMarkerLabel(driver) {
  const fullName = getDriverDisplayName(driver);
  const words = String(fullName).trim().split(/\s+/).filter(Boolean);
  if (words.length >= 2) {
    return `${words[0][0] || ''}${words[1][0] || ''}`.toUpperCase();
  }
  return String(fullName).slice(0, 2).toUpperCase();
}

function normalizeHexColor(hex) {
  if (typeof hex !== 'string') return null;
  const v = hex.trim();
  if (/^#[0-9a-fA-F]{6}$/.test(v)) return v;
  if (/^#[0-9a-fA-F]{3}$/.test(v)) {
    return `#${v[1]}${v[1]}${v[2]}${v[2]}${v[3]}${v[3]}`;
  }
  return null;
}

function blendHexColors(hexA, hexB, amount = 0.5) {
  const a = normalizeHexColor(hexA);
  const b = normalizeHexColor(hexB);
  if (!a || !b) return hexA;
  const t = Math.max(0, Math.min(1, amount));
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const ar = (pa >> 16) & 255;
  const ag = (pa >> 8) & 255;
  const ab = pa & 255;
  const br = (pb >> 16) & 255;
  const bg = (pb >> 8) & 255;
  const bb = pb & 255;
  const rr = Math.round(ar * (1 - t) + br * t);
  const rg = Math.round(ag * (1 - t) + bg * t);
  const rb = Math.round(ab * (1 - t) + bb * t);
  return `#${[rr, rg, rb].map((n) => n.toString(16).padStart(2, '0')).join('')}`;
}

function trackStaleMarkers(companyId, staleCount) {
  if (!staleCount || staleCount <= 0) return;
  const normalizedCompanyId = companyId == null ? 'unknown' : String(companyId);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent('company_realtime_metric', {
        detail: {
          metric: 'company_driver_stale_marker_total',
          labels: { company_id: normalizedCompanyId },
          value: staleCount,
          at: Date.now(),
        },
      })
    );
  }
  if (process.env.NODE_ENV === 'development') {
    // eslint-disable-next-line no-console
    console.info(
      JSON.stringify({
        metric: 'company_driver_stale_marker_total',
        company_id: normalizedCompanyId,
        value: staleCount,
        timestamp: new Date().toISOString(),
      })
    );
  }
}

/** LatLng ou littéral pour bounds / setCenter (Marker classique ou AdvancedMarkerElement). */
function getMarkerLatLngLiteral(marker) {
  if (typeof marker.getPosition === 'function') {
    const p = marker.getPosition();
    if (!p) return null;
    return { lat: p.lat(), lng: p.lng() };
  }
  const p = marker.position;
  if (!p) return null;
  if (typeof p.lat === 'function') return { lat: p.lat(), lng: p.lng() };
  return { lat: p.lat, lng: p.lng };
}

// Popup chauffeur — mini card structurée, classes CSS globales .lirie-popup-*
const createStyledTooltip = (driver, opts = {}) => {
  const { lastSeenSeconds, isStale, status: statusOverride, clientShort, currentBookingId, noGps } = opts;
  const status = statusOverride ?? getDriverStatus(driver);

  const statusConf = {
    available: { label: 'Disponible', dot: AVAILABLE_LIGHT_GREEN, bg: '#dcfce7', color: '#15803d' },
    assigned:  { label: 'Assigné',   dot: '#f59e0b', bg: '#fef3c7', color: '#b45309' },
    busy:      { label: 'En course',  dot: '#00796B', bg: '#e0f2f1', color: '#00695C' },
    offline:   { label: 'Hors-ligne', dot: '#91A3A0', bg: '#f1f5f9', color: '#64748b' },
    emergency: { label: 'Urgence',    dot: '#ef4444', bg: '#fee2e2', color: '#dc2626' },
  };
  const conf = statusConf[status] || statusConf.offline;
  if (noGps) conf.label = 'Sans GPS';

  const displayName = getDriverDisplayName(driver);

  // Ligne meta
  let metaLine = '';
  if (status === 'offline' && (lastSeenSeconds != null || isStale)) {
    metaLine = formatLastSeen(lastSeenSeconds);
  } else if (status === 'busy' || status === 'assigned') {
    const parts = [];
    if (currentBookingId) parts.push(`Mission #${currentBookingId}`);
    if (clientShort) parts.push(clientShort);
    metaLine = parts.join(' · ');
  }

  // Chips optionnels
  const chips = [];
  if (driver.vehicle_name || driver.vehicle_model) {
    chips.push(driver.vehicle_name || driver.vehicle_model);
  }

  return `<div class="lirie-popup">
  <div class="lirie-popup-header">
    <span class="lirie-popup-dot" style="background:${conf.dot}"></span>
    <span class="lirie-popup-name">${displayName}</span>
    <span class="lirie-popup-badge" style="background:${conf.bg};color:${conf.color}">${conf.label}</span>
  </div>${metaLine ? `
  <div class="lirie-popup-meta">${metaLine}</div>` : ''}${chips.length > 0 ? `
  <div class="lirie-popup-chips">${chips.map((c) => `<span class="lirie-popup-chip">${c}</span>`).join('')}</div>` : ''}
</div>`;
};

export default function DriverLiveMap({ drivers: propDrivers }) {
  const { isLoaded: gmLoaded } = useGoogleMapsLoaded();
  const mapRef = useRef(null);
  const markersRef = useRef({});
  const clustererRef = useRef(null);
  const infoWindowRef = useRef(null);
  const locatedIdsRef = useRef(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [mapReady, setMapReady] = useState(false);
  const [mapDebugInfo] = useState(null);
  const [showNoGpsBanner, setShowNoGpsBanner] = useState(false);
  const [locatedCount, setLocatedCount] = useState(0);
  const [mapMarkerCount, setMapMarkerCount] = useState(0);
  const lastStaleMetricAtRef = useRef(0);
  const { company } = useLirieCompany();
  const socketConnected = useCompanySocketConnected();

  const allDrivers = Array.isArray(propDrivers) ? propDrivers : [];
  const drivers = searchQuery
    ? allDrivers.filter((d) => {
        const q = searchQuery.toLowerCase();
        const blob = [
          d.username,
          d.full_name,
          d.first_name,
          d.last_name,
          d.email,
        ]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();
        return blob.includes(q);
      })
    : allDrivers;

  // Coordonnées entreprise comme fallback
  const companyCoords = useMemo(() => {
    const r = normaliseCoords(company?.latitude, company?.longitude);
    return r.center;
  }, [company?.latitude, company?.longitude]);

  // Centre initial de la carte
  const defaultMapCenter = useMemo(() => {
    return companyCoords || SWITZERLAND_CENTER;
  }, [companyCoords]);

  const defaultZoom = companyCoords ? 13 : 9;

  // Créer ou mettre à jour un marqueur (Marker classique si style Lirie JS, sinon AdvancedMarkerElement)
  const upsertMarker = useCallback((id, position, status, isStale, driver, tooltipOpts) => {
    const map = mapRef.current;
    if (!map || !window.google) return;

    const markerColors = { ...STATUS_COLORS, available: AVAILABLE_LIGHT_GREEN };
    const baseColor = markerColors[status] ?? markerColors.available;
    const color = isStale ? blendHexColors(baseColor, '#94A3B8', 0.55) : baseColor;
    const opacity = isStale ? 0.88 : 1;
    const markerLabel = getDriverMarkerLabel(driver);
    const titleStatus = STATUS_TITLE_LABELS[status] || status || 'Inconnu';
    const markerTitle = `${getDriverDisplayName(driver)} · ${titleStatus}${isStale ? ' · signal ancien' : ''}`;
    const iconUrl = makeCircleMarkerIcon(color, opacity, {
      label: markerLabel,
      textColor: '#ffffff',
      ringColor: '#ffffff',
    });

    if (GOOGLE_MAPS_USE_JS_STYLES) {
      if (markersRef.current[id]) {
        const marker = markersRef.current[id];
        marker.setPosition(position);
        marker.setIcon({
          url: iconUrl,
          scaledSize: new window.google.maps.Size(24, 24),
          anchor: new window.google.maps.Point(12, 12),
        });
        marker.setTitle(markerTitle);
        marker._tooltipHtml = createStyledTooltip(driver, tooltipOpts);
        marker._driverStatus = status;
        return marker;
      }

      const clustered = ENABLE_CLUSTERING && clustererRef.current;
      const marker = new window.google.maps.Marker({
        position,
        map: clustered ? null : map,
        icon: {
          url: iconUrl,
          scaledSize: new window.google.maps.Size(24, 24),
          anchor: new window.google.maps.Point(12, 12),
        },
        title: markerTitle,
        optimized: true,
      });

      marker._tooltipHtml = createStyledTooltip(driver, tooltipOpts);
      marker._driverStatus = status;

      marker.addListener('mouseover', () => {
        if (!infoWindowRef.current) {
          infoWindowRef.current = new window.google.maps.InfoWindow({
            disableAutoPan: false,
            maxWidth: 260,
            pixelOffset: new window.google.maps.Size(0, -4),
          });
        }
        infoWindowRef.current.setContent(marker._tooltipHtml);
        infoWindowRef.current.open(map, marker);
      });
      marker.addListener('click', () => {
        if (!infoWindowRef.current) {
          infoWindowRef.current = new window.google.maps.InfoWindow({
            disableAutoPan: false,
            maxWidth: 260,
            pixelOffset: new window.google.maps.Size(0, -4),
          });
        }
        infoWindowRef.current.setContent(marker._tooltipHtml);
        infoWindowRef.current.open(map, marker);
      });
      marker.addListener('mouseout', () => {
        if (infoWindowRef.current) infoWindowRef.current.close();
      });

      markersRef.current[id] = marker;
      if (clustered && clustererRef.current) {
        clustererRef.current.addMarker(marker);
      }
      return marker;
    }

    const AdvancedMarkerElement = window.google?.maps?.marker?.AdvancedMarkerElement;
    if (!AdvancedMarkerElement) return;

    if (markersRef.current[id]) {
      const marker = markersRef.current[id];
      marker.position = position;
      marker.title = markerTitle;
      const img = marker._img;
      if (img) {
        img.src = iconUrl;
        img.style.opacity = String(opacity);
        img.alt = markerTitle;
        img.title = markerTitle;
      }
      marker._tooltipHtml = createStyledTooltip(driver, tooltipOpts);
      marker._driverStatus = status;
      return marker;
    }

    const img = document.createElement('img');
    img.src = iconUrl;
    img.width = 24;
    img.height = 24;
    img.style.display = 'block';
    img.style.opacity = String(opacity);
    img.alt = markerTitle;
    img.title = markerTitle;
    img.draggable = false;

    const clustered = ENABLE_CLUSTERING && clustererRef.current;
    const marker = new AdvancedMarkerElement({
      position,
      map: clustered ? null : map,
      content: img,
      anchorLeft: DRIVER_MARKER_ANCHOR.anchorLeft,
      anchorTop: DRIVER_MARKER_ANCHOR.anchorTop,
      gmpClickable: true,
      title: markerTitle,
    });
    marker._img = img;

    marker._tooltipHtml = createStyledTooltip(driver, tooltipOpts);
    marker._driverStatus = status;

    const onMouseEnter = () => {
      if (!infoWindowRef.current) {
        infoWindowRef.current = new window.google.maps.InfoWindow({
          disableAutoPan: false,
          maxWidth: 260,
          pixelOffset: new window.google.maps.Size(0, -4),
        });
      }
      infoWindowRef.current.setContent(marker._tooltipHtml);
      infoWindowRef.current.open({ map, anchor: marker });
    };
    const onMouseLeave = () => {
      if (infoWindowRef.current) infoWindowRef.current.close();
    };

    img.addEventListener('mouseenter', onMouseEnter);
    img.addEventListener('click', onMouseEnter);
    img.addEventListener('mouseleave', onMouseLeave);
    marker._hoverCleanup = () => {
      img.removeEventListener('mouseenter', onMouseEnter);
      img.removeEventListener('click', onMouseEnter);
      img.removeEventListener('mouseleave', onMouseLeave);
    };

    markersRef.current[id] = marker;

    if (clustered && clustererRef.current) {
      clustererRef.current.addMarker(marker);
    }

    return marker;
  }, []);

  // Supprimer un marqueur
  const removeMarker = useCallback((id) => {
    const marker = markersRef.current[id];
    if (!marker) return;
    if (ENABLE_CLUSTERING && clustererRef.current) {
      clustererRef.current.removeMarker(marker);
    }
    if (typeof marker._hoverCleanup === 'function') marker._hoverCleanup();
    if (GOOGLE_MAPS_USE_JS_STYLES) {
      marker.setMap(null);
    } else {
      marker.map = null;
    }
    delete markersRef.current[id];
  }, []);

  // Fit bounds sur tous les marqueurs visibles
  const fitBoundsToMarkers = useCallback((maxZoom = 14) => {
    const map = mapRef.current;
    if (!map || !window.google) return;
    const entries = Object.values(markersRef.current);
    if (entries.length === 0) return;

    const bounds = new window.google.maps.LatLngBounds();
    entries.forEach((m) => {
      const ll = getMarkerLatLngLiteral(m);
      if (ll) bounds.extend(ll);
    });
    map.fitBounds(bounds, { top: 40, right: 40, bottom: 40, left: 40 });

    const listener = window.google.maps.event.addListenerOnce(map, 'idle', () => {
      if (map.getZoom() > maxZoom) {
        map.setZoom(maxZoom);
      }
    });
    return () => window.google.maps.event.removeListener(listener);
  }, []);

  // Callback quand Google Map est chargée
  const onMapLoad = useCallback((map) => {
    mapRef.current = map;

    if (ENABLE_CLUSTERING) {
      if (GOOGLE_MAPS_USE_JS_STYLES) {
        clustererRef.current = new MarkerClusterer({
          map,
          markers: [],
          renderer: {
            render: ({ count, position }) => {
              const size = count < 10 ? 40 : count < 50 ? 46 : 52;
              return new window.google.maps.Marker({
                position,
                icon: {
                  url: makeClusterIcon(count),
                  scaledSize: new window.google.maps.Size(size, size),
                  anchor: new window.google.maps.Point(size / 2, size / 2),
                },
                zIndex: Number(window.google.maps.Marker.MAX_ZINDEX) + count,
              });
            },
          },
        });
      } else if (window.google?.maps?.marker?.AdvancedMarkerElement) {
        const AdvancedMarkerElement = window.google.maps.marker.AdvancedMarkerElement;
        clustererRef.current = new MarkerClusterer({
          map,
          markers: [],
          renderer: {
            render: ({ count, position }) => {
              const size = count < 10 ? 40 : count < 50 ? 46 : 52;
              const img = document.createElement('img');
              img.src = makeClusterIcon(count);
              img.width = size;
              img.height = size;
              img.style.display = 'block';
              const clusterAnchor = iconAnchorToAdvancedMarkerCss(size / 2, size / 2, size, size);
              return new AdvancedMarkerElement({
                position,
                content: img,
                anchorLeft: clusterAnchor.anchorLeft,
                anchorTop: clusterAnchor.anchorTop,
                zIndex: 1000000 + count,
              });
            },
          },
        });
      }
    }

    setMapReady(true);
    if (MAP_DEBUG) console.log('[DriverLiveMap] Google Map chargée');
  }, []);

  // Placer les positions statiques au chargement + recalculer les localisés (GPS frais < 5 min)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !Array.isArray(drivers)) return;

    let placed = 0;
    let staleMarkersCount = 0;
    const newLocatedIds = new Set();

    drivers.forEach((d) => {
      const resolved = resolveDriverCoords(d, companyCoords);
      if (!resolved) return;

      const { coords, isFallback } = resolved;
      const status = isFallback ? 'offline' : getDriverStatus(d);
      const freshness = getFreshnessStatus(d);
      const isLocated = !isFallback && freshness !== 'offline';
      if (isLocated) newLocatedIds.add(d.id);

      const lastSeenSecondsNumber = Number(d.last_seen_seconds);
      const locStat = String(d.location_status || '').toLowerCase();
      const hasBackendStatus = locStat === 'stale' || locStat === 'offline'
        || locStat === 'live' || locStat === 'recent';
      const staleByAge = !d.location_status
        && Number.isFinite(lastSeenSecondsNumber)
        && lastSeenSecondsNumber > STALE_SECONDS_THRESHOLD;
      const staleByStatus = locStat === 'stale' || locStat === 'offline';
      const isStaleMarker = isFallback
        || (hasBackendStatus ? staleByStatus : staleByAge);
      if (isStaleMarker) staleMarkersCount += 1;

      const tooltipOpts = isFallback
        ? { status: 'offline', isStale: true, noGps: true }
        : {
            lastSeenSeconds: d.last_seen_seconds,
            isStale: isStaleMarker,
            clientShort: d.client_short,
            currentBookingId: d.current_booking_id,
          };
      if (!markersRef.current[d.id]) {
        upsertMarker(d.id, coords, status, isStaleMarker, d, tooltipOpts);
        placed++;
      } else {
        upsertMarker(d.id, coords, status, isStaleMarker, d, tooltipOpts);
      }
    });

    // Synchroniser le compteur localisés
    locatedIdsRef.current = newLocatedIds;
    setLocatedCount(newLocatedIds.size);

    // Supprimer les marqueurs des chauffeurs qui ne sont plus dans la liste
    const driverIds = new Set(drivers.map((d) => d.id));
    Object.keys(markersRef.current).forEach((driverId) => {
      if (!driverIds.has(Number(driverId))) {
        removeMarker(driverId);
      }
    });

    setMapMarkerCount(Object.keys(markersRef.current).length);

    const visibleMarkers = Object.values(markersRef.current);
    if (visibleMarkers.length === 1) {
      const pos = getMarkerLatLngLiteral(visibleMarkers[0]);
      if (pos) map.setCenter(pos);
      map.setZoom(15);
    } else if (visibleMarkers.length > 1) {
      fitBoundsToMarkers(14);
    } else if (placed === 0 && visibleMarkers.length === 0) {
      map.setCenter(companyCoords || SWITZERLAND_CENTER);
      map.setZoom(companyCoords ? 13 : 9);
    }

    const now = Date.now();
    if (staleMarkersCount > 0 && now - lastStaleMetricAtRef.current >= 60000) {
      lastStaleMetricAtRef.current = now;
      trackStaleMarkers(company?.id, staleMarkersCount);
    }
  }, [drivers, companyCoords, upsertMarker, removeMarker, fitBoundsToMarkers, company?.id]);

  useEffect(() => {
    setShowNoGpsBanner(allDrivers.length > 0 && locatedCount === 0);
  }, [locatedCount, allDrivers.length]);

  // Compteur chauffeurs total
  const totalCount = allDrivers.length;
  const noGpsTitle = !socketConnected
    ? 'Temps réel indisponible'
    : 'Aucun GPS récent';
  const noGpsDetail = !socketConnected
    ? 'Données issues du dernier chargement. Vérifiez la connexion réseau.'
    : 'Aucune position fraîche. Vérifiez l’app chauffeur et le GPS.';

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        minHeight: '280px',
        background: '#f0f2f5',
      }}
    >
      {gmLoaded ? (
        <div
          className="lirie-driver-map-enter"
          style={{ width: '100%', height: '100%', minHeight: 280 }}
        >
          <GoogleMap
            mapContainerStyle={CONTAINER_STYLE}
            center={defaultMapCenter}
            zoom={defaultZoom}
            options={DEFAULT_MAP_OPTIONS}
            onLoad={onMapLoad}
          />
        </div>
      ) : (
        <MapPlaceholder style={{ minHeight: 280 }} delayLabelMs={350} />
      )}

      {/* Barre de recherche — responsive */}
      {mapReady && (
        <div
          style={{
            position: 'absolute',
            top: 8,
            right: 8,
            left: 'auto',
            zIndex: 100,
            background: '#fff',
            border: '1px solid #E2E8F0',
            borderRadius: 8,
            padding: '5px 10px',
            boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            maxWidth: 'min(220px, calc(100% - 16px))',
            minWidth: 0,
            transition: 'border-color 0.2s',
          }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
            <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
          </svg>
          <input
            type="text"
            placeholder="Rechercher sur la carte…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              border: 'none',
              outline: 'none',
              background: 'transparent',
              fontSize: 11,
              flex: 1,
              color: '#1E293B',
              fontFamily: "Inter, -apple-system, 'Segoe UI', sans-serif",
              minWidth: 0,
            }}
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery('')}
              style={{
                border: 'none',
                background: '#E2E8F0',
                color: '#64748B',
                borderRadius: '50%',
                width: 16,
                height: 16,
                cursor: 'pointer',
                fontSize: 9,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                lineHeight: 1,
                flexShrink: 0,
              }}
            >
              ✕
            </button>
          )}
        </div>
      )}

      {/* Légende + compteur GPS — responsive */}
      {mapReady && (
        <div
          style={{
            position: 'absolute',
            bottom: 42,
            left: 8,
            zIndex: 100,
            background: 'rgba(255,255,255,0.95)',
            backdropFilter: 'blur(4px)',
            border: '1px solid #E2E8F0',
            borderRadius: 8,
            padding: '6px 10px',
            fontSize: 10,
            fontFamily: "Inter, -apple-system, 'Segoe UI', sans-serif",
            boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
            maxWidth: 'min(200px, calc(100% - 80px))',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: locatedCount > 0 ? '#00796B' : '#91A3A0',
                  display: 'inline-block',
                  flexShrink: 0,
                  animation: locatedCount > 0 ? 'pulse 2s infinite' : 'none',
                }}
              />
              <span
                title={socketConnected ? 'Temps réel (WebSocket) connecté' : 'Temps réel indisponible — snapshot HTTP'}
                style={{
                  fontSize: 8,
                  lineHeight: 1,
                  color: socketConnected ? '#16a34a' : '#94a3b8',
                  flexShrink: 0,
                }}
              >
                {socketConnected ? '●' : '○'}
              </span>
              <div style={{ lineHeight: 1.3, minWidth: 0 }}>
                <div style={{ fontWeight: 600, color: '#1E293B', fontSize: 11 }}>
                  {locatedCount}/{totalCount} localisés
                </div>
                {locatedCount === 0 && totalCount > 0 && (
                  <div style={{ fontSize: 9, color: '#94A3B8', marginTop: 1 }}>
                    {!socketConnected ? 'Dernières données en cache' : 'Aucune position fraîche'}
                  </div>
                )}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, fontSize: 9, color: '#64748B', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: AVAILABLE_LIGHT_GREEN, flexShrink: 0 }} />
                <span>Dispo</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#f59e0b', flexShrink: 0 }} />
                <span>Assigné</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#00796B', flexShrink: 0 }} />
                <span>Course</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#91A3A0', flexShrink: 0 }} />
                <span>Off</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ef4444', flexShrink: 0 }} />
                <span>Urgence</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Bouton Recentrer — responsive */}
      {mapReady && (
        <button
          type="button"
          disabled={mapMarkerCount === 0}
          onClick={() => {
            if (mapMarkerCount === 0) return;
            fitBoundsToMarkers(14);
          }}
          style={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            zIndex: 100,
            padding: '5px 12px',
            fontSize: 11,
            fontWeight: 600,
            color: '#fff',
            background: mapMarkerCount === 0 ? '#94a3b8' : 'linear-gradient(135deg, #00796B 0%, #00695C 100%)',
            border: 'none',
            borderRadius: 8,
            boxShadow: '0 1px 4px rgba(0,0,0,0.12)',
            cursor: mapMarkerCount === 0 ? 'not-allowed' : 'pointer',
            fontFamily: "Inter, -apple-system, 'Segoe UI', sans-serif",
            transition: 'opacity 0.2s',
            opacity: mapMarkerCount === 0 ? 0.75 : 1,
          }}
          title={mapMarkerCount === 0 ? 'Aucun marqueur sur la carte' : 'Recadrer sur les chauffeurs'}
          onMouseEnter={(e) => {
            if (mapMarkerCount > 0) e.currentTarget.style.opacity = '0.9';
          }}
          onMouseLeave={(e) => {
            if (mapMarkerCount > 0) e.currentTarget.style.opacity = '1';
          }}
        >
          Recentrer
        </button>
      )}

      {/* Bannière absence GPS */}
      {mapReady && showNoGpsBanner && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '8px 16px',
            background: '#fff',
            border: '2px solid #E2E8F0',
            color: '#64748B',
            borderRadius: 10,
            fontSize: 12,
            fontWeight: 500,
            zIndex: 100,
            boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
            textAlign: 'center',
            maxWidth: '90%',
            fontFamily: "Inter, -apple-system, 'Segoe UI', sans-serif",
          }}
        >
          <div style={{ color: '#334155', fontWeight: 600 }}>{noGpsTitle}</div>
          <div style={{ fontSize: 11, fontWeight: 400, marginTop: 4, lineHeight: 1.35 }}>{noGpsDetail}</div>
        </div>
      )}

      {/* Panneau debug */}
      {MAP_DEBUG && mapDebugInfo && (
        <div
          style={{
            position: 'absolute',
            bottom: 8,
            left: 8,
            right: 80,
            padding: '8px 10px',
            background: 'rgba(0,0,0,0.75)',
            color: '#eee',
            fontSize: 11,
            fontFamily: 'monospace',
            borderRadius: 6,
            zIndex: 100,
            maxHeight: 120,
            overflow: 'auto',
          }}
        >
          <div><strong>MAP_DEBUG (Google Maps)</strong></div>
          <div>Reçues: {mapDebugInfo.received} | Valides: {mapDebugInfo.valid}</div>
          <div>
            joined_company: {mapDebugInfo.joinedReceived ? '✓ reçu' : '–'}
            {mapDebugInfo.fallbackUsed ? ' | fallback 1.5s utilisé' : ''}
          </div>
          {mapDebugInfo.lastUpdate && <div>Dernière mise à jour: {mapDebugInfo.lastUpdate}</div>}
          {mapDebugInfo.sample && (
            <div>Exemple: lat={mapDebugInfo.sample.lat} lon={mapDebugInfo.sample.lon} id={mapDebugInfo.sample.driver_id}</div>
          )}
          {mapDebugInfo.exclusionReasons?.length > 0 && (
            <div>Exclusions: {mapDebugInfo.exclusionReasons.join(', ')}</div>
          )}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        .lirie-driver-map-enter {
          animation: lirieMapSoftIn 0.45s ease-out;
        }
        @keyframes lirieMapSoftIn {
          from { opacity: 0.86; }
          to { opacity: 1; }
        }
        @media (prefers-reduced-motion: reduce) {
          .lirie-driver-map-enter { animation: none; }
        }
      `}</style>
    </div>
  );
}
