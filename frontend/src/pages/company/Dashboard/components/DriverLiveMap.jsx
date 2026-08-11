// src/pages/company/Dashboard/components/DriverLiveMap.jsx
import React, { useEffect, useLayoutEffect, useRef, useState, useMemo, useCallback, memo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { lirieKeys } from '../../../../queryKeys/lirie';
import { projectDriversForMap } from '../../../../utils/companyDriverProjections';
import {
  buildDriverStructuralSetKey,
  isSameMarkerPosition,
  isDriverConstrained,
  getDriverConstraintReason,
  resolveDriverMapProjection,
  isNonLiveGpsPosition,
  CONSTRAINED_MARKER_COLOR,
} from '../../../../utils/companyDriverProjections';
import { resolveDriverClusteringEnabled } from '../../../../utils/driverMapClustering';
import {
  recordDriverLiveMapRender,
  recordFitBoundsCall,
  recordMarkerCreate,
  recordMarkerPositionUpdate,
} from '../../../../utils/companyDashboardPerfInstrumentation';
import { perfMark } from '../../../../utils/companyDashboardWebPerf';
import { recordMapsOverlayStats } from '../../../../utils/companyDashboardMapsOverlayStats';
import { isCompanyDashboardPerfEnabled } from '../../../../utils/companyDashboardPerfInstrumentation';
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
  getDriverFreshnessLabel,
  formatLastSeen,
  makeCircleMarkerIcon,
  makeClusterIcon,
  iconAnchorToAdvancedMarkerCss,
} from '../../../../utils/mapUtils';
import {
  applyLocalLocationFreshness,
  resolveLocalLocationFreshnessStatus,
} from '../../../../utils/localDriverLocationFreshness';
import {
  interpolateMarkerPosition,
  resolveMarkerMotionDurationMs,
  resolveMotionDurationFromDistance,
  haversineDistanceMeters,
  isApproximateGpsAccuracy,
  MARKER_MOTION_DEFAULT_MS,
} from '../../../../utils/driverMarkerMotion';

/**
 * AGENT FIX (P0 rendu marqueurs) : sur ce setup, les AdvancedMarkerElement
 * restent ATTACHÉS à la carte mais NON PEINTS (sous-arbre content-visibility)
 * → invariant prouvé : drivers=markersRef=attached=5 mais painted=0. Les
 * marqueurs classiques (`google.maps.Marker`, rendus sur le canvas de la carte)
 * ne souffrent pas de ce problème. On force donc le chemin classique pour le
 * rendu des chauffeurs, indépendamment du style de carte (cloud/JS).
 */
const GOOGLE_MAPS_USE_JS_STYLES = true;

/** Cercle chauffeur 24×24, ancrage au centre du disque (12, 12). */
const DRIVER_MARKER_ANCHOR = iconAnchorToAdvancedMarkerCss(12, 12, 24, 24);

const MAP_DEBUG =
  typeof window !== 'undefined' &&
  (window.__MAP_DEBUG === true || sessionStorage.getItem('MAP_DEBUG') === '1');
const AVAILABLE_LIGHT_GREEN = '#4ade80';
/** Au-delà, interpolation désactivée (clustering / perf). */
const MARKER_SMOOTH_MOTION_MAX_DRIVERS = 48;
const STATUS_TITLE_LABELS = {
  available: 'Disponible',
  assigned: 'Assigné',
  busy: 'En course',
  offline: 'Hors-ligne',
  emergency: 'Urgence',
  constrained: 'Position figée',
};

const CONTAINER_STYLE = { width: '100%', height: '100%', minHeight: '280px' };

function createDriverMarkerClusterer(map) {
  if (GOOGLE_MAPS_USE_JS_STYLES) {
    return new MarkerClusterer({
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
  }
  if (window.google?.maps?.marker?.AdvancedMarkerElement) {
    const AdvancedMarkerElement = window.google.maps.marker.AdvancedMarkerElement;
    return new MarkerClusterer({
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
  return null;
}

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

function setMarkerLatLng(marker, latLng) {
  if (typeof marker.setPosition === 'function') {
    marker.setPosition(latLng);
  } else {
    marker.position = latLng;
  }
}

// Popup chauffeur — mini card structurée, classes CSS globales .lirie-popup-*
const escapeHtml = (value) => {
  if (value == null) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
};

const CONSTRAINT_REASON_LABELS = {
  battery_optimized: 'Optimisation batterie OS',
  permission_bg_denied: 'Permission arrière-plan refusée',
  permission_fg_denied: 'Permission GPS refusée',
  fgs_not_running: 'Service tracking arrêté',
  gps_provider_disabled: 'GPS désactivé',
  fix_stale: 'Dernier fix GPS trop ancien',
};

/** Libellé court (badge/marqueur) par contrainte — évite de tout étiqueter « batterie ». */
const CONSTRAINT_REASON_BADGE = {
  battery_optimized: 'Batterie restreinte',
  permission_bg_denied: 'Permission arrière-plan',
  permission_fg_denied: 'Permission GPS',
  fgs_not_running: 'Tracking arrêté',
  gps_provider_disabled: 'GPS désactivé',
  fix_stale: 'Fix GPS ancien',
};

/** Badge court reflétant la vraie contrainte (fallback générique « Position figée »). */
function resolveConstraintBadgeLabel(constraintReason) {
  const key = constraintReason ? String(constraintReason).trim() : '';
  return CONSTRAINT_REASON_BADGE[key] || 'Position figée';
}

/** Phrase meta du tooltip, pilotée par la contrainte réelle (pas « batterie » par défaut). */
function buildConstraintMetaLine(constraintReason, lastSeenSeconds) {
  const key = constraintReason ? String(constraintReason).trim() : '';
  const reasonLabel = key ? (CONSTRAINT_REASON_LABELS[key] || constraintReason) : 'inconnue';
  const seenLabel = Number.isFinite(Number(lastSeenSeconds))
    ? `Dernier signal il y a ${Number(lastSeenSeconds)}s.`
    : 'Dernier signal inconnu.';
  const cause = key === 'battery_optimized'
    ? "l'app du chauffeur signale une optimisation batterie de l'OS"
    : `l'app du chauffeur signale&nbsp;: ${escapeHtml(reasonLabel)}`;
  return `Position figée — ${cause}. ${seenLabel}`;
}

const createStyledTooltip = (driver, opts = {}) => {
  const {
    lastSeenSeconds,
    isStale,
    status: statusOverride,
    clientShort,
    currentBookingId,
    noGps,
    isConstrained,
    constraintReason,
    businessStatus,
    gpsLabel,
    serviceWindowStatus,
  } = opts;
  const status = statusOverride ?? getDriverStatus(driver);
  const biz = businessStatus ?? getDriverStatus(driver);

  const statusConf = {
    available:   { label: 'Disponible',         dot: AVAILABLE_LIGHT_GREEN, bg: '#dcfce7', color: '#15803d' },
    assigned:    { label: 'Assigné',            dot: '#f59e0b',             bg: '#fef3c7', color: '#b45309' },
    busy:        { label: 'En course',          dot: '#00796B',             bg: '#e0f2f1', color: '#00695C' },
    offline:     { label: 'Hors-ligne',         dot: '#91A3A0',             bg: '#f1f5f9', color: '#64748b' },
    emergency:   { label: 'Urgence',            dot: '#ef4444',             bg: '#fee2e2', color: '#dc2626' },
    constrained: { label: 'Batterie restreinte', dot: CONSTRAINED_MARKER_COLOR,   bg: '#ffedd5', color: '#9a3412' },
    off_duty:    { label: 'Hors service',       dot: '#91A3A0',             bg: '#f1f5f9', color: '#64748b' },
  };
  // Badge principal = métier si en course/assigné, sinon statut visuel GPS.
  // TIME-4 : « Hors service » uniquement si service_window_status=off_duty (pas mission_override).
  const isOffDuty = String(serviceWindowStatus || driver?.service_window_status || '') === 'off_duty';
  const badgeKey = (biz === 'busy' || biz === 'assigned')
    ? biz
    : (isOffDuty ? 'off_duty' : status);
  const conf = { ...(statusConf[badgeKey] || statusConf.offline) };
  if (noGps) conf.label = 'Sans GPS';

  const displayName = getDriverDisplayName(driver);
  const isApproximate = Boolean(opts.isApproximate)
    || isApproximateGpsAccuracy(driver?.accuracy);

  // Ligne meta
  let metaLine = '';
  if (status === 'constrained' || (isConstrained && !isStale)) {
    if (!noGps) conf.label = resolveConstraintBadgeLabel(constraintReason);
    metaLine = buildConstraintMetaLine(constraintReason, lastSeenSeconds);
  } else if (gpsLabel) {
    metaLine = escapeHtml(gpsLabel);
  } else if (status === 'offline' && (lastSeenSeconds != null || isStale)) {
    metaLine = formatLastSeen(lastSeenSeconds);
  } else if (status === 'busy' || status === 'assigned' || biz === 'busy' || biz === 'assigned') {
    const parts = [];
    if (currentBookingId) parts.push(`Mission #${Number(currentBookingId) || currentBookingId}`);
    if (clientShort) parts.push(escapeHtml(clientShort));
    metaLine = parts.join(' · ');
  }

  // Chips optionnels
  const chips = [];
  if (driver.vehicle_name || driver.vehicle_model) {
    chips.push(driver.vehicle_name || driver.vehicle_model);
  }
  if ((biz === 'busy' || biz === 'assigned') && (status === 'offline' || isStale) && !noGps) {
    chips.push(conf.label === 'En course' || conf.label === 'Assigné' ? 'GPS hors ligne' : conf.label);
  }
  if (isApproximate && !noGps) {
    chips.push('Position approximative');
  }

  return `<div class="lirie-popup">
  <div class="lirie-popup-header">
    <span class="lirie-popup-dot" style="background:${conf.dot}"></span>
    <span class="lirie-popup-name">${escapeHtml(displayName)}</span>
    <span class="lirie-popup-badge" style="background:${conf.bg};color:${conf.color}">${escapeHtml(conf.label)}</span>
  </div>${metaLine ? `
  <div class="lirie-popup-meta">${metaLine}</div>` : ''}${chips.length > 0 ? `
  <div class="lirie-popup-chips">${chips.map((c) => `<span class="lirie-popup-chip">${escapeHtml(c)}</span>`).join('')}</div>` : ''}
</div>`;
};

/** Export testable (XSS InfoWindow) — ne pas utiliser hors tests / carte. */
export { createStyledTooltip, escapeHtml };

function DriverLiveMap({ drivers: propDrivers }) {
  recordDriverLiveMapRender();

  const { isLoaded: gmLoaded, ensureLoaded } = useGoogleMapsLoaded();

  useEffect(() => {
    ensureLoaded();
  }, [ensureLoaded]);
  const mapShellRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef({});
  const markerMotionRef = useRef({});
  const markerMotionRafRef = useRef(null);
  const markerLastMotionAtRef = useRef({});
  const markerMotionLastFrameRef = useRef(null);
  const smoothMotionEnabledRef = useRef(true);
  const clustererRef = useRef(null);
  const infoWindowRef = useRef(null);
  const locatedIdsRef = useRef(new Set());
  const lastStructuralSetKeyRef = useRef(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [mapReady, setMapReady] = useState(false);
  const [mapDebugInfo] = useState(null);
  const [showNoGpsBanner, setShowNoGpsBanner] = useState(false);
  const [locatedCount, setLocatedCount] = useState(0);
  const [mapMarkerCount, setMapMarkerCount] = useState(0);
  const lastStaleMetricAtRef = useRef(0);
  const firstMarkersMarkedRef = useRef(false);
  const mapConstructorMarkedRef = useRef(false);
  const idleMarkerCaptureScheduledRef = useRef(false);
  const [mapViewLocked, setMapViewLocked] = useState(false);
  const { company } = useLirieCompany();
  const queryClient = useQueryClient();

  const captureOverlayStats = useCallback(() => {
    if (!isCompanyDashboardPerfEnabled()) return;
    const map = mapRef.current;
    const markerEntries = Object.values(markersRef.current);
    let activeMarkerCount = markerEntries.length;
    if (map && window.google?.maps?.LatLng && typeof map.getBounds === 'function') {
      const bounds = map.getBounds();
      if (bounds) {
        activeMarkerCount = 0;
        markerEntries.forEach((m) => {
          const ll = getMarkerLatLngLiteral(m);
          if (!ll) return;
          const latLng = new window.google.maps.LatLng(ll.lat, ll.lng);
          if (bounds.contains(latLng)) activeMarkerCount += 1;
        });
      }
    }
    let clusterCount = null;
    if (clustererRef.current && Array.isArray(clustererRef.current.markers)) {
      clusterCount = clustererRef.current.markers.length;
    } else if (clustererRef.current && clusteringEnabledRef.current) {
      clusterCount = markerEntries.length;
    }
    recordMapsOverlayStats({
      markerCount: markerEntries.length,
      activeMarkerCount,
      overlayCount: 0,
      infoWindowCount: infoWindowRef.current ? 1 : 0,
      clusterCount,
    });
  }, []);

  useLayoutEffect(() => {
    if (!gmLoaded || mapConstructorMarkedRef.current) return;
    mapConstructorMarkedRef.current = true;
    perfMark('gmaps_map_constructor_start');
  }, [gmLoaded]);
  const socketConnected = useCompanySocketConnected();

  const allDrivers = Array.isArray(propDrivers) ? propDrivers : [];
  const clusteringEnabled = useMemo(
    () => resolveDriverClusteringEnabled(allDrivers.length),
    [allDrivers.length],
  );
  const clusteringEnabledRef = useRef(clusteringEnabled);
  clusteringEnabledRef.current = clusteringEnabled;
  smoothMotionEnabledRef.current = allDrivers.length <= MARKER_SMOOTH_MOTION_MAX_DRIVERS;

  const stopMarkerMotionLoop = useCallback(() => {
    if (markerMotionRafRef.current != null) {
      cancelAnimationFrame(markerMotionRafRef.current);
      markerMotionRafRef.current = null;
    }
  }, []);

  const runMarkerMotionFrame = useCallback(() => {
    markerMotionRafRef.current = null;
    const motions = markerMotionRef.current;
    const now = performance.now();
    markerMotionLastFrameRef.current = now;
    let anyActive = false;

    Object.keys(motions).forEach((key) => {
      const motion = motions[key];
      const marker = markersRef.current[key];
      if (!marker) {
        delete motions[key];
        return;
      }

      // Canary : zéro dead reckoning — à l’arrivée sur le fix réel B, on s’arrête.
      if (motion.phase === 'project') {
        setMarkerLatLng(marker, {
          lat: motion.currentLat,
          lng: motion.currentLng,
        });
        delete motions[key];
        recordMarkerPositionUpdate();
        return;
      }

      const progress = (now - motion.startMs) / motion.durationMs;
      if (progress >= 1) {
        setMarkerLatLng(marker, motion.to);
        delete motions[key];
        recordMarkerPositionUpdate();
        return;
      }

      anyActive = true;
      setMarkerLatLng(
        marker,
        interpolateMarkerPosition(motion.from, motion.to, progress)
      );
    });

    if (anyActive) {
      markerMotionRafRef.current = requestAnimationFrame(runMarkerMotionFrame);
    } else {
      markerMotionLastFrameRef.current = null;
    }
  }, []);

  const scheduleMarkerMotionLoop = useCallback(() => {
    if (markerMotionRafRef.current == null) {
      markerMotionRafRef.current = requestAnimationFrame(runMarkerMotionFrame);
    }
  }, [runMarkerMotionFrame]);

  const cancelMarkerMotion = useCallback((id) => {
    const idKey = String(id);
    delete markerMotionRef.current[idKey];
    delete markerLastMotionAtRef.current[idKey];
    if (Object.keys(markerMotionRef.current).length === 0) {
      markerMotionLastFrameRef.current = null;
    }
  }, []);

  const animateMarkerTo = useCallback(
    (id, marker, targetPosition) => {
      const idKey = String(id);
      const target = { lat: Number(targetPosition.lat), lng: Number(targetPosition.lng) };
      const current = getMarkerLatLngLiteral(marker);
      if (isSameMarkerPosition(current, target)) return;

      if (!current) {
        setMarkerLatLng(marker, target);
        markerLastMotionAtRef.current[idKey] = Date.now();
        return;
      }

      const nowMs = Date.now();
      const lastAt = markerLastMotionAtRef.current[idKey];
      const expectedIntervalMs =
        lastAt != null && Number.isFinite(lastAt) ? nowMs - lastAt : MARKER_MOTION_DEFAULT_MS;
      let durationMs = resolveMarkerMotionDurationMs(lastAt, nowMs);
      const from = getMarkerLatLngLiteral(marker) || current;
      durationMs = resolveMotionDurationFromDistance(
        durationMs,
        haversineDistanceMeters(from, target)
      );
      markerLastMotionAtRef.current[idKey] = nowMs;

      markerMotionRef.current[idKey] = {
        phase: 'animate',
        from: { lat: from.lat, lng: from.lng },
        to: target,
        startMs: performance.now(),
        durationMs,
        expectedIntervalMs,
      };
      markerMotionLastFrameRef.current = null;
      scheduleMarkerMotionLoop();
    },
    [scheduleMarkerMotionLoop]
  );

  const applyMarkerPosition = useCallback(
    (id, marker, position, { isStale, status }) => {
      const prevPos = getMarkerLatLngLiteral(marker);
      if (isSameMarkerPosition(prevPos, position)) return;

      const canSmooth =
        smoothMotionEnabledRef.current &&
        !isStale &&
        status !== 'constrained';

      if (canSmooth) {
        animateMarkerTo(id, marker, position);
      } else {
        setMarkerLatLng(marker, position);
        markerLastMotionAtRef.current[String(id)] = Date.now();
        recordMarkerPositionUpdate();
      }
    },
    [animateMarkerTo]
  );

  useEffect(() => () => stopMarkerMotionLoop(), [stopMarkerMotionLoop]);

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

    const markerColors = {
      ...STATUS_COLORS,
      available: AVAILABLE_LIGHT_GREEN,
      constrained: CONSTRAINED_MARKER_COLOR,
    };
    const baseColor = markerColors[status] ?? markerColors.available;
    // Position figée (degraded_constrained / *_constrained) : on garde la couleur orange
    // pleine intensité même si `last_seen_seconds` dépasse le seuil "stale" — c'est précisément
    // l'information visuelle : la position N'EST PAS rafraîchie côté chauffeur.
    const isConstrainedMarker = status === 'constrained';
    const isApproximate = Boolean(tooltipOpts?.isApproximate)
      || isApproximateGpsAccuracy(driver?.accuracy);
    const color = isStale && !isConstrainedMarker
      ? blendHexColors(baseColor, '#94A3B8', 0.55)
      : baseColor;
    let opacity = isStale && !isConstrainedMarker ? 0.88 : 1;
    if (isApproximate && !isStale) opacity = 0.78;
    const markerLabel = getDriverMarkerLabel(driver);
    const titleStatus = status === 'constrained'
      ? resolveConstraintBadgeLabel(getDriverConstraintReason(driver))
      : (STATUS_TITLE_LABELS[status] || status || 'Inconnu');
    const businessLabel = STATUS_TITLE_LABELS[tooltipOpts?.businessStatus]
      || tooltipOpts?.businessStatus
      || null;
    const approxSuffix = isApproximate && !tooltipOpts?.noGps ? ' · position approximative' : '';
    const markerTitle = businessLabel && status === 'offline' && !tooltipOpts?.noGps
      ? `${getDriverDisplayName(driver)} · ${businessLabel} · GPS hors ligne`
      : `${getDriverDisplayName(driver)} · ${titleStatus}${isStale ? ' · signal ancien' : ''}${approxSuffix}`;
    const iconUrl = makeCircleMarkerIcon(color, opacity, {
      label: markerLabel,
      textColor: '#ffffff',
      ringColor: isApproximate ? '#F59E0B' : '#ffffff',
    });

    if (GOOGLE_MAPS_USE_JS_STYLES) {
      if (markersRef.current[id]) {
        const marker = markersRef.current[id];
        const prevPos = getMarkerLatLngLiteral(marker);
        const positionOnly =
          marker._driverStatus === status &&
          marker._iconUrl === iconUrl &&
          isSameMarkerPosition(prevPos, position);
        applyMarkerPosition(id, marker, position, { isStale, status });
        if (!positionOnly) {
          marker.setIcon({
            url: iconUrl,
            scaledSize: new window.google.maps.Size(24, 24),
            anchor: new window.google.maps.Point(12, 12),
          });
          marker.setTitle(markerTitle);
          marker._tooltipHtml = createStyledTooltip(driver, tooltipOpts);
          marker._driverStatus = status;
          marker._iconUrl = iconUrl;
        } else {
          recordMarkerPositionUpdate();
        }
        return marker;
      }

      recordMarkerCreate();
      const clustered = clusteringEnabledRef.current && clustererRef.current;
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
      marker._iconUrl = iconUrl;

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
      markerLastMotionAtRef.current[String(id)] = Date.now();
      if (clustered && clustererRef.current) {
        clustererRef.current.addMarker(marker);
      }
      return marker;
    }

    const AdvancedMarkerElement = window.google?.maps?.marker?.AdvancedMarkerElement;
    if (!AdvancedMarkerElement) return;

    if (markersRef.current[id]) {
      const marker = markersRef.current[id];
      const prevPos = getMarkerLatLngLiteral(marker);
      const positionOnly =
        marker._driverStatus === status &&
        marker._iconUrl === iconUrl &&
        isSameMarkerPosition(prevPos, position);
      applyMarkerPosition(id, marker, position, { isStale, status });
      if (!positionOnly) {
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
        marker._iconUrl = iconUrl;
      } else {
        recordMarkerPositionUpdate();
      }
      return marker;
    }

    recordMarkerCreate();
    const img = document.createElement('img');
    img.src = iconUrl;
    img.width = 24;
    img.height = 24;
    img.style.display = 'block';
    img.style.opacity = String(opacity);
    img.alt = markerTitle;
    img.title = markerTitle;
    img.draggable = false;

    const clustered = clusteringEnabledRef.current && clustererRef.current;
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
    marker._iconUrl = iconUrl;

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
    markerLastMotionAtRef.current[String(id)] = Date.now();

    if (clustered && clustererRef.current) {
      clustererRef.current.addMarker(marker);
    }

    return marker;
  }, [applyMarkerPosition]);

  // Supprimer un marqueur
  const removeMarker = useCallback((id) => {
    cancelMarkerMotion(id);
    const marker = markersRef.current[id];
    if (!marker) return;
    if (clusteringEnabledRef.current && clustererRef.current) {
      clustererRef.current.removeMarker(marker);
    }
    if (typeof marker._hoverCleanup === 'function') marker._hoverCleanup();
    if (GOOGLE_MAPS_USE_JS_STYLES) {
      marker.setMap(null);
    } else {
      marker.map = null;
    }
    delete markersRef.current[id];
  }, [cancelMarkerMotion]);

  // Fit bounds sur tous les marqueurs visibles
  const fitBoundsToMarkers = useCallback((maxZoom = 14, { structural = true } = {}) => {
    const map = mapRef.current;
    if (!map || !window.google) return;
    const entries = Object.values(markersRef.current);
    if (entries.length === 0) return;

    recordFitBoundsCall({ structural });

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

  const onMapLoad = useCallback((map) => {
    mapRef.current = map;
    setMapViewLocked(true);
    setMapReady(true);
    perfMark('gmaps_map_loaded');
    if (MAP_DEBUG) console.log('[DriverLiveMap] Google Map chargée');
    // Double rAF : attendre que le conteneur ait sa largeur finale (layout dashboard)
    // avant de forcer un resize Google Maps, sinon la carte reste « à moitié ».
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (mapRef.current !== map || !window.google?.maps?.event) return;
        const center = typeof map.getCenter === 'function' ? map.getCenter() : null;
        window.google.maps.event.trigger(map, 'resize');
        if (center) map.setCenter(center);
      });
    });
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map || !window.google) return undefined;

    if (clusteringEnabled && !clustererRef.current) {
      const clusterer = createDriverMarkerClusterer(map);
      if (clusterer) {
        clustererRef.current = clusterer;
        Object.values(markersRef.current).forEach((marker) => {
          if (GOOGLE_MAPS_USE_JS_STYLES) {
            marker.setMap(null);
          } else {
            marker.map = null;
          }
          clusterer.addMarker(marker);
        });
      }
    } else if (!clusteringEnabled && clustererRef.current) {
      const clusterer = clustererRef.current;
      Object.values(markersRef.current).forEach((marker) => {
        clusterer.removeMarker(marker);
        if (GOOGLE_MAPS_USE_JS_STYLES) {
          marker.setMap(map);
        } else {
          marker.map = map;
        }
      });
      clusterer.setMap(null);
      clustererRef.current = null;
    }

    return undefined;
  }, [mapReady, clusteringEnabled]);

  const structuralSetKey = useMemo(
    () => buildDriverStructuralSetKey(drivers, searchQuery),
    [drivers, searchQuery]
  );

  const syncMarkersFromCache = useCallback(() => {
    const map = mapRef.current;
    if (!mapReady || !map) return;

    const raw = queryClient.getQueryData(lirieKeys.companyDrivers());
    let list = projectDriversForMap(Array.isArray(raw) ? raw : []);
    const q = searchQuery.trim().toLowerCase();
    if (q) {
      list = list.filter((d) => {
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
      });
    }

    let staleMarkersCount = 0;
    const newLocatedIds = new Set();

    list.forEach((d) => {
      const resolved = resolveDriverCoords(d, companyCoords);
      if (!resolved) return;

      const { coords, isFallback } = resolved;
      const projection = resolveDriverMapProjection(d, { isFallback });
      const isConstrained = !isFallback && isDriverConstrained(d);
      const visualStatus = projection.visualStatus;
      const constraintReason = isConstrained ? getDriverConstraintReason(d) : null;
      const isLocated = !isFallback && !isNonLiveGpsPosition(d, { isFallback });
      if (isLocated) newLocatedIds.add(d.id);

      const agedDriver = applyLocalLocationFreshness(d);
      const localStatus = resolveLocalLocationFreshnessStatus(
        agedDriver.recorded_at ?? agedDriver.timestamp ?? null
      );
      const positionSource = String(d.position_source || '').toLowerCase();
      const staleByLocal =
        localStatus === 'stale' ||
        localStatus === 'offline' ||
        localStatus === 'offline_unknown';
      const staleByProjection =
        projection.visualTreatment === 'gps_stale'
        || projection.visualTreatment === 'gps_stale_constrained'
        || projection.visualTreatment === 'gps_offline'
        || positionSource === 'db_fallback'
        || String(d.location_status || '').toLowerCase() === 'last_known';
      const isStaleMarker = isFallback || staleByLocal || staleByProjection;
      if (isStaleMarker) staleMarkersCount += 1;

      const gpsLabel = getDriverFreshnessLabel(agedDriver);
      const isApproximate = isApproximateGpsAccuracy(d.accuracy);
      const tooltipOpts = isFallback
        ? {
            status: 'offline',
            isStale: true,
            noGps: true,
            businessStatus: projection.businessStatus,
            gpsLabel: 'Sans GPS — fallback entreprise',
          }
        : {
            lastSeenSeconds: d.last_seen_seconds,
            isStale: isStaleMarker,
            clientShort: d.client_short,
            currentBookingId: d.current_booking_id,
            status: visualStatus,
            businessStatus: projection.businessStatus,
            isConstrained,
            constraintReason,
            gpsLabel,
            isApproximate,
            serviceWindowStatus: d.service_window_status,
          };
      // accuracy > 50 m → rendu approximatif uniquement ; coords jamais corrigées.
      upsertMarker(d.id, coords, visualStatus, isStaleMarker, d, tooltipOpts);
    });

    const newLocatedSize = newLocatedIds.size;
    if (newLocatedSize !== locatedIdsRef.current.size) {
      locatedIdsRef.current = newLocatedIds;
      setLocatedCount(newLocatedSize);
    } else {
      locatedIdsRef.current = newLocatedIds;
    }

    const driverIds = new Set(list.map((d) => d.id));
    Object.keys(markersRef.current).forEach((driverId) => {
      if (!driverIds.has(Number(driverId))) {
        removeMarker(driverId);
      }
    });

    const markerCount = Object.keys(markersRef.current).length;
    setMapMarkerCount((prev) => (prev === markerCount ? prev : markerCount));
    if (markerCount > 0 && !firstMarkersMarkedRef.current) {
      firstMarkersMarkedRef.current = true;
      perfMark('gmaps_first_markers');
      captureOverlayStats();
      const mapInstance = mapRef.current;
      if (
        mapInstance &&
        !idleMarkerCaptureScheduledRef.current &&
        window.google?.maps?.event
      ) {
        idleMarkerCaptureScheduledRef.current = true;
        window.google.maps.event.addListenerOnce(mapInstance, 'idle', () => {
          window.setTimeout(() => {
            const count = Object.keys(markersRef.current).length;
            recordMapsOverlayStats({
              markerCount: count,
              markerCountAtIdleMs: count,
              activeMarkerCount: count,
              overlayCount: 0,
              infoWindowCount: infoWindowRef.current ? 1 : 0,
              clusterCount: clustererRef.current?.markers?.length ?? null,
            });
          }, 2000);
        });
      }
    }

    const now = Date.now();
    if (staleMarkersCount > 0 && now - lastStaleMetricAtRef.current >= 60000) {
      lastStaleMetricAtRef.current = now;
      trackStaleMarkers(company?.id, staleMarkersCount);
    }
  }, [
    mapReady,
    searchQuery,
    companyCoords,
    upsertMarker,
    removeMarker,
    company?.id,
    captureOverlayStats,
    queryClient,
  ]);

  // Sync marqueurs via cache TanStack (tick GPS) sans re-render React du composant carte.
  useEffect(() => {
    syncMarkersFromCache();
    const companyDriversKey = lirieKeys.companyDrivers();
    const unsub = queryClient.getQueryCache().subscribe((event) => {
      if (event?.type !== 'updated') return;
      const key = event?.query?.queryKey;
      if (!key || key[0] !== companyDriversKey[0] || key[1] !== companyDriversKey[1]) return;
      syncMarkersFromCache();
    });
    return unsub;
  }, [syncMarkersFromCache, queryClient]);

  // fitBounds uniquement si le set structurel visible change (pas sur tick GPS)
  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map) return;
    if (lastStructuralSetKeyRef.current === structuralSetKey) return;
    lastStructuralSetKeyRef.current = structuralSetKey;

    const visibleMarkers = Object.values(markersRef.current);
    if (visibleMarkers.length === 1) {
      const pos = getMarkerLatLngLiteral(visibleMarkers[0]);
      if (pos) map.setCenter(pos);
      map.setZoom(15);
    } else if (visibleMarkers.length > 1) {
      fitBoundsToMarkers(14, { structural: true });
    } else if (visibleMarkers.length === 0) {
      map.setCenter(companyCoords || SWITZERLAND_CENTER);
      map.setZoom(companyCoords ? 13 : 9);
    }
  }, [mapReady, structuralSetKey, companyCoords, fitBoundsToMarkers]);

  useEffect(() => {
    setShowNoGpsBanner(allDrivers.length > 0 && locatedCount === 0);
  }, [locatedCount, allDrivers.length]);

  useEffect(() => {
    if (!mapReady || !window.google?.maps?.event || !mapShellRef.current) return undefined;
    const map = mapRef.current;
    if (!map || typeof ResizeObserver === 'undefined') return undefined;

    let rafId = null;
    const observer = new ResizeObserver((entries) => {
      const entry = entries?.[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (width <= 0 || height <= 0) return;
      if (rafId != null) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        const center = typeof map.getCenter === 'function' ? map.getCenter() : null;
        window.google.maps.event.trigger(map, 'resize');
        // Sans recentrage, les tuiles restent décalées après un changement de largeur.
        if (center) map.setCenter(center);
      });
    });

    observer.observe(mapShellRef.current);
    return () => {
      observer.disconnect();
      if (rafId != null) cancelAnimationFrame(rafId);
    };
  }, [mapReady]);

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
      ref={mapShellRef}
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
            center={mapViewLocked ? undefined : defaultMapCenter}
            zoom={mapViewLocked ? undefined : defaultZoom}
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
                  {locatedCount}/{totalCount} en direct
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
              <div
                style={{ display: 'flex', alignItems: 'center', gap: 4 }}
                title="Position figée — l'app du chauffeur ne rafraîchit plus sa position (tracking arrêté, optimisation batterie, permissions ou GPS désactivé). Voir le détail au survol du chauffeur."
              >
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: CONSTRAINED_MARKER_COLOR, flexShrink: 0 }} />
                <span>Figé</span>
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
            fitBoundsToMarkers(14, { structural: true });
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

function areDriverLiveMapPropsEqual(prev, next) {
  if (prev === next) return true;
  return buildDriverStructuralSetKey(prev.drivers, '') === buildDriverStructuralSetKey(next.drivers, '');
}

export default memo(DriverLiveMap, areDriverLiveMapPropsEqual);
