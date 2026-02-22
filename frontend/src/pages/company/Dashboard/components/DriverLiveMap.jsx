// src/pages/company/Dashboard/components/DriverLiveMap.jsx
import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { GoogleMap } from '@react-google-maps/api';
import { MarkerClusterer } from '@googlemaps/markerclusterer';
import { getCompanySocket, joinCompanyRoom } from '../../../../services/companySocket';
import { fetchCompanyDriverLocations } from '../../../../services/companyService';
import useCompanyData from '../../../../hooks/useCompanyData';
import { useGoogleMapsLoaded } from '../../../../components/common/GoogleMapsProvider';
import MapPlaceholder from '../../../../components/common/MapPlaceholder';
import {
  SWITZERLAND_CENTER,
  STATUS_COLORS,
  DEFAULT_MAP_OPTIONS,
  normaliseCoords,
  resolveDriverCoords,
  getDriverStatus,
  formatLastSeen,
  makeCircleMarkerIcon,
  makeClusterIcon,
} from '../../../../utils/mapUtils';

const ENABLE_CLUSTERING = process.env.REACT_APP_ENABLE_DRIVER_CLUSTERING === 'true';
const MAP_DEBUG =
  typeof window !== 'undefined' &&
  (window.__MAP_DEBUG === true || sessionStorage.getItem('MAP_DEBUG') === '1');

const CONTAINER_STYLE = { width: '100%', height: '100%', minHeight: '280px' };

// Seuil de fraîcheur GPS (5 minutes) pour considérer un chauffeur comme "localisé"
const LOCATED_THRESHOLD_SEC = 300;

// Popup chauffeur — mini card structurée, classes CSS globales .lirie-popup-*
const createStyledTooltip = (driver, opts = {}) => {
  const { lastSeenSeconds, isStale, status: statusOverride, clientShort, noGps } = opts;
  const status = statusOverride ?? getDriverStatus(driver);

  const statusConf = {
    available: { label: 'Disponible', dot: '#22c55e', bg: '#dcfce7', color: '#15803d' },
    busy:      { label: 'En course',  dot: '#00796B', bg: '#e0f2f1', color: '#00695C' },
    offline:   { label: 'Hors-ligne', dot: '#91A3A0', bg: '#f1f5f9', color: '#64748b' },
    emergency: { label: 'Urgence',    dot: '#ef4444', bg: '#fee2e2', color: '#dc2626' },
  };
  const conf = statusConf[status] || statusConf.offline;
  if (noGps) conf.label = 'Sans GPS';

  const displayName = driver.full_name ||
    (driver.first_name || driver.last_name
      ? `${driver.first_name || ''} ${driver.last_name || ''}`.trim()
      : driver.username || `#${driver.id}`);

  // Ligne meta
  let metaLine = '';
  if (status === 'offline' && (lastSeenSeconds != null || isStale)) {
    metaLine = formatLastSeen(lastSeenSeconds);
  } else if (status === 'busy' && clientShort) {
    metaLine = clientShort;
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
  const hasAutoFittedRef = useRef(false);
  const lastLocationsRef = useRef({});
  const clustererRef = useRef(null);
  const infoWindowRef = useRef(null);
  const locatedIdsRef = useRef(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [mapReady, setMapReady] = useState(false);
  const [mapDebugInfo, setMapDebugInfo] = useState(null);
  const [showNoGpsBanner, setShowNoGpsBanner] = useState(false);
  const [locatedCount, setLocatedCount] = useState(0);
  const { driver: staticDrivers, company } = useCompanyData();

  const allDrivers = propDrivers || staticDrivers;
  const drivers = searchQuery
    ? allDrivers.filter((d) => d.username?.toLowerCase().includes(searchQuery.toLowerCase()))
    : allDrivers;

  // Met à jour le set des chauffeurs réellement localisés (GPS frais < 5 min)
  const updateLocatedSet = useCallback((driverId, isLocated) => {
    const set = locatedIdsRef.current;
    const before = set.size;
    if (isLocated) {
      set.add(driverId);
    } else {
      set.delete(driverId);
    }
    if (set.size !== before) setLocatedCount(set.size);
  }, []);

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

  // Créer ou mettre à jour un marqueur Google Maps
  const upsertMarker = useCallback((id, position, status, isStale, driver, tooltipOpts) => {
    const map = mapRef.current;
    if (!map || !window.google) return;

    const color = isStale ? '#9e9e9e' : (STATUS_COLORS[status] ?? STATUS_COLORS.available);
    const opacity = isStale ? 0.7 : 1;

    if (markersRef.current[id]) {
      const marker = markersRef.current[id];
      marker.setPosition(position);
      marker.setIcon({
        url: makeCircleMarkerIcon(color, opacity),
        scaledSize: new window.google.maps.Size(24, 24),
        anchor: new window.google.maps.Point(12, 12),
      });
      marker._tooltipHtml = createStyledTooltip(driver, tooltipOpts);
      marker._driverStatus = status;
      return marker;
    }

    const marker = new window.google.maps.Marker({
      position,
      map: ENABLE_CLUSTERING ? null : map,
      icon: {
        url: makeCircleMarkerIcon(color, opacity),
        scaledSize: new window.google.maps.Size(24, 24),
        anchor: new window.google.maps.Point(12, 12),
      },
      optimized: true,
    });

    marker._tooltipHtml = createStyledTooltip(driver, tooltipOpts);
    marker._driverStatus = status;

    // InfoWindow au survol — positionnement natif Google Maps
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
    marker.addListener('mouseout', () => {
      if (infoWindowRef.current) infoWindowRef.current.close();
    });

    markersRef.current[id] = marker;

    // Ajouter au clusterer si activé
    if (ENABLE_CLUSTERING && clustererRef.current) {
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
    marker.setMap(null);
    delete markersRef.current[id];
  }, []);

  // Fit bounds sur tous les marqueurs visibles
  const fitBoundsToMarkers = useCallback((maxZoom = 14) => {
    const map = mapRef.current;
    if (!map || !window.google) return;
    const entries = Object.values(markersRef.current);
    if (entries.length === 0) return;

    const bounds = new window.google.maps.LatLngBounds();
    entries.forEach((m) => bounds.extend(m.getPosition()));
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
    }

    setMapReady(true);
    if (MAP_DEBUG) console.log('[DriverLiveMap] Google Map chargée');
  }, []);

  // Placer les positions statiques au chargement
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !Array.isArray(drivers)) return;

    let placed = 0;
    drivers.forEach((d) => {
      if (markersRef.current[d.id]) return;
      const resolved = resolveDriverCoords(d, companyCoords);
      if (!resolved) return;

      const { coords, isFallback } = resolved;
      const status = isFallback ? 'offline' : getDriverStatus(d);
      const tooltipOpts = isFallback ? { status: 'offline', isStale: true, noGps: true } : {};
      upsertMarker(d.id, coords, status, isFallback, d, tooltipOpts);
      placed++;
    });

    // Supprimer les marqueurs des chauffeurs qui ne sont plus dans la liste
    const driverIds = new Set(drivers.map((d) => d.id));
    Object.keys(markersRef.current).forEach((driverId) => {
      if (!driverIds.has(Number(driverId))) {
        removeMarker(driverId);
        updateLocatedSet(Number(driverId), false);
      }
    });

    const visibleMarkers = Object.values(markersRef.current);
    if (visibleMarkers.length === 1) {
      const pos = visibleMarkers[0].getPosition();
      map.setCenter(pos);
      map.setZoom(15);
    } else if (visibleMarkers.length > 1) {
      fitBoundsToMarkers(14);
    } else if (placed === 0 && visibleMarkers.length === 0) {
      map.setCenter(companyCoords || SWITZERLAND_CENTER);
      map.setZoom(companyCoords ? 13 : 9);
    }
  }, [drivers, companyCoords, upsertMarker, removeMarker, fitBoundsToMarkers, updateLocatedSet]);

  // Rejoindre la room entreprise
  useEffect(() => {
    if (company?.id) {
      joinCompanyRoom(company.id).catch(() => {});
    }
  }, [company?.id]);

  // REST: polling des positions
  const POLL_INTERVAL_MS = 5000;
  useEffect(() => {
    if (!company?.id) return;
    let cancelled = false;
    let pollCount = 0;
    let intervalId = null;

    const poll = async () => {
      if (cancelled) return;
      if (typeof document !== 'undefined' && document.visibilityState !== 'visible') return;
      try {
        const locations = await fetchCompanyDriverLocations();
        if (cancelled) return;
        const map = mapRef.current;
        if (!map || !Array.isArray(locations) || locations.length === 0) return;

        pollCount += 1;
        const locMap = {};
        locations.forEach((loc) => {
          const id = loc.driver_id ?? loc.id;
          if (id) locMap[id] = loc;
        });
        lastLocationsRef.current = locMap;

        locations.forEach((loc) => {
          const id = loc.driver_id ?? loc.id;
          const lat = loc.lat ?? loc.latitude;
          const lon = loc.lon ?? loc.longitude;
          const { center: ll } = normaliseCoords(lat, lon);
          if (!id || !ll) return;

          const fullDriver = allDrivers.find((d) => d.id === id) || {
            id,
            first_name: loc.first_name,
            is_active: true,
          };
          const status = loc.status ?? getDriverStatus(fullDriver);
          const isStale = loc.is_stale === true;
          const tooltipOpts = {
            lastSeenSeconds: loc.last_seen_seconds,
            isStale,
            status,
            clientShort: loc.client_short,
          };

          upsertMarker(id, ll, status, isStale, fullDriver, tooltipOpts);

          // Localisé = GPS frais (timestamp disponible et < 5 min)
          const isFreshGps = loc.last_seen_seconds != null && loc.last_seen_seconds < LOCATED_THRESHOLD_SEC;
          updateLocatedSet(id, isFreshGps);

          if (!hasAutoFittedRef.current && Object.keys(markersRef.current).length > 0) {
            fitBoundsToMarkers(14);
            hasAutoFittedRef.current = true;
          }
        });

        if (MAP_DEBUG && pollCount === 1) {
          console.log('[DriverLiveMap] REST: count=', locations.length);
        }
      } catch (e) {
        if (!cancelled && MAP_DEBUG) console.warn('[DriverLiveMap] REST poll error:', e);
      }
    };

    const startPolling = () => {
      if (intervalId) return;
      poll();
      intervalId = setInterval(poll, POLL_INTERVAL_MS);
    };
    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    };

    if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
      startPolling();
    }
    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') startPolling();
      else stopPolling();
    };
    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      cancelled = true;
      stopPolling();
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [company?.id]);

  // Socket: écouter les mises à jour live
  useEffect(() => {
    const socket = getCompanySocket();
    if (!socket) return;

    hasAutoFittedRef.current = false;

    const requestLocations = () => {
      try {
        socket.emit('get_driver_locations');
      } catch (e) {
        console.error('Failed to request driver locations:', e);
      }
    };

    const debug = {
      received: 0,
      valid: 0,
      lastUpdate: null,
      sample: null,
      exclusionReasons: [],
      joinedReceived: false,
      fallbackUsed: false,
    };
    if (MAP_DEBUG) setMapDebugInfo({ ...debug });

    const onLoc = (data) => {
      const map = mapRef.current;
      debug.received += 1;

      if (!map) {
        if (MAP_DEBUG) {
          debug.exclusionReasons = (debug.exclusionReasons.slice(-4).concat('map_not_ready')).slice(-5);
          setMapDebugInfo({ ...debug });
        }
        return;
      }

      const id = data.driver_id ?? data.id;
      const lat = data.lat ?? data.latitude ?? data.current_lat;
      const lon = data.lon ?? data.lng ?? data.longitude ?? data.current_lon;
      const { center: ll, reason } = normaliseCoords(lat, lon);

      if (!id) {
        if (MAP_DEBUG) {
          debug.exclusionReasons = (debug.exclusionReasons.slice(-4).concat('missing_driver_id')).slice(-5);
          setMapDebugInfo({ ...debug });
        }
        return;
      }
      if (!ll) {
        if (MAP_DEBUG) {
          debug.exclusionReasons = (debug.exclusionReasons.slice(-4).concat(reason || 'invalid_coords')).slice(-5);
          setMapDebugInfo({ ...debug });
        }
        return;
      }

      debug.valid += 1;
      if (debug.valid === 1) debug.sample = { driver_id: id, lat: ll.lat, lon: ll.lng };
      debug.lastUpdate = new Date().toISOString();
      if (MAP_DEBUG) setMapDebugInfo({ ...debug });

      const firstName = data.first_name || data.name || `Driver ${id}`;
      const wasEmpty = Object.keys(markersRef.current).length === 0;

      const fullDriver = drivers.find((d) => d.id === id) || {
        id,
        first_name: firstName,
        is_active: true,
      };
      const lastLoc = lastLocationsRef.current[id];
      const status = data.status ?? lastLoc?.status ?? getDriverStatus(fullDriver);
      const isStale = (data.is_stale ?? lastLoc?.is_stale) === true;
      const tooltipOpts = {
        status,
        clientShort: data.client_short ?? lastLoc?.client_short,
        lastSeenSeconds: data.last_seen_seconds ?? lastLoc?.last_seen_seconds,
        isStale,
      };

      upsertMarker(id, ll, status, isStale, fullDriver, tooltipOpts);
      // Donnée socket = GPS temps réel → toujours localisé
      updateLocatedSet(id, true);
      setShowNoGpsBanner(false);

      if (wasEmpty && !hasAutoFittedRef.current) {
        fitBoundsToMarkers(14);
        hasAutoFittedRef.current = true;
      }
    };

    socket.on('driver_location_update', onLoc);

    let fallbackId = null;
    let retryId = null;
    let onJoined = null;

    if (company?.id) {
      try {
        socket.emit('join_company', { company_id: company.id });
      } catch {}
      onJoined = () => {
        if (MAP_DEBUG) setMapDebugInfo((p) => ({ ...(p || {}), joinedReceived: true }));
        requestLocations();
        socket.off('joined_company', onJoined);
      };
      socket.once('joined_company', onJoined);
      fallbackId = setTimeout(() => {
        socket.off('joined_company', onJoined);
        if (MAP_DEBUG) setMapDebugInfo((p) => ({ ...(p || {}), fallbackUsed: true }));
        if (Object.keys(markersRef.current).length === 0) requestLocations();
      }, 1500);
      retryId = setTimeout(() => {
        if (Object.keys(markersRef.current).length === 0) requestLocations();
      }, 3000);
    }

    return () => {
      if (fallbackId != null) clearTimeout(fallbackId);
      if (retryId != null) clearTimeout(retryId);
      if (onJoined) socket.off('joined_company', onJoined);
      socket.off('driver_location_update', onLoc);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [company?.id]);

  // Compteur chauffeurs total
  const totalCount = allDrivers.length;

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
        <GoogleMap
          mapContainerStyle={CONTAINER_STYLE}
          center={defaultMapCenter}
          zoom={defaultZoom}
          options={DEFAULT_MAP_OPTIONS}
          onLoad={onMapLoad}
        />
      ) : (
        <MapPlaceholder style={{ minHeight: 280 }} />
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
            placeholder="Rechercher..."
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
              <div style={{ lineHeight: 1.3, minWidth: 0 }}>
                <div style={{ fontWeight: 600, color: '#1E293B', fontSize: 11 }}>
                  {locatedCount}/{totalCount} localisés
                </div>
                {locatedCount === 0 && totalCount > 0 && (
                  <div style={{ fontSize: 9, color: '#94A3B8', marginTop: 1 }}>Aucun GPS récent</div>
                )}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, fontSize: 9, color: '#64748B', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', flexShrink: 0 }} />
                <span>Dispo</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#00796B', flexShrink: 0 }} />
                <span>Course</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#91A3A0', flexShrink: 0 }} />
                <span>Off</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Bouton Recentrer — responsive */}
      {mapReady && (
        <button
          type="button"
          onClick={() => fitBoundsToMarkers(14)}
          style={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            zIndex: 100,
            padding: '5px 12px',
            fontSize: 11,
            fontWeight: 600,
            color: '#fff',
            background: 'linear-gradient(135deg, #00796B 0%, #00695C 100%)',
            border: 'none',
            borderRadius: 8,
            boxShadow: '0 1px 4px rgba(0,0,0,0.12)',
            cursor: 'pointer',
            fontFamily: "Inter, -apple-system, 'Segoe UI', sans-serif",
            transition: 'opacity 0.2s',
          }}
          title="Recadrer sur les chauffeurs"
          onMouseEnter={(e) => { e.currentTarget.style.opacity = '0.9'; }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = '1'; }}
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
          Aucune position reçue. Vérifiez que les chauffeurs ont activé le GPS.
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

      {/* Pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}
