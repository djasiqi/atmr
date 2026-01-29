// src/pages/company/Dashboard/components/DriverLiveMap.jsx
import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.markercluster';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';
import { getCompanySocket, joinCompanyRoom } from '../../../../services/companySocket';
import { fetchCompanyDriverLocations } from '../../../../services/companyService';
import useCompanyData from '../../../../hooks/useCompanyData';

// Icône Leaflet par défaut (corrige le bug d'icône manquante)
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const defaultCenter = [46.8182, 8.2275]; // CH
const ENABLE_CLUSTERING = process.env.REACT_APP_ENABLE_DRIVER_CLUSTERING === 'true';

// ---- Couleurs par statut (pour CircleMarker, pas d’icône = pas de bug Leaflet default icon) ----
const STATUS_COLORS = {
  available: '#22c55e',   // vert
  busy: '#1976d2',       // bleu
  offline: '#9e9e9e',    // gris
  emergency: '#f44336',
};

/** Crée un CircleMarker pour un chauffeur (évite le bug d’icône Leaflet manquante). */
const createDriverCircleMarker = (latlng, status = 'available', isStale = false) => {
  const color = isStale ? '#9e9e9e' : (STATUS_COLORS[status] ?? STATUS_COLORS.available);
  return L.circleMarker(latlng, {
    radius: 8,
    fillColor: color,
    color: '#fff',
    weight: 2,
    fillOpacity: isStale ? 0.7 : 1,
  });
};

// ---- helpers coords (normalisation + validation) --------------------
const toNumOrNull = (v) => {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

/** Normalise lat/lon (parseFloat, alias lat|latitude, lon|lng|longitude) et rejette invalides. */
const normaliseCoords = (lat, lon) => {
  const la = toNumOrNull(lat);
  const lo = toNumOrNull(lon);
  if (la === null || lo === null) return { center: null, reason: 'missing_or_nan' };
  if (Math.abs(la) > 90) return { center: null, reason: 'lat_out_of_range' };
  if (Math.abs(lo) > 180) return { center: null, reason: 'lon_out_of_range' };
  if (la === 0 && lo === 0) return { center: null, reason: 'zero_coords' };
  return { center: [la, lo], reason: null };
};

const resolveDriverCoords = (d) => {
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
    if (r.center) return r.center;
  }
  return null;
};

// Déterminer le statut du chauffeur
const getDriverStatus = (driver) => {
  if (!driver.is_active) return 'offline';
  if (driver.current_booking_id || driver.status === 'busy') return 'busy';
  if (driver.emergency_mode) return 'emergency';
  return 'available';
};

/** Texte "il y a Xs" ou "Signal inconnu" pour tooltip. Frontend lit is_stale du backend (pas de seuil local). */
const formatLastSeen = (lastSeenSeconds) => {
  if (lastSeenSeconds == null || lastSeenSeconds < 0) return 'Dernier signal inconnu';
  if (lastSeenSeconds < 60) return `il y a ${lastSeenSeconds} s`;
  if (lastSeenSeconds < 3600) {
    const m = Math.floor(lastSeenSeconds / 60);
    return m === 1 ? 'il y a 1 min' : `il y a ${m} min`;
  }
  return '> 1 h';
};

// Créer un tooltip stylé (spec: "Prénom Nom — Disponible" / "En course (client X)" / "Hors-ligne (dernier signal: …)")
const createStyledTooltip = (driver, opts = {}) => {
  const { lastSeenSeconds, isStale, status: statusOverride, clientShort } = opts;
  const status = statusOverride ?? getDriverStatus(driver);
  const statusText = {
    available: 'Disponible',
    busy: clientShort ? `En course (${clientShort})` : 'En course',
    offline: lastSeenSeconds != null || isStale
      ? `Hors-ligne (dernier signal: ${formatLastSeen(lastSeenSeconds)})`
      : 'Hors-ligne',
    emergency: 'Urgence',
  };

  const statusColors = {
    available: '#22c55e',
    busy: '#1976d2',
    offline: '#9e9e9e',
    emergency: '#f44336',
  };

  const lastSeenLine = (lastSeenSeconds != null || isStale) && status !== 'offline'
    ? `<div style="font-size: 10px; color: ${isStale ? '#9e9e9e' : statusColors[status]}; margin-top: 2px;">${formatLastSeen(lastSeenSeconds)}</div>`
    : '';

  const displayName = driver.full_name ||
    (driver.first_name || driver.last_name
      ? `${driver.first_name || ''} ${driver.last_name || ''}`.trim()
      : driver.username || `Chauffeur ${driver.id}`);

  return `
    <div style="
      background: white;
      border: 2px solid ${statusColors[status]};
      border-radius: 6px;
      padding: 4px 8px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.15);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      min-width: 100px;
      text-align: center;
    ">
      <div style="
        font-weight: 600;
        color: #334155;
        margin-bottom: 2px;
        font-size: 12px;
        line-height: 1.2;
      ">
        ${displayName} — ${statusText[status]}
      </div>
      ${lastSeenLine}
    </div>
  `;
};

const MAP_DEBUG =
  typeof window !== 'undefined' &&
  (window.__MAP_DEBUG === true || sessionStorage.getItem('MAP_DEBUG') === '1');

export default function DriverLiveMap({ drivers: propDrivers }) {
  const mapRef = useRef(null);
  const mapElRef = useRef(null);
  const markersRef = useRef({}); // { [driverId]: L.Layer (CircleMarker) }
  const hasAutoFittedRef = useRef(false); // fitBounds une seule fois au premier marker
  const lastLocationsRef = useRef({}); // { [driverId]: loc } — dernière réponse REST pour status/client_short
  const clusterGroupRef = useRef(null); // L.markerClusterGroup si ENABLE_CLUSTERING
  const [searchQuery, setSearchQuery] = useState('');
  const [mapReady, setMapReady] = useState(false);
  const [mapDebugInfo, setMapDebugInfo] = useState(null); // dev only
  const [showNoGpsBanner, setShowNoGpsBanner] = useState(false);
  const { driver: staticDrivers, company } = useCompanyData();

  // Utiliser les drivers passés en props si disponibles, sinon ceux de useCompanyData
  const allDrivers = propDrivers || staticDrivers;

  // Filtrer les drivers selon la recherche
  const drivers = searchQuery
    ? allDrivers.filter((d) => d.username?.toLowerCase().includes(searchQuery.toLowerCase()))
    : allDrivers;

  // petits helpers pour éviter d'appeler Leaflet sur une map détruite
  const addMarkerToLayer = (m) => {
    const cg = clusterGroupRef.current;
    if (cg) cg.addLayer(m);
    else getMap()?.addLayer(m);
  };
  const removeMarkerFromLayer = (m) => {
    const cg = clusterGroupRef.current;
    const map = getMap();
    if (cg) cg.removeLayer(m);
    else if (map) map.removeLayer(m);
  };
  const getMap = () => {
    const m = mapRef.current;
    // ✅ CORRECTION: Vérifier que la carte est complètement initialisée
    // _mapPane est défini une fois la map initialisée et doit avoir un _leaflet_id
    if (!m || !m._mapPane || m._mapPane._leaflet_id === undefined) return null;
    return m;
  };
  const safeSetView = (center, zoom, animate = true) => {
    const m = getMap();
    if (!m) return;
    try {
      m.setView(center, zoom, {
        animate: animate,
        duration: 0.8, // durée de l'animation en secondes
        easeLinearity: 0.25, // rend l'animation plus smooth
      });
    } catch {}
  };
  const fitBoundsToMarkers = (maxZoom = 14, animate = true) => {
    const m = getMap();
    if (!m) return;
    const entries = Object.values(markersRef.current);
    if (entries.length === 0) return;
    try {
      const group = L.featureGroup(entries);
      m.fitBounds(group.getBounds().pad(0.2), {
        animate: animate,
        duration: 0.8, // durée de l'animation en secondes
      });
      if (m.getZoom() > maxZoom) {
        setTimeout(() => {
          m.setZoom(maxZoom, { animate: animate, duration: 0.5 });
        }, 100);
      }
    } catch {}
  };

  // Init carte Leaflet
  useEffect(() => {
    if (mapRef.current) {
      if (MAP_DEBUG) console.log('[DriverLiveMap] ⚠️ Carte déjà initialisée, skip');
      return; // évite double init hors StrictMode
    }
    if (!mapElRef.current) {
      if (MAP_DEBUG) console.log('[DriverLiveMap] ❌ mapElRef.current is null');
      return;
    }

    if (MAP_DEBUG) console.log('[DriverLiveMap] 🗺️ Initialisation de la carte Leaflet...', {
      element: mapElRef.current,
      dimensions: {
        width: mapElRef.current.offsetWidth,
        height: mapElRef.current.offsetHeight,
      },
    });

    try {
      const map = L.map(mapElRef.current, {
        center: defaultCenter,
        zoom: 9,
        zoomControl: true,
        scrollWheelZoom: true,
      });

      if (MAP_DEBUG) console.log('[DriverLiveMap] ✅ Carte Leaflet créée');

      const tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors',
      });

      tileLayer.on('loading', () => {
        console.log('[DriverLiveMap] 📥 Chargement des tiles...');
      });

      tileLayer.on('load', () => {
        console.log('[DriverLiveMap] ✅ Tiles chargées');
      });

      tileLayer.on('tileerror', (error) => {
        console.error('[DriverLiveMap] ❌ Erreur chargement tile:', error);
      });

      tileLayer.addTo(map);

      // Clustering chauffeurs (feature flag, fallback sans cluster si indispo)
      if (ENABLE_CLUSTERING && typeof L.markerClusterGroup === 'function') {
        try {
          const clusterGroup = L.markerClusterGroup({
            disableClusteringAtZoom: 16,
            maxClusterRadius: 40,
            showCoverageOnHover: false,
            spiderfyOnMaxZoom: true,
            iconCreateFunction: (cluster) => {
              const count = cluster.getChildCount();
              return L.divIcon({
                html: `<span style="
                  display: flex; align-items: center; justify-content: center;
                  width: 36px; height: 36px; border-radius: 50%;
                  background: #374151; color: #fff; font-weight: 600; font-size: 12px;
                  border: 2px solid #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                ">${count}</span>`,
                className: 'driver-cluster-icon',
                iconSize: [36, 36],
              });
            },
          });
          clusterGroup.addTo(map);
          clusterGroupRef.current = clusterGroup;
        } catch (e) {
          console.warn('[DriverLiveMap] Clustering indisponible, fallback sans cluster:', e);
          clusterGroupRef.current = null;
        }
      }

      // ✅ CORRECTION: S'assurer que la carte est complètement initialisée avant invalidateSize
      // Vérifier que _mapPane existe avant d'appeler invalidateSize
      const checkAndInvalidateSize = () => {
        // Vérifier que la carte existe et est complètement initialisée
        if (map && map._mapPane && map._mapPane._leaflet_id !== undefined) {
          try {
            map.invalidateSize();
            if (MAP_DEBUG) console.log('[DriverLiveMap] 🔄 Carte redimensionnée');
          } catch (error) {
            console.warn('[DriverLiveMap] ⚠️ Erreur lors du redimensionnement:', error);
          }
        } else {
          // Réessayer après un court délai si pas encore prêt
          setTimeout(checkAndInvalidateSize, 50);
        }
      };

      // Forcer un redimensionnement après un court délai
      setTimeout(checkAndInvalidateSize, 100);

      mapRef.current = map;
      setMapReady(true);
      if (MAP_DEBUG) console.log('[DriverLiveMap] ✅ Carte initialisée avec succès');
    } catch (error) {
      console.error('[DriverLiveMap] ❌ Erreur initialisation carte:', error);
    }

    return () => {
      setMapReady(false);
      clusterGroupRef.current = null;
      try {
        if (mapRef.current) {
          mapRef.current.remove();
        }
      } catch {}
      mapRef.current = null;
      markersRef.current = {};
    };
  }, []);

  // Placer les positions statiques au chargement
  useEffect(() => {
    const map = getMap();
    if (!map || !Array.isArray(drivers)) return;

    let placed = 0;
    drivers.forEach((d) => {
      if (markersRef.current[d.id]) return; // déjà placé (live)
      const ll = resolveDriverCoords(d);
      if (!ll) return; // ignore si pas de coords valides

      const status = getDriverStatus(d);
      const m = createDriverCircleMarker(ll, status);
      addMarkerToLayer(m);

      // Logique intelligente 4 directions : haut/bas ET gauche/droite
      const updateTooltipDirection = () => {
        const bounds = map.getBounds();
        const center = bounds.getCenter();
        const markerLat = ll[0];
        const markerLng = ll[1];

        // Calculer la distance au centre en vertical et horizontal
        const verticalDist = Math.abs(markerLat - center.lat);
        const horizontalDist = Math.abs(markerLng - center.lng);

        // Déterminer quelle direction est prioritaire
        let direction;
        let offset;

        if (verticalDist > horizontalDist) {
          // Position dominante = vertical (haut/bas)
          direction = markerLat > center.lat ? 'bottom' : 'top';
          offset = direction === 'bottom' ? [0, 20] : [0, -20];
        } else {
          // Position dominante = horizontal (gauche/droite)
          direction = markerLng > center.lng ? 'left' : 'right';
          offset = direction === 'left' ? [-10, 0] : [10, 0];
        }

        // Re-bind tooltip avec nouvelle direction
        m.unbindTooltip();
        m.bindTooltip(createStyledTooltip(d), {
          permanent: true,
          direction: direction,
          offset: offset,
          className: 'custom-driver-tooltip',
        }).openTooltip();
      };

      // Appliquer au chargement
      updateTooltipDirection();

      // Mettre à jour quand la carte bouge ou zoom
      map.on('moveend zoomend', updateTooltipDirection);

      markersRef.current[d.id] = m;
      placed++;
    });

    // Supprimer les markers des chauffeurs qui ne sont plus dans la liste filtrée
    const driverIds = new Set(drivers.map((d) => d.id));
    Object.keys(markersRef.current).forEach((driverId) => {
      if (!driverIds.has(Number(driverId))) {
        const marker = markersRef.current[driverId];
        if (marker) {
          try {
            removeMarkerFromLayer(marker);
          } catch {}
        }
        delete markersRef.current[driverId];
      }
    });

    // Zoom intelligent :
    // - Si 1 seul chauffeur : zoom proche sur lui (zoom 15)
    // - Si plusieurs chauffeurs : ajuster la vue pour tous les voir
    // - Si aucun : vue par défaut
    const visibleMarkers = Object.values(markersRef.current).filter((m) => m);
    if (visibleMarkers.length === 1) {
      // Un seul chauffeur : zoom proche
      const marker = visibleMarkers[0];
      const latlng = marker.getLatLng();
      safeSetView([latlng.lat, latlng.lng], 15);
    } else if (visibleMarkers.length > 1) {
      // Plusieurs chauffeurs : ajuster la vue
      fitBoundsToMarkers(14);
    } else if (placed === 0 && visibleMarkers.length === 0) {
      // Aucun marker : vue par défaut
      safeSetView(defaultCenter, 9);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drivers]);

  // Rejoindre la room entreprise au chargement (pour socket + currentCompanyId)
  useEffect(() => {
    if (company?.id) {
      joinCompanyRoom(company.id).catch(() => {});
    }
  }, [company?.id]);

  // REST: polling des positions (fallback si socket vide ou indisponible)
  // Pause si onglet invisible (visibility API)
  const POLL_INTERVAL_MS = 5000;
  useEffect(() => {
    if (!company?.id) return;
    let cancelled = false;
    let pollCount = 0;
    const devLogOnce = { logged: false };
    let intervalId = null;

    const poll = async () => {
      if (cancelled) return;
      if (typeof document !== 'undefined' && document.visibilityState !== 'visible') return;
      try {
        const locations = await fetchCompanyDriverLocations();
        if (cancelled) return;
        const map = getMap();
        if (!map || !Array.isArray(locations) || locations.length === 0) {
          if (MAP_DEBUG && locations.length === 0 && pollCount === 1 && !devLogOnce.logged) {
            devLogOnce.logged = true;
            console.log('[DriverLiveMap] REST: 0 locations reçues (poll #1)');
          }
          return;
        }
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
          // Backend = source de vérité pour status (available|busy|offline)
          const status = loc.status ?? getDriverStatus(fullDriver);
          // Frontend lit is_stale du backend (pas de seuil local, cohérence 90s côté API)
          const isStale = loc.is_stale === true;
          const lastSeenSeconds = loc.last_seen_seconds;
          const tooltipOpts = {
            lastSeenSeconds,
            isStale,
            status,
            clientShort: loc.client_short,
          };
          if (!markersRef.current[id]) {
            const m = createDriverCircleMarker(ll, status, isStale);
            addMarkerToLayer(m);
            m.bindTooltip(createStyledTooltip(fullDriver, tooltipOpts), {
              permanent: true,
              direction: 'top',
              offset: [0, -20],
              className: 'custom-driver-tooltip',
            }).openTooltip();
            markersRef.current[id] = m;
            if (!hasAutoFittedRef.current) {
              fitBoundsToMarkers(14);
              hasAutoFittedRef.current = true;
            }
          } else {
            const m = markersRef.current[id];
            m.setLatLng(ll);
            m.setStyle({
              fillColor: isStale ? '#9e9e9e' : (STATUS_COLORS[status] ?? STATUS_COLORS.available),
              fillOpacity: isStale ? 0.7 : 1,
            });
            m.unbindTooltip();
            m.bindTooltip(createStyledTooltip(fullDriver, tooltipOpts), {
              permanent: true,
              direction: 'top',
              offset: [0, -20],
              className: 'custom-driver-tooltip',
            }).openTooltip();
          }
        });
        if (MAP_DEBUG && pollCount === 1 && !devLogOnce.logged) {
          devLogOnce.logged = true;
          console.log('[DriverLiveMap] REST: count=', locations.length, 'sample=', locations[0] ? { id: locations[0].driver_id, lat: locations[0].latitude ?? locations[0].lat, lon: locations[0].longitude ?? locations[0].lon } : null);
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
      if (document.visibilityState === 'visible') {
        startPolling();
      } else {
        stopPolling();
      }
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
        if (MAP_DEBUG) console.log('[DriverLiveMap] get_driver_locations emitted');
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
      const map = getMap();
      debug.received += 1;
      if (MAP_DEBUG && debug.received <= 5) {
        try {
          console.log('[DriverLiveMap] driver_location_update payload', data);
        } catch (_) {}
      }

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
          debug.lastUpdate = new Date().toISOString();
          setMapDebugInfo({ ...debug });
        }
        return;
      }

      debug.valid += 1;
      if (debug.valid === 1) debug.sample = { driver_id: id, lat: ll[0], lon: ll[1] };
      debug.lastUpdate = new Date().toISOString();
      if (MAP_DEBUG) setMapDebugInfo({ ...debug });

      const firstName = data.first_name || data.name || `Driver ${id}`;
      // wasEmpty = aucun marker valide sur la carte (pas le nombre d’updates reçues)
      const wasEmpty = Object.keys(markersRef.current).length === 0;

      if (!markersRef.current[id]) {
        const fullDriver = drivers.find((d) => d.id === id) || {
          id,
          first_name: firstName,
          is_active: true,
        };
        const lastLoc = lastLocationsRef.current[id];
        const status = data.status ?? lastLoc?.status ?? getDriverStatus(fullDriver);
        const m = createDriverCircleMarker(ll, status);
        addMarkerToLayer(m);

        const updateTooltipDirection = () => {
          const bounds = map.getBounds();
          const center = bounds.getCenter();
          const markerLat = ll[0];
          const markerLng = ll[1];
          const verticalDist = Math.abs(markerLat - center.lat);
          const horizontalDist = Math.abs(markerLng - center.lng);
          let direction;
          let offset;
          if (verticalDist > horizontalDist) {
            direction = markerLat > center.lat ? 'bottom' : 'top';
            offset = direction === 'bottom' ? [0, 20] : [0, -20];
          } else {
            direction = markerLng > center.lng ? 'left' : 'right';
            offset = direction === 'left' ? [-10, 0] : [10, 0];
          }
          m.unbindTooltip();
          m.bindTooltip(createStyledTooltip(fullDriver, {
            status,
            clientShort: data.client_short ?? lastLoc?.client_short,
            lastSeenSeconds: data.last_seen_seconds ?? lastLoc?.last_seen_seconds,
            isStale: (data.is_stale ?? lastLoc?.is_stale) === true,
          }), {
            permanent: true,
            direction,
            offset,
            className: 'custom-driver-tooltip',
          }).openTooltip();
        };

        updateTooltipDirection();
        map.on('moveend zoomend', updateTooltipDirection);

        markersRef.current[id] = m;
        setShowNoGpsBanner(false);

        // fitBounds une seule fois au premier marker valide (évite zooms agressifs à chaque update GPS)
        if (wasEmpty && !hasAutoFittedRef.current) {
          fitBoundsToMarkers(14);
          hasAutoFittedRef.current = true;
        }
      } else {
        const marker = markersRef.current[id];
        marker.setLatLng(ll);

        const fullDriver = drivers.find((d) => d.id === id) || {
          id,
          first_name: firstName,
          is_active: true,
        };
        const lastLoc = lastLocationsRef.current[id];
        const status = data.status ?? lastLoc?.status ?? getDriverStatus(fullDriver);
        const isStale = (data.is_stale ?? lastLoc?.is_stale) === true;
        marker.setStyle({
          fillColor: isStale ? '#9e9e9e' : (STATUS_COLORS[status] ?? STATUS_COLORS.available),
          fillOpacity: isStale ? 0.7 : 1,
        });

        const bounds = map.getBounds();
        const center = bounds.getCenter();
        const markerLat = ll[0];
        const markerLng = ll[1];
        const verticalDist = Math.abs(markerLat - center.lat);
        const horizontalDist = Math.abs(markerLng - center.lng);
        const direction = verticalDist > horizontalDist
          ? (markerLat > center.lat ? 'bottom' : 'top')
          : (markerLng > center.lng ? 'left' : 'right');
        const offset = direction === 'bottom' ? [0, 20] : direction === 'top' ? [0, -20] : direction === 'left' ? [-10, 0] : [10, 0];

        marker.unbindTooltip();
        marker
          .bindTooltip(createStyledTooltip(fullDriver, {
            status,
            clientShort: data.client_short ?? lastLoc?.client_short,
            lastSeenSeconds: data.last_seen_seconds ?? lastLoc?.last_seen_seconds,
            isStale: data.is_stale ?? lastLoc?.is_stale,
          }), {
            permanent: true,
            direction,
            offset,
            className: 'custom-driver-tooltip',
          })
          .openTooltip();
        // pas de fitBounds sur les updates : l’utilisateur garde la main sur la vue
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
      // Cleanup: pas de fuite listeners ni timers (ordre: timers puis listeners)
      if (fallbackId != null) clearTimeout(fallbackId);
      if (retryId != null) clearTimeout(retryId);
      if (onJoined) socket.off('joined_company', onJoined);
      socket.off('driver_location_update', onLoc);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [company?.id]);

  // Ajouter le compteur comme contrôle Leaflet
  useEffect(() => {
    const map = getMap();
    if (!map) return;

    // Créer le contrôle personnalisé avec indicateur GPS temps réel
    const DriverCounterControl = L.Control.extend({
      onAdd: function (_map) {
        const container = L.DomUtil.create('div', 'driver-counter-control');
        container.style.cssText = `
          background: rgba(255,255,255,0.95);
          border: 1px solid #00796b;
          border-radius: 8px;
          padding: 8px 12px;
          font-size: 12px;
          pointer-events: none;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        `;

        const visibleCount = Object.keys(markersRef.current).length;
        const totalCount = allDrivers.length;
        const isActive = visibleCount > 0;

        container.innerHTML = `
          <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 6px;">
              <span style="
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: ${isActive ? '#00c853' : '#9e9e9e'};
                display: inline-block;
                ${isActive ? 'animation: pulse 2s infinite;' : ''}
              "></span>
              <div style="line-height: 1.4;">
                <div style="font-weight: 600; color: #00796b;">
                  <span class="driver-count">${visibleCount}</span> / ${totalCount} GPS actifs
                </div>
                ${visibleCount === 0 && totalCount > 0 ? '<div style="font-size: 10px; color: #f57c00;">⚠️ Aucun chauffeur localisé</div>' : ''}
                ${visibleCount === 0 && totalCount === 0 ? '<div style="font-size: 10px; color: #9e9e9e;">Aucun chauffeur</div>' : ''}
              </div>
            </div>
            <div class="driver-legend" style="display: flex; flex-direction: column; gap: 4px; font-size: 10px; color: #334155;">
              <div style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #22c55e; border: 1px solid #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.2);"></span>
                <span>Disponible</span>
              </div>
              <div style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #1976d2; border: 1px solid #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.2);"></span>
                <span>En course</span>
              </div>
              <div style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #9e9e9e; border: 1px solid #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.2);"></span>
                <span>Hors-ligne</span>
              </div>
            </div>
          </div>
        `;

        // Ajouter l'animation pulse
        const style = document.createElement('style');
        style.textContent = `
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
        `;
        container.appendChild(style);

        return container;
      },

      onRemove: function (_map) {
        // Nettoyage si nécessaire
      },
    });

    // Supprimer l'ancien contrôle s'il existe
    if (map._driverCounterControl) {
      map.removeControl(map._driverCounterControl);
    }

    // Ajouter le nouveau contrôle
    map._driverCounterControl = new DriverCounterControl({
      position: 'bottomleft',
    });
    map.addControl(map._driverCounterControl);

    // Mettre à jour le compteur
    const updateCounter = () => {
      const countElement = map._driverCounterControl.getContainer()?.querySelector('.driver-count');
      if (countElement) {
        countElement.textContent = Object.keys(markersRef.current).length;
      }
    };

    // Mettre à jour le compteur immédiatement et après chaque changement
    updateCounter();
    const interval = setInterval(updateCounter, 1000);

    return () => {
      clearInterval(interval);
      if (map._driverCounterControl) {
        map.removeControl(map._driverCounterControl);
        delete map._driverCounterControl;
      }
    };
  }, [drivers, allDrivers.length]);

  // Ajouter le contrôle de recherche comme contrôle Leaflet (une seule fois)
  useEffect(() => {
    const map = getMap();
    if (!map || map._searchControl) return; // Ne pas recréer si déjà existant

    // Créer le contrôle de recherche personnalisé
    const SearchControl = L.Control.extend({
      onAdd: function (_map) {
        const container = L.DomUtil.create('div', 'search-control');
        container.style.cssText = `
          background: rgba(255,255,255,0.95);
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 8px 12px;
          font-size: 12px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          display: flex;
          align-items: center;
          gap: 8px;
          min-width: 200px;
        `;

        const input = L.DomUtil.create('input', 'search-input');
        input.type = 'text';
        input.placeholder = 'Rechercher un chauffeur...';
        input.value = '';
        input.style.cssText = `
          border: none;
          outline: none;
          background: transparent;
          font-size: 12px;
          flex: 1;
          color: #334155;
        `;

        const clearBtn = L.DomUtil.create('button', 'clear-search');
        clearBtn.innerHTML = '✕';
        clearBtn.style.cssText = `
          border: none;
          background: #e2e8f0;
          color: #64748b;
          border-radius: 50%;
          width: 20px;
          height: 20px;
          cursor: pointer;
          font-size: 12px;
          display: none;
          align-items: center;
          justify-content: center;
        `;

        container.appendChild(input);
        container.appendChild(clearBtn);

        // Événements
        input.addEventListener('input', (e) => {
          setSearchQuery(e.target.value);
          clearBtn.style.display = e.target.value ? 'flex' : 'none';
        });

        clearBtn.addEventListener('click', () => {
          input.value = '';
          setSearchQuery('');
          clearBtn.style.display = 'none';
        });

        // Empêcher la propagation des événements
        L.DomEvent.disableClickPropagation(container);
        L.DomEvent.disableScrollPropagation(container);

        return container;
      },

      onRemove: function (_map) {
        // Nettoyage si nécessaire
      },
    });

    // Ajouter le contrôle
    map._searchControl = new SearchControl({ position: 'topright' });
    map.addControl(map._searchControl);

    return () => {
      if (map._searchControl) {
        map.removeControl(map._searchControl);
        delete map._searchControl;
      }
    };
  }, []);

  useEffect(() => {
    const map = getMap();
    const container = map?._searchControl?.getContainer?.();
    if (!container) return;

    const input = container.querySelector('.search-input');
    const clearBtn = container.querySelector('.clear-search');

    if (input && input.value !== searchQuery) {
      input.value = searchQuery;
    }
    if (clearBtn) {
      clearBtn.style.display = searchQuery ? 'flex' : 'none';
    }
  }, [searchQuery]);

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        minHeight: '400px',
        background: '#e0e0e0', // Fond gris pour debug
      }}
    >
      <div
        ref={mapElRef}
        style={{
          width: '100%',
          height: '100%',
          minHeight: '400px',
          background: '#f5f5f5', // Fond gris clair pour debug
          border: '2px solid #00796b', // Bordure verte pour debug
        }}
      />
      {!mapRef.current && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            padding: '16px 24px',
            background: 'rgba(255, 255, 255, 0.9)',
            borderRadius: '8px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            zIndex: 9999,
            fontSize: '14px',
            fontWeight: '500',
            color: '#00796b',
          }}
        >
          🗺️ Initialisation de la carte...
        </div>
      )}
      {mapReady && showNoGpsBanner && (
        <div
          style={{
            position: 'absolute',
            top: 8,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '8px 16px',
            background: 'rgba(245, 124, 0, 0.95)',
            color: '#fff',
            borderRadius: 8,
            fontSize: 12,
            fontWeight: 500,
            zIndex: 1000,
            boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
            textAlign: 'center',
            maxWidth: '90%',
          }}
        >
          Aucune position reçue. Vérifiez que les chauffeurs ont activé le GPS et l'app en ligne.
        </div>
      )}
      {mapReady && (
        <button
          type="button"
          onClick={() => fitBoundsToMarkers(14)}
          style={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            zIndex: 1000,
            padding: '6px 12px',
            fontSize: 12,
            fontWeight: 600,
            color: '#00796b',
            background: 'rgba(255,255,255,0.95)',
            border: '1px solid #00796b',
            borderRadius: 8,
            boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
            cursor: 'pointer',
          }}
          title="Recadrer sur les chauffeurs"
        >
          Recentrer
        </button>
      )}
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
            zIndex: 1000,
            maxHeight: 120,
            overflow: 'auto',
          }}
        >
          <div><strong>MAP_DEBUG</strong></div>
          <div>Reçues: {mapDebugInfo.received} | Valides: {mapDebugInfo.valid}</div>
          <div>
            joined_company: {mapDebugInfo.joinedReceived ? '✓ reçu' : '–'}
            {mapDebugInfo.fallbackUsed ? ' | fallback 1.5s utilisé' : ''}
          </div>
          {mapDebugInfo.lastUpdate && <div>Dernière mise à jour: {mapDebugInfo.lastUpdate}</div>}
          {mapDebugInfo.sample && (
            <div>Exemple 1er chauffeur: lat={mapDebugInfo.sample.lat} lon={mapDebugInfo.sample.lon} id={mapDebugInfo.sample.driver_id}</div>
          )}
          {mapDebugInfo.exclusionReasons?.length > 0 && (
            <div>Exclusions: {mapDebugInfo.exclusionReasons.join(', ')}</div>
          )}
        </div>
      )}
    </div>
  );
}
