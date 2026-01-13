// src/pages/company/Dashboard/components/DriverLiveMap.jsx
import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { getCompanySocket } from '../../../../services/companySocket';
import useCompanyData from '../../../../hooks/useCompanyData';

// Icône Leaflet par défaut (corrige le bug d'icône manquante)
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const defaultCenter = [46.8182, 8.2275]; // CH

// ---- Icônes personnalisées pour les chauffeurs ----
const createDriverIcon = (status = 'available') => {
  const colors = {
    available: '#00796b', // Vert de marque
    busy: '#ff9800', // Orange
    offline: '#9e9e9e', // Gris
    emergency: '#f44336', // Rouge
  };

  const emojis = {
    available: '🚗',
    busy: '🚕',
    offline: '🚙',
    emergency: '🚨',
  };

  return L.divIcon({
    html: `
      <div style="
        background: ${colors[status]};
        border: 3px solid white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        font-size: 14px;
        position: relative;
      ">
        ${emojis[status]}
      </div>
    `,
    className: 'custom-driver-icon',
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16],
  });
};

// ---- helpers coords -------------------------------------------------
const toNumOrNull = (v) => {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const toLatLngSafe = (lat, lon) => {
  const la = toNumOrNull(lat);
  const lo = toNumOrNull(lon);
  return la !== null && lo !== null ? [la, lo] : null;
};
const resolveDriverCoords = (d) =>
  toLatLngSafe(d.current_lat, d.current_lon) ||
  toLatLngSafe(d.latitude, d.longitude) ||
  toLatLngSafe(d.last_latitude, d.last_longitude) ||
  (d.last_position && toLatLngSafe(d.last_position.lat, d.last_position.lon)) ||
  null;

// Déterminer le statut du chauffeur
const getDriverStatus = (driver) => {
  if (!driver.is_active) return 'offline';
  if (driver.current_booking_id || driver.status === 'busy') return 'busy';
  if (driver.emergency_mode) return 'emergency';
  return 'available';
};

// Créer un tooltip stylé
const createStyledTooltip = (driver) => {
  const status = getDriverStatus(driver);
  const statusText = {
    available: 'Disponible',
    busy: 'En course',
    offline: 'Hors ligne',
    emergency: 'Urgence',
  };

  const statusColors = {
    available: '#00796b',
    busy: '#ff9800',
    offline: '#9e9e9e',
    emergency: '#f44336',
  };

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
        ${driver.full_name ||
          (driver.first_name || driver.last_name
            ? `${driver.first_name || ''} ${driver.last_name || ''}`.trim()
            : driver.username || `Chauffeur ${driver.id}`)}
      </div>
      <div style="
        font-size: 10px;
        color: ${statusColors[status]};
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 3px;
        line-height: 1;
      ">
        <span>${
          status === 'available'
            ? '🟢'
            : status === 'busy'
              ? '🟡'
              : status === 'offline'
                ? '⚫'
                : '🔴'
        }</span>
        ${statusText[status]}
      </div>
    </div>
  `;
};

export default function DriverLiveMap({ drivers: propDrivers }) {
  const mapRef = useRef(null);
  const mapElRef = useRef(null);
  const markersRef = useRef({}); // { [driverId]: L.Marker }
  const [searchQuery, setSearchQuery] = useState('');
  const { driver: staticDrivers, company } = useCompanyData();

  // Utiliser les drivers passés en props si disponibles, sinon ceux de useCompanyData
  const allDrivers = propDrivers || staticDrivers;

  // Filtrer les drivers selon la recherche
  const drivers = searchQuery
    ? allDrivers.filter((d) => d.username?.toLowerCase().includes(searchQuery.toLowerCase()))
    : allDrivers;

  // petits helpers pour éviter d'appeler Leaflet sur une map détruite
  const getMap = () => {
    const m = mapRef.current;
    // _mapPane est défini une fois la map initialisée
    if (!m || !m._mapPane) return null;
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
      console.log('[DriverLiveMap] ⚠️ Carte déjà initialisée, skip');
      return; // évite double init hors StrictMode
    }
    if (!mapElRef.current) {
      console.log('[DriverLiveMap] ❌ mapElRef.current is null');
      return;
    }

    console.log('[DriverLiveMap] 🗺️ Initialisation de la carte Leaflet...', {
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

      console.log('[DriverLiveMap] ✅ Carte Leaflet créée');

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

      // Forcer un redimensionnement après un court délai
      setTimeout(() => {
        if (map) {
          map.invalidateSize();
          console.log('[DriverLiveMap] 🔄 Carte redimensionnée');
        }
      }, 100);

      mapRef.current = map;
      console.log('[DriverLiveMap] ✅ Carte initialisée avec succès');
    } catch (error) {
      console.error('[DriverLiveMap] ❌ Erreur initialisation carte:', error);
    }

    return () => {
      // ⚠️ StrictMode va appeler le cleanup immédiatement en dev -> remets tout à zéro
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
      const m = L.marker(ll, { icon: createDriverIcon(status) }).addTo(map);

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
        if (marker && map) {
          try {
            map.removeLayer(marker);
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

  // Socket: écouter les mises à jour live
  useEffect(() => {
    const socket = getCompanySocket();
    if (!socket) return;

    if (company?.id) {
      try {
        socket.emit('join_company', { company_id: company.id });
      } catch {}
    }

    const onLoc = (data) => {
      const map = getMap();
      if (!map) return;

      // Backend : { driver_id, lat|latitude, lon|lng|longitude, ... }
      const id = data.driver_id ?? data.id;
      const lat = data.lat ?? data.latitude ?? data.current_lat;
      const lon = data.lon ?? data.lng ?? data.longitude ?? data.current_lon;
      const ll = toLatLngSafe(lat, lon);
      if (!id || !ll) return;

      const firstName = data.first_name || data.name || `Driver ${id}`;

      if (!markersRef.current[id]) {
        // Trouver le driver complet pour le statut
        const fullDriver = drivers.find((d) => d.id === id) || {
          id,
          first_name: firstName,
          is_active: true,
        };
        const status = getDriverStatus(fullDriver);
        const m = L.marker(ll, { icon: createDriverIcon(status) }).addTo(map);

        // Fonction pour mettre à jour la direction du tooltip
        const updateTooltipDirection = () => {
          const bounds = map.getBounds();
          const center = bounds.getCenter();
          const markerLat = ll[0];
          const markerLng = ll[1];

          // Calculer la distance au centre en vertical et horizontal
          const verticalDist = Math.abs(markerLat - center.lat);
          const horizontalDist = Math.abs(markerLng - center.lng);

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

          m.unbindTooltip();
          m.bindTooltip(createStyledTooltip(fullDriver), {
            permanent: true,
            direction: direction,
            offset: offset,
            className: 'custom-driver-tooltip',
          }).openTooltip();
        };

        updateTooltipDirection();
        map.on('moveend zoomend', updateTooltipDirection);

        markersRef.current[id] = m;
      } else {
        // Mettre à jour la position
        const marker = markersRef.current[id];
        marker.setLatLng(ll);

        // Mettre à jour le contenu et la direction
        const fullDriver = drivers.find((d) => d.id === id) || {
          id,
          first_name: firstName,
          is_active: true,
        };

        const bounds = map.getBounds();
        const center = bounds.getCenter();
        const markerLat = ll[0];
        const markerLng = ll[1];

        // Calculer la distance au centre
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

        marker.unbindTooltip();
        marker
          .bindTooltip(createStyledTooltip(fullDriver), {
            permanent: true,
            direction: direction,
            offset: offset,
            className: 'custom-driver-tooltip',
          })
          .openTooltip();
      }

      fitBoundsToMarkers(14);
    };

    // ✅ Écouter les mises à jour de position en temps réel
    socket.on('driver_location_update', onLoc);

    // Explicitly request driver locations when component mounts
    try {
      socket.emit('get_driver_locations');
    } catch (e) {
      console.error('Failed to request driver locations:', e);
    }

    return () => {
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
  }, [drivers]);

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
    </div>
  );
}
