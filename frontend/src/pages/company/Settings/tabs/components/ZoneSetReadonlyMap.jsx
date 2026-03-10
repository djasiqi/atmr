import React, { useEffect, useRef, useState } from 'react';
import { useGoogleMapsLoaded } from '../../../../../components/common/GoogleMapsProvider';
import MapPlaceholder from '../../../../../components/common/MapPlaceholder';

const DEFAULT_CENTER = { lat: 46.2044, lng: 6.1432 };
const DEFAULT_ZOOM = 10;
const PALETTE = ['#4F46E5', '#0EA5E9', '#16A34A', '#F59E0B', '#DC2626', '#A855F7', '#14B8A6', '#334155'];
const SVG_FALLBACK_WIDTH = 640;
const SVG_FALLBACK_HEIGHT = 320;

const ZoneSetReadonlyMap = ({
  zoneSetDetail,
  zoneSetDetails = [],
  loading = false,
  active = true,
}) => {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const lastBoundsRef = useRef(null);
  const resizeObserverRef = useRef(null);
  const mapListenersRef = useRef([]);
  const retryInitTimerRef = useRef(null);
  const delayedResizeTimersRef = useRef([]);
  const zoneBoundsByKeyRef = useRef(new Map());
  const didWarnTilesTimeoutRef = useRef(false);
  const [tilesLoaded, setTilesLoaded] = useState(false);
  const [mapWarning, setMapWarning] = useState('');
  const [hoveredZoneKey, setHoveredZoneKey] = useState('');
  const [selectedZoneKey, setSelectedZoneKey] = useState('');
  const { isLoaded, loadError } = useGoogleMapsLoaded();
  const sourceDetails =
    Array.isArray(zoneSetDetails) && zoneSetDetails.length > 0
      ? zoneSetDetails
      : (zoneSetDetail ? [zoneSetDetail] : []);
  const zones = sourceDetails
    .flatMap((detail) => {
      const detailZones = Array.isArray(detail?.zones) ? detail.zones : [];
      return detailZones.map((zone, zoneIdx) => ({
        ...zone,
        __zoneSetLabel: String(detail?.label || detail?.key || ''),
        __zoneKey: String(zone?.id || `${detail?.key || detail?.label || 'zone'}:${zone?.code || zoneIdx}`),
      }));
    })
    .map((zone, globalIdx) => ({
      ...zone,
      // Force one distinct color per displayed zone.
      __legendColor: PALETTE[globalIdx % PALETTE.length],
    }));

  const fitMapToZoneKey = (zoneKey) => {
    const map = mapRef.current;
    if (!map || !zoneKey) return;
    const bounds = zoneBoundsByKeyRef.current.get(String(zoneKey));
    if (bounds && !bounds.isEmpty()) {
      map.fitBounds(bounds, 24);
    }
  };

  useEffect(() => {
    if (!selectedZoneKey) return;
    const exists = zones.some((zone) => String(zone.__zoneKey) === String(selectedZoneKey));
    if (!exists) {
      setSelectedZoneKey('');
    }
  }, [selectedZoneKey, zones]);

  useEffect(() => {
    if (!isLoaded || loadError || !active) return undefined;

    const clearDelayedResizeTimers = () => {
      delayedResizeTimersRef.current.forEach((timerId) => window.clearTimeout(timerId));
      delayedResizeTimersRef.current = [];
    };

    const refreshLayout = () => {
      const map = mapRef.current;
      if (!map) return;
      window.google.maps.event.trigger(map, 'resize');
      if (lastBoundsRef.current && !lastBoundsRef.current.isEmpty()) {
        map.fitBounds(lastBoundsRef.current, 24);
      }
    };

    const ensureMapInitialized = () => {
      const container = mapContainerRef.current;
      if (!container || container.offsetWidth <= 0 || container.offsetHeight <= 0) {
        return false;
      }
      if (!mapRef.current) {
        mapRef.current = new window.google.maps.Map(container, {
          center: DEFAULT_CENTER,
          zoom: DEFAULT_ZOOM,
          mapTypeId: 'roadmap',
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: false,
          backgroundColor: '#f8fafc',
        });
        mapListenersRef.current.push(
          mapRef.current.addListener('tilesloaded', () => {
            setTilesLoaded(true);
            setMapWarning('');
            didWarnTilesTimeoutRef.current = false;
          })
        );
        mapListenersRef.current.push(
          mapRef.current.addListener('idle', () => {
            setTilesLoaded(true);
            setMapWarning('');
            didWarnTilesTimeoutRef.current = false;
          })
        );
        mapListenersRef.current.push(
          mapRef.current.data.addListener('mouseover', (event) => {
            const key = String(event?.feature?.getProperty?.('zone_key') || '');
            if (key) setHoveredZoneKey(key);
          })
        );
        mapListenersRef.current.push(
          mapRef.current.data.addListener('mouseout', () => {
            setHoveredZoneKey('');
          })
        );
        mapListenersRef.current.push(
          mapRef.current.data.addListener('click', (event) => {
            const key = String(event?.feature?.getProperty?.('zone_key') || '');
            if (!key) return;
            setSelectedZoneKey(key);
            fitMapToZoneKey(key);
          })
        );
      }
      clearDelayedResizeTimers();
      delayedResizeTimersRef.current = [
        window.setTimeout(refreshLayout, 0),
        window.setTimeout(refreshLayout, 150),
        window.setTimeout(refreshLayout, 350),
      ];
      return true;
    };

    if (!ensureMapInitialized()) {
      retryInitTimerRef.current = window.setTimeout(() => {
        if (!ensureMapInitialized()) {
          retryInitTimerRef.current = window.setTimeout(ensureMapInitialized, 200);
        }
      }, 150);
    }

    if (!resizeObserverRef.current && typeof ResizeObserver !== 'undefined' && mapContainerRef.current) {
      resizeObserverRef.current = new ResizeObserver(() => {
        refreshLayout();
      });
      resizeObserverRef.current.observe(mapContainerRef.current);
    }

    return () => {
      if (retryInitTimerRef.current) {
        window.clearTimeout(retryInitTimerRef.current);
        retryInitTimerRef.current = null;
      }
      clearDelayedResizeTimers();
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
        resizeObserverRef.current = null;
      }
    };
  }, [isLoaded, loadError, active]);

  useEffect(() => {
    if (!isLoaded || loadError || !active) return undefined;
    if (!mapRef.current) return undefined;
    if (tilesLoaded) return undefined;
    const timeout = window.setTimeout(() => {
      if (didWarnTilesTimeoutRef.current) return;
      didWarnTilesTimeoutRef.current = true;
      setMapWarning('Fond de carte indisponible, zones affichées sans fond.');
      console.warn('[ZoneSetReadonlyMap] Fond de carte indisponible (timeout de chargement tuiles).');
    }, 12000);
    return () => window.clearTimeout(timeout);
  }, [isLoaded, loadError, active, tilesLoaded, zones.length]);

  useEffect(() => () => {
    mapListenersRef.current.forEach((listener) => {
      if (listener?.remove) listener.remove();
    });
    mapListenersRef.current = [];
    if (mapRef.current && window?.google?.maps?.event) {
      window.google.maps.event.clearInstanceListeners(mapRef.current);
    }
    mapRef.current = null;
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isLoaded || !zoneSetDetail || !active) return;
    setTilesLoaded(false);
    setMapWarning('');
    didWarnTilesTimeoutRef.current = false;

    const existing = map.data;
    existing.forEach((feature) => {
      existing.remove(feature);
    });

    const features = [];
    zones.forEach((zone) => {
      const color = String(zone?.__legendColor || zone?.color || '#4F46E5');
      const communes = Array.isArray(zone?.communes) ? zone.communes : [];
      communes.forEach((commune) => {
        const geometry = commune?.geometry;
        const baseProps = {
          zone_code: String(zone?.code || ''),
          zone_key: String(zone?.__zoneKey || ''),
          zone_label: String(zone?.label || zone?.code || ''),
          zone_set_label: String(zone?.__zoneSetLabel || ''),
          commune_name: String(commune?.name || commune?.token || ''),
          fill_color: color,
        };
        if (geometry && typeof geometry === 'object') {
          features.push({
            type: 'Feature',
            properties: baseProps,
            geometry,
          });
          return;
        }
        const lat = Number(commune?.lat);
        const lon = Number(commune?.lon);
        if (Number.isFinite(lat) && Number.isFinite(lon)) {
          features.push({
            type: 'Feature',
            properties: baseProps,
            geometry: { type: 'Point', coordinates: [lon, lat] },
          });
        }
      });
    });

    if (features.length > 0) {
      try {
        map.data.addGeoJson({ type: 'FeatureCollection', features });
      } catch (error) {
        console.warn('[ZoneSetReadonlyMap] Erreur lors du rendu des couches GeoJSON:', error);
      }
    }

    if (features.length > 0) {
      const bounds = new window.google.maps.LatLngBounds();
      const boundsByZoneKey = new Map();
      map.data.forEach((feature) => {
        const geometry = feature.getGeometry();
        if (!geometry) return;
        const zoneKey = String(feature.getProperty('zone_key') || '');
        let zoneBounds = boundsByZoneKey.get(zoneKey);
        if (!zoneBounds) {
          zoneBounds = new window.google.maps.LatLngBounds();
          boundsByZoneKey.set(zoneKey, zoneBounds);
        }
        geometry.forEachLatLng((latLng) => {
          bounds.extend(latLng);
          zoneBounds.extend(latLng);
        });
      });
      lastBoundsRef.current = bounds;
      zoneBoundsByKeyRef.current = boundsByZoneKey;
      if (!bounds.isEmpty()) {
        map.fitBounds(bounds, 24);
      }
    } else {
      lastBoundsRef.current = null;
      zoneBoundsByKeyRef.current = new Map();
      map.setCenter(DEFAULT_CENTER);
      map.setZoom(DEFAULT_ZOOM);
    }
    window.setTimeout(() => window.google.maps.event.trigger(map, 'resize'), 0);
  }, [zoneSetDetail, zones, isLoaded, active]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isLoaded) return;
    map.data.setStyle((feature) => {
      const color = String(feature.getProperty('fill_color') || '#4F46E5');
      const zoneKey = String(feature.getProperty('zone_key') || '');
      const highlighted = Boolean(
        zoneKey
          && (zoneKey === String(hoveredZoneKey || '')
            || zoneKey === String(selectedZoneKey || ''))
      );
      const geometry = feature.getGeometry();
      const gType = geometry?.getType?.() || '';
      if (gType === 'Point') {
        return {
          icon: {
            path: window.google.maps.SymbolPath.CIRCLE,
            fillColor: color,
            fillOpacity: highlighted ? 1 : 0.95,
            strokeColor: '#0f172a',
            strokeWeight: highlighted ? 1.7 : 1,
            scale: highlighted ? 6.6 : 5,
          },
          zIndex: highlighted ? 20 : 1,
        };
      }
      return {
        fillColor: color,
        fillOpacity: highlighted ? 0.5 : 0.35,
        strokeColor: color,
        strokeOpacity: highlighted ? 1 : 0.9,
        strokeWeight: highlighted ? 2.2 : 1.2,
        zIndex: highlighted ? 20 : 1,
      };
    });
  }, [isLoaded, hoveredZoneKey, selectedZoneKey]);

  const hasGeometry = zones.some((zone) => Array.isArray(zone?.communes) && zone.communes.some((item) => item?.geometry));
  const hasCentroids = zones.some(
    (zone) => Array.isArray(zone?.communes)
      && zone.communes.some((item) => Number.isFinite(Number(item?.lat)) && Number.isFinite(Number(item?.lon)))
  );
  const centroidPoints = zones.flatMap((zone) => {
    const communes = Array.isArray(zone?.communes) ? zone.communes : [];
    return communes
      .map((commune) => ({
        zoneLabel: String(zone?.label || zone?.code || ''),
        zoneSetLabel: String(zone?.__zoneSetLabel || ''),
        name: String(commune?.name || commune?.token || ''),
        lat: Number(commune?.lat),
        lon: Number(commune?.lon),
        color: String(zone?.__legendColor || zone?.color || '#4F46E5'),
      }))
      .filter((item) => Number.isFinite(item.lat) && Number.isFinite(item.lon));
  });

  const renderCentroidFallback = () => {
    if (centroidPoints.length === 0) return null;
    const lats = centroidPoints.map((p) => p.lat);
    const lons = centroidPoints.map((p) => p.lon);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);
    const lonSpan = Math.max(maxLon - minLon, 0.0001);
    const latSpan = Math.max(maxLat - minLat, 0.0001);
    const pad = 18;

    const toX = (lon) => pad + ((lon - minLon) / lonSpan) * (SVG_FALLBACK_WIDTH - pad * 2);
    const toY = (lat) => pad + ((maxLat - lat) / latSpan) * (SVG_FALLBACK_HEIGHT - pad * 2);

    return (
      <div
        style={{
          width: '100%',
          height: 320,
          borderRadius: 10,
          border: '1px solid #e5e7eb',
          overflow: 'hidden',
          background: '#f8fafc',
        }}
      >
        <svg viewBox={`0 0 ${SVG_FALLBACK_WIDTH} ${SVG_FALLBACK_HEIGHT}`} width="100%" height="100%">
          <rect x="0" y="0" width={SVG_FALLBACK_WIDTH} height={SVG_FALLBACK_HEIGHT} fill="#f8fafc" />
          {centroidPoints.map((point, idx) => (
            <g key={`${point.zoneSetLabel}-${point.zoneLabel}-${point.name}-${idx}`}>
              <circle
                cx={toX(point.lon)}
                cy={toY(point.lat)}
                r="4.5"
                fill={point.color}
                stroke="#0f172a"
                strokeWidth="0.7"
              >
                <title>{`${point.name} - ${point.zoneLabel}`}</title>
              </circle>
            </g>
          ))}
        </svg>
      </div>
    );
  };
  const renderLegend = () => (
    <div
      style={{
        marginTop: 8,
        border: '1px solid #e5e7eb',
        borderRadius: 10,
        background: '#ffffff',
        padding: 10,
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 6 }}>
        Zones configurées (admin)
      </div>
      {zones.length === 0 ? (
        <div style={{ fontSize: 12, color: '#64748b' }}>Aucune zone définie.</div>
      ) : (
        zones.map((zone) => (
          <div
            key={zone.id || zone.code}
            onMouseEnter={() => setHoveredZoneKey(String(zone.__zoneKey || ''))}
            onMouseLeave={() => setHoveredZoneKey('')}
            onClick={() => {
              const key = String(zone.__zoneKey || '');
              setSelectedZoneKey(key);
              fitMapToZoneKey(key);
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 4,
              cursor: 'pointer',
              padding: '2px 4px',
              borderRadius: 6,
              background:
                String(zone.__zoneKey || '') === String(hoveredZoneKey || '')
                || String(zone.__zoneKey || '') === String(selectedZoneKey || '')
                  ? 'rgba(79,70,229,0.08)'
                  : 'transparent',
            }}
          >
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 999,
                background: String(zone?.__legendColor || zone?.color || '#4F46E5'),
                display: 'inline-block',
                border: '1px solid rgba(15,23,42,0.12)',
              }}
            />
            <span style={{ fontSize: 12, color: '#334155' }}>
              {zone.label || zone.code} ({Number(zone.communes_count || 0)} communes)
              {zone.__zoneSetLabel ? ` - ${zone.__zoneSetLabel}` : ''}
            </span>
          </div>
        ))
      )}
    </div>
  );

  if (loading) {
    return (
      <>
        <MapPlaceholder style={{ minHeight: 260 }} />
        {renderLegend()}
      </>
    );
  }

  if (!isLoaded || loadError) {
    return (
      <>
        {renderCentroidFallback() || <MapPlaceholder style={{ minHeight: 260 }} />}
        {renderLegend()}
      </>
    );
  }

  if (!hasGeometry && !hasCentroids) {
    return (
      <>
        <div
          style={{
            minHeight: 260,
            border: '1px solid #e5e7eb',
            borderRadius: 10,
            background: '#f8fafc',
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'flex-start',
            flexDirection: 'column',
            color: '#475569',
            padding: 14,
            textAlign: 'left',
            gap: 8,
          }}
        >
          <strong>Visualisation des zones</strong>
          <span>
            Les polygones cartographiques ne sont pas encore disponibles pour ce zonage.
          </span>
          {zones.length > 0 && <div style={{ fontSize: 13 }}>Voir la légende ci-dessous.</div>}
        </div>
        {renderLegend()}
      </>
    );
  }

  return (
    <>
      <div
        ref={mapContainerRef}
        style={{
          width: '100%',
          height: 320,
          borderRadius: 10,
          border: '1px solid #e5e7eb',
          overflow: 'hidden',
          position: 'relative',
        }}
      />
      {!tilesLoaded && !mapWarning && (
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 6 }}>
          Chargement du fond de carte...
        </div>
      )}
      {mapWarning && (
        <div style={{ fontSize: 12, color: '#b45309', marginTop: 6 }}>
          {mapWarning}
        </div>
      )}
      {mapWarning && hasCentroids && (
        <div style={{ marginTop: 8 }}>
          {renderCentroidFallback()}
        </div>
      )}
      {renderLegend()}
    </>
  );
};

export default ZoneSetReadonlyMap;
