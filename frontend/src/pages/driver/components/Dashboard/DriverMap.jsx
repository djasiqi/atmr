// src/pages/driver/components/Dashboard/DriverMap.jsx

import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { GoogleMap, Marker, Polyline, InfoWindow } from '@react-google-maps/api';
import { useGoogleMapsLoaded } from '../../../../components/common/GoogleMapsProvider';
import MapPlaceholder from '../../../../components/common/MapPlaceholder';
import {
  SWITZERLAND_CENTER,
  DEFAULT_MAP_OPTIONS,
  MAP_COLORS,
  makePoiMarkerIcon,
  ROUTE_OPTIONS,
  INFOWINDOW_FONT,
} from '../../../../utils/mapUtils';
import myLocationIcon from '../../../../assets/icons/my-location.png';
import styles from './DriverMap.module.css';

const defaultCenter = SWITZERLAND_CENTER;
const CONTAINER_STYLE = { width: '100%', height: '100%' };

// Distance Haversine (m)
const toRad = (v) => (v * Math.PI) / 180;
function haversineDistance(a, b) {
  const R = 6371000;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const x = Math.sin(dLat / 2) ** 2 + Math.sin(dLng / 2) ** 2 * Math.cos(lat1) * Math.cos(lat2);
  const c = 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
  return R * c;
}

// Normalise [lat,lng] ou [lng,lat] vers [{lat,lng}] pour Google Maps
function normalizeToLatLngPairs(arr) {
  if (!Array.isArray(arr) || !arr.length) return [];
  const first = arr[0];
  if (!Array.isArray(first) || first.length < 2) return [];
  const looksLikeLatLng = Math.abs(first[0]) <= 90 && Math.abs(first[1]) <= 180;
  return looksLikeLatLng
    ? arr.map(([lat, lng]) => ({ lat, lng }))
    : arr.map(([lng, lat]) => ({ lat, lng }));
}

// Les marqueurs POI utilisent makePoiMarkerIcon de mapUtils

function fmtTime(t) {
  try {
    const d = new Date(t);
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  } catch {
    return t || '';
  }
}

const DriverMap = ({ assignments = [], myLocation, delays = {}, isLoaded = true }) => {
  const { isLoaded: gmLoaded } = useGoogleMapsLoaded();
  const mapRef = useRef(null);
  const [mapReady, setMapReady] = useState(false);
  const [activeInfoWindow, setActiveInfoWindow] = useState(null);

  // Throttle position
  const lastRenderedPosRef = useRef(null);
  const lastUpdateTsRef = useRef(null);
  const SPEED_THRESHOLD = 0.5;
  const STATIONARY_UPDATE_INTERVAL = 60_000;
  const [displayPos, setDisplayPos] = useState(null);

  // Center initial
  useEffect(() => {
    if (myLocation && !displayPos) {
      setDisplayPos(myLocation);
      lastRenderedPosRef.current = myLocation;
      lastUpdateTsRef.current = Date.now();
    }
  }, [myLocation, displayPos]);

  // Throttle des mises à jour de position
  useEffect(() => {
    if (!myLocation) return;
    const now = Date.now();
    let shouldUpdate = true;
    let timerId;

    if (lastRenderedPosRef.current && lastUpdateTsRef.current) {
      const distance = haversineDistance(lastRenderedPosRef.current, myLocation);
      const timeDelta = Math.max(0.001, (now - lastUpdateTsRef.current) / 1000);
      const speed = distance / timeDelta;
      if (speed < SPEED_THRESHOLD) {
        if (now - lastUpdateTsRef.current < STATIONARY_UPDATE_INTERVAL) {
          shouldUpdate = false;
        }
      }
    }

    const applyUpdate = () => {
      setDisplayPos(myLocation);
      lastRenderedPosRef.current = myLocation;
      lastUpdateTsRef.current = Date.now();
    };

    if (shouldUpdate) {
      applyUpdate();
    } else {
      const remaining = Math.max(
        0,
        STATIONARY_UPDATE_INTERVAL - (now - (lastUpdateTsRef.current || 0))
      );
      timerId = setTimeout(applyUpdate, remaining);
    }

    return () => { if (timerId) clearTimeout(timerId); };
  }, [myLocation]);

  // Helper pour extraire lat/lng
  const getLatLng = (pt) => {
    if (!pt) return null;
    if ('lat' in pt && 'lng' in pt) return { lat: pt.lat, lng: pt.lng };
    if ('latitude' in pt && 'longitude' in pt) return { lat: pt.latitude, lng: pt.longitude };
    if (Array.isArray(pt) && pt.length >= 2) return { lat: pt[0], lng: pt[1] };
    return null;
  };

  const activeAssignments = useMemo(() => {
    return (assignments || []).filter((a) => a && a.is_on_trip);
  }, [assignments]);

  // Polylines par assignation
  const routeData = useMemo(() => {
    return activeAssignments.map((a) => {
      const path = normalizeToLatLngPairs(a.route || []);
      let color = MAP_COLORS.routeDefault;
      const b = a.booking;
      if (b && delays && delays[b.id]) {
        const late = Number(delays[b.id].delay_minutes || 0);
        if (late > 0) color = late < 10 ? MAP_COLORS.warning : MAP_COLORS.danger;
      }
      return { id: a.id, path, color };
    });
  }, [activeAssignments, delays]);

  // Marqueurs Pickup/Dropoff
  const poiMarkers = useMemo(() => {
    const items = [];
    activeAssignments.forEach((a) => {
      const b = a.booking || {};
      const pickup = getLatLng(b.pickup || b.pickup_location || (b.coordinates && b.coordinates.pickup));
      const dropoff = getLatLng(b.dropoff || b.dropoff_location || (b.coordinates && b.coordinates.dropoff));
      if (pickup) items.push({ key: `pickup-${a.id}`, pos: pickup, type: 'P', booking: b, label: 'Pickup' });
      if (dropoff) items.push({ key: `dropoff-${a.id}`, pos: dropoff, type: 'D', booking: b, label: 'Dropoff' });
    });
    return items;
  }, [activeAssignments]);

  // Fit bounds
  const fitMap = useCallback(() => {
    const map = mapRef.current;
    if (!map || !window.google) return;
    const pts = [];
    if (displayPos) pts.push(displayPos);
    activeAssignments.forEach((a) => {
      normalizeToLatLngPairs(a.route || []).forEach((p) => pts.push(p));
      const b = a.booking || {};
      const p1 = getLatLng(b.pickup || b.pickup_location);
      const p2 = getLatLng(b.dropoff || b.dropoff_location);
      if (p1) pts.push(p1);
      if (p2) pts.push(p2);
    });
    if (pts.length < 2) {
      if (displayPos) {
        map.panTo(displayPos);
      }
      return;
    }
    const bounds = new window.google.maps.LatLngBounds();
    pts.forEach((p) => bounds.extend(p));
    map.fitBounds(bounds, { top: 32, right: 32, bottom: 32, left: 32 });
  }, [displayPos, activeAssignments]);

  // Pan sur position conducteur quand pas de routes
  useEffect(() => {
    if (!mapReady || !displayPos || activeAssignments.length > 0) return;
    mapRef.current?.panTo(displayPos);
  }, [mapReady, displayPos, activeAssignments.length]);

  // Fit bounds quand routes changent
  useEffect(() => {
    if (mapReady && activeAssignments.length > 0) fitMap();
  }, [mapReady, activeAssignments, fitMap]);

  const onMapLoad = useCallback((map) => {
    mapRef.current = map;
    setMapReady(true);
  }, []);

  if (!isLoaded || !gmLoaded) {
    return (
      <div className={styles.driverMapContainer}>
        <MapPlaceholder />
      </div>
    );
  }

  const mapCenter = displayPos || defaultCenter;

  return (
    <div className={`${styles.driverMapContainer} ${!mapReady ? styles.loading : ''}`}>
      <GoogleMap
        mapContainerStyle={CONTAINER_STYLE}
        center={mapCenter}
        zoom={12}
        options={DEFAULT_MAP_OPTIONS}
        onLoad={onMapLoad}
      >
        {/* Position conducteur */}
        {displayPos && (
          <Marker
            position={displayPos}
            icon={{
              url: myLocationIcon,
              scaledSize: window.google ? new window.google.maps.Size(40, 40) : undefined,
              anchor: window.google ? new window.google.maps.Point(20, 20) : undefined,
            }}
            title="Ma position"
          />
        )}

        {/* Routes */}
        {routeData.map((r) => (
          <Polyline
            key={`route-${r.id}`}
            path={r.path}
            options={{ ...ROUTE_OPTIONS, strokeColor: r.color }}
          />
        ))}

        {/* Marqueurs Pickup/Dropoff */}
        {poiMarkers.map((poi) => (
          <Marker
            key={poi.key}
            position={poi.pos}
            icon={{
              url: makePoiMarkerIcon(poi.type, poi.type === 'P' ? MAP_COLORS.brand : MAP_COLORS.danger),
              scaledSize: window.google ? new window.google.maps.Size(28, 28) : undefined,
              anchor: window.google ? new window.google.maps.Point(14, 14) : undefined,
            }}
            onClick={() => setActiveInfoWindow(poi.key)}
          />
        ))}

        {/* InfoWindows pour POIs */}
        {poiMarkers.map((poi) =>
          activeInfoWindow === poi.key ? (
            <InfoWindow
              key={`info-${poi.key}`}
              position={poi.pos}
              onCloseClick={() => setActiveInfoWindow(null)}
            >
              <div style={{ fontFamily: INFOWINDOW_FONT, padding: '4px 2px', minWidth: 140, lineHeight: 1.5 }}>
                <div style={{ fontWeight: 600, color: MAP_COLORS.textPrimary, fontSize: 13, marginBottom: 4 }}>
                  {poi.type === 'P' ? 'Prise en charge' : 'Destination'}
                </div>
                {poi.booking.pickup_time && poi.type === 'P' && (
                  <div style={{ fontSize: 12, color: MAP_COLORS.textSecondary }}>Heure : {fmtTime(poi.booking.pickup_time)}</div>
                )}
                {poi.booking.dropoff_time && poi.type === 'D' && (
                  <div style={{ fontSize: 12, color: MAP_COLORS.textSecondary }}>Heure : {fmtTime(poi.booking.dropoff_time)}</div>
                )}
                {poi.booking.client_name && (
                  <div style={{ fontSize: 12, color: MAP_COLORS.textSecondary }}>Client : {poi.booking.client_name}</div>
                )}
              </div>
            </InfoWindow>
          ) : null
        )}
      </GoogleMap>
    </div>
  );
};

export default React.memo(DriverMap);
