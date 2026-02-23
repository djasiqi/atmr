import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { GoogleMap, Marker, Polyline, InfoWindow } from '@react-google-maps/api';
import { useGoogleMapsLoaded } from '../../../../components/common/GoogleMapsProvider';
import MapPlaceholder from '../../../../components/common/MapPlaceholder';
import {
  DEFAULT_MAP_OPTIONS,
  MAP_COLORS,
  makePinMarkerIcon,
  getRouteColor,
  ROUTE_OPTIONS,
  INFOWINDOW_FONT,
} from '../../../../utils/mapUtils';
import styles from './ReservationMapView.module.css';

const CONTAINER_STYLE = { width: '100%', height: '100%' };
const DEFAULT_CENTER = { lat: 46.2044, lng: 6.1432 };

const ReservationMapView = ({ reservations }) => {
  const { isLoaded: gmLoaded } = useGoogleMapsLoaded();
  const mapRef = useRef(null);
  const [geocodingStatus, setGeocodingStatus] = useState('idle');
  const [markerData, setMarkerData] = useState([]);
  const [activeRoute, setActiveRoute] = useState(null);
  const [activeInfoWindow, setActiveInfoWindow] = useState(null);

  const displayDate = useMemo(() => {
    if (reservations.length > 0) {
      const date = new Date(reservations[0].scheduled_time || reservations[0].pickup_time);
      return date.toLocaleDateString('fr-FR', {
        weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
      });
    }
    return new Date().toLocaleDateString('fr-FR', {
      weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
    });
  }, [reservations]);

  const geocodeAddress = useCallback(async (address) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}&limit=1&countrycodes=ch`
      );
      const data = await response.json();
      if (data.length > 0) return { lat: parseFloat(data[0].lat), lng: parseFloat(data[0].lon) };
    } catch (error) {
      console.error('Erreur de géocodification:', error);
    }
    return null;
  }, []);

  const getOSRMRoute = useCallback(async (pickupCoords, dropoffCoords) => {
    try {
      const url =
        `${process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '/api/v1'}/osrm/route?` +
        `pickup_lat=${pickupCoords.lat}&pickup_lon=${pickupCoords.lng}&` +
        `dropoff_lat=${dropoffCoords.lat}&dropoff_lon=${dropoffCoords.lng}`;
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        if (data.route && data.route.length > 0) {
          return data.route.map(([lat, lng]) => ({ lat, lng }));
        }
      }
    } catch (error) {
      console.error('OSRM erreur:', error);
    }
    return [pickupCoords, dropoffCoords];
  }, []);

  // Préparer les données de marqueurs
  useEffect(() => {
    let isCancelled = false;

    const prepareMarkers = async () => {
      setGeocodingStatus('loading');
      const markers = [];

      for (const reservation of reservations) {
        if (isCancelled) break;

        let pickupCoords = null;
        let dropoffCoords = null;

        const pickupLat = reservation.pickup_lat || reservation.pickupLat;
        const pickupLon = reservation.pickup_lon || reservation.pickupLon;
        if (pickupLat && pickupLon && !isNaN(pickupLat) && !isNaN(pickupLon)) {
          pickupCoords = { lat: parseFloat(pickupLat), lng: parseFloat(pickupLon) };
        } else if (reservation.pickup_location) {
          pickupCoords = await geocodeAddress(reservation.pickup_location);
        }

        const dropoffLat = reservation.dropoff_lat || reservation.dropoffLat;
        const dropoffLon = reservation.dropoff_lon || reservation.dropoffLon;
        if (dropoffLat && dropoffLon && !isNaN(dropoffLat) && !isNaN(dropoffLon)) {
          dropoffCoords = { lat: parseFloat(dropoffLat), lng: parseFloat(dropoffLon) };
        } else if (reservation.dropoff_location) {
          dropoffCoords = await geocodeAddress(reservation.dropoff_location);
        }

        if (isCancelled) break;

        if (pickupCoords) {
          markers.push({
            id: `pickup-${reservation.id}`,
            reservationId: reservation.id,
            type: 'pickup',
            position: pickupCoords,
            dropoffCoords,
            reservation,
          });
        }
      }

      if (!isCancelled) {
        setMarkerData(markers);
        setGeocodingStatus(markers.length > 0 ? 'success' : 'no-data');

        // Fit bounds sur tous les marqueurs
        if (markers.length > 0 && mapRef.current && window.google) {
          const bounds = new window.google.maps.LatLngBounds();
          markers.forEach((m) => bounds.extend(m.position));
          mapRef.current.fitBounds(bounds, { top: 50, right: 50, bottom: 50, left: 50 });
        }
      }
    };

    prepareMarkers();
    return () => { isCancelled = true; };
  }, [reservations, geocodeAddress]);

  // Clic sur un marqueur pickup : afficher route + dropoff
  const handlePickupClick = useCallback(async (marker) => {
    setActiveInfoWindow(marker.id);

    if (marker.dropoffCoords) {
      const lineColor = getRouteColor(marker.reservation.status);

      const routePath = await getOSRMRoute(marker.position, marker.dropoffCoords);
      setActiveRoute({
        path: routePath,
        color: lineColor,
        dropoff: marker.dropoffCoords,
        reservation: marker.reservation,
      });

      // Fit bounds sur pickup + dropoff
      if (mapRef.current && window.google) {
        const bounds = new window.google.maps.LatLngBounds();
        bounds.extend(marker.position);
        bounds.extend(marker.dropoffCoords);
        mapRef.current.fitBounds(bounds, { top: 50, right: 50, bottom: 50, left: 50 });
      }
    }
  }, [getOSRMRoute]);

  const onMapLoad = useCallback((map) => {
    mapRef.current = map;
  }, []);

  if (!gmLoaded) {
    return (
      <div className={styles.mapContainer}>
        <MapPlaceholder />
      </div>
    );
  }

  return (
    <div className={styles.mapContainer}>
      <div className={styles.map}>
        <GoogleMap
          mapContainerStyle={CONTAINER_STYLE}
          center={DEFAULT_CENTER}
          zoom={12}
          options={DEFAULT_MAP_OPTIONS}
          onLoad={onMapLoad}
        >
          {/* Marqueurs pickup */}
          {markerData.map((m) => (
            <Marker
              key={m.id}
              position={m.position}
              icon={{
                url: makePinMarkerIcon('pickup'),
                scaledSize: window.google ? new window.google.maps.Size(28, 38) : undefined,
                anchor: window.google ? new window.google.maps.Point(14, 38) : undefined,
              }}
              onClick={() => handlePickupClick(m)}
            />
          ))}

          {/* Marqueur dropoff actif */}
          {activeRoute?.dropoff && (
            <Marker
              position={activeRoute.dropoff}
              icon={{
                url: makePinMarkerIcon('dropoff'),
                scaledSize: window.google ? new window.google.maps.Size(28, 38) : undefined,
                anchor: window.google ? new window.google.maps.Point(14, 38) : undefined,
              }}
            />
          )}

          {/* Route active */}
          {activeRoute?.path && (
            <Polyline
              path={activeRoute.path}
              options={{ ...ROUTE_OPTIONS, strokeColor: activeRoute.color }}
            />
          )}

          {/* InfoWindow sur le marqueur actif */}
          {activeInfoWindow && (() => {
            const m = markerData.find((md) => md.id === activeInfoWindow);
            if (!m) return null;
            const r = m.reservation;
            return (
              <InfoWindow
                position={m.position}
                onCloseClick={() => {
                  setActiveInfoWindow(null);
                  setActiveRoute(null);
                }}
              >
                <div style={{ fontFamily: INFOWINDOW_FONT, padding: '4px 2px', minWidth: 160, lineHeight: 1.5 }}>
                  <div style={{ fontWeight: 600, color: MAP_COLORS.textPrimary, fontSize: 13, marginBottom: 6, borderBottom: `1px solid ${MAP_COLORS.border}`, paddingBottom: 4 }}>
                    Prise en charge
                  </div>
                  <div style={{ fontSize: 12, color: MAP_COLORS.textSecondary, display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <span>Client : {r.client_name || r.client?.full_name || 'N/A'}</span>
                    <span>Adresse : {r.pickup_location}</span>
                    <span>Heure : {new Date(r.scheduled_time).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}</span>
                    <span style={{ color: MAP_COLORS.brand, fontWeight: 500 }}>#{r.id}</span>
                  </div>
                </div>
              </InfoWindow>
            );
          })()}
        </GoogleMap>
      </div>

      {geocodingStatus === 'loading' && (
        <div className={styles.statusMessage}>
          <span>Chargement des positions...</span>
        </div>
      )}
      {geocodingStatus === 'no-data' && (
        <div className={styles.statusMessage}>
          <span>Aucune réservation pour cette journée</span>
        </div>
      )}
      {reservations.length === 0 && geocodingStatus === 'idle' && (
        <div className={styles.statusMessage}>
          <span>Aucune réservation à afficher</span>
        </div>
      )}

      {reservations.length > 0 && (
        <div className={styles.mapInfo}>
          <div className={styles.mapInfoRow}>
            <span className={styles.mapInfoText}>{displayDate}</span>
          </div>
          <div className={styles.mapInfoRow}>
            <span className={styles.mapInfoText} style={{ fontWeight: 600, color: MAP_COLORS.brand }}>
              {reservations.length} réservation{reservations.length > 1 ? 's' : ''}
            </span>
          </div>
        </div>
      )}

      <div className={styles.mapLegend}>
        <div className={styles.legendItem}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: MAP_COLORS.brand, display: 'inline-block' }} />
          <span>Prise en charge</span>
        </div>
        <div className={styles.legendItem}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: MAP_COLORS.danger, display: 'inline-block' }} />
          <span>Destination</span>
        </div>
      </div>
    </div>
  );
};

export default ReservationMapView;
