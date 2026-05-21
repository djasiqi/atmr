import React, { useCallback, useMemo } from 'react';
import { GoogleMap, Polyline } from '@react-google-maps/api';
import { useGoogleMapsLoaded } from '../common/GoogleMapsProvider';
import GoogleMapsAdvancedMarker from '../common/GoogleMapsAdvancedMarker';
import {
  PUBLIC_MAP_OPTIONS,
  MAP_COLORS,
  resolveLiriePointMarkerIcon,
  makePoiMarkerIcon,
} from '../../utils/mapUtils';
import styles from './ClientBookingLiveTrackModal.module.css';

const MAP_BOX_STYLE = { width: '100%', height: 'min(42vh, 360px)', minHeight: '260px' };

function usePickupDriverPositions(booking) {
  return useMemo(() => {
    const plat = Number(booking?.pickup_lat);
    const plon = Number(booking?.pickup_lon);
    const pickup =
      Number.isFinite(plat) && Number.isFinite(plon) ? { lat: plat, lng: plon } : null;
    const dlat = Number(booking?.driver_live_latitude);
    const dlon = Number(booking?.driver_live_longitude);
    const driver =
      Number.isFinite(dlat) && Number.isFinite(dlon) ? { lat: dlat, lng: dlon } : null;
    return { pickup, driver };
  }, [booking]);
}

/**
 * @param {{ booking: object, etaLine: string | null, onClose: () => void }} props
 */
export default function ClientBookingLiveTrackModal({ booking, etaLine, onClose }) {
  const { isLoaded: gmLoaded } = useGoogleMapsLoaded();
  const { pickup, driver } = usePickupDriverPositions(booking);

  const center = useMemo(() => {
    if (pickup && driver) {
      return {
        lat: (pickup.lat + driver.lat) / 2,
        lng: (pickup.lng + driver.lng) / 2,
      };
    }
    if (pickup) return pickup;
    if (driver) return driver;
    return { lat: 46.2044, lng: 6.1432 };
  }, [pickup, driver]);

  const linePath = useMemo(() => {
    if (!pickup || !driver) return [];
    return [driver, pickup];
  }, [pickup, driver]);

  const onMapLoad = useCallback(
    (map) => {
      if (!window.google?.maps?.LatLngBounds) return;
      if (pickup && driver) {
        const b = new window.google.maps.LatLngBounds();
        b.extend(pickup);
        b.extend(driver);
        map.fitBounds(b, 56);
        return;
      }
      if (pickup) {
        map.setCenter(pickup);
        map.setZoom(14);
        return;
      }
      if (driver) {
        map.setCenter(driver);
        map.setZoom(14);
      }
    },
    [pickup, driver]
  );

  return (
    <div
      className={styles.backdrop}
      role="dialog"
      aria-modal="true"
      aria-labelledby="client-live-track-title"
      onClick={onClose}
      onKeyDown={(e) => {
        if (e.key === 'Escape') onClose();
      }}
    >
      <div
        className={styles.panel}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.stopPropagation()}
        role="document"
      >
        <div className={styles.header}>
          <h2 id="client-live-track-title" className={styles.title}>
            Suivi en direct
          </h2>
          <button type="button" className={styles.close} onClick={onClose} aria-label="Fermer">
            ×
          </button>
        </div>
        <p
          className={styles.route}
          title={`${booking?.pickup_location || ''} → ${booking?.dropoff_location || ''}`}
        >
          <strong>Départ :</strong> {booking?.pickup_location || '—'}
        </p>
        {etaLine ? <p className={styles.eta}>{etaLine}</p> : null}
        {!pickup ? (
          <p className={styles.note}>
            Les coordonnées GPS du lieu de prise en charge ne sont pas disponibles : la carte ne peut
            pas s’afficher. L’estimation ci-dessus reste indicative lorsque le transporteur transmet sa
            position.
          </p>
        ) : null}
        {pickup && driver == null ? (
          <p className={styles.note}>
            Position du chauffeur en cours de réception. La carte se centre sur votre lieu de prise en
            charge ; actualisez dans quelques instants.
          </p>
        ) : null}
        {gmLoaded && pickup ? (
          <div className={styles.mapWrap}>
            <GoogleMap
              mapContainerStyle={MAP_BOX_STYLE}
              center={center}
              zoom={13}
              options={PUBLIC_MAP_OPTIONS}
              onLoad={onMapLoad}
            >
              <GoogleMapsAdvancedMarker
                position={pickup}
                icon={{
                  ...resolveLiriePointMarkerIcon(window.google?.maps, 'pickup'),
                }}
                title="Lieu de prise en charge"
              />
              {driver ? (
                <GoogleMapsAdvancedMarker
                  position={driver}
                  icon={{
                    url: makePoiMarkerIcon('V', MAP_COLORS.warning),
                    scaledSize: window.google ? new window.google.maps.Size(28, 28) : undefined,
                    anchor: window.google ? new window.google.maps.Point(14, 14) : undefined,
                  }}
                  title="Chauffeur"
                />
              ) : null}
              {linePath.length === 2 ? (
                <Polyline
                  path={linePath}
                  options={{
                    strokeColor: MAP_COLORS.brand,
                    strokeOpacity: 0.78,
                    strokeWeight: 3,
                    geodesic: true,
                  }}
                />
              ) : null}
            </GoogleMap>
          </div>
        ) : pickup && !gmLoaded ? (
          <p className={styles.note}>Chargement de la carte…</p>
        ) : null}
      </div>
    </div>
  );
}
