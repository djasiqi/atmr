import React, { useEffect, useMemo, useState } from 'react';
import { GoogleMap, Polyline } from '@react-google-maps/api';
import GoogleMapsAdvancedMarker from '../../../components/common/GoogleMapsAdvancedMarker';
import { useGoogleMapsLoaded } from '../../../components/common/GoogleMapsProvider';
import apiClient from '../../../utils/apiClient';
import {
  LIRIE_POINT_MARKER_SIZE_PX,
  MAP_COLORS,
  PUBLIC_MAP_OPTIONS,
  ROUTE_OPTIONS,
  ROUTE_OUTLINE_OPTIONS,
  makePoiMarkerIcon,
  resolveLiriePointMarkerIcon,
} from '../../../utils/mapUtils';
import styles from './Reservations.module.css';

const MAP_STYLE = { width: '100%', height: '100%' };

function stepMarkerIcon(gmaps, label) {
  const size = LIRIE_POINT_MARKER_SIZE_PX;
  const half = size / 2;
  return {
    url: makePoiMarkerIcon(String(label), MAP_COLORS.brand),
    scaledSize: gmaps?.Size ? new gmaps.Size(size, size) : undefined,
    anchor: gmaps?.Point ? new gmaps.Point(half, half) : undefined,
  };
}

function samePoint(left, right) {
  if (!left || !right) return false;
  return Math.abs(left.lat - right.lat) < 0.00001 && Math.abs(left.lng - right.lng) < 0.00001;
}

function geocodePlace(place) {
  const address = String(place || '').trim();
  if (!address || !window.google?.maps?.Geocoder) return Promise.resolve(null);
  const geocoder = new window.google.maps.Geocoder();
  return new Promise((resolve) => {
    geocoder.geocode(
      { address, componentRestrictions: { country: 'CH' } },
      (results, status) => {
        const location = results?.[0]?.geometry?.location;
        if (status !== 'OK' || !location) {
          resolve(null);
          return;
        }
        resolve({ lat: location.lat(), lng: location.lng() });
      }
    );
  });
}

async function roadBetween(from, to) {
  try {
    const response = await apiClient.get('/osrm/route', {
      params: {
        pickup_lat: from.lat,
        pickup_lon: from.lng,
        dropoff_lat: to.lat,
        dropoff_lon: to.lng,
      },
    });
    const route = response?.data?.route;
    if (Array.isArray(route) && route.length > 1) {
      return route
        .map((pair) => ({ lat: Number(pair[0]), lng: Number(pair[1]) }))
        .filter((point) => Number.isFinite(point.lat) && Number.isFinite(point.lng));
    }
  } catch {
    /* tracé direct si l’itinéraire routier est indisponible */
  }
  return [from, to];
}

export default function ReservationRouteMap({ stops }) {
  const { isLoaded, ensureLoaded } = useGoogleMapsLoaded();
  const [points, setPoints] = useState([]);
  const [path, setPath] = useState([]);

  const places = useMemo(
    () =>
      (Array.isArray(stops) ? stops : []).map((stop) => ({
        key: stop.key,
        place: stop.place,
        point: stop.point || null,
      })),
    [stops]
  );

  useEffect(() => {
    ensureLoaded?.();
  }, [ensureLoaded]);

  useEffect(() => {
    if (!isLoaded) return undefined;
    let cancelled = false;
    (async () => {
      const resolved = [];
      for (const stop of places) {
        const known = stop.point;
        const point = known || (await geocodePlace(stop.place));
        if (point) resolved.push(point);
      }
      if (!cancelled) setPoints(resolved);
    })();
    return () => {
      cancelled = true;
    };
  }, [isLoaded, places]);

  useEffect(() => {
    if (points.length < 2) {
      setPath(points);
      return undefined;
    }
    let cancelled = false;
    (async () => {
      const segments = await Promise.all(
        points.slice(1).map((point, index) => roadBetween(points[index], point))
      );
      const line = [];
      segments.forEach((segment) => {
        segment.forEach((point) => {
          const previous = line[line.length - 1];
          if (!samePoint(previous, point)) line.push(point);
        });
      });
      if (!cancelled) setPath(line.length ? line : points);
    })();
    return () => {
      cancelled = true;
    };
  }, [points]);

  const fit = (map) => {
    if (!map || points.length === 0 || !window.google?.maps) return;
    const bounds = new window.google.maps.LatLngBounds();
    points.forEach((point) => bounds.extend(point));
    if (points.length === 1) {
      map.setCenter(points[0]);
      map.setZoom(14);
      return;
    }
    map.fitBounds(bounds, 28);
  };

  return (
    <div className={styles.routeMapCard} aria-label="Aperçu de l’itinéraire">
      {isLoaded && points.length > 0 ? (
        <GoogleMap
          mapContainerStyle={MAP_STYLE}
          center={points[0]}
          zoom={13}
          onLoad={fit}
          options={{
            ...PUBLIC_MAP_OPTIONS,
            disableDefaultUI: true,
            gestureHandling: 'none',
            draggable: false,
            keyboardShortcuts: false,
            clickableIcons: false,
          }}
        >
          {path.length > 1 ? <Polyline path={path} options={ROUTE_OUTLINE_OPTIONS} /> : null}
          {path.length > 1 ? (
            <Polyline path={path} options={{ ...ROUTE_OPTIONS, strokeColor: MAP_COLORS.brand }} />
          ) : null}
          {points.map((point, index) => {
            if (points.findIndex((other) => samePoint(other, point)) !== index) return null;
            const isFirst = index === 0;
            const isLast = index === points.length - 1;
            const kind = isFirst ? 'pickup' : isLast ? 'dropoff' : 'step';
            const icon =
              kind === 'step'
                ? stepMarkerIcon(window.google?.maps, index)
                : resolveLiriePointMarkerIcon(window.google?.maps, kind);
            const title = kind === 'pickup' ? 'Départ' : kind === 'dropoff' ? 'Arrivée' : `Étape ${index}`;
            return (
              <GoogleMapsAdvancedMarker
                key={`${point.lat}-${point.lng}-${index}`}
                position={point}
                icon={icon}
                title={title}
                zIndex={isFirst ? 10 : 11}
              />
            );
          })}
        </GoogleMap>
      ) : (
        <div className={styles.routeMapFallback}>Itinéraire</div>
      )}
    </div>
  );
}
