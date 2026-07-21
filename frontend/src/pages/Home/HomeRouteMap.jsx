import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Circle, GoogleMap, Polyline } from '@react-google-maps/api';
import GoogleMapsAdvancedMarker from '../../components/common/GoogleMapsAdvancedMarker';
import {
  MAP_COLORS,
  resolveLiriePointMarkerIcon,
  HERO_ROUTE_OPTIONS,
  HERO_ROUTE_OUTLINE_OPTIONS,
  HERO_ROUTE_GLOW_OPTIONS,
  HERO_MAP_OPTIONS,
} from '../../utils/mapUtils';

const CONTAINER_STYLE = { width: '100%', height: '100%' };

const GOOGLE_ATTRIBUTION_SELECTORS = [
  '.gmnoprint',
  '.gm-style-cc',
  '.gm-style-mtc',
  'a[href*="maps.google.com"]',
  'a[href*="google.com/maps"]',
  'a[href*="google.com"][target="_blank"]',
  'img[alt="Google"]',
  'img[src*="google_white"]',
  'img[src*="google4"]',
];

function hideGoogleMapAttribution(root) {
  if (!root) return;

  GOOGLE_ATTRIBUTION_SELECTORS.forEach((selector) => {
    root.querySelectorAll(selector).forEach((el) => {
      el.style.setProperty('display', 'none', 'important');
      el.style.setProperty('visibility', 'hidden', 'important');
      el.style.setProperty('opacity', '0', 'important');
    });
  });

  root.querySelectorAll('a[href*="google.com"]').forEach((anchor) => {
    const wrapper = anchor.parentElement;
    if (wrapper && wrapper.childElementCount <= 2) {
      wrapper.style.setProperty('display', 'none', 'important');
    }
  });
}

export default function HomeRouteMap({ center, onMapLoad, routePath, pickupCoord, dropoffCoord }) {
  const [pickupPulseRadius, setPickupPulseRadius] = useState(42);
  const [pickupPulseOpacity, setPickupPulseOpacity] = useState(0.28);
  const attributionObserverRef = useRef(null);
  const pickupLat = pickupCoord?.lat;
  const pickupLon = pickupCoord?.lon;

  const handleMapLoad = useCallback(
    (map) => {
      onMapLoad?.(map);
      const root = map?.getDiv?.();
      if (!root) return;

      hideGoogleMapAttribution(root);
      attributionObserverRef.current?.disconnect();
      attributionObserverRef.current = new MutationObserver(() => hideGoogleMapAttribution(root));
      attributionObserverRef.current.observe(root, { childList: true, subtree: true });
    },
    [onMapLoad]
  );

  useEffect(() => () => attributionObserverRef.current?.disconnect(), []);

  useEffect(() => {
    if (pickupLat == null || pickupLon == null) return;

    const prefersReducedMotion =
      typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (prefersReducedMotion) {
      setPickupPulseRadius(48);
      setPickupPulseOpacity(0.18);
      return undefined;
    }

    let rafId = 0;
    const startedAt = performance.now();
    const duration = 2200;

    const tickPulse = (now) => {
      const progress = Math.min(1, (now - startedAt) / duration);
      const wave = Math.sin(progress * Math.PI);
      setPickupPulseRadius(38 + wave * 52);
      setPickupPulseOpacity(0.1 + wave * 0.22);
      if (progress < 1) rafId = requestAnimationFrame(tickPulse);
    };

    rafId = requestAnimationFrame(tickPulse);
    return () => cancelAnimationFrame(rafId);
  }, [pickupLat, pickupLon]);

  return (
    <GoogleMap
      mapContainerStyle={CONTAINER_STYLE}
      center={center}
      zoom={12}
      options={HERO_MAP_OPTIONS}
      onLoad={handleMapLoad}
    >
      {routePath.length > 0 && <Polyline path={routePath} options={HERO_ROUTE_GLOW_OPTIONS} />}
      {routePath.length > 0 && <Polyline path={routePath} options={HERO_ROUTE_OUTLINE_OPTIONS} />}
      {routePath.length > 0 && (
        <Polyline path={routePath} options={{ ...HERO_ROUTE_OPTIONS, strokeColor: MAP_COLORS.brand, zIndex: 2 }} />
      )}
      {pickupCoord && (
        <Circle
          center={{ lat: pickupCoord.lat, lng: pickupCoord.lon }}
          radius={pickupPulseRadius}
          options={{
            fillColor: MAP_COLORS.brand,
            fillOpacity: pickupPulseOpacity * 0.35,
            strokeColor: MAP_COLORS.brand,
            strokeOpacity: pickupPulseOpacity,
            strokeWeight: 1.5,
            clickable: false,
            zIndex: 8,
          }}
        />
      )}
      {pickupCoord && (
        <GoogleMapsAdvancedMarker
          position={{ lat: pickupCoord.lat, lng: pickupCoord.lon }}
          icon={resolveLiriePointMarkerIcon(window.google?.maps, 'pickup')}
          title="Départ"
          zIndex={10}
        />
      )}
      {dropoffCoord && (
        <GoogleMapsAdvancedMarker
          position={{ lat: dropoffCoord.lat, lng: dropoffCoord.lon }}
          icon={resolveLiriePointMarkerIcon(window.google?.maps, 'dropoff')}
          title="Destination"
          zIndex={11}
        />
      )}
    </GoogleMap>
  );
}
