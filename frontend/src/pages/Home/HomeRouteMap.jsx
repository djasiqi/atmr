import React from 'react';
import { GoogleMap, Polyline } from '@react-google-maps/api';
import GoogleMapsAdvancedMarker from '../../components/common/GoogleMapsAdvancedMarker';
import {
  PUBLIC_MAP_OPTIONS,
  MAP_COLORS,
  resolveLiriePointMarkerIcon,
  ROUTE_OPTIONS,
  ROUTE_OUTLINE_OPTIONS,
} from '../../utils/mapUtils';

const CONTAINER_STYLE = { width: '100%', height: '100%' };

export default function HomeRouteMap({ center, onMapLoad, routePath, pickupCoord, dropoffCoord }) {
  return (
    <GoogleMap
      mapContainerStyle={CONTAINER_STYLE}
      center={center}
      zoom={12}
      options={PUBLIC_MAP_OPTIONS}
      onLoad={onMapLoad}
    >
      {routePath.length > 0 && <Polyline path={routePath} options={ROUTE_OUTLINE_OPTIONS} />}
      {routePath.length > 0 && (
        <Polyline path={routePath} options={{ ...ROUTE_OPTIONS, strokeColor: MAP_COLORS.brand, zIndex: 1 }} />
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
