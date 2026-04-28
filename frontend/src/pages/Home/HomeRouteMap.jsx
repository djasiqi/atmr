import React from 'react';
import { GoogleMap, Polyline } from '@react-google-maps/api';
import GoogleMapsAdvancedMarker from '../../components/common/GoogleMapsAdvancedMarker';
import {
  PUBLIC_MAP_OPTIONS,
  MAP_COLORS,
  makePinMarkerIcon,
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
          icon={{
            url: makePinMarkerIcon('pickup'),
            scaledSize: window.google ? new window.google.maps.Size(40, 52) : undefined,
            anchor: window.google ? new window.google.maps.Point(20, 46) : undefined,
          }}
          title="Départ"
          zIndex={10}
        />
      )}
      {dropoffCoord && (
        <GoogleMapsAdvancedMarker
          position={{ lat: dropoffCoord.lat, lng: dropoffCoord.lon }}
          icon={{
            url: makePinMarkerIcon('dropoff'),
            scaledSize: window.google ? new window.google.maps.Size(40, 52) : undefined,
            anchor: window.google ? new window.google.maps.Point(20, 46) : undefined,
          }}
          title="Destination"
          zIndex={11}
        />
      )}
    </GoogleMap>
  );
}
