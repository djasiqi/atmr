// src/pages/driver/components/Dashboard/DriverMap.jsx

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import myLocationIcon from "../../../../assets/icons/my-location.png";
import styles from "./DriverMap.module.css";

// --- Styles & constants ---
const defaultCenter = { lat: 46.8182, lng: 8.2275 }; // Suisse

// --------- Utils géo ----------
const toRad = (v) => (v * Math.PI) / 180;
// Distance Haversine (m)
function haversineDistance(a, b) {
  const R = 6371000;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.sin(dLng / 2) ** 2 * Math.cos(lat1) * Math.cos(lat2);
  const c = 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
  return R * c;
}

// Détecte si un tableau de coords est [lat,lng] (Leaflet) ou [lng,lat] (GeoJSON)
function normalizeToLatLngPairs(arr) {
  if (!Array.isArray(arr) || !arr.length) return [];
  const first = arr[0];
  if (!Array.isArray(first) || first.length < 2) return [];
  const looksLikeLatLng = Math.abs(first[0]) <= 90 && Math.abs(first[1]) <= 180;
  return looksLikeLatLng
    ? arr.map(([lat, lng]) => [lat, lng])
    : arr.map(([lng, lat]) => [lat, lng]);
}

// --------- Icônes Leaflet ----------
const myIcon = L.icon({
  iconUrl: myLocationIcon,
  iconSize: [40, 40],
  iconAnchor: [20, 20],
  popupAnchor: [0, -20],
});

const pickupDivIcon = L.divIcon({
  className: "pickup-marker",
  html:
    '<div style="width:22px;height:22px;border-radius:50%;background:#52c41a;border:2px solid #fff;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;">P</div>',
  iconSize: [22, 22],
  iconAnchor: [11, 11],
});

const dropoffDivIcon = L.divIcon({
  className: "dropoff-marker",
  html:
    '<div style="width:22px;height:22px;border-radius:50%;background:#f5222d;border:2px solid #fff;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;">D</div>',
  iconSize: [22, 22],
  iconAnchor: [11, 11],
});

// --------- Helper pour pan/fit ----------
const MapPanTo = ({ center }) => {
  const map = useMap();
  useEffect(() => {
    if (center) map.setView(center, map.getZoom(), { animate: true });
  }, [center, map]);
  return null;
};

const MapFitBounds = ({ bounds }) => {
  const map = useMap();
  useEffect(() => {
    if (bounds && bounds.length >= 2) {
      map.fitBounds(bounds, { padding: [32, 32] });
    }
  }, [bounds, map]);
  return null;
};

/**
 * DriverMap
 * - Affiche la position conducteur (throttle si immobile)
 * - Trace la/les routes des assignations en cours (a.route)
 * - Marqueurs Pickup/Dropoff avec popups
 * - Coloration du tronçon “actuel” et teinte retard si `delays[bookingId]` fourni
 *
 * Props attendues:
 *  - myLocation: { lat, lng }
 *  - assignments: [
 *      {
 *        id,
 *        is_on_trip: bool,
 *        route?: Array<[lat,lng]> | Array<[lng,lat]>,
 *        booking?: {
 *          id, status, pickup?: {lat,lng|longitude,latitude}, dropoff?: {...},
 *          pickup_time?, dropoff_time?, pickup_location?, dropoff_location?
 *        }
 *      }, ...
 *    ]
 *  - delays?: { [bookingId]: { delay_minutes:number, is_dropoff?:bool } }
 */
const DriverMap = ({ assignments = [], myLocation, delays = {} }) => {
  const [mapCenter, setMapCenter] = useState(defaultCenter);

  // Dernière position rendue (pour throttle)
  const lastRenderedPosRef = useRef(null);
  const lastUpdateTsRef = useRef(null);

  // Paramètres throttle
  const SPEED_THRESHOLD = 0.5; // m/s (~1.8 km/h)
  const STATIONARY_UPDATE_INTERVAL = 60_000; // 60s immobile

  // Position réellement affichée (throttlée)
  const [displayPos, setDisplayPos] = useState(null);

  // Center initial sur 1re position
  useEffect(() => {
    if (myLocation && !displayPos) {
      setDisplayPos(myLocation);
      setMapCenter(myLocation);
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
      const timeDelta = Math.max(0.001, (now - lastUpdateTsRef.current) / 1000); // s
      const speed = distance / timeDelta; // m/s

      if (speed < SPEED_THRESHOLD) {
        // immobile → n’actualise que toutes les STATIONARY_UPDATE_INTERVAL
        if (now - lastUpdateTsRef.current < STATIONARY_UPDATE_INTERVAL) {
          shouldUpdate = false;
        }
      }
    }

    const applyUpdate = () => {
      setDisplayPos(myLocation);
      setMapCenter(myLocation); // pan léger
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

    return () => {
      if (timerId) clearTimeout(timerId);
    };
  }, [myLocation]);

  // --------- Markers Pickup/Dropoff + polylines ----------
  // On accepte soit booking.pickup {lat,lng}, soit {latitude,longitude}
  const getLatLng = (pt) => {
    if (!pt) return null;
    if ("lat" in pt && "lng" in pt) return [pt.lat, pt.lng];
    if ("latitude" in pt && "longitude" in pt) return [pt.latitude, pt.longitude];
    if (Array.isArray(pt) && pt.length >= 2) return [pt[0], pt[1]];
    return null;
  };

  // Sélectionne les assignments actifs + normalise leur route
  const activeAssignments = useMemo(() => {
    return (assignments || []).filter((a) => a && a.is_on_trip);
  }, [assignments]);

  const routeLayers = useMemo(() => {
    return activeAssignments.map((a) => {
      const latLngs = normalizeToLatLngPairs(a.route || []);

      // Couleur par défaut
      let color = "#3388ff";

      // Si retard connu: jaune si <10min, rouge sinon. On distingue pickup vs dropoff
      const b = a.booking;
      if (b && delays && delays[b.id]) {
        const dl = delays[b.id];
        const late = Number(dl.delay_minutes || 0);
        if (late > 0) {
          color = late < 10 ? "#faad14" : "#f5222d";
        }
      }

      // Tronçon "actuel": on peut renforcer l’opacité + largeur
      const weight = 5;
      const opacity = 0.9;

      return (
        <Polyline
          key={`route-${a.id}`}
          positions={latLngs}
          pathOptions={{ color, weight, opacity }}
        />
      );
    });
  }, [activeAssignments, delays]);

  const poiMarkers = useMemo(() => {
    // Un marker P/D par booking actif
    const items = [];
    activeAssignments.forEach((a) => {
      const b = a.booking || {};
      const pickup = getLatLng(
        b.pickup ||
          b.pickup_location || // compat champ “location”
          (b.coordinates && b.coordinates.pickup)
      );
      const dropoff = getLatLng(
        b.dropoff ||
          b.dropoff_location ||
          (b.coordinates && b.coordinates.dropoff)
      );

      if (pickup) {
        items.push(
          <Marker key={`pickup-${a.id}`} position={pickup} icon={pickupDivIcon}>
            <Popup>
              <strong>Pickup</strong>
              {b.pickup_time ? <div>Heure: {fmtTime(b.pickup_time)}</div> : null}
              {b.client_name ? <div>Client: {b.client_name}</div> : null}
              {b.status ? <div>Statut: {b.status}</div> : null}
            </Popup>
          </Marker>
        );
      }
      if (dropoff) {
        items.push(
          <Marker
            key={`dropoff-${a.id}`}
            position={dropoff}
            icon={dropoffDivIcon}
          >
            <Popup>
              <strong>Dropoff</strong>
              {b.dropoff_time ? (
                <div>Heure: {fmtTime(b.dropoff_time)}</div>
              ) : null}
              {b.client_name ? <div>Client: {b.client_name}</div> : null}
              {b.status ? <div>Statut: {b.status}</div> : null}
            </Popup>
          </Marker>
        );
      }
    });
    return items;
  }, [activeAssignments]);

  // --------- Fit bounds auto sur POIs + route + position conducteur ----------
  const bounds = useMemo(() => {
    const pts = [];

    // position conducteur
    if (displayPos) pts.push([displayPos.lat, displayPos.lng]);

    // points des routes
    activeAssignments.forEach((a) => {
      const latLngs = normalizeToLatLngPairs(a.route || []);
      latLngs.forEach((p) => pts.push(p));
    });

    // POIs pickup/dropoff
    activeAssignments.forEach((a) => {
      const b = a.booking || {};
      const pickup = b.pickup || b.pickup_location;
      const dropoff = b.dropoff || b.dropoff_location;
      const p1 = getLatLng(pickup);
      const p2 = getLatLng(dropoff);
      if (p1) pts.push(p1);
      if (p2) pts.push(p2);
    });

    if (pts.length < 2) return null;

    // calcule [southWest, northEast]
    let minLat = 90,
      maxLat = -90,
      minLng = 180,
      maxLng = -180;
    pts.forEach(([lat, lng]) => {
      minLat = Math.min(minLat, lat);
      maxLat = Math.max(maxLat, lat);
      minLng = Math.min(minLng, lng);
      maxLng = Math.max(maxLng, lng);
    });
    return [
      [minLat, minLng],
      [maxLat, maxLng],
    ];
  }, [displayPos, activeAssignments]);

  return (
    <div className={styles.driverMapContainer}>
      <MapContainer center={mapCenter} zoom={12} className={styles.leafletMap}>
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Pan léger sur changements de position */}
        <MapPanTo center={mapCenter} />
        {/* FitBounds sur route/POIs/driver */}
        {bounds && <MapFitBounds bounds={bounds} />}

        {/* Position conducteur (throttlée) */}
        {displayPos && (
          <Marker position={[displayPos.lat, displayPos.lng]} icon={myIcon}>
            <Popup>Ma position</Popup>
          </Marker>
        )}

        {/* Routes en cours */}
        {routeLayers}

        {/* Markers pickup/dropoff */}
        {poiMarkers}
      </MapContainer>
    </div>
  );
};

// --------- petits helpers d’affichage ----------
function fmtTime(t) {
  try {
    const d = new Date(t);
    return `${String(d.getHours()).padStart(2, "0")}:${String(
      d.getMinutes()
    ).padStart(2, "0")}`;
  } catch {
    return t || "";
  }
}

export default React.memo(DriverMap);
