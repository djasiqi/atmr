// src/pages/Home/Home.jsx
import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import styles from './Home.module.css';
import AddressAutocomplete from '../../components/common/AddressAutocomplete';

// Corrige l’icône par défaut Leaflet (sinon elle n’apparaît pas dans les bundlers)
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const OSRM_BASE = process.env.REACT_APP_OSRM_BASE_URL || 'http://localhost:5001';
const OSRM_PROFILE = process.env.REACT_APP_OSRM_PROFILE || 'driving';

export default function Home() {
  // champs visibles
  const [pickupText, setPickupText] = useState('');
  const [dropoffText, setDropoffText] = useState('');

  // positions choisies
  const [pickupCoord, setPickupCoord] = useState(null); // {lat, lon}
  const [dropoffCoord, setDropoffCoord] = useState(null); // {lat, lon}

  // infos itinéraire
  const [routeInfo, setRouteInfo] = useState(null); // {distanceKm, durationMin}

  // Leaflet
  const mapRef = useRef(null);
  const mapElRef = useRef(null);
  const pickupMarkerRef = useRef(null);
  const dropoffMarkerRef = useRef(null);
  const routeLayerRef = useRef(null);

  const center = useMemo(() => [46.2044, 6.1432], []); // Genève

  // Init carte
  useEffect(() => {
    if (mapRef.current) return;
    mapRef.current = L.map(mapElRef.current, {
      center,
      zoom: 12,
    });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
    }).addTo(mapRef.current);
  }, [center]);

  // Dépose/maj un marker
  const setMarker = useCallback((which, lat, lon, label) => {
    const map = mapRef.current;
    if (!map) return;

    const ref = which === 'pickup' ? pickupMarkerRef : dropoffMarkerRef;
    if (ref.current) {
      ref.current.setLatLng([lat, lon]).setTooltipContent(label || which);
    } else {
      const m = L.marker([lat, lon]).addTo(map);
      m.bindTooltip(label || which, {
        permanent: true,
        direction: 'top',
        offset: [0, -24],
      }).openTooltip();
      ref.current = m;
    }
  }, []);

  // Ajuste la vue aux points/route
  const fitToContent = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const layers = [];

    if (pickupMarkerRef.current) layers.push(pickupMarkerRef.current);
    if (dropoffMarkerRef.current) layers.push(dropoffMarkerRef.current);
    if (routeLayerRef.current) layers.push(routeLayerRef.current);

    if (layers.length === 0) return;

    const group = L.featureGroup(layers);
    map.fitBounds(group.getBounds().pad(0.2));
    if (map.getZoom() > 14) map.setZoom(14);
  }, []);

  // Appel OSRM pour tracer un itinéraire quand 2 points sont définis
  const drawRoute = useCallback(
    async (a, b) => {
      if (!a || !b) return;
      // Nettoie ancienne couche
      if (routeLayerRef.current) {
        try {
          routeLayerRef.current.remove();
        } catch {}
        routeLayerRef.current = null;
      }
      setRouteInfo(null);

      const url = `${OSRM_BASE}/route/v1/${OSRM_PROFILE}/${a.lon},${a.lat};${b.lon},${b.lat}?overview=full&geometries=geojson`;
      try {
        const res = await fetch(url);
        const json = await res.json();
        if (json.code !== 'Ok' || !json.routes?.length) {
          console.warn('OSRM route KO:', json);
          return;
        }
        const route = json.routes[0];
        const coords = route.geometry.coordinates.map(([lon, lat]) => [lat, lon]);

        routeLayerRef.current = L.polyline(coords, { weight: 5 }).addTo(mapRef.current);

        setRouteInfo({
          distanceKm: (route.distance / 1000).toFixed(1),
          durationMin: Math.round(route.duration / 60),
        });

        fitToContent();
      } catch (e) {
        console.error('OSRM route error:', e);
      }
    },
    [fitToContent]
  );

  // Quand les coords changent → placer markers + calculer route
  useEffect(() => {
    if (pickupCoord) {
      setMarker('pickup', pickupCoord.lat, pickupCoord.lon, 'Départ');
    }
    if (dropoffCoord) {
      setMarker('dropoff', dropoffCoord.lat, dropoffCoord.lon, 'Destination');
    }
    if (pickupCoord && dropoffCoord) {
      drawRoute(pickupCoord, dropoffCoord);
    }
  }, [pickupCoord, dropoffCoord, setMarker, drawRoute]);

  // Gestion bouton
  const handleSearch = () => {
    if (pickupCoord && dropoffCoord) {
      drawRoute(pickupCoord, dropoffCoord);
    }
  };

  return (
    <div className={styles.container}>
      <main className={styles.main}>
        <div className={styles.leftSection}>
          <h1 className={styles.title}>Allez où vous voulez avec MonTransport</h1>

          <form className={styles.form} onSubmit={(e) => e.preventDefault()}>
            {/* Lieu de prise en charge */}
            <div className={styles.inputWrapper}>
              <AddressAutocomplete
                placeholder="Lieu de prise en charge"
                value={pickupText}
                onChange={(e) => setPickupText(e.target.value)}
                onSelect={(item) => {
                  // item: { label, lat, lon, ... }
                  setPickupText(item.label || '');
                  if (item.lat && item.lon) {
                    setPickupCoord({ lat: item.lat, lon: item.lon });
                  }
                }}
              />
            </div>

            {/* Destination */}
            <div className={styles.inputWrapper}>
              <AddressAutocomplete
                placeholder="Destination"
                value={dropoffText}
                onChange={(e) => setDropoffText(e.target.value)}
                onSelect={(item) => {
                  setDropoffText(item.label || '');
                  if (item.lat && item.lon) {
                    setDropoffCoord({ lat: item.lat, lon: item.lon });
                  }
                }}
              />
            </div>

            {/* Date, Heure et Option */}
            <div className={styles.dateTime}>
              <div className={styles.inputWrapper}>
                <input type="date" className={styles.input} />
              </div>
              <div className={styles.inputWrapper}>
                <input type="time" className={styles.input} />
              </div>
              <div className={styles.inputWrapper}>
                <select className={styles.select} defaultValue="Départ">
                  <option value="Départ">Départ</option>
                  <option value="Arrivée">Arrivée</option>
                </select>
              </div>
            </div>

            <button
              className={styles.primaryButton}
              type="button"
              onClick={handleSearch}
              disabled={!pickupCoord || !dropoffCoord}
            >
              Réserver maintenant
            </button>

            {routeInfo && (
              <div style={{ marginTop: 10, fontSize: 14 }}>
                Distance ≈ <b>{routeInfo.distanceKm} km</b> • Durée ≈{' '}
                <b>{routeInfo.durationMin} min</b>
              </div>
            )}
          </form>
        </div>

        <div className={styles.rightSection}>
          <div className={styles.mapPlaceholder}>
            <div ref={mapElRef} style={{ width: '100%', height: '100%', minHeight: 400 }} />
          </div>
        </div>

        {/* Suggestions */}
        <div className={styles.suggestionsSection}>
          <h2 className={styles.suggestionsTitle}>Suggestions</h2>
          <div className={styles.suggestionsContainer}>
            <div className={styles.suggestionCard}>
              <div className={styles.cardImage}>🚗</div>
              <h3 className={styles.cardTitle}>Course</h3>
              <p className={styles.cardDescription}>
                Allez où vous voulez avec MonTransport. Commandez une course en un clic et c'est
                parti !
              </p>
              <button className={styles.cardButton}>Détails</button>
            </div>
            <div className={styles.suggestionCard}>
              <div className={styles.cardImage}>🕒</div>
              <h3 className={styles.cardTitle}>Réserver</h3>
              <p className={styles.cardDescription}>
                Réservez votre course à l'avance pour pouvoir vous détendre le jour même.
              </p>
              <button className={styles.cardButton}>Détails</button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
