import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { GoogleMap, Marker, Polyline } from '@react-google-maps/api';
import { useGoogleMapsLoaded } from '../../components/common/GoogleMapsProvider';
import MapPlaceholder from '../../components/common/MapPlaceholder';
import {
  PUBLIC_MAP_OPTIONS,
  MAP_COLORS,
  makePinMarkerIcon,
  ROUTE_OPTIONS,
  ROUTE_OUTLINE_OPTIONS,
} from '../../utils/mapUtils';
import styles from './Home.module.css';
import AddressAutocomplete from '../../components/common/AddressAutocomplete';

const API_URL = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '/api/v1';

const CONTAINER_STYLE = { width: '100%', height: '100%' };

export default function Home() {
  const { isLoaded: gmLoaded } = useGoogleMapsLoaded();
  const [pickupText, setPickupText] = useState('');
  const [dropoffText, setDropoffText] = useState('');
  const [pickupCoord, setPickupCoord] = useState(null);
  const [dropoffCoord, setDropoffCoord] = useState(null);
  const [routeInfo, setRouteInfo] = useState(null);
  const [routePath, setRoutePath] = useState([]);
  const [platformStats, setPlatformStats] = useState(null);

  const mapRef = useRef(null);
  const center = useMemo(() => ({ lat: 46.2044, lng: 6.1432 }), []);

  const onMapLoad = useCallback((map) => {
    mapRef.current = map;
  }, []);

  const fitToContent = useCallback(() => {
    const map = mapRef.current;
    if (!map || !window.google) return;

    const points = [];
    if (pickupCoord) points.push({ lat: pickupCoord.lat, lng: pickupCoord.lon });
    if (dropoffCoord) points.push({ lat: dropoffCoord.lat, lng: dropoffCoord.lon });
    routePath.forEach((p) => points.push(p));

    if (points.length === 0) return;

    if (points.length === 1) {
      map.panTo(points[0]);
      map.setZoom(14);
      return;
    }

    const bounds = new window.google.maps.LatLngBounds();
    points.forEach((p) => bounds.extend(p));
    map.fitBounds(bounds, { top: 40, right: 40, bottom: 40, left: 40 });

    const listener = window.google.maps.event.addListenerOnce(map, 'idle', () => {
      if (map.getZoom() > 14) map.setZoom(14);
    });
    return () => window.google.maps.event.removeListener(listener);
  }, [pickupCoord, dropoffCoord, routePath]);

  const drawRoute = useCallback(
    async (a, b) => {
      if (!a || !b) return;
      setRoutePath([]);
      setRouteInfo(null);

      try {
        const params = new URLSearchParams({
          pickup_lat: a.lat,
          pickup_lon: a.lon,
          dropoff_lat: b.lat,
          dropoff_lon: b.lon,
        });
        const res = await fetch(`${API_URL}/osrm/route?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        if (data.route?.length) {
          setRoutePath(data.route.map(([lat, lng]) => ({ lat, lng })));
        }
        setRouteInfo({
          distanceKm: (data.distance / 1000).toFixed(1),
          durationMin: Math.round(data.duration / 60),
        });
      } catch (err) {
        console.warn('[Home] Route error:', err.message);
      }
    },
    []
  );

  useEffect(() => {
    if (pickupCoord && dropoffCoord) {
      drawRoute(pickupCoord, dropoffCoord);
    } else {
      setRoutePath([]);
      setRouteInfo(null);
    }
  }, [pickupCoord, dropoffCoord, drawRoute]);

  useEffect(() => {
    fetch(`${API_URL}/public/platform-stats`)
      .then((r) => r.ok ? r.json() : null)
      .then((data) => { if (data) setPlatformStats(data); })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fitToContent();
  }, [pickupCoord, dropoffCoord, routePath, fitToContent]);

  const handleSearch = () => {
    if (pickupCoord && dropoffCoord) {
      drawRoute(pickupCoord, dropoffCoord);
    }
  };

  return (
    <div className={styles.page}>
      {/* ── Hero Section ── */}
      <section className={styles.hero}>
        <div className={styles.heroInner}>
          <div className={styles.heroContent}>
            <span className={styles.heroBadge}>Transport médical & adapté</span>
            <h1 className={styles.heroTitle}>
              Déplacez-vous<br />en toute confiance
            </h1>
            <p className={styles.heroSubtitle}>
              Mobilité fiable et bienveillante en Suisse.
              Réservez votre course en quelques clics.
            </p>

            {/* ── Formulaire de réservation ── */}
            <form className={styles.bookingCard} onSubmit={(e) => e.preventDefault()}>
              <div className={styles.fieldGroup}>
                <div className={styles.fieldIcon}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="9" strokeDasharray="4 3"/></svg>
                </div>
                <AddressAutocomplete
                  placeholder="Lieu de prise en charge"
                  value={pickupText}
                  onChange={(e) => setPickupText(e.target.value)}
                  onSelect={(item) => {
                    setPickupText(item.label || '');
                    if (item.lat && item.lon) setPickupCoord({ lat: item.lat, lon: item.lon });
                  }}
                />
              </div>

              <div className={styles.fieldDivider} />

              <div className={styles.fieldGroup}>
                <div className={styles.fieldIcon}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/><circle cx="12" cy="9" r="2.5"/></svg>
                </div>
                <AddressAutocomplete
                  placeholder="Destination"
                  value={dropoffText}
                  onChange={(e) => setDropoffText(e.target.value)}
                  onSelect={(item) => {
                    setDropoffText(item.label || '');
                    if (item.lat && item.lon) setDropoffCoord({ lat: item.lat, lon: item.lon });
                  }}
                />
              </div>

              <div className={styles.dateRow}>
                <input type="date" className={styles.dateInput} />
                <input type="time" className={styles.dateInput} />
                <select className={styles.dateSelect} defaultValue="Départ">
                  <option value="Départ">Départ</option>
                  <option value="Arrivée">Arrivée</option>
                </select>
              </div>

              <button
                className={styles.ctaButton}
                type="button"
                onClick={handleSearch}
                disabled={!pickupCoord || !dropoffCoord}
              >
                Réserver maintenant
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
              </button>

            </form>
          </div>

          {/* ── Hero Visual — Composition ── */}
          <div className={styles.heroVisual}>
            {/* Carte — element principal */}
            <div className={styles.mapShowcase}>
              {gmLoaded ? (
                <GoogleMap
                  mapContainerStyle={CONTAINER_STYLE}
                  center={center}
                  zoom={12}
                  options={PUBLIC_MAP_OPTIONS}
                  onLoad={onMapLoad}
                >
                  {routePath.length > 0 && (
                    <Polyline
                      path={routePath}
                      options={ROUTE_OUTLINE_OPTIONS}
                    />
                  )}
                  {routePath.length > 0 && (
                    <Polyline
                      path={routePath}
                      options={{ ...ROUTE_OPTIONS, strokeColor: MAP_COLORS.brand, zIndex: 1 }}
                    />
                  )}
                  {pickupCoord && (
                    <Marker
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
                    <Marker
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
              ) : (
                <MapPlaceholder />
              )}

              {routeInfo && (
                <div className={styles.mapRouteOverlay}>
                  <div className={styles.overlayItem}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0"/><path d="M12 7v5l3 3"/></svg>
                    <span>{routeInfo.durationMin} min</span>
                  </div>
                  <div className={styles.overlaySep} />
                  <div className={styles.overlayItem}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6L6 18"/><path d="M8 6h10v10"/></svg>
                    <span>{routeInfo.distanceKm} km</span>
                  </div>
                </div>
              )}
            </div>

          </div>
        </div>
      </section>

      {/* ── Section Valeurs ── */}
      <section className={styles.valuesWrap}>
      <div className={styles.values}>
        <span className={styles.sectionTag}>Nos engagements</span>
        <h2 className={styles.valuesTitle}>Pourquoi choisir Lirie ?</h2>
        <p className={styles.valuesSubtitle}>Une mobilité pensée pour les personnes, pas seulement les trajets.</p>

        <div className={styles.valuesGrid}>
          <div className={styles.valueCard}>
            <div className={styles.valueIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>
            </div>
            <h3 className={styles.valueTitle}>Fiabilité</h3>
            <p className={styles.valueDesc}>
              Des partenaires de transport vérifiés et agréés. Chaque course est suivie en temps réel.
            </p>
          </div>

          <div className={styles.valueCard}>
            <div className={styles.valueIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            </div>
            <h3 className={styles.valueTitle}>Ponctualité</h3>
            <p className={styles.valueDesc}>
              Respect strict des horaires. Votre rendez-vous médical n'attend pas.
            </p>
          </div>

          <div className={styles.valueCard}>
            <div className={styles.valueIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
            </div>
            <h3 className={styles.valueTitle}>Accompagnement</h3>
            <p className={styles.valueDesc}>
              Transport adapté pour chaque besoin : mobilité réduite, accompagnement, confort.
            </p>
          </div>
        </div>
      </div>
      </section>

      {/* ── Comment ça marche ── */}
      <section className={styles.howItWorks}>
        <div className={styles.howInner}>
          <span className={styles.sectionTag}>Simple & rapide</span>
          <h2 className={styles.howTitle}>Comment ça marche ?</h2>
          <p className={styles.howSubtitle}>Réservez votre transport en 3 étapes simples.</p>

          <div className={styles.stepsGrid}>
            <div className={styles.step}>
              <div className={styles.stepNumber}><span>1</span></div>
              <div className={styles.stepConnector} aria-hidden="true" />
              <h3 className={styles.stepTitle}>Réservez en ligne</h3>
              <p className={styles.stepDesc}>
                Indiquez vos adresses de départ et d'arrivée, choisissez la date et l'heure qui vous conviennent.
              </p>
            </div>

            <div className={styles.step}>
              <div className={styles.stepNumber}><span>2</span></div>
              <div className={styles.stepConnector} aria-hidden="true" />
              <h3 className={styles.stepTitle}>Mise en relation</h3>
              <p className={styles.stepDesc}>
                Lirie vous connecte avec une entreprise de transport partenaire. Recevez les détails par notification.
              </p>
            </div>

            <div className={styles.step}>
              <div className={styles.stepNumber}><span>3</span></div>
              <h3 className={styles.stepTitle}>Voyagez sereinement</h3>
              <p className={styles.stepDesc}>
                Le transporteur arrive a l'heure convenue. Suivez votre trajet en temps réel et arrivez en toute tranquillité.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Statistiques (temps réel) ── */}
      <section className={styles.stats}>
        <div className={styles.statsInner}>
          {[
            { value: platformStats?.completedBookings, label: 'courses' },
            { value: platformStats?.activeCompanies, label: 'entreprises' },
            { value: platformStats?.activeDrivers, label: 'chauffeurs' },
            { value: platformStats?.citiesServed, label: 'villes' },
          ].map((s, i) => (
            <div key={i} className={styles.statItem}>
              <span className={styles.statNumber}>
                {s.value != null ? s.value.toLocaleString('fr-CH') : '—'}
              </span>
              <span className={styles.statLabel}>{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA Final ── */}
      <section className={styles.ctaSection}>
        <div className={styles.ctaInner}>
          <span className={styles.sectionTag}>Commencer</span>
          <h2 className={styles.ctaTitle}>Prêt à simplifier vos transports ?</h2>
          <p className={styles.ctaDesc}>
            Créez votre compte gratuitement et réservez votre premier transport en quelques minutes.
          </p>
          <div className={styles.ctaActions}>
            <a href="/register" className={styles.ctaPrimary}>Créer un compte</a>
            <a href="/contact" className={styles.ctaSecondary}>Nous contacter</a>
          </div>
        </div>
      </section>
    </div>
  );
}
