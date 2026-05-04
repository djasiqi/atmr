import React, { Suspense, useEffect, useMemo, useRef, useState, useCallback, lazy } from 'react';
import { Link } from 'react-router-dom';
import GoogleMapsProvider, { useGoogleMapsLoaded } from '../../components/common/GoogleMapsProvider';
import styles from './Home.module.css';
import institutionStyles from '../institution/Requests/InstitutionRequestForm.module.css';
import AddressAutocomplete from '../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../components/ui/InlineDatePicker';
import InlineTimePicker from '../../components/ui/InlineTimePicker';
import { getActivePublicId, getActiveUser, hasActiveSession } from '../../utils/webAuthSession';
import { getMobilityProfileForUser, saveLastMobilityProfile } from '../../utils/clientMobilityProfile';

const API_URL = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '/api/v1';
const HomeRouteMap = lazy(() => import('./HomeRouteMap'));
const pad2 = (n) => String(n).padStart(2, '0');
const DROP_OFF_COMMON_HINTS = [
  'Clinique des Hauts d\'Anières, Anières',
  'Clinique de Carouge, Carouge',
  'HUG Maternité, Genève',
  'Hôpital de La Tour, Meyrin',
  'Clinique Générale-Beaulieu, Genève',
];
const MOBILITY_OPTIONS = [
  { key: 'needsWheelchair', label: 'Fauteuil manuel' },
  { key: 'needsElectricWheelchair', label: 'Fauteuil électrique' },
  { key: 'needsWalkingAid', label: 'Aide à la marche' },
  { key: 'needsDoorToDoorAssistance', label: 'Porte-à-porte' },
];

function IcoMapPinHero({ s = 12 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </svg>
  );
}

const LIRIE_ROLE_YES = [
  "Fournit l'outil de coordination et d'organisation des missions",
  "Assure la plateforme, l'hébergement et la disponibilité du service",
  'Vérifie les accréditations des transporteurs partenaires',
  'Met à disposition le suivi de mission aux acteurs habilités',
  'Assure la traçabilité utile à la coordination des transports',
];

const LIRIE_ROLE_NO = [
  "N'exécute pas les prestations de transport sur la voie publique",
  "N'intervient pas en qualité de transporteur",
  'Ne remplace pas le jugement clinique ni les protocoles médicaux',
  'Ne participe pas à la prise en charge médicale des personnes transportées',
  "N'est pas partie aux contrats de transport entre institution et partenaire",
];
const formatIsoDate = (d) => `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
const formatHhMm = (d) => `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
const normalizeText = (v) =>
  (v || '')
    .toString()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();

function getImminentDefaults() {
  const now = new Date();
  const rounded = new Date(now);
  rounded.setSeconds(0, 0);
  const remainder = rounded.getMinutes() % 5;
  if (remainder !== 0) {
    rounded.setMinutes(rounded.getMinutes() + (5 - remainder));
  }
  return {
    date: formatIsoDate(rounded),
    time: formatHhMm(rounded),
  };
}

async function reverseGeocodeClientLocation(lat, lon) {
  try {
    const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${encodeURIComponent(
      lat
    )}&lon=${encodeURIComponent(lon)}&zoom=18&addressdetails=1&accept-language=fr-CH,fr`;
    const response = await fetch(url, {
      headers: {
        Accept: 'application/json',
      },
    });
    if (!response.ok) return '';
    const data = await response.json();
    return data?.display_name || '';
  } catch (_) {
    return '';
  }
}

function buildPlatformStatsUrls(baseUrl) {
  const trimmed = String(baseUrl || '/api/v1').replace(/\/+$/, '');
  if (trimmed.includes('/api/')) {
    return [`${trimmed}/public/platform-stats`];
  }
  return [
    `${trimmed}/api/v1/public/platform-stats`,
    '/api/v1/public/platform-stats',
    `${trimmed}/public/platform-stats`,
  ];
}

function buildOsrmRouteUrls(baseUrl) {
  const trimmed = String(baseUrl || '/api/v1').replace(/\/+$/, '');
  if (trimmed.includes('/api/')) {
    return [`${trimmed}/osrm/route`, '/api/v1/osrm/route', '/api/osrm/route'];
  }
  return [`${trimmed}/api/v1/osrm/route`, '/api/v1/osrm/route', '/api/osrm/route', `${trimmed}/osrm/route`];
}

function HomeInteractiveMap({ center, onMapLoad, routePath, pickupCoord, dropoffCoord }) {
  const { isLoaded: gmLoaded } = useGoogleMapsLoaded();

  if (!gmLoaded) {
    return null;
  }

  return (
    <Suspense fallback={null}>
      <HomeRouteMap
        center={center}
        onMapLoad={onMapLoad}
        routePath={routePath}
        pickupCoord={pickupCoord}
        dropoffCoord={dropoffCoord}
      />
    </Suspense>
  );
}

export default function Home() {
  const imminentDefaults = getImminentDefaults();
  const [pickupText, setPickupText] = useState('');
  const [dropoffText, setDropoffText] = useState('');
  const [pickupCoord, setPickupCoord] = useState(null);
  const [dropoffCoord, setDropoffCoord] = useState(null);
  const [travelDate, setTravelDate] = useState(imminentDefaults.date);
  const [travelTime, setTravelTime] = useState(imminentDefaults.time);
  const [tripKind, setTripKind] = useState('Départ');
  const [scheduleMode, setScheduleMode] = useState('imminent');
  const [isPickupLocating, setIsPickupLocating] = useState(false);
  const [pickupGeoSuggestion, setPickupGeoSuggestion] = useState(null);
  const [attemptedSubmit, setAttemptedSubmit] = useState(false);
  const [routeInfo, setRouteInfo] = useState(null);
  const [routePath, setRoutePath] = useState([]);
  const [platformStats, setPlatformStats] = useState(null);
  const [mobilityProfile, setMobilityProfile] = useState(() =>
    getMobilityProfileForUser({
      publicId: getActivePublicId(),
      email: getActiveUser()?.email,
    })
  );

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
        const candidates = buildOsrmRouteUrls(API_URL);
        let data = null;
        let lastStatus = null;
        for (const base of candidates) {
          const res = await fetch(`${base}?${params}`);
          if (!res.ok) {
            lastStatus = res.status;
            continue;
          }
          data = await res.json();
          break;
        }
        if (!data) throw new Error(`HTTP ${lastStatus || 404}`);

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
    let cancelled = false;
    let idleId = null;
    let timerId = null;

    const fetchStats = async () => {
      const candidates = buildPlatformStatsUrls(API_URL);
      for (const url of candidates) {
        try {
          const response = await fetch(url);
          if (!response.ok) continue;
          const data = await response.json();
          if (!cancelled && data) {
            setPlatformStats(data);
          }
          return;
        } catch (_) {
          // On tente le prochain endpoint candidat sans bruit console.
        }
      }
    };

    // Décale une requête non critique pour protéger le rendu initial/LCP.
    if (typeof window.requestIdleCallback === 'function') {
      idleId = window.requestIdleCallback(fetchStats, { timeout: 2500 });
    } else {
      timerId = window.setTimeout(fetchStats, 1000);
    }

    return () => {
      cancelled = true;
      if (idleId != null && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(idleId);
      }
      if (timerId != null) {
        window.clearTimeout(timerId);
      }
    };
  }, []);

  useEffect(() => {
    fitToContent();
  }, [pickupCoord, dropoffCoord, routePath, fitToContent]);

  const isValidTime = useCallback((v) => /^([01]\d|2[0-3]):([0-5]\d)$/.test(v), []);

  const missingFields = useMemo(() => {
    const missing = [];
    if (!pickupCoord) missing.push('le lieu de prise en charge');
    if (!dropoffCoord) missing.push('la destination');
    if (!travelDate) missing.push('la date');
    if (!isValidTime(travelTime)) missing.push("l'heure");
    return missing;
  }, [pickupCoord, dropoffCoord, travelDate, travelTime, isValidTime]);

  const isBookingReady = missingFields.length === 0;
  const shouldShowValidation = attemptedSubmit && !isBookingReady;
  const completionCount = useMemo(() => {
    let count = 0;
    if (pickupCoord) count += 1;
    if (dropoffCoord) count += 1;
    if (travelDate) count += 1;
    if (travelTime) count += 1;
    return count;
  }, [pickupCoord, dropoffCoord, travelDate, travelTime]);
  const progressPercent = Math.round((completionCount / 4) * 100);
  const helperText = isBookingReady
    ? 'Parfait. Vous pouvez maintenant afficher une estimation puis confirmer la réservation.'
    : `Pour continuer, renseignez ${missingFields.join(', ')}.`;
  const missingPrimaryLabel = useMemo(() => {
    if (isBookingReady) return 'Formulaire complet';
    const firstMissing = missingFields[0] || '';
    if (firstMissing.includes('destination')) return 'Destination manquante';
    if (firstMissing.includes('prise en charge')) return 'Départ manquant';
    if (firstMissing.includes('date')) return 'Date manquante';
    if (firstMissing.includes("heure")) return 'Heure manquante';
    return 'Information manquante';
  }, [isBookingReady, missingFields]);
  const formatProofStat = useCallback((value) => (value != null ? Number(value).toLocaleString('fr-CH') : '—'), []);
  const dropoffQuickSuggestions = useMemo(() => {
    const q = normalizeText(dropoffText).trim();
    if (q.length < 2) return [];
    return DROP_OFF_COMMON_HINTS
      .filter((label) => normalizeText(label).includes(q))
      .slice(0, 5)
      .map((label) => ({
        source: 'quick_dropoff',
        label,
        secondary_text: 'Suggestion rapide',
      }));
  }, [dropoffText]);

  const resolveAddressCoordinates = useCallback(async (label) => {
    try {
      const res = await fetch(`${API_URL}/geocode/autocomplete?q=${encodeURIComponent(label)}&limit=1`);
      if (!res.ok) return null;
      const data = await res.json();
      const first = Array.isArray(data) ? data[0] : null;
      if (!first || first.lat == null || first.lon == null) return null;
      return { lat: Number(first.lat), lon: Number(first.lon) };
    } catch (_) {
      return null;
    }
  }, []);

  const handleSearch = () => {
    setAttemptedSubmit(true);
    if (!isBookingReady) return;
    saveLastMobilityProfile(mobilityProfile);
    if (hasActiveSession()) {
      window.location.assign('/book/new');
      return;
    }
    window.location.assign('/login?next=%2Fbook%2Fnew');
  };

  const handleTripKindKeyDown = useCallback(
    (event) => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        setTripKind((prev) => (prev === 'Départ' ? 'Arrivée' : 'Départ'));
      }
    },
    []
  );

  const requestPickupFromGeolocation = useCallback(async () => {
    if (typeof navigator === 'undefined' || !navigator.geolocation || isPickupLocating) return;
    setIsPickupLocating(true);
    try {
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 12000,
          maximumAge: 60000,
        });
      });
      const lat = position?.coords?.latitude;
      const lon = position?.coords?.longitude;
      if (typeof lat !== 'number' || typeof lon !== 'number') return;

      const label = await reverseGeocodeClientLocation(lat, lon);
      const suggestedLabel = label || `Position actuelle (${lat.toFixed(5)}, ${lon.toFixed(5)})`;
      setPickupGeoSuggestion({
        source: 'geo_suggestion',
        label: suggestedLabel,
        secondary_text: 'Position proposée',
        lat,
        lon,
      });
    } catch (err) {
      // L'utilisateur peut refuser la permission, on garde un fallback manuel silencieux.
      console.info('[Home] Géolocalisation refusée ou indisponible:', err?.message || err);
    } finally {
      setIsPickupLocating(false);
    }
  }, [isPickupLocating]);

  useEffect(() => {
    const refreshMobilityProfile = () => {
      setMobilityProfile(
        getMobilityProfileForUser({
          publicId: getActivePublicId(),
          email: getActiveUser()?.email,
        })
      );
    };
    refreshMobilityProfile();
    window.addEventListener('auth-changed', refreshMobilityProfile);
    return () => {
      window.removeEventListener('auth-changed', refreshMobilityProfile);
    };
  }, []);

  useEffect(() => {
    saveLastMobilityProfile(mobilityProfile);
  }, [mobilityProfile]);

  const handleSelectDeparture = useCallback((withLocationProposal = true) => {
    setTripKind('Départ');
    if (withLocationProposal && !pickupCoord && !pickupText.trim()) {
      requestPickupFromGeolocation();
    }
  }, [pickupCoord, pickupText, requestPickupFromGeolocation]);

  const handleSelectImminentMode = useCallback(() => {
    setScheduleMode('imminent');
    handleSelectDeparture(false);
  }, [handleSelectDeparture]);

  const handlePickupFieldFocus = useCallback(() => {
    if (tripKind !== 'Départ') return;
    if (pickupCoord || pickupText.trim() || pickupGeoSuggestion) return;
    requestPickupFromGeolocation();
  }, [tripKind, pickupCoord, pickupText, pickupGeoSuggestion, requestPickupFromGeolocation]);

  const applyImminentSchedule = useCallback(() => {
    const defaults = getImminentDefaults();
    setTravelDate(defaults.date);
    setTravelTime(defaults.time);
    setTripKind('Départ');
  }, []);

  useEffect(() => {
    if (scheduleMode === 'imminent') {
      applyImminentSchedule();
    }
  }, [scheduleMode, applyImminentSchedule]);

  return (
    <div className={styles.page}>
      {/* ── Hero Section ── */}
      <section className={styles.hero}>
        <div className={styles.heroInner}>
          <div className={styles.heroTop}>
            <div className={styles.heroBadge}>
              <IcoMapPinHero />
              <span className={styles.heroBadgeLabel}>Transport médical &amp; accompagné · Suisse</span>
            </div>
            <h1 className={styles.heroTitle}>
              Déplacez-vous
              <span className={styles.heroTitleAccent}> en toute confiance.</span>
            </h1>
            <p className={styles.heroSubtitle}>
              Lirie coordonne vos trajets médicaux et accompagnés : partenaires vérifiés, demande guidée
              et estimation d&apos;itinéraire — la course est réalisée par une entreprise de transport habilitée.
            </p>
            <div className={styles.heroTrust} role="list">
              <div className={styles.heroTrustItem} role="listitem">
                <span className={styles.heroTrustDot} aria-hidden />
                Chauffeurs habilités &amp; assurés
              </div>
              <span className={styles.heroTrustSep} aria-hidden />
              <div className={styles.heroTrustItem} role="listitem">
                <span className={styles.heroTrustDot} aria-hidden />
                Suivi pour les personnes autorisées
              </div>
              <span className={styles.heroTrustSep} aria-hidden />
              <div className={styles.heroTrustItem} role="listitem">
                <span className={styles.heroTrustDot} aria-hidden />
                Conçu pour la Suisse romande
              </div>
            </div>
            <div className={styles.heroQuickLinks}>
              <Link to="/deplacez-vous" className={styles.heroQuickLink}>
                Patients &amp; proches
              </Link>
              <span className={styles.heroQuickSep} aria-hidden />
              <Link to="/conduire" className={styles.heroQuickLink}>
                Entreprises de transport
              </Link>
              <span className={styles.heroQuickSep} aria-hidden />
              <Link to="/professionnel" className={styles.heroQuickLink}>
                Institutions
              </Link>
            </div>
          </div>

          <div className={styles.heroMainGrid}>
            <div className={styles.bookingColumn}>
            {/* ── Formulaire de réservation ── */}
            <form className={styles.bookingCard} onSubmit={(e) => e.preventDefault()}>
              <div className={styles.bookingHeader}>
                <p className={styles.bookingTitle}>Demande rapide</p>
                <p className={styles.bookingHint}>
                  Renseignez le trajet pour une estimation de distance et de durée, puis poursuivre vers la réservation.
                </p>
              </div>
              <div className={styles.progressBlock}>
                <div className={styles.progressTop}>
                  <span className={styles.progressMissing}>{missingPrimaryLabel}</span>
                  <span className={styles.progressValue} aria-label={`Progression ${completionCount} sur 4`}>
                    {completionCount}/4
                  </span>
                </div>
                <div className={styles.progressTrack} aria-hidden="true">
                  <span className={styles.progressFill} style={{ width: `${progressPercent}%` }} />
                </div>
              </div>
              <div className={styles.fieldBlock}>
                <label htmlFor="home-pickup" className={styles.fieldLabel}>Lieu de prise en charge</label>
                <div className={`${styles.fieldGroup} ${shouldShowValidation && !pickupCoord ? styles.fieldGroupInvalid : ''}`}>
                  <div className={styles.fieldIcon}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="9" strokeDasharray="4 3"/></svg>
                  </div>
                  <AddressAutocomplete
                    name="pickup_location"
                    inputId="home-pickup"
                    placeholder="Ex: HUG, Rue Gabrielle-Perret-Gentil"
                    value={pickupText}
                    prependSuggestions={tripKind === 'Départ' && !pickupText.trim() && pickupGeoSuggestion ? [pickupGeoSuggestion] : []}
                    onFocus={handlePickupFieldFocus}
                    onChange={(e) => {
                      setPickupText(e.target.value);
                      setPickupCoord(null);
                    }}
                    onSelect={(item) => {
                      setPickupText(item.label || '');
                      if (item.lat && item.lon) setPickupCoord({ lat: item.lat, lon: item.lon });
                    }}
                    aria-invalid={shouldShowValidation && !pickupCoord}
                  />
                </div>
                {shouldShowValidation && !pickupCoord && (
                  <p className={styles.fieldError}>Sélectionnez une adresse valide de prise en charge.</p>
                )}
                {tripKind === 'Départ' && isPickupLocating && (
                  <div className={styles.locationAssistRow}>
                    <span className={styles.locationAssistHint} role="status" aria-live="polite">
                      Localisation en cours...
                    </span>
                  </div>
                )}
              </div>

              <div className={styles.fieldBlock}>
                <label htmlFor="home-dropoff" className={styles.fieldLabel}>Destination</label>
                <div className={`${styles.fieldGroup} ${shouldShowValidation && !dropoffCoord ? styles.fieldGroupInvalid : ''}`}>
                  <div className={styles.fieldIcon}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/><circle cx="12" cy="9" r="2.5"/></svg>
                  </div>
                  <AddressAutocomplete
                    name="dropoff_location"
                    inputId="home-dropoff"
                    placeholder="Ex: Clinique de Carouge"
                    minChars={2}
                    prependSuggestions={dropoffQuickSuggestions}
                    value={dropoffText}
                    onChange={(e) => {
                      setDropoffText(e.target.value);
                      setDropoffCoord(null);
                    }}
                    onSelect={async (item) => {
                      setDropoffText(item.label || '');
                      if (item.lat && item.lon) {
                        setDropoffCoord({ lat: item.lat, lon: item.lon });
                        return;
                      }
                      const resolved = await resolveAddressCoordinates(item.label || '');
                      if (resolved) {
                        setDropoffCoord(resolved);
                      } else {
                        setDropoffCoord(null);
                      }
                    }}
                    aria-invalid={shouldShowValidation && !dropoffCoord}
                  />
                </div>
                {shouldShowValidation && !dropoffCoord && (
                  <p className={styles.fieldError}>Sélectionnez une destination valide.</p>
                )}
              </div>

              <div className={`${styles.fieldBlock} ${styles.tripKindFieldScope}`}>
                <span id="home-schedule-mode-label" className={styles.fieldLabelCompact}>Type de départ</span>
                <div className={institutionStyles.missionSegment} role="radiogroup" aria-labelledby="home-schedule-mode-label">
                  <button
                    type="button"
                    className={`${institutionStyles.missionBtn} ${styles.tripKindBtn} ${scheduleMode === 'imminent' ? institutionStyles.missionBtnActive : ''}`}
                    onClick={handleSelectImminentMode}
                    role="radio"
                    aria-checked={scheduleMode === 'imminent'}
                  >
                    Départ imminent
                  </button>
                  <button
                    type="button"
                    className={`${institutionStyles.missionBtn} ${styles.tripKindBtn} ${scheduleMode === 'reservation' ? institutionStyles.missionBtnActive : ''}`}
                    onClick={() => setScheduleMode('reservation')}
                    role="radio"
                    aria-checked={scheduleMode === 'reservation'}
                  >
                    Définir la réservation
                  </button>
                </div>
              </div>

              <div className={styles.schedulePanel}>
                {scheduleMode === 'reservation' ? (
                <div className={styles.dateRow}>
                  <div className={styles.dateField}>
                    <label htmlFor="home-date" className={styles.fieldLabelCompact}>Date du trajet</label>
                    <InlineDatePicker
                      inputId="home-date"
                      value={travelDate}
                      onChange={setTravelDate}
                      className={styles.datePickerChip}
                      inputClassName={styles.datePickerChipInput}
                      invalid={shouldShowValidation && !travelDate}
                    />
                  </div>
                  <div className={styles.dateField}>
                    <label htmlFor="home-time" className={styles.fieldLabelCompact}>Heure du trajet</label>
                    <InlineTimePicker
                      inputId="home-time"
                      value={travelTime}
                      onChange={setTravelTime}
                      onSelectNow={(now) => {
                        setTravelDate(formatIsoDate(now));
                        handleSelectDeparture(false);
                        setScheduleMode('imminent');
                      }}
                      className={styles.timePickerChipInput}
                    />
                  </div>
                  <div className={`${styles.dateField} ${styles.tripKindFieldScope}`}>
                    <span id="home-trip-kind-label" className={styles.fieldLabelCompact}>Type de trajet</span>
                    <div
                      className={institutionStyles.missionSegment}
                      role="radiogroup"
                      aria-labelledby="home-trip-kind-label"
                    >
                      <button
                        type="button"
                        className={`${institutionStyles.missionBtn} ${styles.tripKindBtn} ${tripKind === 'Départ' ? institutionStyles.missionBtnActive : ''}`}
                        onClick={handleSelectDeparture}
                        onKeyDown={handleTripKindKeyDown}
                        role="radio"
                        aria-checked={tripKind === 'Départ'}
                      >
                        Départ
                      </button>
                      <button
                        type="button"
                        className={`${institutionStyles.missionBtn} ${styles.tripKindBtn} ${tripKind === 'Arrivée' ? institutionStyles.missionBtnActive : ''}`}
                        onClick={() => setTripKind('Arrivée')}
                        onKeyDown={handleTripKindKeyDown}
                        role="radio"
                        aria-checked={tripKind === 'Arrivée'}
                      >
                        Arrivée
                      </button>
                    </div>
                  </div>
                </div>
                ) : (
                  <p className={styles.scheduleHint}>Départ imminent activé : date et heure automatiques.</p>
                )}
              </div>

              <div className={styles.fieldBlock}>
                <span className={styles.fieldLabelCompact}>Besoins de mobilité</span>
                <p className={styles.mobilityHint}>
                  Préremplis depuis votre profil, modifiables à chaque réservation.
                </p>
                <div className={styles.mobilityChips}>
                  {MOBILITY_OPTIONS.map((option) => (
                    <button
                      key={option.key}
                      type="button"
                      className={`${styles.mobilityChip} ${mobilityProfile[option.key] ? styles.mobilityChipActive : ''}`}
                      aria-pressed={Boolean(mobilityProfile[option.key])}
                      onClick={() =>
                        setMobilityProfile((prev) => ({
                          ...prev,
                          [option.key]: !prev[option.key],
                        }))
                      }
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
                <div className={styles.mobilityRow}>
                  <input
                    type="text"
                    className={styles.mobilityInput}
                    placeholder="Niveau d'assistance (ex: aide à la montée)"
                    value={mobilityProfile.assistanceLevel || ''}
                    onChange={(e) =>
                      setMobilityProfile((prev) => ({ ...prev, assistanceLevel: e.target.value }))
                    }
                  />
                </div>
                <div className={styles.mobilityRow}>
                  <input
                    type="text"
                    className={styles.mobilityInput}
                    placeholder="Contact proche (nom + téléphone)"
                    value={mobilityProfile.emergencyContact || ''}
                    onChange={(e) =>
                      setMobilityProfile((prev) => ({ ...prev, emergencyContact: e.target.value }))
                    }
                  />
                </div>
                <div className={styles.mobilityRow}>
                  <input
                    type="text"
                    className={styles.mobilityInput}
                    placeholder="Notes utiles (étage, digicode, temps de préparation...)"
                    value={mobilityProfile.notes || ''}
                    onChange={(e) => setMobilityProfile((prev) => ({ ...prev, notes: e.target.value }))}
                  />
                </div>
              </div>

              <button
                className={styles.ctaButton}
                type="button"
                onClick={handleSearch}
                disabled={!isBookingReady}
                aria-describedby="home-booking-helper"
              >
                {isBookingReady ? 'Estimer et réserver' : 'Complétez le formulaire'}
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
              </button>
              <p
                id="home-booking-helper"
                className={`${styles.bookingHelper} ${shouldShowValidation ? styles.bookingHelperWarning : ''}`}
                role="status"
                aria-live="polite"
              >
                {helperText}
              </p>

              <div className={styles.bookingMeta}>
                <span className={styles.bookingMetaItem}>Estimation sans engagement</span>
                <span className={styles.bookingMetaItem}>Transporteurs vérifiés</span>
              </div>

            </form>
            </div>

          {/* ── Hero Visual — Composition ── */}
          <div className={styles.heroVisual}>
            {/* Carte — element principal */}
            <div className={styles.mapShowcase} role="presentation">
              <GoogleMapsProvider>
                <HomeInteractiveMap
                  center={center}
                  onMapLoad={onMapLoad}
                  routePath={routePath}
                  pickupCoord={pickupCoord}
                  dropoffCoord={dropoffCoord}
                />
              </GoogleMapsProvider>

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
        </div>
      </section>

      <section className={styles.trustBand} aria-label="Points forts du service">
        <div className={styles.trustBandInner}>
          {[
            { t: 'Partenaires vérifiés', d: 'Contrôles d\'habilitation et d\'assurance avant mise en relation.' },
            { t: 'Coordination structurée', d: 'Une demande unique, des statuts clairs pour les acteurs habilités.' },
            { t: 'Traçabilité', d: 'Historique des étapes utiles à la coordination institutionnelle.' },
            { t: 'Disponibilité', d: 'Réservation en ligne selon la disponibilité des partenaires sur votre zone.' },
          ].map(({ t, d }) => (
            <div key={t} className={styles.trustBandItem}>
              <div className={styles.trustBandDot} aria-hidden />
              <div>
                <div className={styles.trustBandTitle}>{t}</div>
                <p className={styles.trustBandDesc}>{d}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Section Valeurs ── */}
      <section className={styles.valuesWrap}>
      <div className={styles.values}>
        <span className={styles.sectionTag}>Nos engagements</span>
        <h2 className={styles.valuesTitle}>Pourquoi faire confiance à Lirie ?</h2>
        <p className={styles.valuesSubtitle}>
          Une coordination rigoureuse et humaine, du premier clic à la dépose — sans confondre outil et prestation de transport.
        </p>

        <div className={styles.valuesGrid}>
          <div className={styles.valueCard}>
            <div className={styles.valueIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>
            </div>
            <h3 className={styles.valueTitle}>Partenaires vérifiés</h3>
            <p className={styles.valueDesc}>
              Chaque entreprise de transport suit un processus de vérification avant d&apos;être proposée sur une mission.
            </p>
            <p className={styles.valueProof}>{formatProofStat(platformStats?.activeCompanies)} entreprises partenaires</p>
          </div>

          <div className={styles.valueCard}>
            <div className={styles.valueIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            </div>
            <h3 className={styles.valueTitle}>Coordination précise</h3>
            <p className={styles.valueDesc}>
              Lirie orchestre la demande, les créneaux et les statuts visibles par les personnes autorisées sur la mission.
            </p>
            <p className={styles.valueProof}>{formatProofStat(platformStats?.completedBookings)} courses coordonnées</p>
          </div>

          <div className={styles.valueCard}>
            <div className={styles.valueIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
            </div>
            <h3 className={styles.valueTitle}>Parcours sur mesure</h3>
            <p className={styles.valueDesc}>
              PMR, accompagnement, notes utiles : les besoins sont saisis dès la demande pour limiter les allers-retours.
            </p>
            <p className={styles.valueProof}>{formatProofStat(platformStats?.activeDrivers)} chauffeurs actifs</p>
          </div>
        </div>

        <div className={styles.valuesCta}>
          <a href="#how-it-works" className={styles.valuesCtaBtn}>Découvrir le parcours</a>
        </div>
      </div>
      </section>

      {/* ── Comment ça marche ── */}
      <section id="how-it-works" className={styles.howItWorks}>
        <div className={styles.howInner}>
          <span className={styles.sectionTag}>Simple &amp; rapide</span>
          <h2 className={styles.howTitle}>Comment Lirie coordonne votre transport</h2>
          <p className={styles.howSubtitle}>
            En trois étapes : vous décrivez le trajet, Lirie prépare la mission, le partenaire réalise le transport sur la voie publique.
          </p>
          <p className={styles.howCoordination}>
            Lirie est une <strong>plateforme de coordination</strong> — pas un transporteur. Les courses sont exécutées par des entreprises partenaires habilitées.
          </p>

          <div className={styles.stepsGrid}>
            <div className={styles.step}>
              <div className={styles.stepNumber}><span>1</span></div>
              <div className={styles.stepConnector} aria-hidden="true" />
              <h3 className={styles.stepTitle}>Enregistrez la demande</h3>
              <p className={styles.stepDesc}>
                Adresses validées, créneau (immédiat ou planifié), besoins de mobilité et notes utiles pour le départ ou l&apos;arrivée.
              </p>
            </div>

            <div className={styles.step}>
              <div className={styles.stepNumber}><span>2</span></div>
              <div className={styles.stepConnector} aria-hidden="true" />
              <h3 className={styles.stepTitle}>Assignation partenaire</h3>
              <p className={styles.stepDesc}>
                Un transporteur habilité accepte la mission selon zone, véhicule et disponibilité. Vous êtes informé des étapes prévues par le parcours Lirie.
              </p>
            </div>

            <div className={styles.step}>
              <div className={styles.stepNumber}><span>3</span></div>
              <h3 className={styles.stepTitle}>Mission suivie jusqu&apos;à la dépose</h3>
              <p className={styles.stepDesc}>
                Statuts et informations utiles restent accessibles aux personnes autorisées sur la mission, jusqu&apos;à confirmation de fin de course.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Rôle de Lirie (transparence) ── */}
      <section className={styles.roleSection} aria-labelledby="home-role-title">
        <div className={styles.roleInner}>
          <span className={styles.sectionTag}>Transparence</span>
          <h2 id="home-role-title" className={styles.roleTitle}>
            Le rôle exact de Lirie
          </h2>
          <p className={styles.roleSubtitle}>
            Comprendre ce que Lirie fait — et ne fait pas — clarifie les responsabilités de chacun sur une mission.
          </p>

          <div className={styles.roleCard}>
            <div className={styles.roleCol}>
              <div className={styles.roleColHead}>
                <span className={styles.roleColIconYes} aria-hidden>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </span>
                <span className={styles.roleColTitleYes}>Ce que Lirie assure</span>
              </div>
              <ul className={styles.roleList}>
                {LIRIE_ROLE_YES.map((item) => (
                  <li key={item} className={styles.roleItem}>
                    <span className={`${styles.roleItemGlyph} ${styles.roleItemGlyphYes}`} aria-hidden>
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    </span>
                    <span className={styles.roleItemText}>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className={styles.roleDivider} aria-hidden />

            <div className={styles.roleCol}>
              <div className={styles.roleColHead}>
                <span className={styles.roleColIconNo} aria-hidden>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--danger-dark)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </span>
                <span className={styles.roleColTitleNo}>Ce que Lirie ne fait pas</span>
              </div>
              <ul className={styles.roleList}>
                {LIRIE_ROLE_NO.map((item) => (
                  <li key={item} className={styles.roleItem}>
                    <span className={`${styles.roleItemGlyph} ${styles.roleItemGlyphNo}`} aria-hidden>
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="var(--danger-dark)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="16" y1="8" x2="8" y2="16" />
                        <line x1="8" y1="8" x2="16" y2="16" />
                      </svg>
                    </span>
                    <span className={styles.roleItemText}>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <p className={styles.roleLegal}>
            Pour le cadre juridique complet :{' '}
            <Link to="/conditions" className={styles.roleLegalLink}>
              Conditions générales d&apos;utilisation
            </Link>
            {' · '}
            <Link to="/mentions-legales" className={styles.roleLegalLink}>
              Mentions légales
            </Link>
          </p>
        </div>
      </section>

      <section className={styles.audienceSection}>
        <div className={styles.audienceInner}>
          <span className={styles.sectionTag}>Pour vous</span>
          <h2 className={styles.audienceTitle}>Le bon parcours selon votre rôle</h2>
          <p className={styles.audienceSubtitle}>
            Patient, proche ou institution : chaque page explique les étapes, les attentes et les contacts utiles.
          </p>
          <div className={styles.audienceGrid}>
            <Link to="/deplacez-vous" className={styles.audienceCard}>
              <div className={styles.audienceCardIcon} aria-hidden>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
              </div>
              <h3 className={styles.audienceCardTitle}>Patient ou PMR</h3>
              <p className={styles.audienceCardDesc}>
                Rendez-vous médicaux, rééducation ou retour à domicile : comprendre la réservation et le suivi.
              </p>
              <span className={styles.audienceCardCta}>Déplacez-vous</span>
            </Link>
            <Link to="/conduire" className={styles.audienceCard}>
              <div className={styles.audienceCardIcon} aria-hidden>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="1" y="3" width="15" height="13"/><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/></svg>
              </div>
              <h3 className={styles.audienceCardTitle}>Entreprise de transport</h3>
              <p className={styles.audienceCardDesc}>
                Missions, disponibilités et cadre d&apos;engagement avec Lirie et les donneurs d&apos;ordre.
              </p>
              <span className={styles.audienceCardCta}>Conduire</span>
            </Link>
            <Link to="/professionnel" className={styles.audienceCard}>
              <div className={styles.audienceCardIcon} aria-hidden>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="2" width="16" height="20"/><path d="M9 22V12h6v10"/><line x1="9" y1="7" x2="9.01" y2="7"/><line x1="15" y1="7" x2="15.01" y2="7"/></svg>
              </div>
              <h3 className={styles.audienceCardTitle}>Institution</h3>
              <p className={styles.audienceCardDesc}>
                Coordination, traçabilité et déploiement : la page dédiée aux décideurs organisationnels.
              </p>
              <span className={styles.audienceCardCta}>Professionnel</span>
            </Link>
          </div>
        </div>
      </section>

      {/* ── Statistiques (agrégées) ── */}
      <section className={styles.stats}>
        <div className={styles.statsContent}>
          <p className={styles.statsCaption}>Indicateurs issus de l&apos;activité plateforme (mise à jour régulière).</p>
          <div className={styles.statsInner}>
            {[
              { value: platformStats?.completedBookings, label: 'courses coordonnées' },
              { value: platformStats?.activeCompanies, label: 'entreprises partenaires' },
              { value: platformStats?.activeDrivers, label: 'chauffeurs actifs' },
              { value: platformStats?.cantonsServed, label: 'cantons desservis' },
            ].map((s, i) => (
              <div key={i} className={styles.statItem}>
                <span className={styles.statNumber}>
                  {s.value != null ? s.value.toLocaleString('fr-CH') : '—'}
                </span>
                <span className={styles.statLabel}>{s.label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Final ── */}
      <section className={styles.ctaSection}>
        <div className={styles.ctaInner}>
          <span className={styles.sectionTag}>Commencer</span>
          <h2 className={styles.ctaTitle}>Prêt à simplifier vos transports médicaux ?</h2>
          <p className={styles.ctaDesc}>
            Créez un compte pour réserver, ou écrivez-nous pour une mise en relation institutionnelle ou partenaire.
          </p>
          <div className={styles.ctaActions}>
            <a href="/login?mode=signup" className={styles.ctaPrimary}>Créer un compte</a>
            <a href="/contact" className={styles.ctaSecondary}>Nous contacter</a>
          </div>
          <p className={styles.ctaLegal}>
            <Link to="/conditions">Conditions d&apos;utilisation</Link>
            <span className={styles.ctaLegalSep} aria-hidden>
              ·
            </span>
            <Link to="/mentions-legales">Mentions légales</Link>
            <span className={styles.ctaLegalSep} aria-hidden>
              ·
            </span>
            <Link to="/privacy">Confidentialité</Link>
          </p>
        </div>
      </section>
    </div>
  );
}
