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

// Illustrations stylisées : déposer les visuels dans public/images/home/audience/
// puis renseigner illustrationSrc (ex. '/images/home/audience-patient.webp').
const AUDIENCE_CARDS = [
  {
    id: 'patient',
    to: '/deplacez-vous',
    illustrationSrc: '/images/home/audience/audience-patient.png',
    illustrationAlt: 'Patient ou proche — transport médical accompagné',
    title: 'Patient ou PMR',
    description:
      'Rendez-vous médicaux, rééducation ou retour à domicile : comprendre la réservation et le suivi.',
    cta: 'Déplacez-vous',
    visualTone: 'patient',
    renderWatermark: () => (
      <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    ),
  },
  {
    id: 'transport',
    to: '/conduire',
    illustrationSrc: '/images/home/audience/audience-transport.png',
    illustrationAlt: 'Entreprise de transport médical — véhicule adapté',
    title: 'Entreprise de transport',
    description:
      "Missions, disponibilités et cadre d'engagement avec Lirie et les donneurs d'ordre.",
    cta: 'Conduire',
    visualTone: 'transport',
    renderWatermark: () => (
      <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="1" y="3" width="15" height="13" />
        <polygon points="16 8 20 8 23 11 23 16 16 16 16 8" />
        <circle cx="5.5" cy="18.5" r="2.5" />
        <circle cx="18.5" cy="18.5" r="2.5" />
      </svg>
    ),
  },
  {
    id: 'institution',
    to: '/professionnel',
    illustrationSrc: '/images/home/audience/audience-institution.png',
    illustrationAlt: 'Institution de santé — coordination des transports',
    title: 'Institution',
    description:
      'Coordination, traçabilité et déploiement : la page dédiée aux décideurs organisationnels.',
    cta: 'Professionnel',
    visualTone: 'institution',
    renderWatermark: () => (
      <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="2" width="16" height="20" />
        <path d="M9 22V12h6v10" />
        <line x1="9" y1="7" x2="9.01" y2="7" />
        <line x1="15" y1="7" x2="15.01" y2="7" />
      </svg>
    ),
  },
];

const AUDIENCE_VISUAL_TONE_CLASS = {
  patient: 'audienceVisualPatient',
  transport: 'audienceVisualTransport',
  institution: 'audienceVisualInstitution',
};

function blockAudienceImageSave(event) {
  event.preventDefault();
}

function AudienceCardVisual({ card }) {
  const toneClass = styles[AUDIENCE_VISUAL_TONE_CLASS[card.visualTone]];
  const visualClass = card.illustrationSrc ? styles[`audienceCardVisual_${card.id}`] : '';

  return (
    <div
      className={`${styles.audienceCardVisual} ${visualClass}`.trim()}
      onContextMenu={blockAudienceImageSave}
    >
      {card.illustrationSrc ? (
        <>
          <img
            src={card.illustrationSrc}
            alt={card.illustrationAlt}
            className={styles.audienceCardImage}
            loading="lazy"
            decoding="async"
            draggable={false}
            onDragStart={blockAudienceImageSave}
            onContextMenu={blockAudienceImageSave}
          />
          <span
            className={styles.audienceCardImageShield}
            aria-hidden="true"
            onContextMenu={blockAudienceImageSave}
          />
        </>
      ) : (
        <div className={`${styles.audienceVisualPlaceholder} ${toneClass}`} aria-hidden="true">
          <span className={styles.audienceVisualWatermark}>{card.renderWatermark()}</span>
        </div>
      )}
    </div>
  );
}

function IcoMapPinHero({ s = 12 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </svg>
  );
}

const PLATFORM_STATS_ITEMS = [
  { id: 'bookings', kind: 'number', valueKey: 'completedBookings', label: 'transports coordonnés' },
  { id: 'companies', kind: 'number', valueKey: 'activeCompanies', label: 'entreprises partenaires' },
  { id: 'institutions', kind: 'number', valueKey: 'activeInstitutions', label: 'institutions' },
  { id: 'availability', kind: 'text', displayValue: '24/7', label: 'plateforme disponible' },
];

const WHY_LIRIE_ITEMS = [
  {
    id: 'calls',
    title: 'Moins d’appels',
    description: 'Une demande remplace les allers-retours entre acteurs.',
    icon: 'calls',
  },
  {
    id: 'errors',
    title: 'Moins d’erreurs',
    description: 'Adresses, horaires et besoins sont saisis une fois, au même endroit.',
    icon: 'errors',
  },
  {
    id: 'shared',
    title: 'Une vision commune',
    description: 'Patients, institutions et transporteurs suivent le même parcours.',
    icon: 'shared',
  },
];

const LIRIE_ROLE_YES = [
  "Fournit l'outil de coordination et d'organisation des missions",
  'Vérifie les accréditations des transporteurs partenaires',
  'Met à disposition le suivi et la traçabilité aux acteurs habilités',
];

const LIRIE_ROLE_NO = [
  "N'exécute pas les prestations de transport sur la voie publique",
  "N'intervient pas en qualité de transporteur",
  'Ne remplace pas le jugement clinique ni la prise en charge médicale',
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
  const [bookingStep, setBookingStep] = useState(1);
  const [attemptedSubmit, setAttemptedSubmit] = useState(false);
  const [attemptedContinue, setAttemptedContinue] = useState(false);
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

  const fetchOsrmRoute = useCallback(async (a, b) => {
    if (!a || !b) return null;

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

      return {
        path: data.route?.length ? data.route.map(([lat, lng]) => ({ lat, lng })) : [],
        info: {
          distanceKm: (data.distance / 1000).toFixed(1),
          durationMin: Math.round(data.duration / 60),
        },
      };
    } catch (err) {
      console.warn('[Home] Route error:', err.message);
      return null;
    }
  }, []);

  const drawRoute = useCallback(
    async (a, b) => {
      if (!a || !b) return;
      setRoutePath([]);
      setRouteInfo(null);

      const result = await fetchOsrmRoute(a, b);
      if (!result) return;
      setRoutePath(result.path);
      setRouteInfo(result.info);
    },
    [fetchOsrmRoute]
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

  const missingRouteFields = useMemo(() => {
    const missing = [];
    if (!pickupCoord) missing.push('le lieu de prise en charge');
    if (!dropoffCoord) missing.push('la destination');
    return missing;
  }, [pickupCoord, dropoffCoord]);

  const missingScheduleFields = useMemo(() => {
    const missing = [];
    if (!travelDate) missing.push('la date');
    if (!isValidTime(travelTime)) missing.push("l'heure");
    return missing;
  }, [travelDate, travelTime, isValidTime]);

  const isRouteReady = missingRouteFields.length === 0;
  const isBookingReady = isRouteReady && missingScheduleFields.length === 0;
  const shouldShowRouteValidation = attemptedContinue && !isRouteReady;
  const shouldShowValidation = attemptedSubmit && !isBookingReady;
  const step1HelperText = isRouteReady
    ? 'Itinéraire prêt — poursuivez pour affiner votre demande.'
    : `Indiquez ${missingRouteFields.join(' et ')}.`;
  const step2HelperText = isBookingReady
    ? 'Parfait. Vous pouvez confirmer votre demande de réservation.'
    : `Pour réserver, renseignez ${missingScheduleFields.join(', ')}.`;
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

  const heroMission = useMemo(() => {
    if (!routeInfo) return null;
    return {
      status: 'Trajet visualisé',
      partner: 'Partenaire agréé',
      durationMin: routeInfo.durationMin,
    };
  }, [routeInfo]);

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

  const handleContinue = () => {
    setAttemptedContinue(true);
    if (!isRouteReady) return;
    setBookingStep(2);
  };

  const handleBackToRoute = useCallback(() => {
    setBookingStep(1);
    setAttemptedSubmit(false);
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
          <div className={styles.heroStage}>
            <div className={styles.heroContent}>
              <div className={styles.heroIntro}>
                <div className={styles.heroBadge}>
                  <IcoMapPinHero />
                  <span className={styles.heroBadgeLabel}>Transport médical &amp; accompagné · Suisse</span>
                </div>
                <h1 className={styles.heroTitle}>
                  <span className={styles.heroTitleLine}>Déplacez-vous</span>
                  <span className={styles.heroTitleAccent}> en toute confiance.</span>
                </h1>
                <div className={styles.heroTitleRule} aria-hidden />
                <p className={styles.heroTagline}>
                  Parce qu&apos;un transport médical est bien plus qu&apos;un simple trajet.
                </p>
                <p className={styles.heroSubtitle}>
                  Lirie coordonne vos trajets médicaux et accompagnés : partenaires vérifiés, demande guidée
                  et visualisation d&apos;itinéraire — la course est réalisée par une entreprise de transport habilitée.
                </p>
              </div>

              <div className={styles.heroProofs} role="list" aria-label="Garanties Lirie">
                <div className={styles.heroProofItem} role="listitem">
                  <span className={styles.heroProofMark} aria-hidden />
                  Transporteurs habilités &amp; assurés
                </div>
                <div className={styles.heroProofItem} role="listitem">
                  <span className={styles.heroProofMark} aria-hidden />
                  Suivi pour les personnes autorisées
                </div>
                <div className={styles.heroProofItem} role="listitem">
                  <span className={styles.heroProofMark} aria-hidden />
                  Conçu pour la Suisse romande
                </div>
              </div>

              <form
                className={`${styles.heroBookingForm} ${styles.bookingCard}`}
                onSubmit={(e) => e.preventDefault()}
              >
              <div className={styles.bookingHeader}>
                <p className={styles.bookingTitle}>
                  {bookingStep === 1 ? 'Où allez-vous ?' : 'Affinez votre demande'}
                </p>
                <p className={styles.bookingHint}>
                  {bookingStep === 1
                    ? 'Indiquez votre trajet pour visualiser le parcours et préparer votre demande.'
                    : 'Ces informations sont optionnelles — vous pouvez les compléter plus tard.'}
                </p>
              </div>

              <div className={styles.stepIndicator} aria-label={`Étape ${bookingStep} sur 2`}>
                <span className={`${styles.stepPill} ${bookingStep === 1 ? styles.stepPillActive : styles.stepPillDone}`}>
                  1. Trajet
                </span>
                <span className={styles.stepIndicatorSep} aria-hidden />
                <span className={`${styles.stepPill} ${bookingStep === 2 ? styles.stepPillActive : ''}`}>
                  2. Détails
                </span>
              </div>

              {bookingStep === 1 ? (
                <>
                  <div className={styles.fieldBlock}>
                    <label htmlFor="home-pickup" className={styles.fieldLabel}>Lieu de prise en charge</label>
                    <div className={`${styles.fieldGroup} ${shouldShowRouteValidation && !pickupCoord ? styles.fieldGroupInvalid : ''}`}>
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
                        aria-invalid={shouldShowRouteValidation && !pickupCoord}
                      />
                    </div>
                    {shouldShowRouteValidation && !pickupCoord && (
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
                    <div className={`${styles.fieldGroup} ${shouldShowRouteValidation && !dropoffCoord ? styles.fieldGroupInvalid : ''}`}>
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
                        aria-invalid={shouldShowRouteValidation && !dropoffCoord}
                      />
                    </div>
                    {shouldShowRouteValidation && !dropoffCoord && (
                      <p className={styles.fieldError}>Sélectionnez une destination valide.</p>
                    )}
                  </div>

                  {routeInfo && (
                    <div className={styles.routePreview} role="status">
                      <span>{routeInfo.distanceKm} km</span>
                      <span className={styles.routePreviewSep} aria-hidden />
                      <span>{routeInfo.durationMin} min</span>
                    </div>
                  )}

                  <button
                    className={styles.ctaButton}
                    type="button"
                    onClick={handleContinue}
                    disabled={!isRouteReady}
                    aria-describedby="home-booking-helper"
                  >
                    {isRouteReady ? 'Visualiser l\'itinéraire' : 'Préparer mon trajet'}
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
                  </button>
                  <p
                    id="home-booking-helper"
                    className={`${styles.bookingHelper} ${shouldShowRouteValidation ? styles.bookingHelperWarning : ''}`}
                    role="status"
                    aria-live="polite"
                  >
                    {step1HelperText}
                  </p>
                </>
              ) : (
                <>
                  <div className={styles.routeRecap}>
                    <button type="button" className={styles.routeRecapBack} onClick={handleBackToRoute}>
                      Modifier le trajet
                    </button>
                    <p className={styles.routeRecapText}>
                      <span className={styles.routeRecapLabel}>Départ</span>
                      {pickupText || '—'}
                    </p>
                    <p className={styles.routeRecapText}>
                      <span className={styles.routeRecapLabel}>Destination</span>
                      {dropoffText || '—'}
                    </p>
                    {routeInfo && (
                      <p className={styles.routeRecapMeta}>
                        {routeInfo.distanceKm} km · {routeInfo.durationMin} min
                      </p>
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
                    <p className={styles.mobilityHint}>Optionnel — modifiable à chaque réservation.</p>
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
                    {isBookingReady ? 'Estimer et réserver' : 'Continuer la demande'}
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
                  </button>
                  <p
                    id="home-booking-helper"
                    className={`${styles.bookingHelper} ${shouldShowValidation ? styles.bookingHelperWarning : ''}`}
                    role="status"
                    aria-live="polite"
                  >
                    {step2HelperText}
                  </p>
                </>
              )}

              </form>
            </div>

            <div className={styles.heroMapBackdrop}>
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
                <div className={styles.mapVignette} aria-hidden />
                <div className={styles.heroMapLogoMask} aria-hidden />
                <div className={styles.mapBrandBadge}>
                  <span className={styles.mapBrandDot} aria-hidden />
                  Lirie · Genève
                </div>
              </div>

              {heroMission && (
                <div className={styles.heroMissionStrip} role="status" aria-live="polite">
                  <p className={styles.heroMissionStatus}>{heroMission.status}</p>
                  {(heroMission.partner || heroMission.durationMin != null) && (
                    <p className={styles.heroMissionMeta}>
                      {[heroMission.partner, heroMission.durationMin != null ? `${heroMission.durationMin} min` : null]
                        .filter(Boolean)
                        .join(' · ')}
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ── Preuves chiffrées ── */}
      <section id="platform-stats" className={styles.stats} aria-labelledby="home-stats-title">
        <div className={styles.statsContent}>
          <div className={styles.statsHeader}>
            <h2 id="home-stats-title" className={styles.statsTitle}>
              Lirie aujourd&apos;hui
            </h2>
          </div>
          <div className={styles.statsInner}>
            {PLATFORM_STATS_ITEMS.map((item) => (
              <div key={item.id} className={styles.statItem}>
                <span className={styles.statNumber}>
                  {item.kind === 'text'
                    ? item.displayValue
                    : platformStats?.[item.valueKey] != null
                      ? Number(platformStats[item.valueKey]).toLocaleString('fr-CH')
                      : '—'}
                </span>
                <span className={styles.statLabel}>{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Phrase-pont ── */}
      <section className={styles.valueBridge} aria-label="Proposition de valeur">
        <p className={styles.valueBridgeText}>
          Une plateforme unique qui relie patients, institutions et entreprises de transport autour d&apos;un même parcours coordonné.
        </p>
      </section>

      {/* ── Comment ça marche ── */}
      <section id="how-it-works" className={styles.howItWorks}>
        <div className={styles.howInner}>
          <span className={styles.sectionTag}>Comment ça marche</span>
          <h2 className={styles.howTitle}>Trois étapes pour coordonner un transport</h2>
          <p className={styles.howSubtitle}>
            Vous décrivez le trajet. Lirie organise la mission. Un partenaire habilité réalise le transport.
          </p>

          <div className={styles.stepsGrid}>
            <div className={styles.step}>
              <div className={styles.stepNumber}><span>1</span></div>
              <div className={styles.stepConnector} aria-hidden="true" />
              <h3 className={styles.stepTitle}>Enregistrez la demande</h3>
              <p className={styles.stepDesc}>
                Adresses, créneau et besoins de mobilité : tout ce qu&apos;il faut pour préparer la mission.
              </p>
            </div>

            <div className={styles.step}>
              <div className={styles.stepNumber}><span>2</span></div>
              <div className={styles.stepConnector} aria-hidden="true" />
              <h3 className={styles.stepTitle}>Assignation partenaire</h3>
              <p className={styles.stepDesc}>
                Un transporteur habilité accepte selon zone, véhicule et disponibilité.
              </p>
            </div>

            <div className={styles.step}>
              <div className={styles.stepNumber}><span>3</span></div>
              <h3 className={styles.stepTitle}>Suivi jusqu&apos;à la dépose</h3>
              <p className={styles.stepDesc}>
                Les statuts restent accessibles aux personnes autorisées jusqu&apos;à la fin de course.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Pourquoi Lirie ── */}
      <section className={styles.valuesWrap} aria-labelledby="home-why-title">
        <div className={styles.values}>
          <span className={styles.sectionTag}>Pourquoi Lirie</span>
          <h2 id="home-why-title" className={styles.valuesTitle}>Ce que ça change pour vous</h2>
          <p className={styles.valuesSubtitle}>
            Ce que la coordination change au quotidien.
          </p>

          <div className={styles.valuesGrid}>
            {WHY_LIRIE_ITEMS.map((item) => (
              <div key={item.id} className={styles.valueCard}>
                <div className={styles.valueIcon} aria-hidden>
                  {item.icon === 'calls' && (
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z" />
                    </svg>
                  )}
                  {item.icon === 'errors' && (
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                      <path d="m9 12 2 2 4-4" />
                    </svg>
                  )}
                  {item.icon === 'shared' && (
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="2.5" />
                      <circle cx="5" cy="7" r="2" />
                      <circle cx="19" cy="7" r="2" />
                      <circle cx="12" cy="20" r="2" />
                      <path d="M6.7 8.5 10.2 11M17.3 8.5 13.8 11M12 14.5v3" />
                    </svg>
                  )}
                </div>
                <h3 className={styles.valueTitle}>{item.title}</h3>
                <p className={styles.valueDesc}>{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pour qui ── */}
      <section className={styles.audienceSection}>
        <div className={styles.audienceInner}>
          <span className={styles.sectionTag}>Pour qui</span>
          <h2 className={styles.audienceTitle}>Qui êtes-vous ?</h2>
          <p className={styles.audienceSubtitle}>
            Choisissez le parcours adapté à votre rôle.
          </p>
          <div className={styles.audienceGrid}>
            {AUDIENCE_CARDS.map((card) => (
              <Link key={card.id} to={card.to} className={styles.audienceCard}>
                <AudienceCardVisual card={card} />
                <div className={styles.audienceCardBody}>
                  <h3 className={styles.audienceCardTitle}>{card.title}</h3>
                  <p className={styles.audienceCardDesc}>{card.description}</p>
                  <span className={styles.audienceCardCta}>{card.cta}</span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ── Rôle de Lirie ── */}
      <section className={styles.roleSection} aria-labelledby="home-role-title">
        <div className={styles.roleInner}>
          <span className={styles.sectionTag}>Transparence</span>
          <h2 id="home-role-title" className={styles.roleTitle}>
            Le rôle exact de Lirie
          </h2>
          <p className={styles.roleSubtitle}>
            Ce que Lirie fait — et ne fait pas — pour clarifier les responsabilités de chacun.
          </p>

          <div className={styles.roleGrid}>
            <div className={styles.rolePanel}>
              <div className={styles.roleColHead}>
                <span className={styles.roleColIconYes} aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--brand-primary)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
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

            <div className={styles.rolePanel}>
              <div className={styles.roleColHead}>
                <span className={styles.roleColIconNo} aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--danger-dark)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
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

      {/* ── CTA Final ── */}
      <section className={styles.ctaSection}>
        <div className={styles.ctaInner}>
          <span className={styles.sectionTag}>Commencer</span>
          <h2 className={styles.ctaTitle}>Prêt à simplifier vos transports médicaux ?</h2>
          <p className={styles.ctaDesc}>
            Créez un compte pour réserver, ou écrivez-nous pour une mise en relation institutionnelle ou partenaire.
          </p>
          <p className={styles.ctaReassurance}>
            Création du compte gratuite · Sans engagement avant validation d&apos;une mission.
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
