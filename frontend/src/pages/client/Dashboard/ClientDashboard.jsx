import React, { useEffect, useMemo, useState, useCallback, useRef } from 'react';
import apiClient from '../../../utils/apiClient';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import homeFieldStyles from '../../Home/Home.module.css';
import institutionStyles from '../../institution/Requests/InstitutionRequestForm.module.css';
import './ClientDashboard.css';
import { useMutation } from '@tanstack/react-query';
import { useHybridDataSync } from '../../../hooks/useHybridDataSync';
import { useClientBookingSocketRefresh } from '../../../hooks/useClientBookingSocketRefresh';

// Google Maps
import { GoogleMap, Polyline } from '@react-google-maps/api';
import { useGoogleMapsLoaded } from '../../../components/common/GoogleMapsProvider';
import GoogleMapsAdvancedMarker from '../../../components/common/GoogleMapsAdvancedMarker';
import {
  PUBLIC_MAP_OPTIONS,
  MAP_COLORS,
  makePinMarkerIcon,
  ROUTE_OPTIONS,
} from '../../../utils/mapUtils';
import polyline from '@mapbox/polyline';

// UI
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import { getApiErrorMessage } from '../../../utils/apiErrorMessage';
import { toast } from 'sonner';
import { toastSaferpayCheckoutError } from '../../../utils/saferpayPaymentUi';
import { readAndConsumeSaferpayPayResume } from '../../../utils/clientSaferpayPayResume';
import { startSaferpayHostedCheckout } from '../../../services/clientSaferpayPaymentService';
import {
  getClientBookingToneClass,
  getClientBookingUx,
  getEffectiveClientBookingActions,
  resolveClientBookingDisplayStatus,
} from '../../../utils/clientBookingUx';
import { trackClientKpiEvent } from '../../../utils/clientKpi';
import {
  getActiveAccessToken,
  getActivePublicId,
  hasActiveSession,
} from '../../../utils/webAuthSession';
import { requiresPrivateOnlinePaymentAtBooking } from '../../../utils/clientBookingPayment';
import {
  CLIENT_SURFACE_CONTRACTS,
  reportContractMismatch,
} from '../../../utils/clientSurfaceContracts';

const CONTAINER_STYLE = { width: '100%', height: '100%' };

const MISSING_ADDRESSES_MSG = 'Veuillez saisir le lieu de départ et la destination.';
const MAX_CLIENT_NOTE_LEN = 500;
/** Longueur max par ligne (départ / arrivée) pour rester sous la limite API une fois les libellés ajoutés. */
const MAX_CLIENT_NOTE_LEG = 230;

function buildClientNoteFromLegs(departureHint, arrivalHint) {
  const d = String(departureHint || '').trim();
  const a = String(arrivalHint || '').trim();
  if (!d && !a) return '';
  const parts = [];
  if (d) parts.push(`Prise en charge : ${d}`);
  if (a) parts.push(`Destination : ${a}`);
  return parts.join('\n').slice(0, MAX_CLIENT_NOTE_LEN);
}

/** Jours 0 = lundi … 6 = dimanche (aligné backend `recurrence_days`). */
const RECURRENCE_WEEK_DAYS = [
  { id: 0, short: 'L', label: 'Lundi' },
  { id: 1, short: 'Ma', label: 'Mardi' },
  { id: 2, short: 'Me', label: 'Mercredi' },
  { id: 3, short: 'J', label: 'Jeudi' },
  { id: 4, short: 'V', label: 'Vendredi' },
  { id: 5, short: 'S', label: 'Samedi' },
  { id: 6, short: 'D', label: 'Dimanche' },
];

/** Plancher affiché / envoyé à l’API pour l’indicatif client (CHF). */
const MIN_CLIENT_INDICATIVE_FARE_CHF = 45;

/**
 * Désactivé par défaut : toute l’indicative vient de POST /clients/me/indicative-fare/estimate.
 * N’activer (temporaire) qu’après ticket + retrait planifié — évite double logique locale/serveur.
 */
const LOCAL_INDICATIVE_FARE_FALLBACK_ENABLED =
  process.env.REACT_APP_CLIENT_INDICATIVE_FARE_LOCAL_FALLBACK === 'true' ||
  process.env.REACT_APP_CLIENT_INDICATIVE_FARE_LOCAL_FALLBACK === '1';

/** Message UX unique (web + mobile) pour indisponibilité côté configuration / estimation. */
export const INDICATIVE_FARE_UNAVAILABLE_UX =
  "L'estimation indicative est momentanément indisponible.";

/** Arrondi CHF au 5 centimes (rapen), ex. 48,26 → 48,25. */
function roundChfToFiveRappen(value) {
  const x = Number(value);
  if (!Number.isFinite(x)) return x;
  return Math.round((x + Number.EPSILON) * 20) / 20;
}

/** Forfait fixe dans la formule indicative (CHF). */
const INDICATIVE_BASE_CHF = 18;
/** Participation temps : CHF / minute (inchangée). */
const INDICATIVE_PER_MINUTE_CHF = 0.35;
/**
 * Point d’ancrage : à cette distance et cette durée, le brut = 45 CHF (ex. Anières → HUG ~13,5 km / ~20 min).
 * Le coefficient km est dérivé pour que base + km_ref×coef_km + min_ref×coef_min = 45.
 */
const INDICATIVE_REF_KM = 13.5;
const INDICATIVE_REF_MIN = 20;
const INDICATIVE_PER_KM_CHF =
  (MIN_CLIENT_INDICATIVE_FARE_CHF -
    INDICATIVE_BASE_CHF -
    INDICATIVE_REF_MIN * INDICATIVE_PER_MINUTE_CHF) /
  INDICATIVE_REF_KM;

/**
 * Devis indicatif (CHF) à partir du trajet OSRM (/ai/optimized-route).
 * Brut = base + coef_km×km + 0,35×min (coef_km calibré pour ~45 CHF à 13,5 km et 20 min).
 * Puis max(brut, 45 CHF). Au-delà de ce palier-distance, le montant suit le trajet réel.
 * — uniquement si le feature flag de repli explicite est actif côté build.
 * Pas un tarif contractuel (confirmé par le transporteur).
 */
function computeIndicativeFareChf(distanceM, durationS) {
  if (distanceM == null || Number.isNaN(distanceM) || distanceM <= 0) return null;
  const km = distanceM / 1000;
  const min = (durationS != null && !Number.isNaN(durationS) ? durationS : 0) / 60;
  const raw =
    INDICATIVE_BASE_CHF + km * INDICATIVE_PER_KM_CHF + min * INDICATIVE_PER_MINUTE_CHF;
  const clamped = Math.max(raw, MIN_CLIENT_INDICATIVE_FARE_CHF);
  return roundChfToFiveRappen(clamped);
}

function buildCustomerName(p) {
  if (!p) return 'Client';
  const fn = p.first_name || p.user?.first_name || '';
  const ln = p.last_name || p.user?.last_name || '';
  const n = `${fn} ${ln}`.trim();
  return n || p.user?.username || p.username || 'Client';
}

function homeAddressFromProfile(p) {
  if (!p) return '';
  const dom = p.domicile?.address ? String(p.domicile.address).trim() : '';
  const userAddress = p.user?.address ? String(p.user.address).trim() : '';
  return (dom || userAddress || p.address || p.domicile_address || p.billing_address || '').trim();
}

function formatBookingDate(value) {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) return 'Date inconnue';
  return new Date(parsed).toLocaleString('fr-FR', {
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** Libellé compact pour reprise de trajet (ligne unique date/heure). */
function formatTripResumeWhen(value) {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) return 'Date inconnue';
  const d = new Date(parsed);
  const datePart = d.toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
  const timePart = d.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
  return `${datePart} à ${timePart}`;
}

function formatScheduledSummaryLabel(dateStr, timeStr, asap) {
  if (asap) return 'Dès que possible (selon disponibilité des véhicules)';
  if (!dateStr || !timeStr) return '—';
  const d = new Date(`${dateStr}T${timeStr}:00`);
  if (!Number.isFinite(d.getTime())) return `${dateStr} à ${timeStr}`;
  const datePart = d.toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
  const timePart = d.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
  return `${datePart} · ${timePart}`;
}

/** `Date` → id jour backend `recurrence_days` (0 = lundi … 6 = dimanche). */
function jsDateToRecurrenceDayId(d) {
  return (d.getDay() + 6) % 7;
}

/**
 * Occurrences entre deux dates (inclus), plafonnées à 52 (contrainte API).
 * Sert d’estimation pour `recurrence_series_length` quand une date de fin est saisie.
 */
function estimatedOccurrencesForRecurrence({ startYmd, endYmd, recurrenceType, recurrenceDays }) {
  const start = new Date(`${startYmd}T12:00:00`);
  const end = new Date(`${endYmd}T12:00:00`);
  if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime()) || end < start) {
    return 1;
  }
  if (recurrenceType === 'daily') {
    const msPerDay = 24 * 60 * 60 * 1000;
    const n = Math.floor((end - start) / msPerDay) + 1;
    return Math.min(52, Math.max(1, n));
  }
  if (recurrenceType === 'weekly') {
    let n = 0;
    const cur = new Date(start);
    while (cur <= end) {
      n += 1;
      cur.setDate(cur.getDate() + 7);
    }
    return Math.min(52, Math.max(1, n));
  }
  if (recurrenceType === 'custom' && recurrenceDays.length > 0) {
    const set = new Set(recurrenceDays);
    let n = 0;
    const cur = new Date(start);
    while (cur <= end) {
      if (set.has(jsDateToRecurrenceDayId(cur))) n += 1;
      cur.setDate(cur.getDate() + 1);
    }
    return Math.min(52, Math.max(1, n));
  }
  return 1;
}

function formatRecurrenceYmdShort(ymd) {
  const d = new Date(`${String(ymd).trim()}T12:00:00`);
  if (!Number.isFinite(d.getTime())) return String(ymd || '').trim();
  return d.toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
}

function formatPrice(value) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return '-- CHF';
  return `${roundChfToFiveRappen(amount).toFixed(2)} CHF`;
}

function asBookingBool(value) {
  return value === true || value === 1 || value === '1' || value === 'true';
}

/** Pastille A/R ou retour (champs API `booking.serialize`). */
function getBookingTripKindMeta(booking) {
  if (!booking) return null;
  if (asBookingBool(booking.is_return)) {
    return { variant: 'return', label: 'Retour' };
  }
  if (asBookingBool(booking.is_round_trip) || asBookingBool(booking.has_return)) {
    return { variant: 'roundTrip', label: 'Aller-retour' };
  }
  return null;
}

function deriveAddressErrorMessage(error) {
  const raw = String(
    error?.response?.data?.message ||
      error?.response?.data?.error ||
      error?.message ||
      ''
  ).toLowerCase();
  if (raw.includes('hors zone') || raw.includes('outside') || raw.includes('zone')) {
    return 'Adresse hors zone desservie. Merci de contacter le support.';
  }
  if (raw.includes('imprécis') || raw.includes('imprecis') || raw.includes('ambig') || raw.includes('numéro')) {
    return 'Adresse imprécise. Merci de préciser le numéro.';
  }
  if (raw.includes('unknown') || raw.includes('introuvable') || raw.includes('not found')) {
    return 'Adresse inconnue. Merci de vérifier la saisie.';
  }
  const status = error?.response?.status;
  if (status === 429 || raw.includes('too many') || raw.includes('rate limit')) {
    return 'Trop de demandes d’itinéraire. Patientez quelques instants ou ajustez les adresses.';
  }
  return 'Impossible d’estimer ce trajet pour le moment. Vous pouvez tout de même envoyer votre demande.';
}

function extractPrimaryPlaceLabel(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  const firstSegment = raw
    .split(',')
    .map((part) => part.trim())
    .find(Boolean);
  return firstSegment || raw;
}

/** Destination clairement hôpital / clinique : l’établissement peut être déduit sans champ dupliqué. */
function isHospitalLikeDestination(value) {
  const lower = String(value || '').toLowerCase();
  return ['hôpital', 'hopital', 'hug', 'clinique', 'hospital', 'chuv'].some((k) => lower.includes(k));
}

function destinationHasDoctorHint(value) {
  const lower = String(value || '').toLowerCase();
  return ['docteur', 'dr', 'dr.', 'dr med', 'dr méd', 'médecin'].some((k) => lower.includes(k));
}

/** Normalise la destination pour repérer la plus utilisée dans l’historique. */
function normalizeRecentTripDestination(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ');
}

function isBookingCanceledForRecent(b) {
  const s = String(b?.status || '').toLowerCase();
  return s === 'canceled' || s === 'cancelled';
}

const ClientDashboard = () => {
  const { isLoaded: gmLoaded } = useGoogleMapsLoaded();
  const { id: clientId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const mapRef = useRef(null);

  const [profile, setProfile] = useState(null);
  const [loadingProfile, setLoadingProfile] = useState(true);
  const [upcomingBookings, setUpcomingBookings] = useState([]);
  const [ongoingBookings, setOngoingBookings] = useState([]);
  const [pastBookings, setPastBookings] = useState([]);
  const [loadError, setLoadError] = useState(null);
  const [formError, setFormError] = useState(null);
  const [payOfferBookingId, setPayOfferBookingId] = useState(null);
  const [payingSaferpay, setPayingSaferpay] = useState(false);
  const [bookingSubmitting, setBookingSubmitting] = useState(false);
  const [loadingBookings, setLoadingBookings] = useState(false);
  const [asapMode, setAsapMode] = useState(false);
  const [roundTripEnabled, setRoundTripEnabled] = useState(false);
  const [returnDate, setReturnDate] = useState('');
  const [returnTime, setReturnTime] = useState('');
  const [recurrenceEnabled, setRecurrenceEnabled] = useState(false);
  const [recurrenceType, setRecurrenceType] = useState('weekly');
  const [recurrenceSeriesLength, setRecurrenceSeriesLength] = useState(4);
  const [recurrenceEndDate, setRecurrenceEndDate] = useState('');
  const [recurrenceDays, setRecurrenceDays] = useState([]);
  const [isMobileViewport, setIsMobileViewport] = useState(false);
  const [estimateNotice, setEstimateNotice] = useState('');
  const [reservationFeedback, setReservationFeedback] = useState(null);
  const [payOffer, setPayOffer] = useState(null);
  const [pickup, setPickup] = useState('');
  const [destination, setDestination] = useState('');
  const [pickupSelection, setPickupSelection] = useState(null);
  const [destinationSelection, setDestinationSelection] = useState(null);
  const [routeLatLngs, setRouteLatLngs] = useState([]);
  /** Métriques itinéraire côté OSRM (carte uniquement) — jamais source du montant affiché sauf repli drapeau. */
  const [visualRouteMetrics, setVisualRouteMetrics] = useState(null);
  /**
   * Indicatif serveur (même moteur route que l’affichage carte).
   * Champs: distance_m, duration_s, indicative_amount_chf, config_version (ou erreur d’indispo).
   */
  const [indicativeServer, setIndicativeServer] = useState(null);
  const [indicativeServerLoading, setIndicativeServerLoading] = useState(false);
  const [indicativeUnavailability, setIndicativeUnavailability] = useState('');

  const [medicalFacility, setMedicalFacility] = useState('');
  const [doctorName, setDoctorName] = useState('');
  const [clientNoteDeparture, setClientNoteDeparture] = useState('');
  const [clientNoteArrival, setClientNoteArrival] = useState('');
  const [showMedicalFields, setShowMedicalFields] = useState(false);
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedTime, setSelectedTime] = useState('');

  const center = useMemo(() => ({ lat: 46.2044, lng: 6.1432 }), []);

  const effectiveClientId = useMemo(() => {
    return clientId || getActivePublicId();
  }, [clientId]);
  const accessToken = useMemo(() => getActiveAccessToken({ allowLegacy: true }), []);
  const authHeaders = useMemo(
    () => (accessToken ? { headers: { Authorization: `Bearer ${accessToken}` } } : undefined),
    [accessToken]
  );

  useEffect(() => {
    const saved = readAndConsumeSaferpayPayResume();
    if (!saved) return;
    setPayOfferBookingId(saved.bookingId);
    setPayOffer({
      bookingId: saved.bookingId,
      payerLabel: saved.payerLabel || 'Client',
      finalAmount: saved.finalAmount,
      paymentRequired: true,
      lifecycleLabel:
        saved.lifecycleLabel || getClientBookingUx('awaiting_client_payment').label,
      checkoutError: null,
    });
  }, []);

  /** Préremplissage départ / destination (ex. « Recommander » ou « Modifier » depuis Mes courses). */
  useEffect(() => {
    const pb = location.state?.prefillFromBooking;
    if (!pb || typeof pb !== 'object') return;
    const pu = String(pb.pickup_location || '').trim();
    const dd = String(pb.dropoff_location || '').trim();
    if (!pu && !dd) return;
    if (pu) setPickup(pu);
    if (dd) setDestination(dd);
    setFormError(null);
    navigate(location.pathname, { replace: true, state: null });
  }, [location.pathname, location.state, navigate]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (typeof window.matchMedia !== 'function') return;
    const media = window.matchMedia('(max-width: 768px)');
    if (!media || typeof media.matches !== 'boolean') return;
    const handleMedia = (event) => setIsMobileViewport(event.matches);
    setIsMobileViewport(media.matches);
    if (typeof media.addEventListener === 'function') {
      media.addEventListener('change', handleMedia);
      return () => media.removeEventListener('change', handleMedia);
    }
    if (typeof media.addListener === 'function') {
      media.addListener(handleMedia);
      return () => media.removeListener(handleMedia);
    }
    return undefined;
  }, []);

  useEffect(() => {
    const now = new Date();
    const oneHourLater = new Date(now.getTime() + 60 * 60 * 1000);
    oneHourLater.setMinutes(Math.ceil(oneHourLater.getMinutes() / 5) * 5, 0, 0);
    const defaultDate = oneHourLater.toISOString().split('T')[0];
    const defaultHours = String(oneHourLater.getHours()).padStart(2, '0');
    const defaultMinutes = String(oneHourLater.getMinutes()).padStart(2, '0');
    const defaultTime = `${defaultHours}:${defaultMinutes}`;
    setSelectedDate((prev) => prev || defaultDate);
    setSelectedTime((prev) => prev || defaultTime);
  }, []);

  const onMapLoad = useCallback((map) => {
    mapRef.current = map;
  }, []);

  // Fit bounds quand route change (marge gauche pour laisser la route lisible derrière le formulaire)
  useEffect(() => {
    if (routeLatLngs.length > 0 && mapRef.current && window.google) {
      const bounds = new window.google.maps.LatLngBounds();
      routeLatLngs.forEach(([lat, lng]) => bounds.extend({ lat, lng }));
      const w = typeof window !== 'undefined' ? window.innerWidth : 1200;
      const leftPad = Math.min(500, Math.round(w * 0.34) + 48);
      mapRef.current.fitBounds(bounds, { top: 56, right: 48, bottom: 56, left: leftPad });
    }
  }, [routeLatLngs]);

  // Profil client
  useEffect(() => {
    if (!hasActiveSession()) {
      navigate('/login');
      return;
    }
    if (!effectiveClientId) {
      setLoadingProfile(false);
      setLoadError('Identifiant client introuvable.');
      return;
    }
    setLoadingProfile(true);
    apiClient
      .get(`/clients/${effectiveClientId}`, authHeaders)
      .then((response) => setProfile(response.data))
      .catch((err) => {
        console.error('Erreur profil :', err);
        setLoadError('Impossible de charger le profil utilisateur.');
      })
      .finally(() => setLoadingProfile(false));
  }, [effectiveClientId, navigate, authHeaders]);

  // Mutation: optimisation d'itinéraire
  const { mutate: triggerOptimizeRoute } = useMutation({
    mutationFn: async () => {
      if (!pickup || !destination) return null;
      const response = await apiClient.post('/ai/optimized-route', {
        pickup,
        dropoff: destination,
      });
      return response.data;
    },
    onSuccess: (data) => {
      setEstimateNotice('');
      try {
        let latlngs = [];
        if (data?.polyline) {
          latlngs = polyline.decode(data.polyline).map(([lat, lng]) => [lat, lng]);
        } else if (data?.route?.polyline) {
          latlngs = polyline.decode(data.route.polyline).map(([lat, lng]) => [lat, lng]);
        } else if (Array.isArray(data?.route)) {
          latlngs = data.route;
        } else if (data?.route?.coordinates) {
          latlngs = data.route.coordinates.map(([lng, lat]) => [lat, lng]);
        } else if (data?.geometry?.coordinates) {
          latlngs = data.geometry.coordinates.map(([lng, lat]) => [lat, lng]);
        }

        if (!latlngs.length) throw new Error("Format d'itinéraire inconnu");
        setRouteLatLngs(latlngs);
        const dm = data?.distance_m ?? data?.distance_meters ?? data?.route?.distance_m;
        const ds = data?.duration_s ?? data?.duration_seconds ?? data?.route?.duration_s;
        if (dm != null && Number(dm) > 0) {
          setVisualRouteMetrics({ distance_m: Number(dm), duration_s: ds != null ? Number(ds) : 0 });
        } else {
          setVisualRouteMetrics(null);
        }
      } catch (e) {
        console.error('Parsing itinéraire:', e);
        setEstimateNotice(
          'Impossible d’estimer ce trajet pour le moment. Vous pouvez tout de même envoyer votre demande.'
        );
        setRouteLatLngs([]);
        setVisualRouteMetrics(null);
      }
    },
    onError: (error) => {
      setEstimateNotice(deriveAddressErrorMessage(error));
      setRouteLatLngs([]);
      setVisualRouteMetrics(null);
    },
  });

  // Ne pas dépendre de isOptimizing : à chaque fin de requête ça replanifiait un POST
  // après 2s (même adresses) → rafale et 429 côté API.
  useEffect(() => {
    if (!pickup || !destination) return;
    const t = setTimeout(() => {
      triggerOptimizeRoute();
    }, 2000);
    return () => clearTimeout(t);
  }, [pickup, destination, triggerOptimizeRoute]);

  // Indicatif CHF 100 % serveur (débounced comme la carte) — ne pas en déduire de /ai/optimized-route.
  useEffect(() => {
    if (!pickup || !destination) {
      setIndicativeServer(null);
      setIndicativeUnavailability('');
      setIndicativeServerLoading(false);
      return;
    }

    // Paire d'adresses modifiée : invalider l'indicatif précédent
    // (évite un montant obsolète pendant le debounce).
    setIndicativeServer(null);
    setIndicativeUnavailability('');

    const t = setTimeout(() => {
      (async () => {
        if (!authHeaders) {
          setIndicativeServer(null);
          setIndicativeUnavailability('');
          return;
        }
        setIndicativeServerLoading(true);
        setIndicativeUnavailability('');
        try {
          const response = await apiClient.post(
            '/clients/me/indicative-fare/estimate',
            { pickup_location: pickup, dropoff_location: destination },
            authHeaders
          );
          setIndicativeServer(response.data);
        } catch (err) {
          setIndicativeServer(null);
          const st = err?.response?.status;
          const code = err?.response?.data?.error;
          if (st === 412 && code === 'indicative_fare_disabled') {
            setIndicativeUnavailability(INDICATIVE_FARE_UNAVAILABLE_UX);
            return;
          }
          if (st === 503) {
            setIndicativeUnavailability(INDICATIVE_FARE_UNAVAILABLE_UX);
            return;
          }
          if (st === 400 && (code === 'indicative_fare_route_error' || code)) {
            setIndicativeUnavailability(INDICATIVE_FARE_UNAVAILABLE_UX);
            return;
          }
          setIndicativeUnavailability(INDICATIVE_FARE_UNAVAILABLE_UX);
        } finally {
          setIndicativeServerLoading(false);
        }
      })();
    }, 2000);
    return () => clearTimeout(t);
  }, [pickup, destination, authHeaders]);

  const toggleRecurrenceDay = useCallback((dayId) => {
    setRecurrenceDays((prev) =>
      prev.includes(dayId)
        ? prev.filter((d) => d !== dayId)
        : [...prev, dayId].sort((a, b) => a - b)
    );
  }, []);

  const handleSwapAddresses = useCallback(() => {
    setPickup(destination);
    setDestination(pickup);
    setPickupSelection(destinationSelection);
    setDestinationSelection(pickupSelection);
    setFormError(null);
  }, [pickup, destination, pickupSelection, destinationSelection]);

  const loadBookings = useCallback(
    async (quiet = false) => {
      if (!effectiveClientId) return null;
      if (!quiet) setLoadingBookings(true);
      try {
        const response = await apiClient.get(`/clients/${effectiveClientId}/bookings`, authHeaders);
        const bookingsArray = response.data;
        const now = Date.now();
        const ongoing = bookingsArray.filter((b) => {
          const status = String(b.status || '').toLowerCase();
          if (status === 'in_progress' || status === 'assigned') return true;
          const scheduledTime = Date.parse(b.scheduled_time);
          return Number.isFinite(scheduledTime) && Math.abs(scheduledTime - now) <= 90 * 60 * 1000;
        });
        const upcoming = bookingsArray.filter((b) => Date.parse(b.scheduled_time) > now);
        const past = bookingsArray.filter((b) => Date.parse(b.scheduled_time) <= now);
        setUpcomingBookings(upcoming);
        setPastBookings(past);
        setOngoingBookings(ongoing);
        setLoadError(null);
        return bookingsArray;
      } catch (err) {
        if (!quiet) {
          console.error('Erreur réservations :', err);
          setLoadError('Impossible de charger les réservations.');
        }
        throw err;
      } finally {
        if (!quiet) setLoadingBookings(false);
      }
    },
    [effectiveClientId, authHeaders]
  );

  // Réservations: snapshot HTTP initial
  useEffect(() => {
    if (!effectiveClientId) return;
    loadBookings(false).catch(() => {});
  }, [effectiveClientId, loadBookings]);

  // Fallback polling: re-synchronise l'état si aucune source live n'est disponible.
  useHybridDataSync({
    fetchFn: () => loadBookings(true),
    enabled: process.env.NODE_ENV !== 'test',
    staleThreshold: 120000,
    pollIntervalDisconnected: 45000,
    pollIntervalConnected: 180000,
    dependencies: [effectiveClientId],
  });

  useClientBookingSocketRefresh(loadBookings, Boolean(effectiveClientId));

  const nearestUpcomingBooking = useMemo(
    () =>
      [...upcomingBookings].sort(
        (a, b) => Date.parse(a.scheduled_time) - Date.parse(b.scheduled_time)
      )[0] || null,
    [upcomingBookings]
  );

  const nearestOngoingBooking = useMemo(() => {
    const now = Date.now();
    const valid = ongoingBookings
      .filter((b) => {
        const scheduled = Date.parse(b.scheduled_time);
        return Number.isFinite(scheduled) && scheduled >= now - 3 * 60 * 60 * 1000;
      })
      .sort((a, b) => Date.parse(a.scheduled_time) - Date.parse(b.scheduled_time));
    return valid[0] || null;
  }, [ongoingBookings]);

  const nextBooking = nearestOngoingBooking || nearestUpcomingBooking || null;
  const hasActiveOrFutureBooking = Boolean(nextBooking);
  /** Dernier trajet passé + trajet le plus récent vers la destination la plus fréquente (max 2, sans doublon). */
  const recentTrips = useMemo(() => {
    const list = pastBookings.filter(
      (b) => b && !isBookingCanceledForRecent(b) && String(b.dropoff_location || '').trim()
    );
    if (list.length === 0) return [];

    const sortedByTime = [...list].sort(
      (a, b) => Date.parse(b.scheduled_time) - Date.parse(a.scheduled_time)
    );
    const lastTrip = sortedByTime[0];

    /** @type {Map<string, { count: number, best: (typeof list)[0] }>} */
    const byDest = new Map();
    for (const b of list) {
      const key = normalizeRecentTripDestination(b.dropoff_location);
      if (!key) continue;
      const ts = Date.parse(b.scheduled_time);
      const cur = byDest.get(key);
      if (!cur) {
        byDest.set(key, { count: 1, best: b });
      } else {
        cur.count += 1;
        if (Number.isFinite(ts) && ts > Date.parse(cur.best.scheduled_time)) {
          cur.best = b;
        }
      }
    }

    let favoriteTrip = null;
    let bestCount = -1;
    let bestLatestTs = -Infinity;
    for (const { count, best } of byDest.values()) {
      const ts = Date.parse(best.scheduled_time);
      if (
        count > bestCount ||
        (count === bestCount && Number.isFinite(ts) && ts > bestLatestTs)
      ) {
        bestCount = count;
        favoriteTrip = best;
        bestLatestTs = Number.isFinite(ts) ? ts : bestLatestTs;
      }
    }

    const out = [];
    if (lastTrip) out.push(lastTrip);
    if (favoriteTrip && favoriteTrip.id !== lastTrip?.id) out.push(favoriteTrip);
    return out;
  }, [pastBookings]);
  const hasRecentTrips = recentTrips.length > 0;

  const todayDateMin = useMemo(() => new Date().toISOString().split('T')[0], []);

  /** Date du 1er départ (récurrent) : jour du trajet planifié, ou aujourd’hui si « dès que possible ». */
  const recurrenceStartYmd = useMemo(() => {
    if (!asapMode && selectedDate && String(selectedDate).trim()) {
      return String(selectedDate).trim();
    }
    return todayDateMin;
  }, [asapMode, selectedDate, todayDateMin]);

  const timeMinForSelectedDate = useMemo(() => {
    if (!selectedDate || selectedDate !== todayDateMin) return undefined;
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    return `${hh}:${mm}`;
  }, [selectedDate, todayDateMin]);

  const timeMinForReturn = useMemo(() => {
    if (!returnDate) return undefined;
    let minMinutes = null;
    if (returnDate === todayDateMin) {
      const now = new Date();
      minMinutes = now.getHours() * 60 + now.getMinutes();
    }
    if (!asapMode && selectedDate && selectedTime && returnDate === selectedDate) {
      const outbound = new Date(`${selectedDate}T${selectedTime}:00`);
      if (Number.isFinite(outbound.getTime())) {
        const afterOutbound = outbound.getHours() * 60 + outbound.getMinutes() + 1;
        minMinutes = minMinutes == null ? afterOutbound : Math.max(minMinutes, afterOutbound);
      }
    }
    if (minMinutes == null) return undefined;
    const hh = Math.floor(minMinutes / 60);
    const mm = minMinutes % 60;
    return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
  }, [returnDate, todayDateMin, asapMode, selectedDate, selectedTime]);

  useEffect(() => {
    if (!roundTripEnabled) {
      setReturnDate('');
      setReturnTime('');
      return;
    }
    setReturnDate((prev) => {
      if (prev && String(prev).trim()) return prev;
      if (selectedDate && String(selectedDate).trim()) return selectedDate;
      return todayDateMin;
    });
  }, [roundTripEnabled, selectedDate, todayDateMin]);

  useEffect(() => {
    if (!destination) {
      setShowMedicalFields(false);
      setMedicalFacility('');
      setDoctorName('');
      return;
    }
    const lower = destination.toLowerCase();
    const medicalKeywords = ['hôpital', 'hopital', 'hug', 'ems', 'cabinet', 'clinique', 'médecin', 'docteur'];
    const doctorKeywords = ['docteur', 'dr', 'dr.', 'dr med', 'dr méd', 'médecin'];

    const isMedicalFacility = medicalKeywords.some((k) => lower.includes(k));
    const isDoctor = doctorKeywords.some((k) => lower.includes(k));

    const primaryPlaceLabel = extractPrimaryPlaceLabel(destination);

    let showMedical = false;
    if (isDoctor) {
      setDoctorName(primaryPlaceLabel);
      showMedical = true;
    } else {
      setDoctorName('');
    }
    if (isMedicalFacility) {
      setMedicalFacility(primaryPlaceLabel);
      showMedical = true;
    } else {
      setMedicalFacility('');
    }
    setShowMedicalFields(showMedical);
  }, [destination]);

  const indicativeAmount = useMemo(() => {
    if (indicativeServer && typeof indicativeServer.indicative_amount_chf === 'number') {
      return indicativeServer.indicative_amount_chf;
    }
    if (LOCAL_INDICATIVE_FARE_FALLBACK_ENABLED) {
      const m = visualRouteMetrics;
      if (m?.distance_m) {
        return computeIndicativeFareChf(m.distance_m, m.duration_s);
      }
    }
    return null;
  }, [indicativeServer, visualRouteMetrics]);

  const serverIndicativeLineMetrics = useMemo(() => {
    if (indicativeServer?.distance_m != null && Number(indicativeServer.distance_m) > 0) {
      return {
        distance_m: Number(indicativeServer.distance_m),
        duration_s:
          indicativeServer.duration_s != null && Number.isFinite(Number(indicativeServer.duration_s))
            ? Number(indicativeServer.duration_s)
            : 0,
      };
    }
    if (LOCAL_INDICATIVE_FARE_FALLBACK_ENABLED && visualRouteMetrics?.distance_m) {
      return visualRouteMetrics;
    }
    return null;
  }, [indicativeServer, visualRouteMetrics]);

  /** Multiplicateur indicatif : date de fin prioritaire sur le nombre de répétitions. */
  const recurrenceSeriesMultiplier = useMemo(() => {
    if (!recurrenceEnabled) return 1;
    const end = String(recurrenceEndDate || '').trim();
    if (end) {
      return Math.max(
        1,
        estimatedOccurrencesForRecurrence({
          startYmd: recurrenceStartYmd,
          endYmd: end,
          recurrenceType,
          recurrenceDays,
        })
      );
    }
    const n = Math.min(52, Math.max(1, Math.floor(Number(recurrenceSeriesLength)) || 1));
    if (recurrenceType === 'custom' && recurrenceDays.length > 0) {
      return Math.max(1, n * recurrenceDays.length);
    }
    return Math.max(1, n);
  }, [
    recurrenceEnabled,
    recurrenceEndDate,
    recurrenceStartYmd,
    recurrenceSeriesLength,
    recurrenceType,
    recurrenceDays,
  ]);

  const indicativeAmountForDisplay = useMemo(() => {
    if (indicativeAmount == null) return null;
    let v = indicativeAmount;
    if (roundTripEnabled) v *= 2;
    if (recurrenceSeriesMultiplier > 1) v *= recurrenceSeriesMultiplier;
    return roundChfToFiveRappen(v);
  }, [indicativeAmount, roundTripEnabled, recurrenceSeriesMultiplier]);

  const sidebarEstimateLegal = useMemo(() => {
    if (indicativeAmount == null) return '';
    const chunks = [];
    if (roundTripEnabled) chunks.push('aller + retour (×2)');
    if (recurrenceEnabled) {
      chunks.push(
        recurrenceSeriesMultiplier > 1
          ? `série décrite (×${recurrenceSeriesMultiplier})`
          : 'série indiquée'
      );
    }
    const tail =
      " Indicatif, non contractuel. Le prix final est confirmé à la prévisualisation (avant demande de transport).";
    if (!chunks.length) {
      return `Indicatif avant validation transporteur.${tail}`;
    }
    return `Indicatif : ${chunks.join(' · ')}, ordre de grandeur${tail}`;
  }, [
    indicativeAmount,
    roundTripEnabled,
    recurrenceEnabled,
    recurrenceSeriesMultiplier,
  ]);

  const handleBooking = async () => {
    if (bookingSubmitting) return;
    const token = getActiveAccessToken({ allowLegacy: true });
    setFormError(null);
    setReservationFeedback(null);
    setPayOfferBookingId(null);
    setPayOffer(null);
    if (!token && !hasActiveSession()) {
      setFormError("Token d'authentification manquant.");
      return;
    }
    if (!pickup || !destination) {
      setFormError(MISSING_ADDRESSES_MSG);
      return;
    }
    trackClientKpiEvent('reserve_cta_clicked', {
      clientPublicId: effectiveClientId,
      asapMode,
      roundTrip: roundTripEnabled,
    });
    if (!asapMode && (!selectedDate || !selectedTime)) {
      setFormError('Veuillez sélectionner une date et une heure.');
      return;
    }

    /** ISO UTC pour l’API (évite le double décalage getTimezoneOffset). */
    let scheduledTimeIso = null;
    let outboundMs = Date.now();
    if (!asapMode) {
      const scheduledDateTime = new Date(`${selectedDate}T${selectedTime}:00`);
      if (!Number.isFinite(scheduledDateTime.getTime())) {
        setFormError('Date/heure invalide.');
        return;
      }
      if (scheduledDateTime.getTime() < Date.now() - 60 * 1000) {
        setFormError('Veuillez choisir une date et une heure futures.');
        return;
      }
      scheduledTimeIso = scheduledDateTime.toISOString();
      outboundMs = scheduledDateTime.getTime();
    }

    let returnTimeIso = null;
    if (roundTripEnabled) {
      if (!returnDate || !String(returnDate).trim()) {
        setFormError(
          'Pour un aller-retour, indiquez au moins la date de retour (l’heure peut rester à définir).'
        );
        return;
      }
      const outboundDateStr = !asapMode && selectedDate ? selectedDate : todayDateMin;
      if (String(returnDate).trim() < String(outboundDateStr).trim()) {
        setFormError('La date de retour ne peut pas être antérieure au départ prévu.');
        return;
      }
      if (returnTime && String(returnTime).trim()) {
        const returnDateTime = new Date(`${returnDate}T${returnTime}:00`);
        if (!Number.isFinite(returnDateTime.getTime())) {
          setFormError('Date/heure de retour invalides.');
          return;
        }
        if (returnDateTime.getTime() <= outboundMs) {
          setFormError('L’heure de retour doit être après le départ prévu.');
          return;
        }
        if (returnDateTime.getTime() < Date.now() - 60 * 1000) {
          setFormError('Choisissez une heure de retour dans le futur.');
          return;
        }
        returnTimeIso = returnDateTime.toISOString();
      }
    }

    const recurrenceEndTrim = recurrenceEndDate.trim();
    const recurrenceStartForSeries = !asapMode && selectedDate ? selectedDate : todayDateMin;

    if (recurrenceEnabled) {
      if (recurrenceType === 'custom' && recurrenceDays.length === 0) {
        setFormError('Pour des jours personnalisés, sélectionnez au moins un jour de la semaine.');
        return;
      }
      if (recurrenceEndTrim) {
        if (recurrenceEndTrim < recurrenceStartForSeries) {
          setFormError('La date de fin de série ne peut pas précéder la date du premier départ.');
          return;
        }
      } else {
        const rep = Math.min(52, Math.max(1, Math.floor(Number(recurrenceSeriesLength)) || 0));
        if (!rep) {
          setFormError('Indiquez le nombre de répétitions (1 à 52), ou choisissez une date de fin de série.');
          return;
        }
      }
    }

    /** Même logique que `indicativeAmountForDisplay` / encadré « Estimation transport » (total indicatif payé). */
    const baseAmount = indicativeAmount != null ? indicativeAmount : MIN_CLIENT_INDICATIVE_FARE_CHF;
    let amountCalc = baseAmount * (roundTripEnabled ? 2 : 1);
    if (recurrenceEnabled && recurrenceSeriesMultiplier > 1) {
      amountCalc *= recurrenceSeriesMultiplier;
    }
    const amountForApi = roundChfToFiveRappen(amountCalc);
    const seriesLen = recurrenceEndTrim
      ? estimatedOccurrencesForRecurrence({
          startYmd: recurrenceStartForSeries,
          endYmd: recurrenceEndTrim,
          recurrenceType,
          recurrenceDays,
        })
      : Math.min(52, Math.max(1, Math.floor(Number(recurrenceSeriesLength)) || 1));

    const clientNotePayload = buildClientNoteFromLegs(clientNoteDeparture, clientNoteArrival);

    const bookingData = {
      customer_name: buildCustomerName(profile),
      pickup_location: pickup,
      dropoff_location: destination,
      scheduled_time: scheduledTimeIso,
      asap: asapMode,
      amount: amountForApi,
      medical_facility: medicalFacility,
      doctor_name: doctorName,
      ...(clientNotePayload ? { client_note: clientNotePayload } : {}),
      is_round_trip: roundTripEnabled,
      ...(roundTripEnabled && returnDate && String(returnDate).trim()
        ? { return_date: String(returnDate).trim() }
        : {}),
      ...(returnTimeIso ? { return_time: returnTimeIso } : {}),
      is_recurring: recurrenceEnabled,
      ...(recurrenceEnabled
        ? {
            recurrence_type: recurrenceType,
            recurrence_series_length: seriesLen,
            ...(recurrenceEndTrim ? { recurrence_end_date: recurrenceEndTrim } : {}),
            ...(recurrenceType === 'custom' && recurrenceDays.length > 0
              ? { recurrence_days: [...recurrenceDays] }
              : {}),
          }
        : {}),
    };

    setBookingSubmitting(true);
    try {
      const previewPayload = {
        ...bookingData,
      };
      delete previewPayload.customer_name;
      const previewResponse = await apiClient.post('/clients/me/bookings/preview', previewPayload, {
        headers: { 'Content-Type': 'application/json' },
      });
      const previewRoot = previewResponse.data || {};
      const previewContracts = previewRoot.contracts || {};
      if (
        previewContracts.status_dictionary_version &&
        previewContracts.status_dictionary_version !==
          CLIENT_SURFACE_CONTRACTS.statusDictionaryVersion
      ) {
        reportContractMismatch({
          contract: 'status',
          expected: CLIENT_SURFACE_CONTRACTS.statusDictionaryVersion,
          received: previewContracts.status_dictionary_version,
        });
      }
      if (
        previewContracts.pricing_contract_version &&
        previewContracts.pricing_contract_version !==
          CLIENT_SURFACE_CONTRACTS.pricingContractVersion
      ) {
        reportContractMismatch({
          contract: 'pricing',
          expected: CLIENT_SURFACE_CONTRACTS.pricingContractVersion,
          received: previewContracts.pricing_contract_version,
        });
      }
      if (
        previewContracts.canonical_address_contract_version &&
        previewContracts.canonical_address_contract_version !==
          CLIENT_SURFACE_CONTRACTS.canonicalAddressContractVersion
      ) {
        reportContractMismatch({
          contract: 'canonical_address',
          expected: CLIENT_SURFACE_CONTRACTS.canonicalAddressContractVersion,
          received: previewContracts.canonical_address_contract_version,
        });
      }
      const previewPricing = previewRoot.pricing || {};
      const previewCanonical = previewRoot.canonical_addresses || {};
      const previewWorkflow = previewRoot.workflow || {};
      const canonicalPickup = previewCanonical.pickup?.label || bookingData.pickup_location;
      const canonicalDropoff = previewCanonical.dropoff?.label || bookingData.dropoff_location;
      const previewAmount = Number(previewPricing.amount);
      if (!Number.isFinite(previewAmount) || previewAmount <= 0) {
        setFormError('Prévisualisation tarifaire indisponible. Réessayez dans quelques instants.');
        return;
      }
      const blockedPrecisionLevels = new Set(['locality', 'approximate']);
      const pickupPrecision = String(previewCanonical.pickup?.precision_level || '').toLowerCase();
      const dropoffPrecision = String(previewCanonical.dropoff?.precision_level || '').toLowerCase();
      if (
        !previewCanonical.pickup?.canonical_hash ||
        !previewCanonical.dropoff?.canonical_hash ||
        blockedPrecisionLevels.has(pickupPrecision) ||
        blockedPrecisionLevels.has(dropoffPrecision)
      ) {
        setFormError(
          "Les adresses doivent être canonisées avec une précision suffisante avant la soumission."
        );
        return;
      }

      const response = await apiClient.post(`/clients/${effectiveClientId}/bookings`, {
        ...bookingData,
        pickup_location: canonicalPickup,
        dropoff_location: canonicalDropoff,
        amount: previewAmount,
        preview_amount: previewAmount,
      }, {
        headers: { 'Content-Type': 'application/json' },
      });
      const root = response.data || {};
      const payload = root.data !== undefined ? root.data : root;
      const bookingId = payload.booking_id ?? root.booking_id;
      const resolvedBooking = payload.booking || root.booking || {};
      const previewPaymentRequired = Boolean(previewWorkflow.payment_required);
      const needPrivateOnlinePay =
        Boolean(bookingId) &&
        (previewPaymentRequired || requiresPrivateOnlinePaymentAtBooking(resolvedBooking));

      if (needPrivateOnlinePay) {
        toast.success('Demande enregistrée. Finalisez le paiement Saferpay dans le formulaire ci-dessous.', {
          duration: 7000,
        });
      } else if (previewWorkflow.transmission_requires_client_action) {
        toast.success(
          "Demande enregistrée. Une action de votre part est encore requise avant transmission à l'entreprise.",
          { duration: 8000 }
        );
      } else {
        toast.success(
          "Demande enregistrée. Votre course est en attente de confirmation par l'entreprise de transport.",
          { duration: 8000 }
        );
      }
      setReservationFeedback({
        pickup,
        destination,
        scheduledLabel: asapMode ? 'Dès que possible' : `${selectedDate} ${selectedTime}`,
        statusLabel: getClientBookingUx('pending').label,
        billingLabel:
          resolvedBooking.payer_label ||
          resolvedBooking.coverage_label ||
          'Payeur non défini',
      });
      setUpcomingBookings((prev) => {
        const incomingBooking = payload.booking || root.booking;
        if (incomingBooking?.pickup_location) {
          return [...prev, incomingBooking];
        }
        return prev;
      });
      setPickup('');
      setDestination('');
      setPickupSelection(null);
      setDestinationSelection(null);
      if (!asapMode) {
        setSelectedDate('');
        setSelectedTime('');
      }
      setRoundTripEnabled(false);
      setReturnDate('');
      setReturnTime('');
      setRecurrenceEnabled(false);
      setRecurrenceType('weekly');
      setRecurrenceSeriesLength(4);
      setRecurrenceEndDate('');
      setRecurrenceDays([]);
      setMedicalFacility('');
      setDoctorName('');
      setClientNoteDeparture('');
      setClientNoteArrival('');
      setRouteLatLngs([]);
      setVisualRouteMetrics(null);
      setIndicativeServer(null);
      setIndicativeUnavailability('');
      trackClientKpiEvent('booking_created', {
        clientPublicId: effectiveClientId,
        bookingId: bookingId ? Number(bookingId) : null,
      });
      await loadBookings(true).catch(() => {});

      if (needPrivateOnlinePay && bookingId) {
        trackClientKpiEvent('payment_required_seen', {
          clientPublicId: effectiveClientId,
          bookingId: Number(bookingId),
        });
        trackClientKpiEvent('payment_redirect_started', {
          clientPublicId: effectiveClientId,
          bookingId: Number(bookingId),
        });
        const bid = Number(bookingId);
        const fallbackAmount = Number(resolvedBooking.amount ?? previewAmount);
        const payerLabel =
          resolvedBooking.payer_label || resolvedBooking.coverage_label || 'Client';
        setPayOfferBookingId(bid);
        setPayOffer({
          bookingId: bid,
          payerLabel,
          finalAmount: fallbackAmount,
          paymentRequired: true,
          lifecycleLabel: getClientBookingUx('awaiting_client_payment').label,
          checkoutError: null,
        });
        setPayingSaferpay(true);
        try {
          await startSaferpayHostedCheckout(bid);
        } catch (pe) {
          toastSaferpayCheckoutError(toast, pe);
          setPayOffer((prev) =>
            prev && prev.bookingId === bid
              ? {
                  ...prev,
                  checkoutError:
                    pe?.message || "Le paiement sécurisé n'a pas pu s'ouvrir. Réessayez ci-dessous.",
                }
              : prev
          );
        } finally {
          setPayingSaferpay(false);
        }
      }
    } catch (err) {
      console.error('Erreur réservation :', err);
      const msg = getApiErrorMessage(err, 'Une erreur est survenue lors de la réservation.');
      setFormError(msg);
      toast.error(msg, { duration: 6000 });
    } finally {
      setBookingSubmitting(false);
    }
  };

  const handlePayNowOffer = () => {
    if (!payOfferBookingId || !payOffer || payingSaferpay) return;
    const id = payOfferBookingId;
    trackClientKpiEvent('pay_now_clicked', {
      clientPublicId: effectiveClientId,
      bookingId: id,
    });
    setFormError(null);
    setPayOffer((prev) => (prev ? { ...prev, checkoutError: null } : prev));
    setPayingSaferpay(true);
    startSaferpayHostedCheckout(id)
      .catch((pe) => {
        toastSaferpayCheckoutError(toast, pe);
        setPayOffer((prev) =>
          prev && prev.bookingId === id
            ? {
                ...prev,
                checkoutError:
                  pe?.message || "Le paiement sécurisé n'a pas pu s'ouvrir. Réessayez ci-dessous.",
              }
            : prev
        );
      })
      .finally(() => {
        setPayingSaferpay(false);
      });
  };

  // Convertir routeLatLngs pour Google Maps
  const googleRoutePath = useMemo(() => {
    return routeLatLngs.map(([lat, lng]) => ({ lat, lng }));
  }, [routeLatLngs]);
  const hasValidatedPickup = Boolean(
    pickupSelection?.validated || parseCoordInput(pickup)
  );
  const hasValidatedDestination = Boolean(
    destinationSelection?.validated || parseCoordInput(destination)
  );
  const hasRouteInputs = hasValidatedPickup && hasValidatedDestination;

  const showPickupFieldInvalid =
    formError === MISSING_ADDRESSES_MSG && (!pickup.trim() || !hasValidatedPickup);
  const showDropoffFieldInvalid =
    formError === MISSING_ADDRESSES_MSG && (!destination.trim() || !hasValidatedDestination);

  // Parse des coordonnées depuis le texte du champ
  function parseCoordInput(text) {
    if (/^-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?$/.test(text)) {
      const [lat, lng] = text.split(',').map(Number);
      return { lat, lng };
    }
    return null;
  }

  const pickupMarkerPos =
    pickupSelection?.lat != null && pickupSelection?.lon != null
      ? { lat: Number(pickupSelection.lat), lng: Number(pickupSelection.lon) }
      : parseCoordInput(pickup);
  const destinationMarkerPos =
    destinationSelection?.lat != null && destinationSelection?.lon != null
      ? { lat: Number(destinationSelection.lat), lng: Number(destinationSelection.lon) }
      : parseCoordInput(destination);
  const displayBookingStatus = useMemo(
    () => resolveClientBookingDisplayStatus(nextBooking),
    [nextBooking]
  );
  const bookingUx = useMemo(() => getClientBookingUx(displayBookingStatus), [displayBookingStatus]);
  const nextTripKindMeta = useMemo(() => getBookingTripKindMeta(nextBooking), [nextBooking]);
  const currentStatusLabel = bookingUx.label;
  const actionsByStatus = useMemo(
    () => getEffectiveClientBookingActions(nextBooking),
    [nextBooking]
  );
  const statusToneClass = useMemo(
    () =>
      getClientBookingToneClass(bookingUx.label, {
        statusPending: 'statusPending',
        statusConfirmed: 'statusConfirmed',
        statusOnRoute: 'statusOnRoute',
        statusInProgress: 'statusInProgress',
        statusCompleted: 'statusCompleted',
        statusCancelled: 'statusCancelled',
      }),
    [bookingUx.label]
  );

  const dashboardDateParts = useMemo(() => {
    const d = new Date();
    return {
      display: d.toLocaleDateString('fr-CH', {
        weekday: 'long',
        day: '2-digit',
        month: 'long',
        year: 'numeric',
      }),
      iso: d.toISOString(),
    };
  }, []);
  const [estimateAmountPulse, setEstimateAmountPulse] = useState(false);
  const prevIndicativeAmountRef = useRef(null);

  useEffect(() => {
    if (indicativeAmountForDisplay == null) {
      prevIndicativeAmountRef.current = null;
      return;
    }
    const prev = prevIndicativeAmountRef.current;
    prevIndicativeAmountRef.current = indicativeAmountForDisplay;
    if (prev != null && prev !== indicativeAmountForDisplay) {
      setEstimateAmountPulse(true);
      const t = window.setTimeout(() => setEstimateAmountPulse(false), 180);
      return () => window.clearTimeout(t);
    }
  }, [indicativeAmountForDisplay]);

  const tripDraftSummary = useMemo(() => {
    const p = pickup.trim();
    const d = destination.trim();
    if (!p || !d) return null;
    if (!asapMode && (!selectedDate || !selectedTime)) return null;
    const extras = [];
    if (roundTripEnabled) {
      if (returnDate && returnTime) {
        extras.push(`Retour prévu : ${returnDate} à ${returnTime}`);
      } else if (returnDate) {
        extras.push(`Retour prévu le ${returnDate} (heure à définir)`);
      } else {
        extras.push('Aller-retour');
      }
    }
    if (recurrenceEnabled) {
      const typeLabel =
        recurrenceType === 'daily'
          ? 'tous les jours'
          : recurrenceType === 'weekly'
            ? 'toutes les semaines'
            : 'jours personnalisés';
      const endTrim = recurrenceEndDate.trim();
      if (endTrim) {
        extras.push(
          `Récurrence : ${typeLabel} jusqu’au ${formatRecurrenceYmdShort(endTrim)}`
        );
      } else {
        extras.push(
          `Récurrence : ${typeLabel}, ${Math.min(52, Math.max(1, Math.floor(Number(recurrenceSeriesLength)) || 1))} répétition(s)`
        );
      }
    }
    return {
      pickup: p,
      destination: d,
      whenLabel: formatScheduledSummaryLabel(selectedDate, selectedTime, asapMode),
      extras,
    };
  }, [
    pickup,
    destination,
    selectedDate,
    selectedTime,
    asapMode,
    roundTripEnabled,
    returnDate,
    returnTime,
    recurrenceEnabled,
    recurrenceType,
    recurrenceSeriesLength,
    recurrenceEndDate,
  ]);

  const recurrenceHintText = useMemo(() => {
    if (!recurrenceEnabled) return null;
    const endTrim = String(recurrenceEndDate || '').trim();
    if (endTrim) {
      const endLabel = formatRecurrenceYmdShort(endTrim);
      if (recurrenceType === 'custom' && recurrenceDays.length > 0) {
        return `Une réservation est créée par cette demande. La série est décrite jusqu’au ${endLabel} (jours choisis) : le transporteur confirmera les passages réels.`;
      }
      const cadence =
        recurrenceType === 'daily' ? 'chaque jour' : 'chaque semaine (même jour de la semaine)';
      return `Une réservation est créée par cette demande. La série est prévue jusqu’au ${endLabel} (${cadence}) : le transporteur confirmera les occurrences.`;
    }
    const n = Math.min(52, Math.max(1, Math.floor(Number(recurrenceSeriesLength)) || 1));
    if (recurrenceType === 'custom' && recurrenceDays.length > 0) {
      const total = n * recurrenceDays.length;
      return `Une réservation est créée par cette demande. Vous décrivez environ ${total} passage${total > 1 ? 's' : ''} (${n} cycle${n > 1 ? 's' : ''} × ${recurrenceDays.length} jour${recurrenceDays.length > 1 ? 's' : ''}) : le transporteur confirmera la série.`;
    }
    return `Une réservation est créée par cette demande. Vous indiquez ${n} répétition${n > 1 ? 's' : ''} (${recurrenceType === 'daily' ? 'quotidien' : 'hebdomadaire'}) : le transporteur confirmera la série.`;
  }, [
    recurrenceEnabled,
    recurrenceEndDate,
    recurrenceType,
    recurrenceSeriesLength,
    recurrenceDays,
  ]);

  const estimateInlineParts = useMemo(() => {
    if (indicativeAmount == null) return null;
    const parts = [];
    if (serverIndicativeLineMetrics?.duration_s) {
      parts.push(`≈ ${Math.round(serverIndicativeLineMetrics.duration_s / 60)} min`);
    }
    if (serverIndicativeLineMetrics?.distance_m != null) {
      parts.push(`${(serverIndicativeLineMetrics.distance_m / 1000).toFixed(1)} km`);
    }
    return parts;
  }, [indicativeAmount, serverIndicativeLineMetrics]);

  const estimateMetaJoined = useMemo(() => {
    if (!estimateInlineParts?.length) return null;
    return estimateInlineParts.join(' • ');
  }, [estimateInlineParts]);

  const clientNotePreview = useMemo(
    () => buildClientNoteFromLegs(clientNoteDeparture, clientNoteArrival),
    [clientNoteDeparture, clientNoteArrival]
  );
  const clientNotePreviewLen = clientNotePreview.length;

  useEffect(() => {
    const home = homeAddressFromProfile(profile);
    if (!home) return;
    setPickup((prev) => {
      if (String(prev || '').trim().length > 0) return prev;
      return home;
    });
  }, [profile]);

  const handleBookingAction = useCallback(
    (action, booking) => {
      if (!booking?.id) return;
      if (action === 'Recommander') {
        setPickup(String(booking.pickup_location || ''));
        setDestination(String(booking.dropoff_location || ''));
        setFormError(null);
        return;
      }
      const actionQuery = encodeURIComponent(action.toLowerCase());
      navigate(`/reservations/${effectiveClientId}?bookingId=${booking.id}&action=${actionQuery}`);
    },
    [effectiveClientId, navigate]
  );

  useEffect(() => {
    trackClientKpiEvent('reserve_opened', { clientPublicId: effectiveClientId });
  }, [effectiveClientId]);

  return (
    <div className="container">
      {(() => {
        const p = profile || {};
        const userName = p.first_name ?? p.firstName ?? p.username ?? p.user?.first_name ?? p.user?.username ?? 'Utilisateur';
        return <HeaderDashboard userName={userName} />;
      })()}

      <div
        className="mobileCanonWebHint"
        role="status"
        title="Réservations et suivi : l’application mobile LIRIE complète ce portail web (canon multi-surface)."
      >
        Astuce : l&apos;app mobile LIRIE complète ce portail pour le suivi au quotidien.
      </div>

      <div className="clientDashboardContentStack">
        {loadingProfile && <p>Chargement du profil…</p>}
        {loadingBookings && <div className="loadingSkeleton" aria-hidden />}
        {loadError && (
          <p className="error" role="alert">
            {loadError}
          </p>
        )}
        <main className={`clientDashboardPage${gmLoaded ? ' clientDashboardPage--mapBackdrop' : ''}`}>
        {gmLoaded ? (
          <div className="clientDashboardMapBackdrop" aria-hidden="true">
            <div className="clientDashboardMapBackdropMap">
              <div className="mapStack mapStackFullscreen">
                <GoogleMap
                  mapContainerStyle={CONTAINER_STYLE}
                  center={center}
                  zoom={hasRouteInputs ? 12 : 11}
                  options={PUBLIC_MAP_OPTIONS}
                  onLoad={onMapLoad}
                >
                  {pickupMarkerPos && (
                    <GoogleMapsAdvancedMarker
                      position={pickupMarkerPos}
                      icon={{
                        url: makePinMarkerIcon('pickup'),
                        scaledSize: window.google ? new window.google.maps.Size(28, 38) : undefined,
                        anchor: window.google ? new window.google.maps.Point(14, 38) : undefined,
                      }}
                      title="Départ"
                    />
                  )}
                  {destinationMarkerPos && (
                    <GoogleMapsAdvancedMarker
                      position={destinationMarkerPos}
                      icon={{
                        url: makePinMarkerIcon('dropoff'),
                        scaledSize: window.google ? new window.google.maps.Size(28, 38) : undefined,
                        anchor: window.google ? new window.google.maps.Point(14, 38) : undefined,
                      }}
                      title="Arrivée"
                    />
                  )}
                  {googleRoutePath.length > 0 && (
                    <Polyline
                      path={googleRoutePath}
                      options={{ ...ROUTE_OPTIONS, strokeColor: MAP_COLORS.brand }}
                    />
                  )}
                </GoogleMap>
              </div>
            </div>
            <div className="clientDashboardMapBackdropScrim" aria-hidden="true" />
          </div>
        ) : null}
        <div className="mainRow clientDashboardMainRow">
            <section className="leftSection card bookingFormCard">
              <div className="dashboardHeader bookingHeaderPro">
                <div className="headerLeft">
                  <div className="bookingHeaderLead">
                    <div className="bookingHeaderTitles">
                      <h1 className="title bookingHeaderTitle">Demande de transport médical</h1>
                    </div>
                    <time
                      className="headerMeta headerMetaPill"
                      dateTime={dashboardDateParts.iso}
                      aria-label="Journée affichée (référence)"
                    >
                      {dashboardDateParts.display}
                    </time>
                  </div>
                </div>
              </div>
              <div className="cardBody">
                <form className="form formDense">
                  {reservationFeedback ? (
                    <div className="bookingFeedback" role="status" aria-live="polite">
                      <div className="bookingFeedbackTop">
                        <span className="bookingFeedbackKicker">Demande envoyée</span>
                        <span className="bookingFeedbackPill">{reservationFeedback.statusLabel}</span>
                      </div>
                      <dl className="bookingFeedbackGrid">
                        <div className="bookingFeedbackItem">
                          <dt>Trajet</dt>
                          <dd>
                            <span className="bookingFeedbackRoute">{reservationFeedback.pickup}</span>
                            <span className="bookingFeedbackArrow" aria-hidden="true">
                              →
                            </span>
                            <span className="bookingFeedbackRoute">{reservationFeedback.destination}</span>
                          </dd>
                        </div>
                        <div className="bookingFeedbackItem">
                          <dt>Horaire</dt>
                          <dd>{reservationFeedback.scheduledLabel}</dd>
                        </div>
                        <div className="bookingFeedbackItem">
                          <dt>Couverture</dt>
                          <dd>{reservationFeedback.billingLabel}</dd>
                        </div>
                      </dl>
                    </div>
                  ) : null}

                  <div className="addressFieldsBlock addressFieldsBlockPrimary">
                    <div className="bookingAddressSwapRow">
                      <div className="bookingAddressSwapFields">
                        <div className={`${homeFieldStyles.fieldBlock} bookingFormAddressFieldBlock`}>
                          <label htmlFor="client-dashboard-pickup" className={homeFieldStyles.fieldLabel}>
                            Lieu de prise en charge
                          </label>
                          <div
                            className={`${homeFieldStyles.fieldGroup} bookingFormAddressFieldGroup${
                              showPickupFieldInvalid ? ` ${homeFieldStyles.fieldGroupInvalid}` : ''
                            }`}
                          >
                            <div className={homeFieldStyles.fieldIcon} aria-hidden>
                              <svg
                                width="18"
                                height="18"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                              >
                                <circle cx="12" cy="12" r="3" />
                                <circle cx="12" cy="12" r="9" strokeDasharray="4 3" />
                              </svg>
                            </div>
                            <AddressAutocomplete
                              flushInput
                              inputId="client-dashboard-pickup"
                              name="pickup"
                              value={pickup}
                              aria-invalid={showPickupFieldInvalid}
                              onChange={(e) => {
                                setPickup(e.target.value);
                                setPickupSelection(null);
                                setFormError(null);
                              }}
                              onSelect={(item) => {
                                setPickup(item.label || '');
                                setPickupSelection({
                                  validated: Boolean(item?.lat != null && item?.lon != null),
                                  lat: item?.lat,
                                  lon: item?.lon,
                                });
                                setFormError(null);
                              }}
                              placeholder="Ex: HUG, Rue Gabrielle-Perret-Gentil"
                            />
                          </div>
                        </div>
                        <div
                          className={`${homeFieldStyles.fieldBlock} bookingFormAddressFieldBlock`}
                          title={
                            isHospitalLikeDestination(destination)
                              ? 'L’établissement médical est déduit de l’adresse de destination.'
                              : undefined
                          }
                        >
                          <label htmlFor="client-dashboard-dropoff" className={homeFieldStyles.fieldLabel}>
                            Destination
                          </label>
                          <div
                            className={`${homeFieldStyles.fieldGroup} bookingFormAddressFieldGroup${
                              showDropoffFieldInvalid ? ` ${homeFieldStyles.fieldGroupInvalid}` : ''
                            }`}
                          >
                            <div className={homeFieldStyles.fieldIcon} aria-hidden>
                              <svg
                                width="18"
                                height="18"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                              >
                                <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" />
                                <circle cx="12" cy="9" r="2.5" />
                              </svg>
                            </div>
                            <AddressAutocomplete
                              flushInput
                              inputId="client-dashboard-dropoff"
                              name="dropoff"
                              value={destination}
                              aria-invalid={showDropoffFieldInvalid}
                              onChange={(e) => {
                                setDestination(e.target.value);
                                setDestinationSelection(null);
                                setFormError(null);
                              }}
                              onSelect={(item) => {
                                setDestination(item.label || '');
                                setDestinationSelection({
                                  validated: Boolean(item?.lat != null && item?.lon != null),
                                  lat: item?.lat,
                                  lon: item?.lon,
                                });
                                setFormError(null);
                              }}
                              placeholder="Ex: Clinique de Carouge"
                            />
                          </div>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="bookingAddressSwapButton"
                        onClick={handleSwapAddresses}
                        title="Inverser prise en charge et destination"
                        aria-label="Inverser le lieu de prise en charge et la destination"
                      >
                        ↕
                      </button>
                    </div>
                  </div>

                  <details className="bookingClientNoteFold bookingClientNoteFold--afterAddresses">
                    <summary className="bookingClientNoteSummary">
                      <span className="bookingClientNoteSummaryChevron" aria-hidden="true">
                        <svg
                          width="14"
                          height="14"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.25"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="m6 9 6 6 6-6" />
                        </svg>
                      </span>
                      <span className="bookingClientNoteSummaryLabel">
                        Précisions pour le transporteur{' '}
                        <span className="bookingClientNoteSummaryOptional">(optionnel)</span>
                      </span>
                    </summary>
                    <div className="bookingClientNoteBody">
                      <div className="bookingClientNoteRows">
                        <div className="bookingClientNoteRow">
                          <label
                            className="bookingClientNoteRowLabel"
                            htmlFor="client-booking-note-departure"
                          >
                            Au départ
                          </label>
                          <input
                            id="client-booking-note-departure"
                            type="text"
                            className="input bookingClientNoteLineInput"
                            value={clientNoteDeparture}
                            onChange={(e) =>
                              setClientNoteDeparture(
                                e.target.value.slice(0, MAX_CLIENT_NOTE_LEG)
                              )
                            }
                            maxLength={MAX_CLIENT_NOTE_LEG}
                            spellCheck="true"
                            autoComplete="off"
                            placeholder="Ex. RDV 9h, parking visiteurs"
                          />
                        </div>
                        <div className="bookingClientNoteRow">
                          <label className="bookingClientNoteRowLabel" htmlFor="client-booking-note-arrival">
                            À l’arrivée
                          </label>
                          <input
                            id="client-booking-note-arrival"
                            type="text"
                            className="input bookingClientNoteLineInput"
                            value={clientNoteArrival}
                            onChange={(e) =>
                              setClientNoteArrival(e.target.value.slice(0, MAX_CLIENT_NOTE_LEG))
                            }
                            maxLength={MAX_CLIENT_NOTE_LEG}
                            spellCheck="true"
                            autoComplete="off"
                            placeholder="Ex. Bât. B, 3e étage, accueil radiologie"
                          />
                        </div>
                      </div>
                      <div className="bookingClientNoteFooter">
                        <span
                          className={`bookingClientNoteCounter${
                            clientNotePreviewLen >= 450 ? ' bookingClientNoteCounter--near' : ''
                          }`}
                          aria-live="polite"
                        >
                          {clientNotePreviewLen} / {MAX_CLIENT_NOTE_LEN}
                        </span>
                      </div>
                    </div>
                  </details>

                  <div
                    className={`${homeFieldStyles.fieldBlock} ${homeFieldStyles.tripKindFieldScope} bookingScheduleModeField`}
                  >
                    <span id="client-dashboard-schedule-mode-label" className={homeFieldStyles.fieldLabelCompact}>
                      Horaire du transport
                    </span>
                    <div
                      className={institutionStyles.missionSegment}
                      role="radiogroup"
                      aria-labelledby="client-dashboard-schedule-mode-label"
                    >
                      <button
                        type="button"
                        className={`${institutionStyles.missionBtn} ${homeFieldStyles.tripKindBtn} ${
                          asapMode ? institutionStyles.missionBtnActive : ''
                        }`}
                        role="radio"
                        aria-checked={asapMode}
                        onClick={() => setAsapMode(true)}
                      >
                        Dès que possible
                      </button>
                      <button
                        type="button"
                        className={`${institutionStyles.missionBtn} ${homeFieldStyles.tripKindBtn} ${
                          !asapMode ? institutionStyles.missionBtnActive : ''
                        }`}
                        role="radio"
                        aria-checked={!asapMode}
                        onClick={() => setAsapMode(false)}
                      >
                        Planifier un horaire
                      </button>
                    </div>
                  </div>

                  {(!asapMode || !isMobileViewport) && (
                    <div className={`dateTime dateTimeFade ${asapMode ? 'dateTimeDimmed' : ''}`}>
                      <div className="inputWrapper dateField">
                        <label className="inputLabel" htmlFor="client-booking-date">
                          Date
                        </label>
                        <input
                          id="client-booking-date"
                          type="date"
                          value={selectedDate}
                          onChange={(e) => setSelectedDate(e.target.value)}
                          className="input"
                          min={todayDateMin}
                          disabled={asapMode}
                        />
                      </div>
                      <div className="inputWrapper dateField dateFieldNarrow">
                        <label className="inputLabel" htmlFor="client-booking-time">
                          Heure
                        </label>
                        <input
                          id="client-booking-time"
                          type="time"
                          value={selectedTime}
                          onChange={(e) => setSelectedTime(e.target.value)}
                          className="input"
                          min={timeMinForSelectedDate}
                          disabled={asapMode}
                        />
                      </div>
                    </div>
                  )}

                  <div
                    className={`${homeFieldStyles.fieldBlock} ${homeFieldStyles.tripKindFieldScope} bookingTripTypeField`}
                  >
                    <span id="client-dashboard-trip-type-label" className={homeFieldStyles.fieldLabelCompact}>
                      Aller / retour
                    </span>
                    <div
                      className={institutionStyles.missionSegment}
                      role="radiogroup"
                      aria-labelledby="client-dashboard-trip-type-label"
                    >
                      <button
                        type="button"
                        className={`${institutionStyles.missionBtn} ${homeFieldStyles.tripKindBtn} ${
                          !roundTripEnabled ? institutionStyles.missionBtnActive : ''
                        }`}
                        role="radio"
                        aria-checked={!roundTripEnabled}
                        onClick={() => setRoundTripEnabled(false)}
                      >
                        Aller simple
                      </button>
                      <button
                        type="button"
                        className={`${institutionStyles.missionBtn} ${homeFieldStyles.tripKindBtn} ${
                          roundTripEnabled ? institutionStyles.missionBtnActive : ''
                        }`}
                        role="radio"
                        aria-checked={roundTripEnabled}
                        onClick={() => setRoundTripEnabled(true)}
                      >
                        Avec retour planifié
                      </button>
                    </div>
                  </div>

                  {roundTripEnabled ? (
                    <div className="dateTime bookingReturnDateTime">
                      <div className="inputWrapper dateField">
                        <label className="inputLabel" htmlFor="client-booking-return-date">
                          Date du retour
                        </label>
                        <input
                          id="client-booking-return-date"
                          type="date"
                          value={returnDate}
                          onChange={(e) => setReturnDate(e.target.value)}
                          className="input"
                          min={todayDateMin}
                        />
                      </div>
                      <div className="inputWrapper dateField dateFieldNarrow">
                        <label className="inputLabel" htmlFor="client-booking-return-time">
                          Heure du retour (optionnel)
                        </label>
                        <input
                          id="client-booking-return-time"
                          type="time"
                          value={returnTime}
                          onChange={(e) => setReturnTime(e.target.value)}
                          className="input"
                          min={timeMinForReturn}
                        />
                      </div>
                    </div>
                  ) : null}

                  <div className="bookingRecurrenceBlock">
                    <div
                      className="bookingRecurrenceToolbar"
                      role="group"
                      aria-labelledby="client-booking-recurrence-legend"
                    >
                      <span
                        id="client-booking-recurrence-legend"
                        className="bookingRecurrenceToolbarLabel"
                      >
                        Récurrence
                      </span>
                      <button
                        type="button"
                        className={`bookingRecurrenceChip${
                          recurrenceEnabled ? ' bookingRecurrenceChip--active' : ''
                        }`}
                        aria-pressed={recurrenceEnabled}
                        onClick={() => setRecurrenceEnabled((v) => !v)}
                      >
                        Récurrente
                      </button>
                    </div>

                    {recurrenceEnabled ? (
                      <div className="bookingRecurrenceConfig">
                        <div className="inputWrapper bookingRecurrenceGroup">
                          <label className="inputLabel" htmlFor="client-booking-recurrence-type">
                            Type de récurrence
                          </label>
                          <select
                            id="client-booking-recurrence-type"
                            className="input bookingRecurrenceSelect"
                            value={recurrenceType}
                            onChange={(e) => setRecurrenceType(e.target.value)}
                          >
                            <option value="daily">Tous les jours</option>
                            <option value="weekly">Toutes les semaines</option>
                            <option value="custom">Jours personnalisés</option>
                          </select>
                        </div>

                        {recurrenceType === 'custom' ? (
                          <div className="bookingRecurrenceGroup">
                            <span className="inputLabel" id="client-booking-recurrence-days-label">
                              Jours concernés
                            </span>
                            <div
                              className="bookingRecurrenceDayStrip"
                              role="group"
                              aria-labelledby="client-booking-recurrence-days-label"
                            >
                              {RECURRENCE_WEEK_DAYS.map((day) => (
                                <button
                                  key={day.id}
                                  type="button"
                                  className={`bookingRecurrenceDayBtn${
                                    recurrenceDays.includes(day.id) ? ' bookingRecurrenceDayBtn--on' : ''
                                  }`}
                                  onClick={() => toggleRecurrenceDay(day.id)}
                                  title={day.label}
                                  aria-pressed={recurrenceDays.includes(day.id)}
                                >
                                  {day.short}
                                </button>
                              ))}
                            </div>
                            {recurrenceDays.length === 0 ? (
                              <p className="bookingRecurrenceWarning" role="status">
                                Sélectionnez au moins un jour.
                              </p>
                            ) : null}
                          </div>
                        ) : null}

                        <div className="inputWrapper bookingRecurrenceGroup">
                          <label className="inputLabel" htmlFor="client-booking-recurrence-end">
                            Jusqu’au
                            <span className="bookingRecurrenceLabelOptional"> (optionnel)</span>
                          </label>
                          <input
                            id="client-booking-recurrence-end"
                            type="date"
                            className="input"
                            value={recurrenceEndDate}
                            min={recurrenceStartYmd}
                            onChange={(e) => setRecurrenceEndDate(e.target.value)}
                            aria-describedby={
                              recurrenceHintText ? 'client-booking-recurrence-hint' : undefined
                            }
                          />
                        </div>

                        {!String(recurrenceEndDate || '').trim() ? (
                          <div className="inputWrapper bookingRecurrenceGroup">
                            <label className="inputLabel" htmlFor="client-booking-recurrence-count">
                              Nombre de répétitions
                            </label>
                            <input
                              id="client-booking-recurrence-count"
                              type="number"
                              inputMode="numeric"
                              min={1}
                              max={52}
                              step={1}
                              value={recurrenceSeriesLength}
                              onChange={(e) => {
                                const n = Number(e.target.value);
                                if (Number.isNaN(n)) {
                                  setRecurrenceSeriesLength(1);
                                  return;
                                }
                                setRecurrenceSeriesLength(Math.min(52, Math.max(1, Math.floor(n))));
                              }}
                              className="input bookingRecurrenceCountInput"
                              placeholder="Ex. 4"
                              aria-describedby={
                                recurrenceHintText ? 'client-booking-recurrence-hint' : undefined
                              }
                            />
                          </div>
                        ) : null}
                        {recurrenceHintText ? (
                          <p id="client-booking-recurrence-hint" className="bookingRecurrenceHint">
                            {recurrenceHintText}
                          </p>
                        ) : null}
                      </div>
                    ) : null}
                  </div>

                  {showMedicalFields ? (
                    <>
                      {!isHospitalLikeDestination(destination) ? (
                        <div className="inputWrapper">
                          <label className="inputLabel" htmlFor="client-medical-facility">
                            Établissement (optionnel)
                          </label>
                          <input
                            id="client-medical-facility"
                            type="text"
                            value={medicalFacility}
                            onChange={(e) => setMedicalFacility(e.target.value)}
                            className="input"
                            placeholder="Nom de l’établissement"
                          />
                        </div>
                      ) : null}
                      {destinationHasDoctorHint(destination) ? (
                        <div className="inputWrapper">
                          <label className="inputLabel" htmlFor="client-doctor">
                            Nom du médecin (optionnel)
                          </label>
                          <input
                            id="client-doctor"
                            type="text"
                            value={doctorName}
                            onChange={(e) => setDoctorName(e.target.value)}
                            className="input"
                            placeholder="Nom du médecin"
                          />
                        </div>
                      ) : null}
                    </>
                  ) : null}

                  {payOfferBookingId != null && payOffer ? (
                    <div
                      className={`bookingPaymentPanel${
                        payOffer.checkoutError ? ' bookingPaymentPanel--error' : ''
                      }`}
                      role="region"
                      aria-labelledby="client-booking-pay-title"
                    >
                      <div className="bookingPaymentPanelTop">
                        <div className="bookingPaymentPanelHeading">
                          <h2 id="client-booking-pay-title" className="bookingPaymentPanelTitle">
                            Paiement sécurisé
                          </h2>
                          <span className="bookingPaymentPanelVendor">Saferpay</span>
                        </div>
                        <span className="bookingPaymentLifecycle">{payOffer.lifecycleLabel}</span>
                      </div>
                      {payingSaferpay ? (
                        <p className="bookingPaymentStatus" role="status">
                          Redirection vers la page de paiement sécurisée…
                        </p>
                      ) : payOffer.checkoutError ? (
                        <p className="bookingPaymentError">{payOffer.checkoutError}</p>
                      ) : (
                        <p className="bookingPaymentLead">
                          Finalisez le règlement en ligne via le bouton ci-dessous.
                        </p>
                      )}
                      <dl className="bookingPaymentFacts">
                        {payOffer.payerLabel ? (
                          <>
                            <dt>Payeur</dt>
                            <dd>{payOffer.payerLabel}</dd>
                          </>
                        ) : null}
                        <dt>Montant</dt>
                        <dd className="bookingPaymentAmount">{formatPrice(payOffer.finalAmount)}</dd>
                      </dl>
                      <div className="bookingPaymentActions">
                        <button
                          type="button"
                          className="primaryButton"
                          onClick={handlePayNowOffer}
                          disabled={payingSaferpay}
                          aria-busy={payingSaferpay}
                        >
                          {payingSaferpay ? 'Redirection…' : 'Ouvrir le paiement'}
                        </button>
                      </div>
                    </div>
                  ) : null}

                  {formError ? (
                    <p className="error" role="alert">
                      {formError}
                    </p>
                  ) : null}
                  {indicativeUnavailability ? (
                    <p className="networkHint" role="status">
                      {indicativeUnavailability}
                    </p>
                  ) : null}
                  {estimateNotice ? (
                    <p className="networkHint" role="status">
                      {estimateNotice}
                    </p>
                  ) : null}
                  {indicativeServerLoading ? (
                    <p className="networkHint" role="status" aria-live="polite">
                      Indicatif en cours de calcul…
                    </p>
                  ) : null}

                  <div className="formActions formActionsPrimary">
                    <button
                      type="button"
                      className={`${homeFieldStyles.ctaButton} bookingDashboardCta`}
                      onClick={handleBooking}
                      disabled={bookingSubmitting || loadingProfile || loadingBookings || !effectiveClientId}
                      aria-busy={bookingSubmitting}
                    >
                      {bookingSubmitting ? (
                        <>
                          <span className="btnInlineSpinner" aria-hidden="true" />
                          Validation en cours…
                        </>
                      ) : (
                        <>
                          Valider la demande de transport
                          <svg
                            width="18"
                            height="18"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            aria-hidden
                          >
                            <path d="M5 12h14" />
                            <path d="m12 5 7 7-7 7" />
                          </svg>
                        </>
                      )}
                    </button>
                  </div>
                </form>
              </div>
            </section>
            <aside className="bookingSidebar" aria-label="Contexte trajet et reprise">
              {indicativeAmount != null || tripDraftSummary ? (
                <div
                  className="sidebarJourneyCard"
                  role="region"
                  aria-label={
                    indicativeAmount != null && tripDraftSummary
                      ? 'Estimation et récapitulatif du trajet'
                      : indicativeAmount != null
                        ? 'Estimation du transport'
                        : 'Récapitulatif du trajet'
                  }
                >
                  {indicativeAmount != null ? (
                    <section
                      className="sidebarJourneyCardEstimate"
                      aria-labelledby="sidebar-journey-estimate-title"
                      role="status"
                      aria-live="polite"
                    >
                      <h3 id="sidebar-journey-estimate-title" className="sidebarEstimateTitle">
                        Estimation transport
                      </h3>
                      <div
                        className={`sidebarEstimateAmount${estimateAmountPulse ? ' sidebarEstimateAmount--pulse' : ''}`}
                      >
                        {indicativeAmountForDisplay.toFixed(2)} CHF
                      </div>
                      {estimateMetaJoined ? (
                        <p className="sidebarEstimateMetaLine">{estimateMetaJoined}</p>
                      ) : null}
                      <p className="sidebarEstimateLegal">{sidebarEstimateLegal}</p>
                    </section>
                  ) : null}

                  {tripDraftSummary ? (
                    <section
                      className={`sidebarJourneyCardRecap${indicativeAmount != null ? ' sidebarJourneyCardRecap--afterEstimate' : ''}`}
                      aria-labelledby="sidebar-journey-recap-kicker"
                    >
                      <header className="tripSummaryProHeader">
                        <span id="sidebar-journey-recap-kicker" className="tripSummaryProKicker">
                          Récapitulatif
                        </span>
                        <span className="tripSummaryProHeaderTitle">Trajet sélectionné</span>
                      </header>
                      <ul className="tripSummaryProPath">
                        <li className="tripSummaryProLeg">
                          <span className="tripSummaryProLegMark" aria-hidden="true" />
                          <div className="tripSummaryProLegBody">
                            <span className="tripSummaryProLegLabel">Prise en charge</span>
                            <span className="tripSummaryProLegText">{tripDraftSummary.pickup}</span>
                          </div>
                        </li>
                        <li className="tripSummaryProLeg">
                          <span
                            className="tripSummaryProLegMark tripSummaryProLegMark--arrival"
                            aria-hidden="true"
                          />
                          <div className="tripSummaryProLegBody">
                            <span className="tripSummaryProLegLabel">Destination</span>
                            <span className="tripSummaryProLegText">{tripDraftSummary.destination}</span>
                          </div>
                        </li>
                      </ul>
                      <footer className="tripSummaryProSchedule">
                        <span className="tripSummaryProScheduleLabel">Horaire affiché</span>
                        <p className="tripSummaryProWhen">{tripDraftSummary.whenLabel}</p>
                      </footer>
                      {tripDraftSummary.extras?.length ? (
                        <div className="tripSummaryProExtras">
                          {tripDraftSummary.extras.map((line) => (
                            <p key={line} className="tripSummaryProExtraLine">
                              {line}
                            </p>
                          ))}
                        </div>
                      ) : null}
                    </section>
                  ) : null}
                </div>
              ) : null}

              {hasRecentTrips ? (
                <section className="rightSection card recentResumeCard sidebarRecentCard">
                  <div className="cardHeader">
                    <h2 className="cardTitle">Reprendre un trajet récent</h2>
                  </div>
                  <div className="cardBody">
                    <div className="recentTripsList recentTripsListCompact">
                      {recentTrips.map((trip) => {
                        const tripKindMeta = getBookingTripKindMeta(trip);
                        return (
                          <article
                            key={trip.id}
                            className="recentTripCard recentTripCardCompact"
                          >
                            <div className="recentTripCardInner">
                              <div className="recentTripInfo">
                                <div className="recentTripCardTop">
                                  {tripKindMeta ? (
                                    <span
                                      className={`bookingTripKindChip bookingTripKindChip--${tripKindMeta.variant}`}
                                    >
                                      {tripKindMeta.label}
                                    </span>
                                  ) : null}
                                  <time
                                    className="recentTripWhenLine"
                                    dateTime={
                                      Number.isFinite(Date.parse(trip.scheduled_time))
                                        ? new Date(trip.scheduled_time).toISOString()
                                        : undefined
                                    }
                                  >
                                    {formatTripResumeWhen(trip.scheduled_time)}
                                  </time>
                                </div>
                                <div className="recentTripRouteStack" aria-label="Trajet enregistré">
                                  <div className="recentTripLeg">
                                    <span className="recentTripLegDot" aria-hidden />
                                    <span className="recentTripLegText">{trip.pickup_location}</span>
                                  </div>
                                  <div className="recentTripLeg recentTripLeg--arrival">
                                    <span className="recentTripLegDot" aria-hidden />
                                    <span className="recentTripLegText">{trip.dropoff_location}</span>
                                  </div>
                                </div>
                              </div>
                              <button
                                type="button"
                                className="secondaryButton recentTripReuseBtnSober"
                                onClick={() => handleBookingAction('Recommander', trip)}
                              >
                                Réutiliser ce trajet
                              </button>
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  </div>
                </section>
              ) : null}

              <section className="rightSection card sidebarSupportCard">
                <div className="cardHeader">
                  <h2 className="cardTitle">Support rapide</h2>
                </div>
                <div className="cardBody">
                  <p className="sidebarSupportText">Une question sur ce trajet ou votre dossier ?</p>
                  <button
                    type="button"
                    className="sidebarSupportBtnOutline"
                    onClick={() => navigate('/contact/support')}
                  >
                    Contacter le support
                  </button>
                </div>
              </section>
            </aside>
          </div>
          {hasActiveOrFutureBooking && nextBooking ? (
            <section className="activityContainer card clientDashboardBelowRow nextBookingCard">
              <div className="cardHeader nextBookingCardHeader">
                <h2 className="cardTitle">Prochaine course</h2>
              </div>
              <div className="cardBody nextBookingCardBody">
                <div className="nextBookingInner">
                  <div className="nextBookingTopRow">
                    <p className={`activityLabel ${statusToneClass}`}>{currentStatusLabel}</p>
                    {nextTripKindMeta ? (
                      <span
                        className={`bookingTripKindChip bookingTripKindChip--${nextTripKindMeta.variant}`}
                      >
                        {nextTripKindMeta.label}
                      </span>
                    ) : null}
                  </div>

                  <div className="nextBookingRoute" aria-label="Trajet">
                    <div className="nextBookingLeg">
                      <span className="nextBookingLegRail" aria-hidden="true">
                        <span className="nextBookingLegDot nextBookingLegDot--pickup" />
                        <span className="nextBookingLegLine" />
                        <span className="nextBookingLegDot nextBookingLegDot--dropoff" />
                      </span>
                      <div className="nextBookingLegStack">
                        <div className="nextBookingLegBlock">
                          <span className="nextBookingLegEyebrow">Départ</span>
                          <span className="nextBookingLegAddr">{nextBooking.pickup_location}</span>
                        </div>
                        <div className="nextBookingLegBlock">
                          <span className="nextBookingLegEyebrow">Arrivée</span>
                          <span className="nextBookingLegAddr">{nextBooking.dropoff_location}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <dl className="nextBookingStats">
                    <div className="nextBookingStat">
                      <dt>Date</dt>
                      <dd>{formatBookingDate(nextBooking.scheduled_time)}</dd>
                    </div>
                    <div className="nextBookingStat">
                      <dt>Montant</dt>
                      <dd>{formatPrice(nextBooking.amount)}</dd>
                    </div>
                    {nextBooking.eta_minutes != null ? (
                      <div className="nextBookingStat nextBookingStat--eta">
                        <dt>ETA chauffeur</dt>
                        <dd>{Math.max(0, Number(nextBooking.eta_minutes))} min</dd>
                      </div>
                    ) : null}
                  </dl>

                  <div className="nextBookingActions">
                    {actionsByStatus.map((action) => (
                      <button
                        key={action}
                        type="button"
                        className={
                          action === 'Voir'
                            ? 'secondaryButton nextBookingActionBtn'
                            : action === 'Annuler'
                              ? 'nextBookingActionBtn nextBookingActionBtnAnnuler'
                              : 'primaryButton nextBookingActionBtn'
                        }
                        onClick={() => handleBookingAction(action, nextBooking)}
                      >
                        {action}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          ) : null}
        </main>
      </div>

      <Footer />
    </div>
  );
};

export default ClientDashboard;
