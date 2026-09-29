import React, { useEffect, useMemo, useState, useCallback, useRef } from 'react';
import { flushSync } from 'react-dom';
import apiClient from '../../../utils/apiClient';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import homeFieldStyles from '../../Home/Home.module.css';
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
  resolveLiriePointMarkerIcon,
  ROUTE_OPTIONS,
} from '../../../utils/mapUtils';
import polyline from '@mapbox/polyline';

// UI
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import Modal from '../../../components/common/Modal';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import InlineTimePicker from '../../../components/ui/InlineTimePicker';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import {
  composeProfilePickupAccess,
  classifyPortalMedicalPlace,
  destinationLooksMedical,
} from './portalTransportProfile';
import { getApiErrorMessage } from '../../../utils/apiErrorMessage';
import { toast } from 'sonner';
import { toastSaferpayCheckoutError } from '../../../utils/saferpayPaymentUi';
import { readAndConsumeSaferpayPayResume } from '../../../utils/clientSaferpayPayResume';
import { startSaferpayHostedCheckout } from '../../../services/clientSaferpayPaymentService';
import {
  getClientBookingToneClass,
  getClientBookingUx,
  getEffectiveClientBookingActions,
  normalizeClientBookingStatus,
  resolveClientBookingDisplayStatus,
} from '../../../utils/clientBookingUx';
import { trackClientKpiEvent } from '../../../utils/clientKpi';
import {
  formatPortalCancellationPolicyForDisplay,
  formatPortalOfferChf,
  isPortalOfferConfirmable,
  PORTAL_DV_COPY,
} from '../../../utils/portalDoubleValidationUi';
import {
  downloadPortalTermsDocument,
  portalTermsDocumentLabel,
  printPortalTermsDocument,
} from '../../../utils/portalTermsDocument';
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
import { foldClientRouteRequests } from '../../../utils/clientRouteRequest';

const CONTAINER_STYLE = { width: '100%', height: '100%' };

const MISSING_ADDRESSES_MSG = 'Veuillez saisir le lieu de départ et la destination.';
/** Longueur max par ligne (départ / arrivée), affichée comme sur le formulaire. */
const MAX_CLIENT_NOTE_LEG = 250;

const PORTAL_ROUTE_TIPS = [
  'Ajoutez vos destinations dans l’ordre de votre trajet.',
  'Indiquez l’heure de prise en charge ou celle du rendez-vous.',
  'Si les deux heures sont remplies, la prise en charge fait foi.',
  'Le rendez-vous doit être après l’heure de prise en charge.',
  'La date du transport se choisit une seule fois, en haut de la demande.',
  'Le départ du retour peut rester sans heure.',
  'L’adresse de retour reprend le point de départ.',
  'Pour une étape un autre jour, utilisez « Autre jour » à côté de l’heure.',
  'Activez « Destination médicale » pour l’établissement, le service ou le médecin.',
  'Un nom de médecin remplit le champ Médecin, l’établissement devient « Cabinet médical ».',
  'Précisez l’accès seulement s’il y a une entrée, un code ou un étage particulier.',
  'Un fauteuil personnel et un fauteuil à fournir ne se combinent pas.',
  'L’assistance peut s’ajouter au fauteuil : indiquez alors le type d’aide.',
  'La récurrence décrit la série. Le transporteur confirme chaque passage.',
];

const PORTAL_ROUTE_TIP_STORAGE_KEY = 'portal-route-tip-index';

function readStoredRouteTipIndex() {
  try {
    const index = Number(sessionStorage.getItem(PORTAL_ROUTE_TIP_STORAGE_KEY));
    if (Number.isInteger(index) && index >= 0 && index < PORTAL_ROUTE_TIPS.length) {
      return index;
    }
  } catch {
    /* sessionStorage indisponible */
  }
  return null;
}

/** À chaque chargement, le conseil suivant. Le premier affichage est tiré au sort. */
function initialRouteTipIndex() {
  const previous = readStoredRouteTipIndex();
  const count = PORTAL_ROUTE_TIPS.length;
  const next = previous == null ? Math.floor(Math.random() * count) : (previous + 1) % count;
  try {
    sessionStorage.setItem(PORTAL_ROUTE_TIP_STORAGE_KEY, String(next));
  } catch {
    /* sessionStorage indisponible */
  }
  return next;
}

function extraStopIsMedical(stop) {
  if (stop?.medicalOptOut) return false;
  return (
    Boolean(stop?.detailsOpen) ||
    destinationLooksMedical(stop?.address) ||
    Boolean(String(stop?.facility || '').trim()) ||
    Boolean(String(stop?.service || '').trim()) ||
    Boolean(String(stop?.doctor || '').trim())
  );
}

/** Recopie le nom du lieu (ex. Clinique de Joli-Mont) dans Établissement, tant que le client n’a pas modifié le champ. */
function applyExtraStopPlace(stop, address) {
  const text = String(address || '');
  const next = { ...stop, address: text };
  if (stop?.medicalOptOut) return next;
  if (!text.trim() || !destinationLooksMedical(text)) {
    if (!stop?.facilityTouched) next.facility = '';
    if (!stop?.doctorTouched) next.doctor = '';
    return next;
  }
  const place = classifyPortalMedicalPlace(text);
  if (!stop?.facilityTouched) next.facility = place.facility.slice(0, 200);
  if (!stop?.doctorTouched) next.doctor = place.doctor.slice(0, 200);
  return next;
}

function PortalMark({ name }) {
  const props = {
    width: 16,
    height: 16,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 2,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    'aria-hidden': true,
  };
  if (name === 'calendar') {
    return (
      <svg {...props}>
        <rect x="3" y="5" width="18" height="16" rx="2" />
        <path d="M3 10h18M8 3v4M16 3v4" />
      </svg>
    );
  }
  if (name === 'pin') {
    return (
      <svg {...props}>
        <path d="M12 21s7-6.2 7-11a7 7 0 1 0-14 0c0 4.8 7 11 7 11z" />
        <circle cx="12" cy="10" r="2.2" />
      </svg>
    );
  }
  if (name === 'info') {
    return (
      <svg {...props} width="14" height="14">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v5M12 8h.01" />
      </svg>
    );
  }
  if (name === 'crosshair') {
    return (
      <svg {...props} width="16" height="16">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 3v3M12 18v3M3 12h3M18 12h3" />
      </svg>
    );
  }
  if (name === 'plus') {
    return (
      <svg {...props} width="16" height="16">
        <path d="M12 5v14M5 12h14" />
      </svg>
    );
  }
  if (name === 'person') {
    return (
      <svg {...props}>
        <circle cx="12" cy="8" r="3" />
        <path d="M6 19c1.2-3 3.2-4.5 6-4.5S16.8 16 18 19" />
      </svg>
    );
  }
  if (name === 'wheelchair') {
    return (
      <svg {...props}>
        <circle cx="16" cy="6" r="1.6" />
        <path d="M8 19a5 5 0 1 0 4.2-7.4L11 9h6M11 12h4" />
      </svg>
    );
  }
  return (
    <svg {...props}>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 8v4.2l2.4 1.5" />
    </svg>
  );
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
  const fn = String(p.first_name || p.user?.first_name || '').trim();
  const rawLast = String(p.last_name || p.user?.last_name || '').trim();
  const ln = rawLast && rawLast !== 'Non spécifié' ? rawLast.toLocaleUpperCase('fr-CH') : '';
  return `${fn} ${ln}`.trim() || 'Client';
}

function homeAddressFromProfile(p) {
  if (!p) return '';
  const dom = p.domicile?.address ? String(p.domicile.address).trim() : '';
  const userAddress = p.user?.address ? String(p.user.address).trim() : '';
  return (dom || userAddress || p.address || p.domicile_address || p.billing_address || '').trim();
}

function habitualMobilityFromProfile(p) {
  const mobility = p?.mobility || {};
  const own = Boolean(mobility.wheelchair_client_has);
  const need = Boolean(mobility.wheelchair_need) && !own;
  const assistance = Boolean(mobility.needs_assistance);
  const detail = assistance ? String(mobility.assistance_detail || '').trim().slice(0, 200) : '';
  return { own, need, assistance, detail };
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

function formatPortalLegWhen(prefix, dateStr, timeStr) {
  const date = String(dateStr || '').trim();
  const time = String(timeStr || '').trim();
  if (!date && !time) return prefix;
  const parsed = new Date(time ? `${date}T${time}:00` : `${date}T12:00:00`);
  if (!Number.isFinite(parsed.getTime())) {
    return [prefix, date, time].filter(Boolean).join(' · ');
  }
  const datePart = parsed.toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
  if (!time) return `${prefix} · ${datePart}`;
  const timePart = parsed.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
  return `${prefix} · ${datePart} · ${timePart}`;
}

function formatScheduledSummaryLabel(dateStr, timeStr, asap, scheduleAnchor = 'arrival') {
  if (asap) return 'Dès que possible (selon disponibilité des véhicules)';
  if (!dateStr || !timeStr) return '—';
  const d = new Date(`${dateStr}T${timeStr}:00`);
  const prefix = scheduleAnchor === 'departure' ? 'Départ souhaité' : 'Rendez-vous';
  if (!Number.isFinite(d.getTime())) return `${prefix} · ${dateStr} à ${timeStr}`;
  const datePart = d.toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
  const timePart = d.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
  return `${prefix} · ${datePart} · ${timePart}`;
}

function verifiedProfilePhone(profile) {
  if (!profile || profile.phone_verified === false) return '';
  return String(
    profile.phone || profile.user?.phone || profile.mobile_phone || profile.mobile || ''
  ).trim();
}

function composeMedicalContact(service, doctor) {
  return [String(service || '').trim(), String(doctor || '').trim()].filter(Boolean).join(' – ');
}

function mobilityNeedsLabel({
  wheelchairOwn,
  wheelchairRequired,
  assistanceRequired,
  assistanceDetail,
}) {
  const parts = [];
  if (wheelchairOwn) parts.push('Fauteuil personnel');
  if (wheelchairRequired) parts.push('Fauteuil à fournir');
  if (assistanceRequired) {
    const detail = String(assistanceDetail || '').trim();
    parts.push(detail ? `Assistance · ${detail}` : 'Assistance');
  }
  return parts.join(' · ');
}

function portalSubmitWaitCopy(index, transportCount) {
  const count = Math.max(1, Number(transportCount) || 1);
  const record =
    count > 1 ? `Enregistrement des ${count} transports` : 'Enregistrement du transport';
  const steps = ['Vérification des adresses', record, 'Transmission de la demande'];
  const settled = index >= steps.length;
  const active = Math.min(index, steps.length - 1);
  return {
    title: settled ? 'Traitement toujours en cours' : steps[active],
    steps,
    active,
    settled,
    hint: settled
      ? 'Le traitement continue. Restez sur cette page : la confirmation arrive dès que la demande est enregistrée.'
      : 'Restez sur cette page. La confirmation s’affiche dès que la demande est enregistrée.',
  };
}

/** Jour calendaire local (YYYY-MM-DD), sans le décalage UTC de toISOString. */
function localCalendarYmd(date = new Date()) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function formatPortalDayLabel(dateStr) {
  const d = new Date(`${dateStr}T12:00:00`);
  if (!Number.isFinite(d.getTime())) return dateStr;
  const label = d.toLocaleDateString('fr-CH', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function formatPortalClock(timeStr) {
  const match = String(timeStr || '').match(/^(\d{2}):(\d{2})/);
  return match ? `${match[1]}:${match[2]}` : String(timeStr || '').trim();
}

/** Récapitulatif en langage humain, avant la confirmation. */
function buildPortalReviewNarrative({
  asap,
  scheduleAnchor,
  selectedDate,
  selectedTime,
  departureTime = '',
  appointmentTime = '',
  pickup,
  destination,
  roundTrip,
  returnTime,
  needsLabel,
  contactDetail,
  contactName,
  contactPhone,
  recurrenceLabel,
  extraStopLabels = [],
  establishmentLabel = '',
}) {
  const lines = [];
  if (!asap && selectedDate) {
    lines.push({ kind: 'date', strong: true, text: formatPortalDayLabel(selectedDate) });
  }
  lines.push({ kind: 'route', text: `${pickup} → ${destination}` });
  if (asap) {
    lines.push({ kind: 'schedule', strong: true, text: 'Dès que possible' });
  } else if (scheduleAnchor === 'departure') {
    lines.push({
      kind: 'schedule',
      strong: true,
      text: `Prise en charge à ${formatPortalClock(selectedTime)}`,
    });
    if (appointmentTime) {
      lines.push({ kind: 'schedule', text: `Rendez-vous à ${formatPortalClock(appointmentTime)}` });
    }
  } else {
    lines.push({
      kind: 'schedule',
      strong: true,
      text: `Rendez-vous à ${formatPortalClock(selectedTime)}`,
    });
    if (departureTime) {
      lines.push({ kind: 'schedule', text: `Départ souhaité à ${formatPortalClock(departureTime)}` });
    } else {
      lines.push({ kind: 'schedule', text: 'Prise en charge : à déterminer par le transporteur' });
    }
  }
  if (roundTrip) {
    lines.push({
      kind: 'trip',
      text: returnTime
        ? `Aller-retour · départ à ${formatPortalClock(returnTime)}`
        : 'Aller-retour · départ non précisé',
    });
  } else {
    lines.push({ kind: 'trip', text: 'Aller simple' });
  }
  extraStopLabels.forEach((label) => lines.push({ kind: 'stop', text: label }));
  if (needsLabel) lines.push({ kind: 'needs', text: needsLabel });
  if (establishmentLabel) {
    lines.push({ kind: 'place', text: `Établissement : ${establishmentLabel}` });
  }
  if (contactDetail) lines.push({ kind: 'place', text: contactDetail });
  if (recurrenceLabel) lines.push({ kind: 'meta', text: recurrenceLabel });
  if (contactName) {
    lines.push({
      kind: 'meta',
      text: contactPhone ? `Contact : ${contactName} • ${contactPhone}` : `Contact : ${contactName}`,
    });
  }
  return lines;
}

function buildPortalOrderSteps({
  pickup,
  destination,
  extraStops = [],
  roundTrip,
  asap,
  hasDepartureTime,
  hasAppointmentTime,
  departureTime,
  appointmentTime,
  returnTime,
  pickupAccess,
  dropoffAccess,
  facilityText,
  serviceText,
  doctorText,
  medicalDestinationActive,
}) {
  const steps = [];
  const departLines = [];
  if (asap) departLines.push('Dès que possible');
  else if (hasDepartureTime) departLines.push(`Prise en charge à ${formatPortalClock(departureTime)}`);
  else departLines.push('Prise en charge : à déterminer par le transporteur');
  if (pickupAccess) departLines.push(`Accès départ : ${pickupAccess}`);
  steps.push({ key: 'depart', label: 'Départ', address: pickup, lines: departLines });

  const arrivalLines = [];
  if (!asap && hasAppointmentTime) {
    arrivalLines.push(`Rendez-vous à ${formatPortalClock(appointmentTime)}`);
  }
  if (medicalDestinationActive && facilityText) arrivalLines.push(`Établissement : ${facilityText}`);
  if (medicalDestinationActive && serviceText) arrivalLines.push(serviceText);
  if (medicalDestinationActive && doctorText) arrivalLines.push(doctorText);
  if (dropoffAccess) arrivalLines.push(`Accès destination : ${dropoffAccess}`);
  const hasExtra = extraStops.some((stop) => String(stop.address || '').trim());
  steps.push({
    key: 'arrival',
    label: hasExtra ? 'Étape 1' : 'Arrivée',
    address: destination,
    lines: arrivalLines,
  });

  extraStops.forEach((stop, index) => {
    const address = String(stop.address || '').trim();
    if (!address) return;
    const lines = [];
    const time = String(stop.time || '').trim();
    if (time) lines.push(`Heure de départ à ${formatPortalClock(time)}`);
    if (extraStopIsMedical(stop)) {
      const facility = String(stop.facility || '').trim();
      const service = String(stop.service || '').trim();
      const doctor = String(stop.doctor || '').trim();
      if (facility) lines.push(`Établissement : ${facility}`);
      if (service) lines.push(service);
      if (doctor) lines.push(doctor);
    }
    const access = String(stop.access || '').trim();
    if (access) lines.push(`Accès : ${access}`);
    steps.push({
      key: stop.key || `stop-${index}`,
      label: `Étape ${index + 2}`,
      address,
      lines,
    });
  });

  if (roundTrip) {
    const lines = [
      returnTime
        ? `Aller-retour · départ à ${formatPortalClock(returnTime)}`
        : 'Aller-retour · départ non précisé',
    ];
    if (pickupAccess) lines.push(`Accès retour : ${pickupAccess}`);
    steps.push({ key: 'return', label: 'Retour', address: pickup, lines });
  }
  return steps;
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
  const legs = Number(booking.route_request_transport_count) || 0;
  if (legs > 2) {
    return { variant: 'roundTrip', label: `${legs} trajets` };
  }
  if (asBookingBool(booking.is_return)) {
    return { variant: 'return', label: 'Retour' };
  }
  if (asBookingBool(booking.is_round_trip) || asBookingBool(booking.has_return)) {
    return { variant: 'roundTrip', label: 'Aller-retour' };
  }
  return null;
}

function recentTripStops(trip) {
  const folded = Array.isArray(trip?.route_request_stops) ? trip.route_request_stops : [];
  const places = folded.filter((stop) => String(stop?.place || '').trim());
  if (places.length > 1) return places;
  return [
    { key: 'pickup', place: trip?.pickup_location },
    { key: 'dropoff', place: trip?.dropoff_location },
  ].filter((stop) => String(stop.place || '').trim());
}

function splitReuseDetail(detail) {
  const parts = String(detail || '')
    .split('·')
    .map((part) => part.trim())
    .filter(Boolean);
  let service = '';
  let doctor = '';
  parts.forEach((part) => {
    if (/^(dr\.?|docteur|prof\.?)\b/i.test(part)) doctor = doctor || part;
    else if (!service) service = part;
    else if (!doctor) doctor = part;
  });
  return { service, doctor };
}

function blankExtraStop(address, index, detail = '') {
  const medical = splitReuseDetail(detail);
  return applyExtraStopPlace(
    {
      key: `reuse-${Date.now()}-${index}`,
      address: '',
      asap: true,
      scheduleOpen: false,
      anchor: 'departure',
      date: '',
      time: '',
      facility: '',
      service: medical.service,
      doctor: medical.doctor,
      doctorTouched: Boolean(medical.doctor),
      access: '',
      otherDay: false,
      detailsOpen: false,
      medicalOptOut: false,
    },
    address
  );
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
  const [routeTipIndex, setRouteTipIndex] = useState(initialRouteTipIndex);
  useEffect(() => {
    try {
      sessionStorage.setItem(PORTAL_ROUTE_TIP_STORAGE_KEY, String(routeTipIndex));
    } catch {
      /* sessionStorage indisponible */
    }
  }, [routeTipIndex]);
  const [payOfferBookingId, setPayOfferBookingId] = useState(null);
  const [payingSaferpay, setPayingSaferpay] = useState(false);
  const [bookingSubmitting, setBookingSubmitting] = useState(false);
  const [submitWaitIndex, setSubmitWaitIndex] = useState(0);
  useEffect(() => {
    if (!bookingSubmitting) {
      setSubmitWaitIndex(0);
      return undefined;
    }
    const timers = [1400, 3200, 7000].map((delay, index) =>
      window.setTimeout(() => setSubmitWaitIndex(index + 1), delay)
    );
    return () => {
      timers.forEach((id) => window.clearTimeout(id));
    };
  }, [bookingSubmitting]);
  const [phoneGate, setPhoneGate] = useState({
    open: false,
    code: '',
    sending: false,
    verifying: false,
    message: '',
    error: '',
    maskedPhone: '',
  });
  const [loadingBookings, setLoadingBookings] = useState(false);
  const [departureTime, setDepartureTime] = useState('');
  const [appointmentTime, setAppointmentTime] = useState('');
  const [wheelchairOwn, setWheelchairOwn] = useState(false);
  const [wheelchairRequired, setWheelchairRequired] = useState(false);
  const [assistanceRequired, setAssistanceRequired] = useState(false);
  const [assistanceDetail, setAssistanceDetail] = useState('');
  const [hospitalService, setHospitalService] = useState('');
  const [doctorName, setDoctorName] = useState('');
  const [extraStops, setExtraStops] = useState([]);
  const facilityTouchedRef = useRef(false);
  const doctorTouchedRef = useRef(false);
  const medicalAutoOpenedRef = useRef(false);
  const locatingPickupRef = useRef(false);
  const medicalOptOutRef = useRef(false);
  const medicalDestinationKeyRef = useRef('');
  const profileSnapshotApplied = useRef(false);
  const [roundTripEnabled, setRoundTripEnabled] = useState(true);
  const [returnOtherDay, setReturnOtherDay] = useState(false);
  const [returnDate, setReturnDate] = useState('');
  const [returnTime, setReturnTime] = useState('');
  const [medicalEditorOpen, setMedicalEditorOpen] = useState(false);
  const [medicalOptOut, setMedicalOptOut] = useState(false);
  const [recurrenceEnabled, setRecurrenceEnabled] = useState(false);
  const [recurrenceEndMode, setRecurrenceEndMode] = useState('count');
  const [recurrenceType, setRecurrenceType] = useState('weekly');
  const [recurrenceSeriesLength, setRecurrenceSeriesLength] = useState(4);
  const [recurrenceEndDate, setRecurrenceEndDate] = useState('');
  const [recurrenceDays, setRecurrenceDays] = useState([]);
  const [estimateNotice, setEstimateNotice] = useState('');
  const [reservationFeedback, setReservationFeedback] = useState(null);
  const [portalReview, setPortalReview] = useState(null);

  const presentPortalCard = useCallback((apply) => {
    const reduceMotion =
      window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches === true;
    const card = document.querySelector('.bookingFormCard');
    if (reduceMotion || typeof document.startViewTransition !== 'function' || !card) {
      apply();
      return;
    }
    card.style.setProperty('view-transition-name', 'booking-form-card');
    const transition = document.startViewTransition(() => {
      flushSync(apply);
    });
    transition.finished.finally(() => {
      card.style.removeProperty('view-transition-name');
    });
  }, []);
  const [termsCatalog, setTermsCatalog] = useState([]);
  const [termsAcceptances, setTermsAcceptances] = useState([]);
  const [termsStatus, setTermsStatus] = useState(null);
  const [termsAcceptChecked, setTermsAcceptChecked] = useState(false);
  const [termsAccepting, setTermsAccepting] = useState(false);
  const [openTermsDoc, setOpenTermsDoc] = useState(null);
  const [doubleValidationEnabled, setDoubleValidationEnabled] = useState(false);
  const [conditionalOrderEnabled, setConditionalOrderEnabled] = useState(false);
  const [maximumAcceptedAmount, setMaximumAcceptedAmount] = useState('');
  const [pricingCeiling, setPricingCeiling] = useState(null);
  const [eligibleCarriers, setEligibleCarriers] = useState([]);
  const [pricingCeilingLoading, setPricingCeilingLoading] = useState(false);
  const [pricingCeilingError, setPricingCeilingError] = useState('');
  const [pendingCarrierOffer, setPendingCarrierOffer] = useState(null);
  const [confirmingTransport, setConfirmingTransport] = useState(false);
  const portalCeilingFlowEnabled = doubleValidationEnabled || conditionalOrderEnabled;
  const submitLockRef = useRef(false);
  const portalIdempotencyKeyRef = useRef(null);
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
  const [clientNoteDeparture, setClientNoteDeparture] = useState('');
  const [clientNoteArrival, setClientNoteArrival] = useState('');
  const [showMedicalFields, setShowMedicalFields] = useState(false);
  const [selectedDate, setSelectedDate] = useState('');
  const [todayDateMin, setTodayDateMin] = useState(() => localCalendarYmd());
  const departureClock = String(departureTime || '').trim();
  const appointmentClock = String(appointmentTime || '').trim();
  const hasDepartureTime = Boolean(departureClock);
  const hasAppointmentTime = Boolean(appointmentClock);
  const asapMode = !hasDepartureTime && !hasAppointmentTime;
  const scheduleAnchor = hasDepartureTime
    ? 'departure'
    : hasAppointmentTime
      ? 'arrival'
      : 'departure';
  const selectedTime = scheduleAnchor === 'arrival' ? appointmentClock : departureClock;

  const center = useMemo(() => ({ lat: 46.2044, lng: 6.1432 }), []);

  const effectiveClientId = useMemo(() => {
    return clientId || getActivePublicId();
  }, [clientId]);
  const isPortalPrivateClient =
    String(profile?.client_type || 'PORTAL').toUpperCase() === 'PORTAL';
  const requiredTermsDocs = (termsStatus?.documents || []).filter(
    (doc) => doc.acceptance_required
  );
  const termsReacceptanceRequired =
    isPortalPrivateClient && termsStatus?.status === 'reacceptance_required';
  const accessToken = useMemo(() => getActiveAccessToken({ allowLegacy: true }), []);
  const authHeaders = useMemo(
    () => (accessToken ? { headers: { Authorization: `Bearer ${accessToken}` } } : undefined),
    [accessToken]
  );

  useEffect(() => {
    // Même règle que isPortalPrivateClient (défaut PORTAL). Ne pas exiger
    // client_type déjà peuplé sinon le GET statut CGU n'est jamais lancé.
    if (!profile || !isPortalPrivateClient) {
      return undefined;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await apiClient.get('/clients/me/portal-terms-status');
        if (!cancelled) {
          setTermsStatus(res.data?.data || null);
          setTermsAcceptChecked(false);
        }
      } catch (err) {
        console.warn('Impossible de charger le statut des conditions PORTAL:', err);
        if (!cancelled) setTermsStatus(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [profile, isPortalPrivateClient]);

  useEffect(() => {
    if (!isPortalPrivateClient) {
      setDoubleValidationEnabled(false);
      setConditionalOrderEnabled(false);
      return undefined;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await apiClient.get('/clients/me/contract-flow');
        if (!cancelled) {
          setDoubleValidationEnabled(Boolean(res.data?.portal_double_validation_enabled));
          setConditionalOrderEnabled(Boolean(res.data?.portal_conditional_order_enabled));
        }
      } catch {
        if (!cancelled) {
          setDoubleValidationEnabled(false);
          setConditionalOrderEnabled(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isPortalPrivateClient]);

  useEffect(() => {
    if (!isPortalPrivateClient || !doubleValidationEnabled || conditionalOrderEnabled) {
      setPendingCarrierOffer(null);
      return undefined;
    }
    const candidates = [...(upcomingBookings || []), ...(ongoingBookings || [])];
    if (!candidates.length) {
      setPendingCarrierOffer(null);
      return undefined;
    }
    let cancelled = false;
    (async () => {
      try {
        const pending = candidates.find((b) => {
          const st = String(b.status || '').toUpperCase();
          return st === 'PENDING' && !b.company_id && !b.company_name;
        });
        if (!pending?.id) {
          if (!cancelled) setPendingCarrierOffer(null);
          return;
        }
        const res = await apiClient.get(`/clients/me/bookings/${pending.id}/pending-offer`);
        if (!cancelled) {
          const offer = res.data?.offer || null;
          if (offer && isPortalOfferConfirmable(offer)) {
            setPendingCarrierOffer({ ...offer, bookingId: pending.id });
          } else {
            setPendingCarrierOffer(null);
          }
        }
      } catch {
        if (!cancelled) setPendingCarrierOffer(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isPortalPrivateClient, doubleValidationEnabled, conditionalOrderEnabled, upcomingBookings, ongoingBookings]);

  const handleConfirmTransportOffer = async () => {
    if (
      !pendingCarrierOffer?.id ||
      !pendingCarrierOffer.bookingId ||
      confirmingTransport ||
      !isPortalOfferConfirmable(pendingCarrierOffer)
    ) {
      return;
    }
    setConfirmingTransport(true);
    setFormError(null);
    try {
      await apiClient.post(
        `/clients/me/bookings/${pendingCarrierOffer.bookingId}/confirm-transport`,
        {
          carrier_offer_id: pendingCarrierOffer.id,
          offer_content_hash: pendingCarrierOffer.offer_content_hash,
        }
      );
      setPendingCarrierOffer(null);
      toast.success(PORTAL_DV_COPY.transportConfirmed);
      // Recharge via invalidation habituelle si disponible
      window.location.reload();
    } catch (err) {
      setFormError(
        getApiErrorMessage(err, 'Impossible de confirmer cette proposition de transport.')
      );
    } finally {
      setConfirmingTransport(false);
    }
  };

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
    const extras = Array.isArray(pb.extra_stops) ? pb.extra_stops : [];
    setExtraStops(
      extras.map((stop, index) => blankExtraStop(stop?.place || '', index, stop?.detail || ''))
    );
    if (typeof pb.round_trip === 'boolean') setRoundTripEnabled(pb.round_trip);
    const detail = splitReuseDetail(pb.dropoff_detail);
    if (detail.service) setHospitalService(detail.service);
    if (detail.doctor) {
      doctorTouchedRef.current = true;
      setDoctorName(detail.doctor);
    }
    setFormError(null);
    navigate(location.pathname, { replace: true, state: null });
  }, [location.pathname, location.state, navigate]);

  useEffect(() => {
    const syncTransportDate = () => {
      const now = new Date();
      const today = localCalendarYmd(now);
      const oneHourLater = localCalendarYmd(new Date(now.getTime() + 60 * 60 * 1000));
      const fallback = oneHourLater < today ? today : oneHourLater;
      setTodayDateMin(today);
      setSelectedDate((prev) => {
        if (!prev || prev < today) return fallback;
        return prev;
      });
    };
    syncTransportDate();
    const id = window.setInterval(syncTransportDate, 30 * 1000);
    return () => window.clearInterval(id);
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

  const loadBookings = useCallback(
    async (quiet = false) => {
      if (!effectiveClientId) return null;
      if (!quiet) setLoadingBookings(true);
      try {
        const response = await apiClient.get(`/clients/${effectiveClientId}/bookings`, authHeaders);
        const bookingsArray = response.data;
        const now = Date.now();
        const isFinishedBooking = (booking) => {
          const norm = normalizeClientBookingStatus(booking?.status);
          return norm === 'completed' || norm === 'cancelled';
        };
        const ongoing = bookingsArray.filter((b) => {
          if (isFinishedBooking(b)) return false;
          const status = String(b.status || '').toLowerCase();
          if (status === 'in_progress' || status === 'assigned') return true;
          const scheduledTime = Date.parse(b.scheduled_time);
          return Number.isFinite(scheduledTime) && Math.abs(scheduledTime - now) <= 90 * 60 * 1000;
        });
        const upcoming = bookingsArray.filter(
          (b) => !isFinishedBooking(b) && Date.parse(b.scheduled_time) > now
        );
        const pastGroups = new Set();
        bookingsArray.forEach((b) => {
          const scheduledTime = Date.parse(b.scheduled_time);
          const status = String(b.status || '').toLowerCase();
          const isPast =
            isFinishedBooking(b) ||
            (Number.isFinite(scheduledTime) && scheduledTime <= now) ||
            (!Number.isFinite(scheduledTime) &&
              (status === 'completed' || status === 'cancelled' || status === 'canceled'));
          if (isPast && b.route_group_id) pastGroups.add(String(b.route_group_id));
        });
        const past = bookingsArray.filter((b) => {
          if (isFinishedBooking(b)) return true;
          const scheduledTime = Date.parse(b.scheduled_time);
          if (Number.isFinite(scheduledTime)) return scheduledTime <= now;
          return Boolean(b.route_group_id) && pastGroups.has(String(b.route_group_id));
        });
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
  /** Dernier trajet, puis l’itinéraire distinct le plus répété. */
  const recentTrips = useMemo(() => {
    const list = foldClientRouteRequests(pastBookings).filter(
      (b) => b && !isBookingCanceledForRecent(b) && String(b.dropoff_location || '').trim()
    );
    if (list.length === 0) return [];

    const instantOf = (trip) => {
      const latest = Date.parse(trip?.route_request_latest_time || trip?.scheduled_time);
      return Number.isFinite(latest) ? latest : 0;
    };
    const signatureOf = (trip) =>
      recentTripStops(trip)
        .map((stop) => normalizeRecentTripDestination(stop.place))
        .filter(Boolean)
        .join('>');

    /** @type {Map<string, { count: number, best: (typeof list)[0], latest: number }>} */
    const byRoute = new Map();
    for (const trip of list) {
      const key = signatureOf(trip);
      if (!key) continue;
      const latest = instantOf(trip);
      const current = byRoute.get(key);
      if (!current) {
        byRoute.set(key, { count: 1, best: trip, latest });
      } else {
        current.count += 1;
        if (latest >= current.latest) {
          current.best = trip;
          current.latest = latest;
        }
      }
    }

    const routes = [...byRoute.values()];
    if (routes.length === 0) return [];
    routes.sort((left, right) => right.latest - left.latest);
    const last = routes[0];
    const other = routes
      .filter((route) => route !== last)
      .sort((left, right) => right.count - left.count || right.latest - left.latest)[0];

    const out = [{ ...last.best, recentTripRole: 'Dernier' }];
    if (other && other.best.id !== last.best.id) {
      out.push({
        ...other.best,
        recentTripRole: other.count > last.count ? 'Le plus utilisé' : 'Précédent',
      });
    }
    return out;
  }, [pastBookings]);
  const hasRecentTrips = recentTrips.length > 0;

  /** Date du 1er départ (récurrent) : jour du trajet planifié, ou aujourd’hui si « dès que possible ». */
  const recurrenceStartYmd = useMemo(() => {
    if (selectedDate && String(selectedDate).trim()) {
      return String(selectedDate).trim();
    }
    return todayDateMin;
  }, [selectedDate, todayDateMin]);

  useEffect(() => {
    if (!roundTripEnabled) {
      setReturnDate('');
      setReturnTime('');
      setReturnOtherDay(false);
      return;
    }
    if (returnOtherDay) return;
    const outbound =
      selectedDate && String(selectedDate).trim() ? String(selectedDate).trim() : todayDateMin;
    setReturnDate(outbound);
  }, [roundTripEnabled, returnOtherDay, selectedDate, todayDateMin]);

  useEffect(() => {
    const text = String(destination || '').trim();
    if (medicalDestinationKeyRef.current !== text) {
      medicalDestinationKeyRef.current = text;
      if (medicalOptOutRef.current) {
        medicalOptOutRef.current = false;
        setMedicalOptOut(false);
      }
    }
    if (!text) {
      setShowMedicalFields(false);
      if (!facilityTouchedRef.current) setMedicalFacility('');
      if (!doctorTouchedRef.current) setDoctorName('');
      if (medicalAutoOpenedRef.current) {
        medicalAutoOpenedRef.current = false;
        setMedicalEditorOpen(false);
      }
      return;
    }
    if (destinationLooksMedical(text)) {
      if (medicalOptOutRef.current) return;
      medicalAutoOpenedRef.current = true;
      setShowMedicalFields(true);
      const place = classifyPortalMedicalPlace(text);
      if (!facilityTouchedRef.current) setMedicalFacility(place.facility.slice(0, 200));
      if (!doctorTouchedRef.current) setDoctorName(place.doctor.slice(0, 200));
      if (!hospitalService.trim() && !place.doctor) setMedicalEditorOpen(true);
      return;
    }
    setShowMedicalFields(false);
    if (!facilityTouchedRef.current) setMedicalFacility('');
    if (!doctorTouchedRef.current) setDoctorName('');
  }, [destination, hospitalService, doctorName]);

  useEffect(() => {
    setExtraStops((prev) => {
      let changed = false;
      const next = prev.map((stop) => {
        const updated = applyExtraStopPlace(stop, stop.address);
        if (updated.facility !== stop.facility || updated.doctor !== stop.doctor) {
          changed = true;
          return updated;
        }
        return stop;
      });
      return changed ? next : prev;
    });
  }, [extraStops]);

  const medicalTextFilled = Boolean(
    String(medicalFacility || '').trim() ||
      String(hospitalService || '').trim() ||
      String(doctorName || '').trim()
  );
  const medicalCardOn =
    !medicalOptOut && (showMedicalFields || medicalEditorOpen || medicalTextFilled);

  const setMedicalDestinationOn = (next) => {
    medicalAutoOpenedRef.current = false;
    medicalOptOutRef.current = !next;
    setMedicalOptOut(!next);
    setShowMedicalFields(next);
    setMedicalEditorOpen(next);
  };

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

  // Après recurrenceSeriesMultiplier (évite TDZ / ReferenceError).
  useEffect(() => {
    if (!isPortalPrivateClient || !portalCeilingFlowEnabled) {
      setPricingCeiling(null);
      setPricingCeilingError('');
      setMaximumAcceptedAmount('');
      setEligibleCarriers([]);
      return undefined;
    }
    const pickupText = String(pickup || '').trim();
    const dropoffText = String(destination || '').trim();
    if (!pickupText || !dropoffText) {
      setPricingCeiling(null);
      setPricingCeilingError('');
      setMaximumAcceptedAmount('');
      return undefined;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      setPricingCeilingLoading(true);
      setPricingCeilingError('');
      try {
        const res = await apiClient.post('/clients/me/portal-pricing-ceiling', {
          pickup_location: pickupText,
          dropoff_location: dropoffText,
          pickup_lat: pickupSelection?.lat ?? pickupSelection?.latitude ?? null,
          pickup_lon: pickupSelection?.lng ?? pickupSelection?.lon ?? pickupSelection?.longitude ?? null,
          dropoff_lat: destinationSelection?.lat ?? destinationSelection?.latitude ?? null,
          dropoff_lon:
            destinationSelection?.lng ??
            destinationSelection?.lon ??
            destinationSelection?.longitude ??
            null,
          is_round_trip: Boolean(roundTripEnabled),
          series_occurrences: recurrenceEnabled
            ? Math.max(1, Math.min(52, Number(recurrenceSeriesMultiplier) || 1))
            : 1,
          scheduled_time: asapMode
            ? null
            : selectedDate && selectedTime
              ? `${selectedDate}T${selectedTime}:00`
              : null,
        });
        if (cancelled) return;
        const data = res.data?.data || res.data || null;
        const max = Number(data?.maximum_accepted_amount);
        if (!Number.isFinite(max) || max <= 0) {
          setPricingCeiling(null);
          setMaximumAcceptedAmount('');
          setPricingCeilingError(
            'Impossible de calculer un prix maximum pour cette course.'
          );
          return;
        }
        setPricingCeiling(data);
        setMaximumAcceptedAmount(max.toFixed(2));
        setEligibleCarriers(
          Array.isArray(data?.eligible_carriers) ? data.eligible_carriers : []
        );
      } catch (err) {
        if (cancelled) return;
        setPricingCeiling(null);
        setMaximumAcceptedAmount('');
        setEligibleCarriers([]);
        // Timeout / API restart : message métier plafond, pas le bandeau « maintenance ».
        const status = err?.response?.status;
        const timedOut =
          err?.code === 'ECONNABORTED' ||
          err?.code === 'ERR_NETWORK' ||
          status === 502 ||
          status === 503 ||
          status === 504;
        setPricingCeilingError(
          timedOut
            ? 'Calcul du prix maximum trop long ou service momentanément saturé. Réessayez dans quelques secondes.'
            : getApiErrorMessage(
                err,
                'Impossible de calculer un prix maximum pour cette course.'
              )
        );
      } finally {
        if (!cancelled) setPricingCeilingLoading(false);
      }
    }, 450);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    isPortalPrivateClient,
    portalCeilingFlowEnabled,
    pickup,
    destination,
    pickupSelection,
    destinationSelection,
    roundTripEnabled,
    recurrenceEnabled,
    recurrenceSeriesMultiplier,
    asapMode,
    selectedDate,
    selectedTime,
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

  const closePhoneGate = () => {
    setPhoneGate({
      open: false,
      code: '',
      sending: false,
      verifying: false,
      message: '',
      error: '',
      maskedPhone: '',
    });
  };

  const openAccountPhoneVerification = async (maskedPhone = '') => {
    setPhoneGate({
      open: true,
      code: '',
      sending: true,
      verifying: false,
      message: '',
      error: '',
      maskedPhone,
    });
    try {
      const smsRes = await apiClient.post('/auth/phone/send-code', {});
      setPhoneGate((prev) => ({
        ...prev,
        sending: false,
        message: smsRes?.data?.message || 'Code SMS envoyé pour vérifier le compte.',
        maskedPhone: smsRes?.data?.masked_phone || prev.maskedPhone,
      }));
    } catch (smsErr) {
      setPhoneGate((prev) => ({
        ...prev,
        sending: false,
        error: getApiErrorMessage(smsErr, 'SMS temporairement indisponible. Réessayez.'),
      }));
    }
  };

  const handlePhoneGateVerify = async () => {
    const code = String(phoneGate.code || '').trim();
    if (!/^\d{6}$/.test(code)) {
      setPhoneGate((prev) => ({ ...prev, error: 'Entrez un code SMS à 6 chiffres.' }));
      return;
    }
    setPhoneGate((prev) => ({ ...prev, verifying: true, error: '' }));
    try {
      await apiClient.post('/auth/phone/verify-code', { code });
      closePhoneGate();
      setProfile((prev) => (prev ? { ...prev, phone_verified: true } : prev));
      toast.success('Numéro vérifié. Vous pouvez confirmer la demande.');
    } catch (err) {
      setPhoneGate((prev) => ({
        ...prev,
        verifying: false,
        error: getApiErrorMessage(err, 'Code SMS invalide ou expiré.'),
      }));
    }
  };

  const handlePhoneGateResend = async () => {
    setPhoneGate((prev) => ({ ...prev, sending: true, error: '' }));
    try {
      const smsRes = await apiClient.post('/auth/phone/send-code', {});
      setPhoneGate((prev) => ({
        ...prev,
        sending: false,
        message: smsRes?.data?.message || 'Code SMS renvoyé.',
        maskedPhone: smsRes?.data?.masked_phone || prev.maskedPhone,
      }));
    } catch (err) {
      setPhoneGate((prev) => ({
        ...prev,
        sending: false,
        error: getApiErrorMessage(err, 'Impossible d’envoyer le SMS.'),
      }));
    }
  };

  const termsDocumentTitle = (documentType) =>
    portalTermsDocumentLabel(documentType, { variant: 'order' });

  const toggleOpenTermsDoc = (doc) => {
    setOpenTermsDoc((current) =>
      current?.document_type === doc.document_type ? null : doc
    );
  };

  const renderPortalTermsDocPanel = (doc, { variant = 'update' } = {}) => {
    const version = doc.current_version || doc.terms_version;
    const label = portalTermsDocumentLabel(doc.document_type, { variant });
    const title = version ? `${label} — version ${version}` : label;
    const isOpen = openTermsDoc?.document_type === doc.document_type;
    return (
      <div
        key={doc.document_type}
        className={`portalTermsUpdateDoc${isOpen ? ' is-open' : ''}`}
        role="listitem"
      >
        <button
          type="button"
          className="portalTermsDocToggle"
          aria-expanded={isOpen}
          onClick={() => toggleOpenTermsDoc(doc)}
        >
          <span className="portalTermsDocToggleMain">
            <span className="portalTermsDocToggleLabel">{label}</span>
            {version ? (
              <span className="portalTermsDocVersion">v{version}</span>
            ) : null}
          </span>
          <span className="portalTermsDocToggleHint" aria-hidden="true">
            <span className="portalTermsDocChevron" />
          </span>
        </button>
        {isOpen ? (
          <div className="portalTermsDocPanel">
            <div className="portalTermsDocToolbar">
              <button
                type="button"
                className="portalTermsDocAction"
                onClick={() => {
                  void downloadPortalTermsDocument(doc).then((ok) => {
                    if (!ok) {
                      toast.error('Impossible de télécharger le PDF des conditions.');
                    }
                  });
                }}
              >
                PDF
              </button>
              <button
                type="button"
                className="portalTermsDocAction"
                onClick={() => printPortalTermsDocument(doc, { label: title })}
              >
                Imprimer
              </button>
            </div>
            <pre className="portalTermsBody" tabIndex={0}>
              {doc.canonical_body}
            </pre>
          </div>
        ) : null}
      </div>
    );
  };

  const requiredTermsAcceptLabel = () => {
    const labels = requiredTermsDocs.map((doc) => termsDocumentTitle(doc.document_type));
    if (labels.length === 2) {
      return 'J’ai lu et j’accepte les Conditions générales d’utilisation et les Conditions générales de transport applicables.';
    }
    if (labels.length === 1) {
      return `J’ai lu et j’accepte les ${labels[0]} applicables.`;
    }
    return 'J’ai lu et j’accepte les conditions applicables.';
  };

  const handleAcceptRequiredTerms = async () => {
    if (!termsAcceptChecked || termsAccepting) return;
    setTermsAccepting(true);
    setFormError(null);
    try {
      await apiClient.post('/clients/me/terms-acceptances', {
        accept_current_required_terms: true,
      });
      setTermsAcceptChecked(false);
      const res = await apiClient.get('/clients/me/portal-terms-status');
      setTermsStatus(res.data?.data || null);
      await loadPortalTerms();
    } catch (err) {
      setFormError(getApiErrorMessage(err, 'Impossible d’enregistrer l’acceptation.'));
    } finally {
      setTermsAccepting(false);
    }
  };

  const loadPortalTerms = async () => {
    try {
      const [catalogRes, acceptanceRes] = await Promise.all([
        apiClient.get('/clients/me/portal-terms'),
        apiClient.get('/clients/me/terms-acceptances'),
      ]);
      const catalog = catalogRes.data?.data || [];
      const rows = acceptanceRes.data?.data || [];
      setTermsCatalog(Array.isArray(catalog) ? catalog : []);
      setTermsAcceptances(Array.isArray(rows) ? rows : []);
    } catch {
      setTermsCatalog([]);
      setTermsAcceptances([]);
    }
  };

  const handleBooking = async ({ confirm = false } = {}) => {
    if (bookingSubmitting || submitLockRef.current) return;
    if (termsReacceptanceRequired) {
      setFormError('Acceptez les conditions en vigueur avant une nouvelle réservation.');
      return;
    }
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
    if (!selectedDate || !String(selectedDate).trim()) {
      setFormError('Indiquez la date du transport.');
      return;
    }
    if (!hasDepartureTime && !hasAppointmentTime) {
      setFormError('Indiquez l’heure de prise en charge ou l’heure du rendez-vous.');
      return;
    }
    /** ISO UTC pour l’API (évite le double décalage getTimezoneOffset). */
    let scheduledTimeIso = null;
    let outboundMs = Date.now();
    if (!asapMode) {
      const departureDateTime = hasDepartureTime
        ? new Date(`${selectedDate}T${departureTime}:00`)
        : null;
      const appointmentDateTime = hasAppointmentTime
        ? new Date(`${selectedDate}T${appointmentTime}:00`)
        : null;
      if (
        (departureDateTime && !Number.isFinite(departureDateTime.getTime())) ||
        (appointmentDateTime && !Number.isFinite(appointmentDateTime.getTime()))
      ) {
        setFormError('Date/heure invalide.');
        return;
      }
      const clocks = [departureDateTime, appointmentDateTime].filter(Boolean);
      if (clocks.some((when) => when.getTime() < Date.now() - 60 * 1000)) {
        setFormError('Veuillez choisir une date et une heure futures.');
        return;
      }
      if (
        departureDateTime &&
        appointmentDateTime &&
        appointmentDateTime.getTime() <= departureDateTime.getTime()
      ) {
        setFormError('Le rendez-vous doit être après l’heure de départ.');
        return;
      }
      const scheduledDateTime = appointmentDateTime || departureDateTime;
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
      const outboundDateStr = selectedDate && String(selectedDate).trim() ? selectedDate : todayDateMin;
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

    const recurrenceEndTrim = recurrenceEndMode === 'date' ? recurrenceEndDate.trim() : '';
    const recurrenceStartForSeries = selectedDate && String(selectedDate).trim() ? selectedDate : todayDateMin;

    if (recurrenceEnabled) {
      if (recurrenceType === 'custom' && recurrenceDays.length === 0) {
        setFormError('Pour des jours personnalisés, sélectionnez au moins un jour de la semaine.');
        return;
      }
      if (recurrenceEndMode === 'date') {
        if (!recurrenceEndTrim) {
          setFormError('Indiquez la date de fin de la série.');
          return;
        }
        if (recurrenceEndTrim < recurrenceStartForSeries) {
          setFormError('La date de fin de série ne peut pas précéder la date du premier départ.');
          return;
        }
      } else if (recurrenceEndMode === 'count') {
        const rep = Math.min(52, Math.max(1, Math.floor(Number(recurrenceSeriesLength)) || 0));
        if (!rep) {
          setFormError('Indiquez le nombre de transports (1 à 52).');
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
    const seriesLen =
      recurrenceEndMode === 'date' && recurrenceEndTrim
        ? estimatedOccurrencesForRecurrence({
            startYmd: recurrenceStartForSeries,
            endYmd: recurrenceEndTrim,
            recurrenceType,
            recurrenceDays,
          })
        : recurrenceEndMode === 'open'
          ? 52
          : Math.min(52, Math.max(1, Math.floor(Number(recurrenceSeriesLength)) || 1));

    const pickupAccess = String(clientNoteDeparture || '').trim();
    const dropoffAccess = String(clientNoteArrival || '').trim();
    const facilityText = String(medicalFacility || '').trim();
    const serviceText = String(hospitalService || '').trim();
    const doctorText = String(doctorName || '').trim();
    const contactDetail = composeMedicalContact(serviceText, doctorText);
    const medicalDestinationActive =
      !medicalOptOut &&
      (showMedicalFields || medicalEditorOpen || Boolean(facilityText || serviceText || doctorText));
    if (medicalDestinationActive && (!facilityText || !contactDetail)) {
      setFormError('Indiquez l’établissement et le service ou le médecin.');
      return;
    }
    if (assistanceRequired && !String(assistanceDetail || '').trim()) {
      setFormError('Indiquez le type d’assistance.');
      return;
    }
    for (const stop of extraStops) {
      if (!String(stop.address || '').trim()) {
        setFormError('Chaque étape doit avoir une adresse.');
        return;
      }
      if (String(stop.time || '').trim() && stop.otherDay && !String(stop.date || '').trim()) {
        setFormError('Indiquez le jour de l’étape dont l’heure est précisée.');
        return;
      }
      if (
        extraStopIsMedical(stop) &&
        (!String(stop.facility || '').trim() ||
          !composeMedicalContact(stop.service, stop.doctor))
      ) {
        setFormError('Chaque destination médicale doit indiquer l’établissement et le service ou le médecin.');
        return;
      }
    }
    const serviceOrDoctor = contactDetail;
    const needsLabel = mobilityNeedsLabel({
      wheelchairOwn,
      wheelchairRequired,
      assistanceRequired,
      assistanceDetail,
    });
    const contactPhone = verifiedProfilePhone(profile).slice(0, 30);
    const effectiveScheduleAnchor = asapMode ? 'departure' : scheduleAnchor;
    let recurrenceLabel = '';
    if (recurrenceEnabled) {
      const typeLabel =
        recurrenceType === 'daily'
          ? 'tous les jours'
          : recurrenceType === 'weekly'
            ? 'toutes les semaines'
            : 'jours personnalisés';
      const endTrim = String(recurrenceEndDate || '').trim();
      recurrenceLabel = endTrim
        ? `Récurrence : ${typeLabel} jusqu’au ${formatRecurrenceYmdShort(endTrim)}`
        : `Récurrence : ${typeLabel}, ${Math.min(52, Math.max(1, Math.floor(Number(recurrenceSeriesLength)) || 1))} répétition(s)`;
    }

    if (isPortalPrivateClient && !confirm) {
      const fingerprint = JSON.stringify({
        pickup,
        destination,
        scheduledTimeIso,
        scheduleAnchor: effectiveScheduleAnchor,
        amount: amountForApi,
        maximum: portalCeilingFlowEnabled ? maximumAcceptedAmount : '',
        roundTripEnabled,
        returnDate: returnDate || '',
        returnTimeIso,
        wheelchairOwn,
        wheelchairRequired,
        assistanceRequired,
        assistanceDetail: assistanceRequired ? assistanceDetail.trim() : '',
        pickupAccess,
        dropoffAccess,
        serviceOrDoctor: medicalDestinationActive ? serviceOrDoctor : '',
      });
      const storageKey = `portal-order-idempotency:${fingerprint}`;
      let key = null;
      try {
        key = sessionStorage.getItem(storageKey);
      } catch {
        key = null;
      }
      if (!key) {
        key =
          window.crypto?.randomUUID?.() ||
          `portal-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        try {
          sessionStorage.setItem(storageKey, key);
        } catch {
          /* sessionStorage indisponible : la clé reste en mémoire pour ce clic */
        }
      }
      portalIdempotencyKeyRef.current = key;
      if (portalCeilingFlowEnabled) {
        const maxNum = Number(String(maximumAcceptedAmount).replace(',', '.'));
        if (!Number.isFinite(maxNum) || maxNum <= 0) {
          setFormError(
            pricingCeilingError ||
              'Le prix maximum de cette demande n’est pas encore disponible. Vérifiez les adresses.'
          );
          return;
        }
      }
      presentPortalCard(() => setPortalReview({
        amountLabel: Number(amountForApi).toFixed(2),
        maximumLabel: portalCeilingFlowEnabled
          ? Number(String(maximumAcceptedAmount).replace(',', '.')).toFixed(2)
          : null,
        eligibleCarriers: conditionalOrderEnabled ? eligibleCarriers : [],
        narrative: buildPortalReviewNarrative({
          asap: asapMode,
          scheduleAnchor: effectiveScheduleAnchor,
          selectedDate,
          selectedTime,
          departureTime: hasDepartureTime && hasAppointmentTime ? departureTime : '',
          appointmentTime: scheduleAnchor === 'departure' && hasAppointmentTime ? appointmentTime : '',
          pickup,
          destination,
          roundTrip: roundTripEnabled,
          returnTime: returnTime || '',
          needsLabel,
          contactDetail: medicalDestinationActive ? contactDetail : '',
          contactName: buildCustomerName(profile),
          contactPhone,
          recurrenceLabel,
          extraStopLabels: extraStops.map(
            (stop, index) => `Étape ${index + 2} : ${String(stop.address || '').trim()}`
          ),
          establishmentLabel: medicalDestinationActive ? facilityText : '',
        }),
        accessLines: [
          pickupAccess && `Accès départ : ${pickupAccess}`,
          dropoffAccess && `Accès destination : ${dropoffAccess}`,
        ].filter(Boolean),
        storageKey,
        steps: buildPortalOrderSteps({
          pickup,
          destination,
          extraStops,
          roundTrip: roundTripEnabled,
          asap: asapMode,
          hasDepartureTime,
          hasAppointmentTime,
          departureTime,
          appointmentTime,
          returnTime: returnTime || '',
          pickupAccess,
          dropoffAccess,
          facilityText,
          serviceText,
          doctorText,
          medicalDestinationActive,
        }),
        transportCount:
          1 +
          extraStops.filter((stop) => String(stop.address || '').trim()).length +
          (roundTripEnabled ? 1 : 0),
        maximumAuthorizedLabel: roundChfToFiveRappen(
          (indicativeAmount != null ? indicativeAmount : MIN_CLIENT_INDICATIVE_FARE_CHF) *
            (1 +
              extraStops.filter((stop) => String(stop.address || '').trim()).length +
              (roundTripEnabled ? 1 : 0)) *
            (recurrenceEnabled && recurrenceSeriesMultiplier > 1 ? recurrenceSeriesMultiplier : 1)
        ).toFixed(2),
      }));
      setOpenTermsDoc(null);
      void loadPortalTerms();
      return;
    }

    if (portalCeilingFlowEnabled && isPortalPrivateClient && confirm) {
      const maxNum = Number(String(maximumAcceptedAmount).replace(',', '.'));
      if (!Number.isFinite(maxNum) || maxNum <= 0) {
        setFormError(
          pricingCeilingError ||
            'Le prix maximum de cette demande n’est pas encore disponible.'
        );
        return;
      }
    }

    const bookingData = {
      customer_name: buildCustomerName(profile),
      pickup_location: pickup,
      dropoff_location: destination,
      scheduled_time: scheduledTimeIso,
      asap: asapMode,
      amount: amountForApi,
      ...(portalCeilingFlowEnabled
        ? {
            // 7B.2 : le serveur recalcule le plafond ; valeur affichée non autoritaire.
            accept_pricing_ceiling: true,
          }
        : {}),
      medical_facility: medicalDestinationActive ? medicalFacility : '',
      doctor_name: medicalDestinationActive ? doctorText.slice(0, 200) : '',
      hospital_service: medicalDestinationActive ? serviceText.slice(0, 255) : '',
      ...(medicalDestinationActive
        ? {
            medical_destination: true,
            medical_destination_detail: contactDetail.slice(0, 255),
          }
        : {}),
      ...(extraStops.length
        ? {
            route_steps: extraStops.map((stop) => {
              const facility = String(stop.facility || '').trim();
              const service = String(stop.service || '').trim();
              const doctor = String(stop.doctor || '').trim();
              const detail = composeMedicalContact(service, doctor);
              const medical = extraStopIsMedical(stop);
              const stopTime = String(stop.time || '').trim();
              const planned = Boolean(stopTime);
              const stopDate =
                stop.otherDay && String(stop.date || '').trim()
                  ? String(stop.date).trim()
                  : String(selectedDate || '').trim();
              let scheduled = null;
              if (planned && stopDate && stopTime) {
                const when = new Date(`${stopDate}T${stopTime}:00`);
                if (Number.isFinite(when.getTime())) scheduled = when.toISOString();
              }
              return {
                address: String(stop.address || '').trim(),
                asap: !planned,
                scheduled_time_type: 'departure',
                scheduled_time: scheduled,
                medical_destination: medical,
                medical_facility: medical ? facility : '',
                hospital_service: medical ? service.slice(0, 255) : '',
                doctor_name: medical ? doctor.slice(0, 200) : '',
                medical_destination_detail: medical ? detail.slice(0, 255) : '',
                access_notes: String(stop.access || '').trim(),
              };
            }),
          }
        : {}),
      scheduled_time_type: effectiveScheduleAnchor,
      is_urgent: asapMode,
      wheelchair_client_has: wheelchairOwn,
      wheelchair_need: wheelchairRequired,
      needs_assistance: assistanceRequired,
      ...(assistanceRequired && assistanceDetail.trim()
        ? { assistance_detail: assistanceDetail.trim().slice(0, 200) }
        : {}),
      ...(pickupAccess ? { pickup_access_notes: pickupAccess } : {}),
      ...(dropoffAccess ? { dropoff_access_notes: dropoffAccess } : {}),
      requester_name: buildCustomerName(profile),
      ...(contactPhone ? { requester_phone: contactPhone } : {}),
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

    submitLockRef.current = true;
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

      if (isPortalPrivateClient && profile?.phone_verified === false) {
        setFormError(
          'Votre numéro de téléphone doit être vérifié avant votre prochaine réservation. La demande saisie est conservée.'
        );
        await openAccountPhoneVerification(profile?.phone || '');
        return;
      }

      const response = await apiClient.post(`/clients/${effectiveClientId}/bookings`, {
        ...bookingData,
        pickup_location: canonicalPickup,
        dropoff_location: canonicalDropoff,
        amount: previewAmount,
        preview_amount: previewAmount,
      }, {
        headers: {
          'Content-Type': 'application/json',
          ...(isPortalPrivateClient && portalIdempotencyKeyRef.current
            ? { 'Idempotency-Key': portalIdempotencyKeyRef.current }
            : {}),
        },
      });
      const root = response.data || {};
      const payload = root.data !== undefined ? root.data : root;
      const bookingId = payload.booking_id ?? root.booking_id;
      const resolvedBooking = payload.booking || root.booking || {};
      const previewPaymentRequired = Boolean(previewWorkflow.payment_required);
      const needPrivateOnlinePay =
        !isPortalPrivateClient &&
        Boolean(bookingId) &&
        (previewPaymentRequired || requiresPrivateOnlinePaymentAtBooking(resolvedBooking));

      if (needPrivateOnlinePay) {
        toast.success('Demande enregistrée. Finalisez le paiement Saferpay dans le formulaire ci-dessous.', {
          duration: 7000,
        });
      } else if (conditionalOrderEnabled && isPortalPrivateClient) {
        toast.success(
          'Commande transmise. Aucun transporteur n’a encore accepté. Le contrat n’est pas encore formé.',
          { duration: 8000 }
        );
      } else if (doubleValidationEnabled && isPortalPrivateClient) {
        toast.success(PORTAL_DV_COPY.firstClick, { duration: 8000 });
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
        reference: bookingId ? `#${bookingId}` : null,
        billingLabel: isPortalPrivateClient
          ? buildCustomerName(profile)
          : resolvedBooking.payer_label ||
            resolvedBooking.coverage_label ||
            'Payeur non défini',
      });
      if (isPortalPrivateClient && portalReview?.storageKey) {
        try {
          sessionStorage.removeItem(portalReview.storageKey);
        } catch {
          /* ignore */
        }
      }
      portalIdempotencyKeyRef.current = null;
      presentPortalCard(() => setPortalReview(null));
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
      }
      setDepartureTime('');
      setAppointmentTime('');
      setRoundTripEnabled(true);
      setReturnDate('');
      setReturnTime('');
      setReturnOtherDay(false);
      setRecurrenceEnabled(false);
      setRecurrenceType('weekly');
      setRecurrenceSeriesLength(4);
      setRecurrenceEndDate('');
      setRecurrenceDays([]);
      setMedicalFacility('');
      setHospitalService('');
      setDoctorName('');
      facilityTouchedRef.current = false;
      doctorTouchedRef.current = false;
      medicalAutoOpenedRef.current = false;
      setMedicalEditorOpen(false);
      medicalOptOutRef.current = false;
      setMedicalOptOut(false);
      const habitual = habitualMobilityFromProfile(profile);
      setWheelchairOwn(habitual.own);
      setWheelchairRequired(habitual.need);
      setAssistanceRequired(habitual.assistance);
      setAssistanceDetail(habitual.detail);
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
      const apiError = err?.response?.data?.error;
      if (apiError === 'terms_reacceptance_required') {
        setFormError('Acceptez les conditions en vigueur avant une nouvelle réservation.');
        try {
          const res = await apiClient.get('/clients/me/portal-terms-status');
          setTermsStatus(res.data?.data || null);
          setTermsAcceptChecked(false);
        } catch {
          /* le message suffit si le statut est momentanément illisible */
        }
        return;
      }
      if (apiError === 'phone_verification_required') {
        const details = err?.response?.data?.details || {};
        setFormError(
          'Votre numéro de téléphone doit être vérifié avant votre prochaine réservation. La demande saisie est conservée.'
        );
        await openAccountPhoneVerification(details.masked_phone || '');
        return;
      }
      const msg = getApiErrorMessage(err, 'Une erreur est survenue lors de la réservation.');
      setFormError(msg);
      toast.error(msg, { duration: 6000 });
    } finally {
      submitLockRef.current = false;
      setBookingSubmitting(false);
    }
  };

  const handlePayNowOffer = () => {
    if (isPortalPrivateClient || !payOfferBookingId || !payOffer || payingSaferpay) return;
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

  const missionDayLabel = useMemo(() => {
    const ymd = selectedDate && String(selectedDate).trim() ? String(selectedDate).trim() : todayDateMin;
    const parsed = new Date(`${ymd}T12:00:00`);
    if (!Number.isFinite(parsed.getTime())) return ymd;
    return parsed.toLocaleDateString('fr-CH', {
      weekday: 'long',
      day: '2-digit',
      month: 'long',
      year: 'numeric',
    });
  }, [selectedDate, todayDateMin]);
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
    const hasStops = extraStops.some((stop) => String(stop.address || '').trim());
    const detailedPath = hasStops || roundTripEnabled;
    const legs = [
      {
        key: 'pickup',
        label: 'Prise en charge',
        text: p,
        when: detailedPath
          ? asapMode
            ? 'Dès que possible'
            : hasDepartureTime
              ? formatPortalLegWhen('Prise en charge', selectedDate, departureClock)
              : 'Prise en charge à déterminer'
          : '',
      },
      {
        key: 'destination',
        label: hasStops ? 'Étape 1' : 'Destination',
        text: d,
        detail: medicalCardOn
          ? [String(hospitalService || '').trim(), String(doctorName || '').trim()]
              .filter(Boolean)
              .join(' · ')
          : '',
        when:
          detailedPath && hasAppointmentTime
            ? formatPortalLegWhen('Rendez-vous', selectedDate, appointmentClock)
            : '',
      },
    ];
    extraStops.forEach((stop, index) => {
      const address = String(stop.address || '').trim();
      if (!address) return;
      const stopTime = String(stop.time || '').trim();
      const stopDate =
        stop.otherDay && String(stop.date || '').trim()
          ? String(stop.date).trim()
          : String(selectedDate || '').trim();
      legs.push({
        key: stop.key || `stop-${index}`,
        label: `Étape ${index + 2}`,
        text: address,
        detail: extraStopIsMedical(stop)
          ? [String(stop.service || '').trim(), String(stop.doctor || '').trim()]
              .filter(Boolean)
              .join(' · ')
          : '',
        when: stopTime ? formatPortalLegWhen('Heure de départ', stopDate, stopTime) : '',
      });
    });
    if (roundTripEnabled) {
      const backDate = String(returnDate || selectedDate || '').trim();
      const backTime = String(returnTime || '').trim();
      legs.push({
        key: 'return',
        label: 'Retour',
        text: p,
        when: backTime
          ? formatPortalLegWhen('Heure de départ', backDate, backTime)
          : formatPortalLegWhen('Heure à définir', backDate, ''),
      });
    }
    const extras = [];
    const needs = mobilityNeedsLabel({
      wheelchairOwn,
      wheelchairRequired,
      assistanceRequired,
      assistanceDetail,
    });
    if (needs) extras.push(needs);
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
      legs,
      showScheduleFooter: !detailedPath,
      whenLabel: formatScheduledSummaryLabel(
        selectedDate,
        selectedTime,
        asapMode,
        asapMode ? 'departure' : scheduleAnchor
      ),
      extras,
    };
  }, [
    pickup,
    destination,
    selectedDate,
    selectedTime,
    asapMode,
    scheduleAnchor,
    departureClock,
    appointmentClock,
    hasDepartureTime,
    hasAppointmentTime,
    extraStops,
    medicalCardOn,
    hospitalService,
    doctorName,
    wheelchairOwn,
    wheelchairRequired,
    assistanceRequired,
    assistanceDetail,
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
    if (recurrenceEndMode === 'open') {
      return 'Une réservation est créée par cette demande. La série n’a pas de date de fin : elle est décrite jusqu’à 52 transports, et le transporteur confirmera les passages.';
    }
    const endTrim = recurrenceEndMode === 'date' ? String(recurrenceEndDate || '').trim() : '';
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
    recurrenceEndMode,
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

  useEffect(() => {
    const home = homeAddressFromProfile(profile);
    if (!home) return;
    setPickup((prev) => {
      if (String(prev || '').trim().length > 0) return prev;
      return home;
    });
  }, [profile]);

  useEffect(() => {
    if (!profile || profileSnapshotApplied.current) return;
    profileSnapshotApplied.current = true;
    const access = composeProfilePickupAccess(profile);
    if (access) {
      setClientNoteDeparture((prev) => (String(prev || '').trim() ? prev : access));
    }
    const habitual = habitualMobilityFromProfile(profile);
    setWheelchairOwn(habitual.own);
    setWheelchairRequired(habitual.need);
    setAssistanceRequired(habitual.assistance);
    setAssistanceDetail(habitual.detail);
  }, [profile]);

  const handleBookingAction = useCallback(
    (action, booking) => {
      if (!booking?.id) return;
      if (action === 'Recommander') {
        const stops = recentTripStops(booking);
        const hasReturn = stops.some((stop) => stop.label === 'Retour');
        const routeStops = (hasReturn ? stops.slice(0, -1) : stops).filter((stop) =>
          String(stop?.place || '').trim()
        );
        const dropoffs = routeStops.slice(1);
        const firstDetail = splitReuseDetail(dropoffs[0]?.detail);
        setPickup(String(routeStops[0]?.place || booking.pickup_location || ''));
        setDestination(String(dropoffs[0]?.place || booking.dropoff_location || ''));
        if (firstDetail.service) setHospitalService(firstDetail.service);
        if (firstDetail.doctor) {
          doctorTouchedRef.current = true;
          setDoctorName(firstDetail.doctor);
        }
        setExtraStops(
          dropoffs.slice(1).map((stop, index) => blankExtraStop(stop.place, index, stop.detail))
        );
        setRoundTripEnabled(
          hasReturn || asBookingBool(booking.is_round_trip) || asBookingBool(booking.has_return)
        );
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
        const p = profile || null;
        const named = p ? buildCustomerName(p) : '';
        const headerName = named && named !== 'Client' ? named : undefined;
        return <HeaderDashboard userName={headerName} />;
      })()}

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
                      icon={resolveLiriePointMarkerIcon(window.google?.maps, 'pickup')}
                      title="Départ"
                    />
                  )}
                  {destinationMarkerPos && (
                    <GoogleMapsAdvancedMarker
                      position={destinationMarkerPos}
                      icon={resolveLiriePointMarkerIcon(window.google?.maps, 'dropoff')}
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
            <section
              className={`leftSection card bookingFormCard${
                portalReview ? ' bookingFormCard--review' : ''
              }`}
            >
              <div className="dashboardHeader bookingHeaderPro">
                <div className="headerLeft">
                  <div className="bookingHeaderLead">
                    <div className="bookingHeaderTitles">
                      <h1 className="title bookingHeaderTitle">Demande de transport</h1>
                      <p className="bookingHeaderSubtitle">
                        {portalReview
                          ? 'Vérifiez le récapitulatif, puis confirmez ou modifiez la demande.'
                          : 'Indiquez les lieux de votre transport et les horaires associés.'}
                      </p>
                    </div>
                    <div
                      className="headerMeta headerMetaPill portalHeaderDate"
                      onClick={(event) => {
                        const target = event.target;
                        if (target instanceof Element && target.closest('[role="dialog"]')) return;
                        const toggle = event.currentTarget.querySelector(
                          'button[aria-haspopup="dialog"]'
                        );
                        if (!(toggle instanceof HTMLButtonElement)) return;
                        if (event.target === toggle || toggle.contains(event.target)) return;
                        toggle.click();
                      }}
                    >
                      <span className="portalDateIcon" aria-hidden="true">
                        <PortalMark name="calendar" />
                      </span>
                      <span className="portalDateText">
                        <span className="portalDateKicker">
                          Date du transport <span className="portalDateReq">*</span>
                        </span>
                        <span className="portalDateValue" aria-hidden="true">{missionDayLabel}</span>
                      </span>
                      <span className="portalDateChevron" aria-hidden="true">▾</span>
                      <InlineDatePicker
                        className="portalHeaderDatePicker"
                        inputId="client-booking-date"
                        value={selectedDate}
                        onChange={setSelectedDate}
                        minDate={todayDateMin}
                        ariaLabel="Date du transport"
                      />
                    </div>
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
                        {reservationFeedback.reference ? (
                          <div className="bookingFeedbackItem">
                            <dt>Référence</dt>
                            <dd>{reservationFeedback.reference}</dd>
                          </div>
                        ) : null}
                        <div className="bookingFeedbackItem">
                          <dt>Facturé à</dt>
                          <dd>{reservationFeedback.billingLabel}</dd>
                        </div>
                      </dl>
                    </div>
                  ) : null}

                  <div className="bookingFormRoute">
                  <div className="portalRouteHead">
                  <h2 className="portalRouteHeading">Parcours</h2>
                  <p className="portalRouteTip">
                    <span className="portalRouteTipMark" aria-hidden="true">
                      <PortalMark name="pin" />
                    </span>
                    <span className="portalRouteTipLabel">Conseil</span>
                    <span className="portalRouteTipItem" aria-live="polite">
                      {PORTAL_ROUTE_TIPS[routeTipIndex]}
                    </span>
                    <span className="portalRouteTipNav">
                      <button
                        type="button"
                        className="portalRouteTipNavBtn"
                        aria-label="Conseil précédent"
                        onClick={() =>
                          setRouteTipIndex(
                            (current) =>
                              (current + PORTAL_ROUTE_TIPS.length - 1) % PORTAL_ROUTE_TIPS.length
                          )
                        }
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                          <path d="m15 18-6-6 6-6" />
                        </svg>
                      </button>
                      <button
                        type="button"
                        className="portalRouteTipNavBtn"
                        aria-label="Conseil suivant"
                        onClick={() =>
                          setRouteTipIndex((current) => (current + 1) % PORTAL_ROUTE_TIPS.length)
                        }
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                          <path d="m9 18 6-6-6-6" />
                        </svg>
                      </button>
                    </span>
                  </p>
                  </div>
                  <div className="portalRouteList">
                    <article className="portalRouteStep">
                      <div className="portalRouteRail" aria-hidden="true">
                        <span className="portalRouteDot">1</span>
                        <span className="portalRouteConnector" />
                      </div>
                      <div className="portalRouteStepBody">
                        <div className="portalStepSplit">
                        <div className="portalStepPlace">
                        <div className="portalStepHeading">
                          <label htmlFor="client-dashboard-pickup" className="portalRouteStepTitle">
                            Départ
                          </label>
                        </div>
                          <div
                            className={`${homeFieldStyles.fieldGroup} bookingFormAddressFieldGroup${
                              showPickupFieldInvalid ? ` ${homeFieldStyles.fieldGroupInvalid}` : ''
                            }`}
                          >
                            <button
                              type="button"
                              className={`${homeFieldStyles.fieldIcon} portalHomeAddressBtn`}
                              aria-label="Utiliser mon adresse de domicile"
                              title="Adresse de domicile"
                              onClick={() => {
                                const home = homeAddressFromProfile(profile);
                                if (!home) {
                                  setFormError('Aucune adresse de domicile n’est enregistrée.');
                                  return;
                                }
                                const lat = profile?.domicile?.lat ?? profile?.domicile_lat;
                                const lon = profile?.domicile?.lon ?? profile?.domicile_lon;
                                setPickup(home);
                                setPickupSelection(
                                  lat != null && lat !== '' && lon != null && lon !== ''
                                    ? { validated: true, lat, lon }
                                    : null
                                );
                                setFormError(null);
                              }}
                            >
                              <svg
                                width="18"
                                height="18"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <path d="M4 10.5 12 4l8 6.5V20a1 1 0 0 1-1 1h-5v-6H10v6H5a1 1 0 0 1-1-1z" />
                              </svg>
                            </button>
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
                            {pickup ? (
                              <button
                                type="button"
                                className="portalFieldAction"
                                aria-label="Effacer le lieu de prise en charge"
                                onClick={() => {
                                  setPickup('');
                                  setPickupSelection(null);
                                }}
                              >
                                ×
                              </button>
                            ) : null}
                            <button
                              type="button"
                              className="portalFieldAction"
                              aria-label="Utiliser ma position actuelle"
                              title="Ma position"
                              onClick={() => {
                                if (locatingPickupRef.current) return;
                                if (!navigator.geolocation) {
                                  setFormError('La géolocalisation n’est pas disponible sur cet appareil.');
                                  return;
                                }
                                locatingPickupRef.current = true;
                                setFormError(null);
                                navigator.geolocation.getCurrentPosition(
                                  (position) => {
                                    const lat = position.coords.latitude;
                                    const lon = position.coords.longitude;
                                    apiClient
                                      .get('/geocode/reverse', { params: { lat, lon } })
                                      .then((response) => {
                                        const item = response?.data || {};
                                        const label = String(item.label || item.address || '').trim();
                                        if (!label) {
                                          setFormError('Aucune adresse trouvée pour cette position.');
                                          return;
                                        }
                                        setPickup(label);
                                        setPickupSelection({
                                          validated: true,
                                          lat: item.lat ?? lat,
                                          lon: item.lon ?? lon,
                                        });
                                        setFormError(null);
                                      })
                                      .catch(() => {
                                        setFormError(
                                          'Impossible de déterminer l’adresse à partir de votre position.'
                                        );
                                      })
                                      .finally(() => {
                                        locatingPickupRef.current = false;
                                      });
                                  },
                                  (err) => {
                                    locatingPickupRef.current = false;
                                    setFormError(
                                      err?.code === 1
                                        ? 'Autorisez la localisation pour remplir le lieu de prise en charge.'
                                        : 'Impossible d’obtenir votre position.'
                                    );
                                  },
                                  { enableHighAccuracy: true, timeout: 12000, maximumAge: 30000 }
                                );
                              }}
                            >
                              <PortalMark name="crosshair" />
                            </button>
                          </div>
                        <details className="portalAccessFold">
                          <summary>Préciser l’accès (optionnel)</summary>
                          <input
                            type="text"
                            className="input bookingClientNoteLineInput"
                            value={clientNoteDeparture}
                            onChange={(e) =>
                              setClientNoteDeparture(e.target.value.slice(0, MAX_CLIENT_NOTE_LEG))
                            }
                            maxLength={MAX_CLIENT_NOTE_LEG}
                            placeholder="Entrée, code, étage, instructions…"
                            autoComplete="off"
                            aria-label="Accès au départ"
                          />
                        </details>
                        </div>
                        <aside className="portalTimeCard">
                          <p className="portalTimeCardTitle">
                            Heure de prise en charge
                            {!hasDepartureTime && !hasAppointmentTime ? (
                              <span className="portalDateReq">*</span>
                            ) : null}
                          </p>
                          <InlineTimePicker
                            inputId="client-booking-departure-time"
                            value={departureTime}
                            onChange={setDepartureTime}
                            ariaLabel="Heure de départ"
                          />
                        </aside>
                        </div>
                      </div>
                    </article>
                    <article className="portalRouteStep">
                      <div className="portalRouteRail" aria-hidden="true">
                        <span className="portalRouteDot">2</span>
                        {extraStops.length > 0 || roundTripEnabled ? (
                          <span className="portalRouteConnector" />
                        ) : null}
                      </div>
                      <div className="portalRouteStepBody">
                        <div className="portalStepSplit">
                        <div className="portalStepPlace">
                        <div className="portalStepHeading">
                          <label htmlFor="client-dashboard-dropoff" className="portalRouteStepTitle">
                            {extraStops.length > 0 ? 'Étape 1' : 'Destination'}
                          </label>
                        </div>
                        <div className={`${homeFieldStyles.fieldBlock} bookingFormAddressFieldBlock`}>
                          <div
                            className={`${homeFieldStyles.fieldGroup} bookingFormAddressFieldGroup${
                              showDropoffFieldInvalid ? ` ${homeFieldStyles.fieldGroupInvalid}` : ''
                            }`                            }
                          >
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
                            {destination ? (
                              <button
                                type="button"
                                className="portalFieldAction"
                                aria-label="Effacer la destination"
                                onClick={() => {
                                  setDestination('');
                                  setDestinationSelection(null);
                                }}
                              >
                                ×
                              </button>
                            ) : null}
                            <button
                              type="button"
                              className="portalFieldAction"
                              aria-label="Rechercher la destination"
                              onClick={() => document.getElementById('client-dashboard-dropoff')?.focus()}
                            >
                              <PortalMark name="crosshair" />
                            </button>
                          </div>
                        </div>
                        <div
                          className={`portalMedicalCard${medicalCardOn ? '' : ' portalMedicalCard--closed'}`}
                        >
                          <div className="portalMedicalCardHead">
                            <button
                              type="button"
                              role="switch"
                              className={`portalMedicalSwitch${medicalCardOn ? ' is-on' : ''}`}
                              aria-checked={medicalCardOn}
                              aria-label="Destination médicale"
                              onClick={() => setMedicalDestinationOn(!medicalCardOn)}
                            />
                            <span>Destination médicale</span>
                          </div>
                          {medicalCardOn ? (
                            <div className="portalMedicalFields">
                              <label htmlFor="client-booking-establishment">
                                <span className="portalMedicalFieldLabel">
                                  Établissement <span className="portalMedicalReq">*</span>
                                </span>
                                <input
                                  id="client-booking-establishment"
                                  className="portalMedicalInput"
                                  value={medicalFacility}
                                  onChange={(e) => {
                                    facilityTouchedRef.current = true;
                                    setMedicalFacility(e.target.value.slice(0, 200));
                                  }}
                                  maxLength={200}
                                  placeholder="Clinique de Carouge"
                                  autoComplete="organization"
                                />
                              </label>
                              <label htmlFor="client-booking-service">
                                <span className="portalMedicalFieldLabel">
                                  Service
                                  {!hospitalService.trim() && !doctorName.trim() ? (
                                    <span className="portalMedicalReq">*</span>
                                  ) : null}
                                </span>
                                <input
                                  id="client-booking-service"
                                  className="portalMedicalInput"
                                  value={hospitalService}
                                  onChange={(e) => setHospitalService(e.target.value.slice(0, 255))}
                                  maxLength={255}
                                  placeholder="Cardiologie"
                                  autoComplete="off"
                                />
                              </label>
                              <label htmlFor="client-booking-doctor">
                                <span className="portalMedicalFieldLabel">
                                  Médecin
                                  {!hospitalService.trim() && !doctorName.trim() ? (
                                    <span className="portalMedicalReq">*</span>
                                  ) : null}
                                </span>
                                <input
                                  id="client-booking-doctor"
                                  className="portalMedicalInput"
                                  value={doctorName}
                                  onChange={(e) => {
                                    doctorTouchedRef.current = true;
                                    setDoctorName(e.target.value.slice(0, 200));
                                  }}
                                  maxLength={200}
                                  placeholder="Dr Martin"
                                  autoComplete="off"
                                />
                              </label>
                            </div>
                          ) : null}
                        </div>
                        <details className="portalAccessFold">
                          <summary>Préciser l’accès (optionnel)</summary>
                          <input
                            type="text"
                            className="input bookingClientNoteLineInput"
                            value={clientNoteArrival}
                            onChange={(e) =>
                              setClientNoteArrival(e.target.value.slice(0, MAX_CLIENT_NOTE_LEG))
                            }
                            maxLength={MAX_CLIENT_NOTE_LEG}
                            placeholder="Service, étage, accueil, instructions…"
                            autoComplete="off"
                            aria-label="Accès à destination"
                          />
                        </details>
                        </div>
                        <aside className="portalTimeCard">
                          <p className="portalTimeCardTitle">
                            Rendez-vous
                            {!hasDepartureTime && !hasAppointmentTime ? (
                              <span className="portalDateReq">*</span>
                            ) : null}
                          </p>
                          <InlineTimePicker
                            inputId="client-booking-time"
                            value={appointmentTime}
                            onChange={setAppointmentTime}
                            ariaLabel="Heure du rendez-vous"
                          />
                        </aside>
                        </div>
                      </div>
                    </article>

                  {extraStops.map((stop, index) => {
                    const stopMedicalOn = extraStopIsMedical(stop);
                    const stopLabel = `Étape ${index + 2}`;
                    return (
                    <article className="portalRouteStep" key={stop.key}>
                      <div className="portalRouteRail" aria-hidden="true">
                        <span className="portalRouteDot">{index + 3}</span>
                        {index < extraStops.length - 1 || roundTripEnabled ? (
                          <span className="portalRouteConnector" />
                        ) : null}
                      </div>
                      <div className="portalRouteStepBody">
                        <div className="portalStepSplit">
                          <div className="portalStepPlace">
                            <div className="portalStepHeading">
                              <label
                                htmlFor={`client-booking-stop-${stop.key}`}
                                className="portalRouteStepTitle"
                              >
                                {stopLabel}
                              </label>
                              <button
                                type="button"
                                className="portalExtraStopRemove"
                                onClick={() =>
                                  setExtraStops((prev) => prev.filter((item) => item.key !== stop.key))
                                }
                              >
                                Retirer
                              </button>
                            </div>
                            <div className={`${homeFieldStyles.fieldBlock} bookingFormAddressFieldBlock`}>
                              <div className={`${homeFieldStyles.fieldGroup} bookingFormAddressFieldGroup`}>
                                <AddressAutocomplete
                                  flushInput
                                  inputId={`client-booking-stop-${stop.key}`}
                                  name={`stop-${stop.key}`}
                                  value={stop.address}
                                  onChange={(e) =>
                                    setExtraStops((prev) =>
                                      prev.map((item) =>
                                        item.key === stop.key
                                          ? applyExtraStopPlace(item, e.target.value)
                                          : item
                                      )
                                    )
                                  }
                                  onSelect={(item) =>
                                    setExtraStops((prev) =>
                                      prev.map((entry) =>
                                        entry.key === stop.key
                                          ? applyExtraStopPlace(entry, item.label || '')
                                          : entry
                                      )
                                    )
                                  }
                                  placeholder="Ex: Clinique de Carouge"
                                />
                                {stop.address ? (
                                  <button
                                    type="button"
                                    className="portalFieldAction"
                                    aria-label={`Effacer l’adresse de l’étape ${index + 2}`}
                                    onClick={() =>
                                      setExtraStops((prev) =>
                                        prev.map((item) =>
                                          item.key === stop.key ? applyExtraStopPlace(item, '') : item
                                        )
                                      )
                                    }
                                  >
                                    ×
                                  </button>
                                ) : null}
                                <button
                                  type="button"
                                  className="portalFieldAction"
                                  aria-label={`Rechercher l’adresse de l’étape ${index + 2}`}
                                  onClick={() =>
                                    document.getElementById(`client-booking-stop-${stop.key}`)?.focus()
                                  }
                                >
                                  <PortalMark name="crosshair" />
                                </button>
                              </div>
                            </div>
                            <div
                              className={`portalMedicalCard${stopMedicalOn ? '' : ' portalMedicalCard--closed'}`}
                            >
                              <div className="portalMedicalCardHead">
                                <button
                                  type="button"
                                  role="switch"
                                  className={`portalMedicalSwitch${stopMedicalOn ? ' is-on' : ''}`}
                                  aria-checked={stopMedicalOn}
                                  aria-label="Destination médicale"
                                  onClick={() =>
                                    setExtraStops((prev) =>
                                      prev.map((item) =>
                                        item.key === stop.key
                                          ? {
                                              ...item,
                                              medicalOptOut: stopMedicalOn,
                                              detailsOpen: !stopMedicalOn,
                                            }
                                          : item
                                      )
                                    )
                                  }
                                />
                                <span>Destination médicale</span>
                              </div>
                              {stopMedicalOn ? (
                                <div className="portalMedicalFields">
                                  <label htmlFor={`client-booking-stop-facility-${stop.key}`}>
                                    <span className="portalMedicalFieldLabel">
                                      Établissement <span className="portalMedicalReq">*</span>
                                    </span>
                                    <input
                                      id={`client-booking-stop-facility-${stop.key}`}
                                      className="portalMedicalInput"
                                      value={stop.facility}
                                      onChange={(e) =>
                                        setExtraStops((prev) =>
                                          prev.map((item) =>
                                            item.key === stop.key
                                              ? {
                                                  ...item,
                                                  facilityTouched: true,
                                                  facility: e.target.value.slice(0, 200),
                                                }
                                              : item
                                          )
                                        )
                                      }
                                      maxLength={200}
                                      placeholder="Clinique de Carouge"
                                      autoComplete="organization"
                                    />
                                  </label>
                                  <label htmlFor={`client-booking-stop-service-${stop.key}`}>
                                    <span className="portalMedicalFieldLabel">
                                      Service
                                      {!String(stop.service || '').trim() && !String(stop.doctor || '').trim() ? (
                                        <span className="portalMedicalReq">*</span>
                                      ) : null}
                                    </span>
                                    <input
                                      id={`client-booking-stop-service-${stop.key}`}
                                      className="portalMedicalInput"
                                      value={stop.service}
                                      onChange={(e) =>
                                        setExtraStops((prev) =>
                                          prev.map((item) =>
                                            item.key === stop.key
                                              ? { ...item, service: e.target.value.slice(0, 255) }
                                              : item
                                          )
                                        )
                                      }
                                      maxLength={255}
                                      placeholder="Cardiologie"
                                      autoComplete="off"
                                    />
                                  </label>
                                  <label htmlFor={`client-booking-stop-doctor-${stop.key}`}>
                                    <span className="portalMedicalFieldLabel">
                                      Médecin
                                      {!String(stop.service || '').trim() && !String(stop.doctor || '').trim() ? (
                                        <span className="portalMedicalReq">*</span>
                                      ) : null}
                                    </span>
                                    <input
                                      id={`client-booking-stop-doctor-${stop.key}`}
                                      className="portalMedicalInput"
                                      value={stop.doctor}
                                      onChange={(e) =>
                                        setExtraStops((prev) =>
                                          prev.map((item) =>
                                            item.key === stop.key
                                              ? {
                                                  ...item,
                                                  doctorTouched: true,
                                                  doctor: e.target.value.slice(0, 200),
                                                }
                                              : item
                                          )
                                        )
                                      }
                                      maxLength={200}
                                      placeholder="Dr Martin"
                                      autoComplete="off"
                                    />
                                  </label>
                                </div>
                              ) : null}
                            </div>
                            <details className="portalAccessFold">
                              <summary>Préciser l’accès (optionnel)</summary>
                              <input
                                type="text"
                                className="input bookingClientNoteLineInput"
                                value={stop.access}
                                onChange={(e) =>
                                  setExtraStops((prev) =>
                                    prev.map((item) =>
                                      item.key === stop.key
                                        ? { ...item, access: e.target.value.slice(0, MAX_CLIENT_NOTE_LEG) }
                                        : item
                                    )
                                  )
                                }
                                maxLength={MAX_CLIENT_NOTE_LEG}
                                placeholder="Entrée, étage, accueil…"
                                autoComplete="off"
                                aria-label={`Accès à l’étape ${index + 2}`}
                              />
                            </details>
                          </div>
                          <aside className="portalTimeCard">
                            <div className="portalTimeCardHead">
                              <p className="portalTimeCardTitle">Heure de départ</p>
                              {!stop.otherDay ? (
                                <button
                                  type="button"
                                  className="portalOtherDay"
                                  onClick={() =>
                                    setExtraStops((prev) =>
                                      prev.map((item) =>
                                        item.key === stop.key ? { ...item, otherDay: true } : item
                                      )
                                    )
                                  }
                                >
                                  Autre jour
                                </button>
                              ) : null}
                            </div>
                            <InlineTimePicker
                              value={stop.time}
                              onChange={(hhmm) =>
                                setExtraStops((prev) =>
                                  prev.map((item) =>
                                    item.key === stop.key ? { ...item, time: hhmm } : item
                                  )
                                )
                              }
                              ariaLabel={`Heure de départ de l’étape ${index + 2}`}
                            />
                            {stop.otherDay ? (
                              <input
                                type="date"
                                className="input"
                                aria-label={`Jour de l’étape ${index + 2}`}
                                value={stop.date}
                                min={selectedDate || todayDateMin}
                                onChange={(e) =>
                                  setExtraStops((prev) =>
                                    prev.map((item) =>
                                      item.key === stop.key ? { ...item, date: e.target.value } : item
                                    )
                                  )
                                }
                              />
                            ) : null}
                          </aside>
                        </div>
                      </div>
                    </article>
                    );
                  })}

                  </div>
                  <div className={`portalRouteActions${roundTripEnabled ? ' portalRouteActions--linked' : ''}`}>
                  {roundTripEnabled ? (
                    <div className="portalRouteRail" aria-hidden="true">
                      <span className="portalRouteConnector" />
                    </div>
                  ) : null}
                  <div className="portalAddStopWrap">
                  <button
                    type="button"
                    className="portalAddStop"
                    onClick={() =>
                      setExtraStops((prev) => [
                        ...prev,
                        {
                          key: `${Date.now()}-${prev.length}`,
                          address: '',
                          asap: true,
                          scheduleOpen: false,
                          anchor: 'departure',
                          date: '',
                          time: '',
                          facility: '',
                          service: '',
                          doctor: '',
                          access: '',
                          otherDay: false,
                          detailsOpen: false,
                          medicalOptOut: false,
                        },
                      ])
                    }
                  >
                    <PortalMark name="plus" />
                    Ajouter une destination
                  </button>
                  </div>
                  <div className="portalRoundTripToggle">
                    <button
                      type="button"
                      role="switch"
                      className={`portalMedicalSwitch${roundTripEnabled ? ' is-on' : ''}`}
                      aria-checked={roundTripEnabled}
                      aria-label="Aller-retour"
                      onClick={() => setRoundTripEnabled((current) => !current)}
                    />
                    <span>Aller-retour</span>
                  </div>
                  </div>
                  {roundTripEnabled ? (
                    <article className="portalRouteStep portalRouteReturnStep">
                      <div className="portalRouteRail" aria-hidden="true">
                        <span className="portalRouteDot portalRouteDotReturn">
                          {extraStops.length + 3}
                        </span>
                      </div>
                      <div className="portalRouteStepBody">
                        <div className="portalStepSplit">
                        <div className="portalStepPlace">
                        <div className="portalStepHeading">
                          <p className="portalRouteStepTitle" id="client-booking-return-label">
                            Retour
                          </p>
                          <span className="portalStepKicker">Identique au point de départ</span>
                        </div>
                        <div className={`${homeFieldStyles.fieldBlock} bookingFormAddressFieldBlock`}>
                        <div
                          className={`${homeFieldStyles.fieldGroup} bookingFormAddressFieldGroup portalRouteReadonlyGroup`}
                        >
                          <input
                            type="text"
                            className="portalRouteReadonly"
                            readOnly
                            tabIndex={-1}
                            aria-label="Adresse de retour, identique au départ"
                            value={pickup}
                          />
                        </div>
                        </div>
                        <details className="portalAccessFold">
                          <summary>Préciser l’accès (optionnel)</summary>
                          <input
                            type="text"
                            className="input bookingClientNoteLineInput"
                            value={clientNoteDeparture}
                            onChange={(e) =>
                              setClientNoteDeparture(e.target.value.slice(0, MAX_CLIENT_NOTE_LEG))
                            }
                            maxLength={MAX_CLIENT_NOTE_LEG}
                            placeholder="Entrée, code, étage, instructions…"
                            autoComplete="off"
                            aria-label="Accès au retour"
                          />
                        </details>
                        </div>
                        <aside className="portalTimeCard">
                          <div className="portalTimeCardHead">
                            <p className="portalTimeCardTitle">Heure de départ (optionnel)</p>
                            {!returnOtherDay ? (
                              <button
                                type="button"
                                className="portalOtherDay"
                                onClick={() => setReturnOtherDay(true)}
                              >
                                Autre jour
                              </button>
                            ) : null}
                          </div>
                          <InlineTimePicker
                            inputId="client-booking-return-time"
                            value={returnTime}
                            onChange={setReturnTime}
                            ariaLabel="Heure de départ du retour, facultatif"
                          />
                          <input
                            id="client-booking-return-date"
                            type={returnOtherDay ? 'date' : 'hidden'}
                            className={returnOtherDay ? 'input' : undefined}
                            aria-label={returnOtherDay ? 'Jour du retour' : undefined}
                            value={returnDate}
                            min={returnOtherDay ? selectedDate || todayDateMin : undefined}
                            onChange={(e) => setReturnDate(e.target.value)}
                          />
                        </aside>
                        </div>
                      </div>
                    </article>
                  ) : null}
                  </div>

                  <div className="bookingFormWhen">
                  <h2 className="portalRouteHeading">Options trajet</h2>
                  <div className="portalRecurrenceRow">
                    <div className="portalRecurrenceCopy">
                      <p className="portalRecurrenceTitle">
                        Récurrence <span>(optionnel)</span>
                      </p>
                      <p className="portalRecurrenceLead">
                        Plusieurs transports sur le même trajet.
                      </p>
                    </div>
                    <button
                      type="button"
                      role="switch"
                      className={`portalMedicalSwitch${recurrenceEnabled ? ' is-on' : ''}`}
                      aria-checked={recurrenceEnabled}
                      aria-label="Récurrence"
                      onClick={() => setRecurrenceEnabled((current) => !current)}
                    />
                  </div>
                  {recurrenceEnabled ? (
                      <div className="bookingRecurrenceConfig">
                        <div className="portalRecurBlock">
                          <p className="portalRecurLabel" id="client-booking-recurrence-type-label">
                            Répéter
                          </p>
                          <div
                            id="client-booking-recurrence-type"
                            className="portalRecurSegments"
                            role="radiogroup"
                            aria-labelledby="client-booking-recurrence-type-label"
                          >
                            {[
                              ['daily', 'Tous les jours'],
                              ['weekly', 'Chaque semaine'],
                              ['custom', 'Jours choisis'],
                            ].map(([value, label]) => (
                              <button
                                key={value}
                                type="button"
                                role="radio"
                                aria-checked={recurrenceType === value}
                                className={recurrenceType === value ? 'is-on' : undefined}
                                onClick={() => setRecurrenceType(value)}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </div>
                        {recurrenceType === 'custom' ? (
                          <div className="portalRecurBlock">
                            <p className="portalRecurLabel" id="client-booking-recurrence-days-label">
                              Jours concernés
                            </p>
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
                          </div>
                        ) : null}
                        <div className="portalRecurBlock">
                          <p className="portalRecurLabel" id="portal-repeat-end-label">
                            La série se termine
                          </p>
                          <div
                            className="portalRecurSegments"
                            role="radiogroup"
                            aria-labelledby="portal-repeat-end-label"
                          >
                            {[
                              ['count', 'Après'],
                              ['date', 'À une date'],
                              ['open', 'Sans fin'],
                            ].map(([value, label]) => (
                              <button
                                key={value}
                                type="button"
                                role="radio"
                                name="portal-repeat-end"
                                aria-checked={recurrenceEndMode === value}
                                className={recurrenceEndMode === value ? 'is-on' : undefined}
                                onClick={() => setRecurrenceEndMode(value)}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                          {recurrenceEndMode === 'count' ? (
                            <div className="portalRecurValue">
                              <span>Nombre de transports</span>
                              <div className="portalRecurStepper">
                                <button
                                  type="button"
                                  aria-label="Diminuer le nombre de transports"
                                  onClick={() =>
                                    setRecurrenceSeriesLength((current) => Math.max(1, current - 1))
                                  }
                                >
                                  −
                                </button>
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
                                  className="bookingRecurrenceCountInput"
                                  aria-label="Nombre de transports"
                                />
                                <button
                                  type="button"
                                  aria-label="Augmenter le nombre de transports"
                                  onClick={() =>
                                    setRecurrenceSeriesLength((current) => Math.min(52, current + 1))
                                  }
                                >
                                  +
                                </button>
                              </div>
                            </div>
                          ) : null}
                          {recurrenceEndMode === 'date' ? (
                            <div className="portalRecurValue portalRecurValue--date">
                              <InlineDatePicker
                                inputId="client-booking-recurrence-end"
                                className="portalRecurDate"
                                value={recurrenceEndDate}
                                minDate={recurrenceStartYmd}
                                onChange={setRecurrenceEndDate}
                                ariaLabel="Date de fin de la série"
                              />
                            </div>
                          ) : null}
                        </div>
                        {recurrenceHintText ? (
                          <p id="client-booking-recurrence-hint" className="bookingRecurrenceHint">
                            {recurrenceHintText}
                          </p>
                        ) : null}
                      </div>
                    ) : null}
                  </div>




                  {!isPortalPrivateClient && payOfferBookingId != null && payOffer ? (
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

                  {portalCeilingFlowEnabled &&
                  isPortalPrivateClient &&
                  !portalReview ? (
                    <div className="portalMaximumField">
                      <div className="portalMaximumFieldLabel">
                        Prix maximum de la demande
                      </div>
                      <div className="portalMaximumFieldValue" aria-live="polite">
                        {pricingCeilingLoading
                          ? 'Calcul…'
                          : maximumAcceptedAmount
                            ? `CHF ${Number(maximumAcceptedAmount).toFixed(2)}`
                            : '—'}
                      </div>
                      <p id="portal-maximum-accepted-hint" className="portalMaximumFieldHint">
                        {pricingCeilingError ||
                          (Number(pricingCeiling?.series_occurrences) > 1
                            ? `Calculé selon les tarifs applicables (trajet × ${Number(
                                pricingCeiling.series_occurrences
                              )} passages décrits sur cette demande). Le prix définitif sera celui de l’entreprise que vous confirmerez et ne dépassera pas ce montant.`
                            : 'Calculé selon les tarifs applicables des entreprises de transport susceptibles de prendre en charge cette demande. Le prix définitif sera celui de l’entreprise que vous confirmerez et ne dépassera pas ce montant.')}
                      </p>
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

                  {phoneGate.open ? (
                    <div className="phoneGateBox" role="dialog" aria-label="Vérification du numéro de compte">
                      <p>
                        Votre numéro
                        {phoneGate.maskedPhone ? ` ${phoneGate.maskedPhone}` : ''} doit être
                        vérifié une seule fois avant la prochaine réservation. Cette étape
                        concerne le compte. La demande saisie reste affichée.
                      </p>
                      {phoneGate.message ? <p className="networkHint">{phoneGate.message}</p> : null}
                      {phoneGate.error ? <p className="error">{phoneGate.error}</p> : null}
                      <div className="formActions">
                        <input
                          type="text"
                          inputMode="numeric"
                          maxLength={6}
                          className="input"
                          placeholder="Code à 6 chiffres"
                          value={phoneGate.code}
                          onChange={(e) =>
                            setPhoneGate((prev) => ({
                              ...prev,
                              code: e.target.value.replace(/[^\d]/g, ''),
                            }))
                          }
                          disabled={phoneGate.sending || phoneGate.verifying}
                        />
                        <button
                          type="button"
                          className={homeFieldStyles.ctaButton}
                          onClick={handlePhoneGateVerify}
                          disabled={phoneGate.sending || phoneGate.verifying}
                        >
                          Valider le code
                        </button>
                        <button
                          type="button"
                          className="ghostButton"
                          onClick={handlePhoneGateResend}
                          disabled={phoneGate.sending || phoneGate.verifying}
                        >
                          Renvoyer le SMS
                        </button>
                      </div>
                    </div>
                  ) : null}

                  {isPortalPrivateClient && portalReview ? (
                    <section
                      className="portalOrderReview"
                      aria-labelledby="portal-order-review-title"
                    >
                      <h2 id="portal-order-review-title" className="portalOrderReviewTitle">
                        Récapitulatif de la demande
                      </h2>
                      {(() => {
                        const narrative = portalReview.narrative || [];
                        const dateLine = narrative.find((line) => line.kind === 'date');
                        const legs = tripDraftSummary?.legs?.length
                          ? tripDraftSummary.legs
                          : (portalReview.steps || []).map((step) => ({
                              key: step.key,
                              label: step.label,
                              text: step.address,
                              detail: '',
                              when: (step.lines || []).join(' · '),
                            }));
                        const extras = [
                          ...(tripDraftSummary?.extras || []),
                          ...narrative
                            .filter(
                              (line) =>
                                line.kind === 'meta' &&
                                String(line.text || '').startsWith('Contact')
                            )
                            .map((line) => line.text),
                          ...(portalReview.accessLines || []),
                        ].filter(Boolean);
                        return (
                          <>
                            {dateLine ? (
                              <p className="portalOrderReviewDate">{dateLine.text}</p>
                            ) : null}
                            {legs.length ? (
                              <ol className="portalOrderRoute">
                                {legs.map((leg) => (
                                  <li key={leg.key}>
                                    <span>{leg.label}</span>
                                    <strong>{leg.text}</strong>
                                    {leg.detail ? (
                                      <p className="portalOrderRouteNote">{leg.detail}</p>
                                    ) : null}
                                    {leg.when ? (
                                      <p className="portalOrderRouteNote portalOrderRouteNote--when">
                                        {leg.when}
                                      </p>
                                    ) : null}
                                  </li>
                                ))}
                              </ol>
                            ) : null}
                            {tripDraftSummary?.showScheduleFooter && tripDraftSummary.whenLabel ? (
                              <p className="portalOrderReviewSchedule">{tripDraftSummary.whenLabel}</p>
                            ) : null}
                            {extras.length ? (
                              <div className="portalOrderFacts">
                                {extras.map((line) => (
                                  <p key={line} className="portalOrderFact">
                                    {line}
                                  </p>
                                ))}
                              </div>
                            ) : null}
                          </>
                        );
                      })()}
                      {portalReview.transportCount > 2 && portalReview.maximumAuthorizedLabel ? (
                        <p className="portalOrderEstimate">
                          Prix maximum autorisé : CHF {portalReview.maximumAuthorizedLabel}
                        </p>
                      ) : portalCeilingFlowEnabled && portalReview.maximumLabel ? (
                        <p className="portalOrderEstimate">
                          Prix maximum accepté (pas le prix final) : CHF{' '}
                          {portalReview.maximumLabel}
                        </p>
                      ) : (
                        <p className="portalOrderEstimate">
                          Estimation indicative : CHF {portalReview.amountLabel}
                        </p>
                      )}
                      {conditionalOrderEnabled &&
                      Array.isArray(portalReview.eligibleCarriers) &&
                      portalReview.eligibleCarriers.length > 0 ? (
                        <details className="portalOrderEstimate">
                          <summary>
                            Entreprises susceptibles de prendre en charge cette demande
                          </summary>
                          <ul>
                            {portalReview.eligibleCarriers.map((c) => (
                              <li key={c.company_id || c.legal_name}>
                                {c.legal_name}
                              </li>
                            ))}
                          </ul>
                          <p className="portalOrderEstimateNote">
                            Le premier transporteur qui accepte dans les conditions de
                            votre commande deviendra votre cocontractant.
                          </p>
                        </details>
                      ) : null}
                      {portalReview.transportCount > 2 ? (
                        <p className="portalOrderEstimateNote">
                          {portalReview.transportCount} transports. En confirmant, vous
                          acceptez de payer jusqu’à ce montant.
                        </p>
                      ) : conditionalOrderEnabled && portalReview.maximumLabel ? (
                        <p className="portalOrderEstimateNote">
                          Cette commande vous engage si une entreprise de transport
                          l’accepte. Le contrat sera alors automatiquement conclu avec
                          cette entreprise à son propre tarif, dans la limite de CHF{' '}
                          {portalReview.maximumLabel}. Le montant de CHF{' '}
                          {portalReview.maximumLabel} est un plafond et non le prix de
                          la course.
                        </p>
                      ) : (
                        <p className="portalOrderEstimateNote">
                          {doubleValidationEnabled
                            ? 'Ce plafond n’est pas le prix à payer. Le montant définitif sera celui proposé par l’entreprise que vous confirmerez (≤ plafond).'
                            : 'Indicative — le montant final sera facturé par l’entreprise de transport.'}
                        </p>
                      )}
                      <p className="portalOrderLegal">
                        {termsAcceptances.some((row) => row.document_type === 'terms_of_service') &&
                        termsAcceptances.some((row) => row.document_type === 'transport_terms')
                          ? conditionalOrderEnabled
                            ? 'En commandant, vous autorisez LIRIE à transmettre cette demande aux entreprises éligibles. Le contrat se forme à l’acceptation du premier transporteur éligible — pas au moment de votre clic.'
                            : doubleValidationEnabled
                              ? 'En confirmant, vous enregistrez une demande de transport (pas encore un contrat). Le contrat se forme au second clic après proposition du transporteur.'
                              : 'En confirmant cette demande, vous passez une commande de transport soumise aux conditions acceptées pour votre compte.'
                          : 'Aucune acceptation des conditions n’est enregistrée pour ce compte. Confirmer cette demande n’en crée pas.'}
                      </p>
                      <p className="portalOrderLegalLinks">
                        {termsCatalog.map((doc) => (
                          <button
                            key={doc.document_type}
                            type="button"
                            className="portalOrderTermsLink"
                            onClick={() => toggleOpenTermsDoc(doc)}
                          >
                            {portalTermsDocumentLabel(doc.document_type, {
                              variant: 'order',
                            })}
                            {doc.terms_version ? ` ${doc.terms_version}` : ''}
                          </button>
                        ))}
                      </p>
                      {openTermsDoc &&
                      termsCatalog.some(
                        (doc) => doc.document_type === openTermsDoc.document_type
                      )
                        ? renderPortalTermsDocPanel(openTermsDoc, {
                            variant: 'order',
                          })
                        : null}
                      {bookingSubmitting ? (
                        (() => {
                          const wait = portalSubmitWaitCopy(
                            submitWaitIndex,
                            portalReview.transportCount
                          );
                          return (
                            <div
                              className="portalOrderWait"
                              role="status"
                              aria-live="polite"
                              aria-busy="true"
                            >
                              <div className="portalOrderWaitTrack" aria-hidden="true">
                                <span className="portalOrderWaitBar" />
                              </div>
                              <p className="portalOrderWaitTitle">{wait.title}</p>
                              <ol className="portalOrderWaitSteps">
                                {wait.steps.map((label, stepIndex) => {
                                  const state =
                                    wait.settled || stepIndex < wait.active
                                      ? 'done'
                                      : stepIndex === wait.active
                                        ? 'current'
                                        : 'pending';
                                  return (
                                    <li
                                      key={label}
                                      className={`portalOrderWaitStep portalOrderWaitStep--${state}`}
                                    >
                                      <span className="portalOrderWaitMark" aria-hidden="true" />
                                      {label}
                                    </li>
                                  );
                                })}
                              </ol>
                              <p className="portalOrderWaitHint">{wait.hint}</p>
                            </div>
                          );
                        })()
                      ) : (
                      <div className="formActions formActionsPrimary">
                        <button
                          type="button"
                          className={`${homeFieldStyles.ctaButton} bookingDashboardCta`}
                          onClick={() => handleBooking({ confirm: true })}
                        >
                          {portalReview?.transportCount > 2 &&
                          portalReview?.maximumAuthorizedLabel
                            ? `Confirmer et accepter CHF ${portalReview.maximumAuthorizedLabel}`
                            : conditionalOrderEnabled && portalReview?.maximumLabel
                              ? `Commander jusqu’à CHF ${portalReview.maximumLabel}.–`
                              : 'Confirmer la demande de transport'}
                        </button>
                        <button
                          type="button"
                          className="ghostButton"
                          onClick={() => presentPortalCard(() => setPortalReview(null))}
                        >
                          Modifier la demande
                        </button>
                      </div>
                      )}
                    </section>
                  ) : null}


                  <div className="bookingFormMore">
                  <div
                    className={`${homeFieldStyles.fieldBlock} ${homeFieldStyles.tripKindFieldScope} bookingNeedsBlock`}
                  >
                    <span id="client-dashboard-needs-label" className="portalSectionHeading">
                      Besoins pour le transport
                    </span>
                    {profile?.mobility &&
                    (profile.mobility.wheelchair_client_has ||
                      profile.mobility.wheelchair_need ||
                      profile.mobility.needs_assistance) ? (
                      <p className="portalNeedsHint">Selon vos informations habituelles</p>
                    ) : null}
                    <div className="bookingNeedsGrid">
                      <div
                        id="client-booking-wheelchair"
                        className="bookingWheelchairChips"
                        role="group"
                        aria-label="Fauteuil"
                      >
                          {[
                            ['none', 'Aucun', !wheelchairOwn && !wheelchairRequired],
                            ['own', 'En fauteuil', wheelchairOwn],
                            ['provide', 'Fournir fauteuil', wheelchairRequired],
                          ].map(([value, label, pressed]) => (
                            <button
                              key={value}
                              type="button"
                              className="bookingWheelchairChip"
                              aria-pressed={pressed}
                              onClick={() => {
                                setWheelchairOwn(value === 'own');
                                setWheelchairRequired(value === 'provide');
                              }}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                      <div className="portalRoundTripToggle bookingNeedsSwitch">
                        <button
                          type="button"
                          role="switch"
                          id="client-booking-assistance"
                          className={`portalMedicalSwitch${assistanceRequired ? ' is-on' : ''}`}
                          aria-checked={assistanceRequired}
                          aria-label="Assistance"
                          onClick={() => {
                            setAssistanceRequired((current) => {
                              if (current) setAssistanceDetail('');
                              return !current;
                            });
                          }}
                        />
                        <span>Assistance</span>
                      </div>
                      {assistanceRequired ? (
                        <input
                          id="client-booking-assistance-detail"
                          className="bookingNeedsAssistInput"
                          type="text"
                          maxLength={200}
                          required
                          aria-required="true"
                          value={assistanceDetail}
                          placeholder="Type d'assistance *"
                          aria-label="Type d'assistance"
                          onChange={(e) => setAssistanceDetail(e.target.value)}
                        />
                      ) : null}
                    </div>
                  </div>
                  </div>
                  <div className="formActions formActionsPrimary">
                    <button
                      type="button"
                      className={`${homeFieldStyles.ctaButton} bookingDashboardCta`}
                      onClick={() => handleBooking(isPortalPrivateClient ? { confirm: false } : undefined)}
                      disabled={
                        bookingSubmitting ||
                        loadingProfile ||
                        loadingBookings ||
                        !effectiveClientId ||
                        termsReacceptanceRequired ||
                        (doubleValidationEnabled &&
                          isPortalPrivateClient &&
                          (pricingCeilingLoading ||
                            !maximumAcceptedAmount ||
                            Boolean(pricingCeilingError)))
                      }
                      aria-busy={bookingSubmitting}
                      hidden={isPortalPrivateClient && Boolean(portalReview)}
                    >
                      {bookingSubmitting ? (
                        <>
                          <span className="btnInlineSpinner" aria-hidden="true" />
                          Validation en cours…
                        </>
                      ) : (
                        <>
                          {isPortalPrivateClient
                            ? 'Vérifier la demande'
                            : 'Valider la demande de transport'}
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
              {indicativeAmount != null ? (
                <div
                  className="sidebarJourneyCard"
                  role="region"
                  aria-label="Estimation du transport"
                >
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
                                {trip.recentTripRole ? (
                                  <p className="recentTripRole">{trip.recentTripRole}</p>
                                ) : null}
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
                                  {recentTripStops(trip).map((stop, index, all) => {
                                    const label =
                                      stop.label || (index === 0 ? 'Départ' : 'Arrivée');
                                    const dotKind =
                                      index === 0
                                        ? ' recentTripLegDot--origin'
                                        : index === all.length - 1
                                          ? ' recentTripLegDot--end'
                                          : '';
                                    return (
                                      <div
                                        key={stop.key || `${trip.id}-${index}`}
                                        className="recentTripLeg"
                                      >
                                        <span className="recentTripLegRail" aria-hidden="true">
                                          <span className={`recentTripLegDot${dotKind}`} />
                                          {index < all.length - 1 ? (
                                            <span className="recentTripLegLine" />
                                          ) : null}
                                        </span>
                                        <span className="recentTripLegCopy">
                                          <span className="recentTripLegLabel">{label}</span>
                                          <span className="recentTripLegText">{stop.place}</span>
                                        </span>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                              <button
                                type="button"
                                className="recentTripReuseBtnSober"
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

      {termsReacceptanceRequired ? (
        <Modal
          size="lg"
          ariaLabel="Mise à jour des conditions"
          onClose={() => {}}
          className="portalTermsUpdateModal"
        >
          <section
            className="portalTermsUpdate"
            aria-labelledby="portal-terms-update-title"
          >
            <h2 id="portal-terms-update-title" className="portalTermsUpdateTitle">
              Mise à jour des conditions
            </h2>
            <p className="portalTermsUpdateLead">
              Une nouvelle acceptation est requise avant toute nouvelle demande de
              transport.
            </p>
            <div className="portalTermsUpdateList" role="list">
              {requiredTermsDocs.map((doc) => renderPortalTermsDocPanel(doc))}
            </div>
            <label className="portalTermsAccept" htmlFor="portal-terms-reaccept">
              <input
                id="portal-terms-reaccept"
                type="checkbox"
                checked={termsAcceptChecked}
                onChange={(event) => setTermsAcceptChecked(event.target.checked)}
              />
              <span>{requiredTermsAcceptLabel()}</span>
            </label>
            <button
              type="button"
              className={`${homeFieldStyles.ctaButton} portalTermsAcceptBtn`}
              onClick={handleAcceptRequiredTerms}
              disabled={!termsAcceptChecked || termsAccepting}
            >
              {termsAccepting ? 'Enregistrement…' : 'Accepter les conditions'}
            </button>
          </section>
        </Modal>
      ) : null}

      {pendingCarrierOffer ? (
        <Modal
          size="md"
          ariaLabel="Proposition de transport"
          onClose={() => {}}
          className="portalCarrierOfferModal"
        >
          <section
            className="portalCarrierOffer"
            aria-labelledby="portal-carrier-offer-title"
          >
            <header className="portalCarrierOfferHeader">
              <p className="portalCarrierOfferEyebrow">Proposition reçue</p>
              <h2 id="portal-carrier-offer-title">
                {PORTAL_DV_COPY.carrierOfferedTitle}
              </h2>
              <p className="portalCarrierOfferLead">
                Vérifiez le prix et les conditions d’annulation avant de
                confirmer. Ce n’est pas encore un transport confirmé.
              </p>
            </header>

            <div className="portalCarrierOfferPriceCard">
              <div className="portalCarrierOfferPriceMain">
                <span className="portalCarrierOfferPriceLabel">Prix proposé</span>
                <span className="portalCarrierOfferPriceValue">
                  {formatPortalOfferChf(pendingCarrierOffer.offered_amount)}
                </span>
              </div>
              <dl className="portalCarrierOfferMeta">
                <div>
                  <dt>Entreprise de transport</dt>
                  <dd>{pendingCarrierOffer.company_name}</dd>
                </div>
                {pendingCarrierOffer.maximum_accepted_amount != null ? (
                  <div>
                    <dt>Prix maximum de la demande</dt>
                    <dd>
                      {formatPortalOfferChf(
                        pendingCarrierOffer.maximum_accepted_amount
                      )}
                    </dd>
                  </div>
                ) : null}
              </dl>
            </div>

            <details className="portalCarrierOfferPolicy" open>
              <summary className="portalCarrierOfferPolicySummary">
                Conditions d’annulation / no-show / attente
              </summary>
              <div
                className="portalCarrierOfferPolicyBody"
                tabIndex={0}
                role="region"
                aria-label="Texte des conditions d’annulation"
              >
                {formatPortalCancellationPolicyForDisplay(
                  pendingCarrierOffer.cancellation_policy_text
                )}
              </div>
            </details>

            <div className="portalCarrierOfferActions">
              <button
                type="button"
                className={`${homeFieldStyles.ctaButton} portalCarrierOfferConfirmBtn`}
                onClick={handleConfirmTransportOffer}
                disabled={confirmingTransport}
              >
                {confirmingTransport
                  ? 'Confirmation…'
                  : `Confirmer le transport à ${formatPortalOfferChf(
                      pendingCarrierOffer.offered_amount
                    )}`}
              </button>
              <p className="portalCarrierOfferFootnote">
                En confirmant, vous acceptez ce prix et les conditions ci-dessus.
                Le contrat de transport est alors formé.
              </p>
            </div>
          </section>
        </Modal>
      ) : null}
    </div>
  );
};

export default ClientDashboard;
