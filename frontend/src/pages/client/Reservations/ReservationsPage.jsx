import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { exportBookingsPDF, fetchBookings } from '../../../services/bookingService';
import { fetchClient } from '../../../services/clientService';
import { useHybridDataSync } from '../../../hooks/useHybridDataSync';
import styles from './Reservations.module.css';
import '../Dashboard/ClientDashboard.css';
import { toast } from 'sonner';

import apiClient from '../../../utils/apiClient';
import { getApiErrorMessage } from '../../../utils/apiErrorMessage';
// ✅ SUPPRIMÉ: mergeInvoiceAndQRBill - Génération PDF déplacée vers backend
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import {
  getClientBookingToneClass,
  getClientBookingUx,
  getEffectiveClientBookingActions,
  normalizeClientBookingStatus,
  resolveClientBookingDisplayStatus,
} from '../../../utils/clientBookingUx';
import { getActivePublicId } from '../../../utils/webAuthSession';
import { trackClientKpiEvent } from '../../../utils/clientKpi';
import { requiresPrivateOnlinePaymentAtBooking } from '../../../utils/clientBookingPayment';
import {
  isPortalContractFlow,
  portalReservationAmountDisplay,
} from '../../../utils/portalDoubleValidationUi';
import { useClientBookingSocketRefresh } from '../../../hooks/useClientBookingSocketRefresh';
import ClientBookingLiveTrackModal from '../../../components/client/ClientBookingLiveTrackModal';
import ClientTransportContactModal from '../../../components/client/ClientTransportContactModal';
import ClientTripEventReportModal from './ClientTripEventReportModal';
import ReservationRouteMap from './ReservationRouteMap';
import { foldClientRouteRequests } from '../../../utils/clientRouteRequest';
import { fetchClientMessageNotifications } from '../../../services/clientService';
import { ensureClientPortalSocket } from '../../../services/clientPortalSocket';
import {
  CLIENT_MESSAGE_READ_EVENT,
  bookingHasUnreadCarrierMessage,
  readClientMessageIds,
  rememberClientMessageRead,
} from '../../../utils/clientMessageNotifRead';

/** Libellé facturation / couverture (sans répéter le montant affiché à côté). */
function getBillingCoverageLabel(booking) {
  if (isPortalContractFlow(booking)) {
    return 'Client — facturation par le transporteur';
  }
  const raw = String(
    booking?.payer_label ||
      booking?.coverage_label ||
      booking?.covered_by ||
      booking?.payment_coverage ||
      ''
  ).trim();
  const n = raw.toLowerCase();
  if (n.includes('assur')) return 'Assurance';
  if (n.includes('instit')) return 'Institution / tiers payeur';
  if (requiresPrivateOnlinePaymentAtBooking(booking)) return 'Client — règlement en ligne';
  const clientType = String(booking?.client?.client_type || booking?.client_type || '')
    .trim()
    .toUpperCase();
  if (clientType === 'PORTAL') {
    return 'Client — facturation par le transporteur';
  }
  if (raw) return raw;
  return 'Client';
}

/** Nom affichable chauffeur (API `driver_name` ou objet `driver` imbriqué du serialize). */
function getClientDriverDisplayName(booking) {
  return String(
    booking?.driver_name ||
      booking?.driver?.full_name ||
      `${booking?.driver?.first_name || ''} ${booking?.driver?.last_name || ''}`.trim() ||
      ''
  ).trim();
}

function hasDriverAssigned(booking) {
  const n = getClientDriverDisplayName(booking);
  if (!n) return false;
  if (/^chauffeur\s+\d+$/i.test(n)) return false;
  if (n.toLowerCase() === 'non assigné') return false;
  return true;
}

function hasTrackingUrl(booking) {
  const u = String(booking?.tracking_url || '').trim();
  return u.length > 5 && /^https?:\/\//i.test(u);
}

function isReturnLegTerminalStatus(statusRaw) {
  const n = normalizeClientBookingStatus(statusRaw);
  return n === 'completed' || n === 'cancelled';
}

function isClientRoundTripOutbound(booking) {
  return Boolean(
    booking &&
      !booking.is_return &&
      (booking.is_round_trip ||
        booking.has_return ||
        (booking.return_booking && Number(booking.return_booking.id) > 0))
  );
}

/** Total facturé du dossier (aller + segment retour lié), pour tri / PDF / past. */
function clientRoundTripTotalChf(booking) {
  if (Number.isFinite(Number(booking?.route_request_amount))) {
    return Number(booking.route_request_amount);
  }
  if (!booking || booking.is_return) return Number(booking?.amount || 0);
  const base = Number(booking.amount || 0);
  const rb = booking.return_booking;
  const r = rb != null ? Number(rb.amount || 0) : 0;
  return base + (Number.isFinite(r) ? r : 0);
}

/** Aller déjà clos mais retour encore ouvert → le dossier reste « à venir » pour le client. */
function clientRoundTripAwaitingReturnCompletion(booking) {
  if (!isClientRoundTripOutbound(booking)) return false;
  const rs = booking.return_booking;
  if (!rs || !Number(rs.id)) return false;
  if (!isReturnLegTerminalStatus(booking.status)) return false;
  return !isReturnLegTerminalStatus(rs.status);
}

function clientTripListSection(booking, nowTs) {
  if (clientRoundTripAwaitingReturnCompletion(booking)) return 'upcoming';
  const shown = normalizeClientBookingStatus(resolveClientBookingDisplayStatus(booking));
  if (shown === 'completed' || shown === 'cancelled') return 'past';
  const t = Date.parse(booking.scheduled_time);
  if (!Number.isFinite(t)) return 'past';
  return t > nowTs ? 'upcoming' : 'past';
}

function clientUpcomingSortKey(booking) {
  if (clientRoundTripAwaitingReturnCompletion(booking) && booking.return_booking?.scheduled_time) {
    const r = Date.parse(booking.return_booking.scheduled_time);
    if (Number.isFinite(r)) return r;
  }
  const o = Date.parse(booking.scheduled_time);
  return Number.isFinite(o) ? o : 0;
}

/** Carte interne (Google) : lieu de prise en charge géocodé côté serveur. */
function canClientLiveTrack(booking) {
  const norm = normalizeClientBookingStatus(booking?.status);
  if (norm !== 'driver_on_the_way' && norm !== 'in_progress') return false;
  const plat = Number(booking?.pickup_lat);
  const plon = Number(booking?.pickup_lon);
  return Number.isFinite(plat) && Number.isFinite(plon);
}

function formatDriverEtaLine(booking) {
  const norm = normalizeClientBookingStatus(booking?.status);

  const formatWithPrefix = (etaPrefix, etaMin, rawTimeFields) => {
    if (etaMin != null && Number.isFinite(Number(etaMin))) {
      const m = Math.max(0, Math.round(Number(etaMin)));
      return m <= 0 ? `${etaPrefix} : imminente` : `${etaPrefix} : ~${m} min`;
    }
    const raw = String(rawTimeFields || '').trim();
    if (!raw) return null;
    const t = Date.parse(raw);
    if (!Number.isFinite(t)) return null;
    return `${etaPrefix} vers ${new Date(t).toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' })}`;
  };

  if (norm === 'in_progress') {
    return formatWithPrefix(
      'Arrivée à destination estimée',
      booking?.eta_minutes,
      booking?.estimated_dropoff_arrival || ''
    );
  }

  // Vers le lieu de prise en charge : seulement quand le chauffeur est réellement en route (pas « assigné »).
  if (norm !== 'driver_on_the_way') {
    return null;
  }

  return formatWithPrefix(
    'Arrivée au lieu de prise en charge estimée',
    booking?.eta_minutes,
    booking?.estimated_pickup_arrival || booking?.driver_eta || ''
  );
}

function onlinePaymentCompletedForTimeline(booking) {
  const st = (booking?.online_payment?.status || '').toLowerCase();
  return st === 'completed';
}

/** Même logique que `canPayOnline` pour positionner l’étape « Paiement requis » sur un `pending`. */
function bookingNeedsClientOnlinePayment(booking) {
  const status = (booking?.status || '').toLowerCase();
  if (status === 'canceled' || status === 'cancelled') return false;
  if (status !== 'pending' && status !== 'awaiting_client_payment') return false;
  if (onlinePaymentCompletedForTimeline(booking)) return false;
  if (!requiresPrivateOnlinePaymentAtBooking(booking)) return false;
  return true;
}

/**
 * Flux compte TRANSPORT (Saferpay) :
 * 1. Paiement en ligne exigé en premier ;
 * 2. Après paiement validé, la demande est transmise aux entreprises ;
 * 3. Entreprise retenue → chauffeur → en route → terminée.
 */
const TIMELINE_PAYMENT_FIRST = [
  { id: 'payment', label: 'Paiement requis' },
  { id: 'broadcast', label: 'Demande transmise aux entreprises' },
  { id: 'company', label: 'Entreprise retenue' },
  { id: 'driver', label: 'Chauffeur désigné par l’entreprise' },
  { id: 'moving', label: 'En route' },
  { id: 'done', label: 'Terminée' },
];

/**
 * Compte PORTAL : l’entreprise facture après la course. Aucun encaissement
 * n’a lieu à la commande, donc l’étape paiement n’est pas affichée.
 */
const TIMELINE_CARRIER_INVOICE = [
  { id: 'broadcast', label: 'Demande transmise aux entreprises' },
  { id: 'company', label: 'Entreprise retenue' },
  { id: 'driver', label: 'Chauffeur désigné par l’entreprise' },
  { id: 'moving', label: 'En route' },
  { id: 'done', label: 'Terminée' },
];

/**
 * Flux PORTAL double_validation_v2 — aucun paiement LIRIE.
 * 1. Demande enregistrée ;
 * 2. Transmission aux entreprises ;
 * 3. Proposition transporteur ;
 * 4. Confirmation client ;
 * 5. Chauffeur → en route → terminée.
 */
const TIMELINE_PORTAL_DV = [
  { id: 'recorded', label: 'Demande enregistrée' },
  { id: 'broadcast', label: 'Transmission aux entreprises' },
  { id: 'offer', label: 'Proposition d’une entreprise' },
  { id: 'confirmed', label: 'Confirmation du client' },
  { id: 'driver', label: 'Chauffeur désigné' },
  { id: 'moving', label: 'En route' },
  { id: 'done', label: 'Terminée' },
];

function getTimelineSteps(booking) {
  const displayNorm = normalizeClientBookingStatus(
    resolveClientBookingDisplayStatus(booking)
  );
  const isPortalDv = isPortalContractFlow(booking);
  const skipsOnlinePayment = !requiresPrivateOnlinePaymentAtBooking(booking);
  const base = isPortalDv
    ? TIMELINE_PORTAL_DV
    : skipsOnlinePayment
      ? TIMELINE_CARRIER_INVOICE
      : TIMELINE_PAYMENT_FIRST;

  if (displayNorm === 'cancelled') {
    return [{ id: 'cancelled', label: 'Course annulée', state: 'cancelled' }];
  }
  if (displayNorm === 'completed') {
    return base.map((s) => ({ ...s, state: 'done' }));
  }
  if (displayNorm === 'round_trip_return_pending') {
    const last = base.length - 1;
    return base.map((s, i) => ({
      ...s,
      state: i < last ? 'done' : i === last ? 'current' : 'upcoming',
      label: i === last ? 'Retour à venir' : s.label,
    }));
  }

  let currentIndex = 0;

  if (isPortalDv) {
    // Indices TIMELINE_PORTAL_DV
    switch (displayNorm) {
      case 'pending':
        currentIndex = booking?.company_id ? 3 : 1;
        if (!booking?.company_id && booking?.has_pending_portal_offer) {
          currentIndex = 2;
        }
        break;
      case 'confirmed':
        currentIndex = hasDriverAssigned(booking) ? 4 : 3;
        break;
      case 'driver_on_the_way':
      case 'in_progress':
        currentIndex = 5;
        break;
      default:
        currentIndex = booking?.company_id ? 3 : 1;
        break;
    }
  } else if (skipsOnlinePayment) {
    switch (displayNorm) {
      case 'confirmed':
        currentIndex = hasDriverAssigned(booking) ? 2 : 1;
        break;
      case 'driver_on_the_way':
      case 'in_progress':
        currentIndex = 3;
        break;
      default:
        currentIndex = 0;
        break;
    }
  } else {
    const needsPay = bookingNeedsClientOnlinePayment(booking);
    switch (displayNorm) {
      case 'awaiting_payment':
        currentIndex = 0;
        break;
      case 'pending':
        currentIndex = needsPay ? 0 : 1;
        break;
      case 'confirmed':
        currentIndex = hasDriverAssigned(booking) ? 3 : 2;
        break;
      case 'driver_on_the_way':
      case 'in_progress':
        currentIndex = 4;
        break;
      default:
        currentIndex = needsPay ? 0 : 1;
        break;
    }
  }

  return base.map((s, i) => {
    let state = 'upcoming';
    if (i < currentIndex) state = 'done';
    else if (i === currentIndex) state = 'current';
    return { ...s, state };
  });
}

function BookingStatusTimeline({ booking, layout = 'vertical' }) {
  const steps = getTimelineSteps(booking);
  const isSingleCancelled = steps.length === 1 && steps[0].state === 'cancelled';
  const horizontal = layout === 'horizontal' && !isSingleCancelled;
  const listClass = horizontal ? styles.statusTimelineHorizontal : styles.statusTimeline;
  return (
    <ol className={listClass} aria-label="Progression de la course">
      {steps.map((step) => (
        <li
          key={step.id}
          className={`${styles.timelineStep} ${horizontal ? styles.timelineStepHorizontal : ''} ${
            styles[`timelineStep_${step.state}`] || ''
          }`}
        >
          {horizontal ? (
            <span className={styles.timelineNode} aria-hidden>
              {step.state === 'done' ? (
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <polyline
                    points="20 6 9 17 4 12"
                    stroke="white"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              ) : null}
            </span>
          ) : (
            <span className={styles.timelineDot} aria-hidden />
          )}
          <span className={styles.timelineLabel}>{step.label}</span>
        </li>
      ))}
    </ol>
  );
}

/** Icône type « produit » (Uber) — transport médical, sans dépendance externe. */
function SvgMedicalTrip({ className }) {
  return (
    <svg className={className} viewBox="0 0 48 48" width="48" height="48" aria-hidden>
      <rect x="2" y="14" width="44" height="22" rx="4" fill="#e2e8f0" />
      <rect x="5" y="17" width="18" height="10" rx="2" fill="#ffffff" />
      <path
        d="M28 20h10v4H28v-4zm2 7h6v3h-6v-3zM8 38h6a3 3 0 0 0 6 0h14a3 3 0 0 0 6 0h2"
        stroke="#64748b"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
      />
      <path d="M12 22h6v4h-6v-4z" fill="#ef4444" opacity="0.9" />
    </svg>
  );
}

function startOfLocalDay(d) {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
}

/** Libellé relatif (Aujourd'hui, Demain, Hier, 11 avr.) pour l'entête « prochaine course ». */
function formatRelativeDayLabel(iso) {
  const parsed = Date.parse(iso);
  if (!Number.isFinite(parsed)) return '';
  const d = new Date(parsed);
  const today = new Date();
  const diffDays = Math.round((startOfLocalDay(d) - startOfLocalDay(today)) / 86400000);
  if (diffDays === 0) return "Aujourd'hui";
  if (diffDays === 1) return 'Demain';
  if (diffDays === -1) return 'Hier';
  return d.toLocaleDateString('fr-CH', { day: 'numeric', month: 'short' });
}

/** Date longue capitalisée (ex. « lundi 13 avril 2026 » → « Lundi 13 avril 2026 »). */
function formatLongWeekdayDateFr(iso) {
  const parsed = Date.parse(iso);
  if (!Number.isFinite(parsed)) return '';
  const raw = new Date(parsed).toLocaleDateString('fr-CH', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
  if (!raw) return '';
  return raw.charAt(0).toUpperCase() + raw.slice(1);
}

function ClientRouteStops({ stops, label = 'Trajet' }) {
  if (!Array.isArray(stops) || stops.length === 0) return null;
  return (
    <div className={styles.routeBlock} aria-label={label}>
      {stops.map((stop, index, all) => {
        const isLast = index === all.length - 1;
        return (
          <div className={styles.routeLeg} key={stop.key}>
            <div className={styles.routeIconCol} aria-hidden>
              <div
                className={`${styles.routeDot} ${
                  index === 0 ? styles.routeDotOrigin : styles.routeDotDestination
                }`}
              />
              {isLast ? null : <div className={styles.routeLine} />}
            </div>
            <div className={`${styles.routeContent} ${isLast ? styles.routeContentLast : ''}`}>
              <div className={styles.routeHead}>
                <div className={styles.routeLabel}>{stop.label}</div>
                {stop.when ? <div className={styles.routeWhen}>{glanceStopWhen(stop.when)}</div> : null}
              </div>
              <div className={styles.routePlace}>
                {stop.place}
                {stop.detail ? <span className={styles.routeDetail}> · {stop.detail}</span> : null}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function historyMapPoint(lat, lon) {
  const latitude = Number(lat);
  const longitude = Number(lon);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
  if (latitude === 0 && longitude === 0) return null;
  return { lat: latitude, lng: longitude };
}

function historyMedicalDetail(booking) {
  return [booking?.hospital_service, booking?.doctor_name]
    .map((value) => String(value || '').trim())
    .filter((value) => value && value !== 'Non spécifié' && value !== 'Aucune note')
    .join(' · ');
}

/** Heure réelle : prise en charge chauffeur, ou dépôt confirmé (chauffeur ou entreprise). */
function operationalStopWhen(prefix, iso) {
  const clock = formatSpotlightTime(iso);
  return clock ? `${prefix} ${clock}` : '';
}

function historyOperationalWhen(stop) {
  if (stop?.label === 'Prise en charge') {
    return operationalStopWhen('Prise en charge', stop.boarded_at);
  }
  return operationalStopWhen('Dépôt', stop.completed_at);
}

/** Étapes repliées d’un groupe, ou départ → arrivée d’une course simple. */
function historyRouteStops(booking) {
  const folded = Array.isArray(booking?.route_request_stops) ? booking.route_request_stops : [];
  if (folded.length > 1) {
    return folded.map((stop) => ({ ...stop, when: historyOperationalWhen(stop) }));
  }
  const pickup = String(booking?.pickup_location || '').trim();
  const dropoff = String(booking?.dropoff_location || '').trim();
  if (!pickup && !dropoff) return [];
  return [
    {
      key: `pickup-${booking?.id || 'trip'}`,
      label: 'Prise en charge',
      place: pickup || 'Départ non renseigné',
      detail: '',
      point: historyMapPoint(booking?.pickup_lat, booking?.pickup_lon),
      boarded_at: booking?.boarded_at || null,
      completed_at: null,
      when: operationalStopWhen('Prise en charge', booking?.boarded_at),
    },
    {
      key: `dropoff-${booking?.id || 'trip'}`,
      label: 'Destination',
      place: dropoff || 'Destination non renseignée',
      detail: historyMedicalDetail(booking),
      point: historyMapPoint(booking?.dropoff_lat, booking?.dropoff_lon),
      boarded_at: null,
      completed_at: booking?.completed_at || null,
      when: operationalStopWhen('Dépôt', booking?.completed_at),
    },
  ];
}

function glanceStopWhen(when) {
  const text = String(when || '').trim();
  if (!text) return '';
  if (text === 'Prise en charge à déterminer') return 'À déterminer';
  if (text.startsWith('Heure à définir')) return 'Heure à définir';
  const clock = text.match(/(\d{2}:\d{2})\s*$/);
  if (text.startsWith('Rendez-vous') && clock) return `Rendez-vous ${clock[1]}`;
  if (text.startsWith('Heure de départ') && clock) return `Départ ${clock[1]}`;
  if (text.startsWith('Prise en charge') && clock) return `Prise en charge ${clock[1]}`;
  if (text.startsWith('Dépôt') && clock) return `Dépôt ${clock[1]}`;
  return text;
}

function formatSpotlightTime(iso) {
  const parsed = Date.parse(iso);
  if (!Number.isFinite(parsed)) return '';
  return new Date(parsed).toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
}

function SvgIconCreditCard({ className, size = 12 }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <rect x="1" y="4" width="22" height="16" rx="2" ry="2" stroke="currentColor" strokeWidth="2" />
      <line x1="1" y1="10" x2="23" y2="10" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

function SvgIconTruck({ className, size = 12 }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <rect x="1" y="3" width="15" height="13" stroke="currentColor" strokeWidth="2" />
      <polygon
        points="16 8 20 8 23 11 23 16 16 16 16 8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <circle cx="5.5" cy="18.5" r="2.5" stroke="currentColor" strokeWidth="2" />
      <circle cx="18.5" cy="18.5" r="2.5" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

function SvgIconShield({ className, size = 12 }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

function SvgIconUserOutline({ className, size = 22 }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <path
        d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function SvgIconInfo({ className, size = 14 }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
      <line x1="12" y1="8" x2="12" y2="12" stroke="currentColor" strokeWidth="2" />
      <line x1="12" y1="16" x2="12.01" y2="16" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

function SvgIconClockSmall({ className, size = 12 }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
      <polyline points="12 6 12 12 16 14" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

/** Ambulance / véhicule médicalisé (ligne) — statut terminé. */
function SvgPastTripDone({ className, size = 18 }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <path d="M10 10H6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path
        d="M14 18V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v11a1 1 0 0 0 1 1h2"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path
        d="M19 18h2a1 1 0 0 0 1-1v-3.28a1 1 0 0 0-.684-.948l-1.923-.641a1 1 0 0 1-.578-.502l-1.539-3.076A1 1 0 0 0 16.382 8H14"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="7" cy="18" r="2" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="17" cy="18" r="2" stroke="currentColor" strokeWidth="1.5" />
      <path d="M9 10v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M7 12h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/** Ambulance simplifiée — course annulée. */
function SvgPastTripCanceled({ className, size = 18 }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <path
        d="M5 17H3a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v9a2 2 0 0 1-2 2h-3"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="7.5" cy="17.5" r="2.5" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="17.5" cy="17.5" r="2.5" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function UpcomingEmptyCard({ clientPublicId }) {
  const navigate = useNavigate();
  const handleReserve = () => {
    if (!clientPublicId) return;
    navigate(`/dashboard/client/${encodeURIComponent(clientPublicId)}`);
  };
  return (
    <button type="button" className={styles.upcomingEmptyCard} onClick={handleReserve}>
      <div className={styles.upcomingEmptyIllustration} aria-hidden>
        <SvgMedicalTrip className={styles.upcomingEmptySvg} />
      </div>
      <div className={styles.upcomingEmptyTitle}>{`Vous n'avez aucune course à venir`}</div>
      <div className={styles.upcomingEmptyCta}>
        <span className={styles.upcomingEmptyCtaIcon} aria-hidden>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path
              d="M5 12h12M13 6l6 6-6 6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        Réserver une course
      </div>
    </button>
  );
}

function PastBookingRow({
  booking,
  isSelected,
  isExpanded,
  onToggleDetails,
  onContact,
  hasNewMessage = false,
  title,
  relativeDayLabel,
  timeLabel,
  statusPill,
  amountContent,
  children,
}) {
  const displayPast = resolveClientBookingDisplayStatus(booking);
  const normPast = normalizeClientBookingStatus(displayPast);
  const iconTerminated = normPast === 'completed';
  return (
    <div
      id={`booking-${booking.id}`}
      className={`${styles.pastRowOuter} ${isSelected ? styles.pastRowOuterSelected : ''}`}
    >
      <div className={styles.pastRow} onClick={onToggleDetails} aria-expanded={isExpanded}>
        <div
          className={`${styles.pastRowIcon} ${
            iconTerminated ? styles.pastRowIconDone : styles.pastRowIconCanceled
          }`}
          aria-hidden
        >
          {iconTerminated ? (
            <SvgPastTripDone className={styles.pastRowIconSvg} />
          ) : (
            <SvgPastTripCanceled className={styles.pastRowIconSvg} />
          )}
        </div>
        <div className={styles.pastRowMain}>
          <div className={styles.pastRowTitle}>{title}</div>
          <div className={styles.pastRowMeta}>
            <span>{relativeDayLabel}</span>
            <span className={styles.pastRowMetaSep}>·</span>
            <span className={styles.pastRowMetaClock} aria-hidden>
              <SvgIconClockSmall className={styles.pastRowMetaClockSvg} size={12} />
            </span>
            <span>{timeLabel}</span>
            <span className={styles.pastRowMetaSep}>·</span>
            {statusPill}
          </div>
        </div>
        <div className={styles.pastRowRight}>
          <span className={styles.pastRowAmount}>{amountContent}</span>
          <div
            className={styles.pastRowActions}
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              className={styles.pastGhostLink}
              onClick={(e) => {
                e.stopPropagation();
                onContact();
              }}
            >
              Contacter
              {hasNewMessage ? (
                <span className={styles.contactNewDot} aria-label="Nouveau message" />
              ) : null}
            </button>
            <button
              type="button"
              className={styles.pastDetailsBtn}
              onClick={(e) => {
                e.stopPropagation();
                onToggleDetails();
              }}
              aria-expanded={isExpanded}
            >
              <span>Détails</span>
              <span className={`${styles.pastDetailsChevron} ${isExpanded ? styles.pastDetailsChevronOpen : ''}`}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <polyline
                    points="6 9 12 15 18 9"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </span>
            </button>
          </div>
        </div>
      </div>
      {isExpanded ? (
        <div
          className={styles.pastRowPanel}
          role="region"
          aria-label="Détails et suivi de la course"
        >
          {children}
        </div>
      ) : null}
    </div>
  );
}

/** Règles affichées sur les réservations « à venir » (modification / annulation). */
function ModificationAnnulationPolicy({ portalDoubleValidation = false }) {
  if (portalDoubleValidation) {
    return (
      <details className={styles.bookingPolicyDetails}>
        <summary className={styles.bookingPolicySummary}>Modification et annulation</summary>
        <div className={styles.bookingPolicyBody}>
          <p className={styles.bookingPolicyLead}>
            Tant que le chauffeur n’est pas <strong>en route</strong> et que la course n’est pas{' '}
            <strong>en cours</strong>, vous pouvez <strong>modifier</strong> ou <strong>annuler</strong> depuis
            cette page.
          </p>
          <p className={styles.bookingPolicyP}>
            <strong>Modification</strong> : lieux de prise en charge et de destination, date et heure. Si le nouveau
            prix maximum est <strong>identique ou inférieur</strong>, aucun complément n’est dû. S’il est{' '}
            <strong>supérieur</strong>, vous devez accepter le nouveau plafond avant validation. LIRIE n’encaisse
            pas la course : le transporteur facture après la prestation.
          </p>
          <p className={styles.bookingPolicyP}>
            <strong>Annulation</strong> : les éventuels frais ou avoirs suivent les conditions du transporteur
            retenues au moment de votre confirmation (pas de remboursement via un paiement en ligne LIRIE, car aucun
            encaissement plateforme n’a lieu sur ce parcours).
          </p>
        </div>
      </details>
    );
  }

  return (
    <details className={styles.bookingPolicyDetails}>
      <summary className={styles.bookingPolicySummary}>Modification et annulation</summary>
      <div className={styles.bookingPolicyBody}>
        <p className={styles.bookingPolicyLead}>
          Tant que le chauffeur n’est pas <strong>en route</strong> et que la course n’est pas{' '}
          <strong>en cours</strong>, vous pouvez <strong>modifier</strong> ou <strong>annuler</strong> depuis cette
          page.
        </p>
        <p className={styles.bookingPolicyP}>
          <strong>Modification</strong> : lieux de prise en charge et de destination, date et heure. Si le nouveau tarif
          est <strong>identique ou inférieur</strong>, aucun complément n’est dû. Si le tarif est{' '}
          <strong>supérieur</strong>, la différence doit être réglée avant validation. Aucun frais de modification ne
          s’applique tant que le chauffeur n’est pas en route (y compris à <strong>moins de 24 h</strong> du départ) ;
          seule une hausse de tarif peut donner lieu au paiement de la différence.
        </p>
        <p className={styles.bookingPolicyP}>
          <strong>Annulation</strong> (montants indicatifs ; en priorité <strong>avoir</strong> sur votre compte, sinon
          remboursement sur le moyen de paiement utilisé) :
        </p>
        <ul className={styles.bookingPolicyList}>
          <li>
            <strong>Plus de 24 h</strong> avant le départ : remboursement intégral du dossier (aller et, le cas échéant,
            retour).
          </li>
          <li>
            <strong>Entre 4 h et 24 h</strong> avant le départ : <strong>50 % du montant de l’aller</strong> est
            retenu ; 50 % de l’aller est remboursé (avoir ou remboursement). En <strong>aller-retour</strong>, le{' '}
            <strong>retour est remboursé intégralement</strong>.
          </li>
          <li>
            <strong>Moins de 4 h</strong> avant le départ : <strong>totalité du montant de l’aller</strong> est retenue ;
            en <strong>aller-retour</strong>, le <strong>retour est remboursé intégralement</strong>.
          </li>
        </ul>
      </div>
    </details>
  );
}

const CLIENT_INLINE_NOTE_MAX = 1000;

function scheduledIsoToDatetimeLocalValue(iso) {
  if (!iso || !Number.isFinite(Date.parse(iso))) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

function normalizeNotesMedicalForEdit(raw) {
  const s = String(raw ?? '').trim();
  if (!s || s === 'Aucune note') return '';
  return s;
}

function ClientBookingInlineEditForm({ booking, onClose, onSaved, clientPublicId }) {
  const rawNotes = booking.notes_medical;
  const [pickup, setPickup] = useState(() => String(booking.pickup_location || '').trim());
  const [dropoff, setDropoff] = useState(() => String(booking.dropoff_location || '').trim());
  const [dtLocal, setDtLocal] = useState(() => scheduledIsoToDatetimeLocalValue(booking.scheduled_time));
  const [notes, setNotes] = useState(() =>
    normalizeNotesMedicalForEdit(rawNotes).slice(0, CLIENT_INLINE_NOTE_MAX)
  );
  const [saving, setSaving] = useState(false);

  const roundTripOutbound = isClientRoundTripOutbound(booking);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const pu = pickup.trim();
    const dr = dropoff.trim();
    if (pu.length < 1 || dr.length < 1) {
      toast.error('Indiquez le lieu de prise en charge et la destination.');
      return;
    }
    if (!dtLocal) {
      toast.error('Choisissez une date et une heure pour la course.');
      return;
    }
    const scheduled = new Date(dtLocal);
    if (Number.isNaN(scheduled.getTime())) {
      toast.error('Date ou heure invalide.');
      return;
    }
    setSaving(true);
    try {
      await apiClient.put(`/bookings/${booking.id}`, {
        pickup_location: pu,
        dropoff_location: dr,
        scheduled_time: scheduled.toISOString(),
        notes_medical: notes.trim(),
      });
      trackClientKpiEvent('client_booking_inline_edit_saved', {
        bookingId: booking.id,
        clientPublicId: clientPublicId || getActivePublicId(),
      });
      await onSaved();
    } catch (err) {
      console.error(err);
      toast.error(getApiErrorMessage(err, 'Enregistrement impossible pour le moment.'), {
        duration: 7000,
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className={styles.bookingInlineEditWrap}>
      <form className={styles.bookingInlineEdit} onSubmit={handleSubmit} noValidate aria-labelledby={`inline-edit-title-${booking.id}`}>
        <h3 className={styles.bookingInlineEditTitle} id={`inline-edit-title-${booking.id}`}>
          Modifier cette course
        </h3>
        <p className={styles.bookingInlineEditHint}>
          Ajustez les lieux, l&apos;horaire de l&apos;aller ou les précisions pour le transporteur. Si le nouveau tarif
          est supérieur, un complément pourra vous être demandé avant validation.
        </p>
        {roundTripOutbound && booking.return_booking ? (
          <p className={styles.bookingInlineEditHint}>
            Pour le <strong>retour</strong> (lieu ou heure), contactez directement le transporteur ou le support : ce
            formulaire concerne l&apos;aller.
          </p>
        ) : null}
        <div className={styles.bookingInlineEditGrid}>
          <div>
            <label className={styles.bookingInlineEditLabel} htmlFor={`inline-pickup-${booking.id}`}>
              Lieu de prise en charge
            </label>
            <textarea
              id={`inline-pickup-${booking.id}`}
              className={styles.bookingInlineEditTextarea}
              value={pickup}
              onChange={(ev) => setPickup(ev.target.value)}
              rows={2}
              autoComplete="street-address"
              maxLength={500}
            />
          </div>
          <div>
            <label className={styles.bookingInlineEditLabel} htmlFor={`inline-drop-${booking.id}`}>
              Destination
            </label>
            <textarea
              id={`inline-drop-${booking.id}`}
              className={styles.bookingInlineEditTextarea}
              value={dropoff}
              onChange={(ev) => setDropoff(ev.target.value)}
              rows={2}
              autoComplete="street-address"
              maxLength={500}
            />
          </div>
          <div>
            <label className={styles.bookingInlineEditLabel} htmlFor={`inline-when-${booking.id}`}>
              Date et heure (aller)
            </label>
            <input
              id={`inline-when-${booking.id}`}
              className={styles.bookingInlineEditInput}
              type="datetime-local"
              value={dtLocal}
              onChange={(ev) => setDtLocal(ev.target.value)}
            />
          </div>
          <div>
            <label className={styles.bookingInlineEditLabel} htmlFor={`inline-notes-${booking.id}`}>
              Complément d&apos;information pour le transporteur
            </label>
            <textarea
              id={`inline-notes-${booking.id}`}
              className={styles.bookingInlineEditTextarea}
              value={notes}
              onChange={(ev) => setNotes(ev.target.value.slice(0, CLIENT_INLINE_NOTE_MAX))}
              rows={3}
              maxLength={CLIENT_INLINE_NOTE_MAX}
              placeholder="Ex. code d'accès, étage, personne à prévenir, matériel…"
            />
          </div>
        </div>
        <div className={styles.bookingInlineEditActions}>
          <button type="submit" className="primaryButton" disabled={saving}>
            {saving ? 'Enregistrement…' : 'Enregistrer les modifications'}
          </button>
          <button type="button" className="secondaryButton" onClick={onClose} disabled={saving}>
            Annuler
          </button>
        </div>
      </form>
    </div>
  );
}

function DriverMiniCard({ booking, emphasized }) {
  const norm = normalizeClientBookingStatus(booking?.status);
  if (norm === 'cancelled' || norm === 'completed') return null;
  const hasDriver = hasDriverAssigned(booking);
  const etaLine = formatDriverEtaLine(booking);
  const wrapClass = emphasized ? `${styles.driverMini} ${styles.driverMiniEmphasis}` : styles.driverMini;

  return (
    <div className={wrapClass} id={`booking-driver-${booking.id}`}>
      <div className={styles.driverMiniInner}>
        <div className={hasDriver ? styles.driverAvatarAssigned : styles.driverAvatar}>
          <SvgIconUserOutline
            size={20}
            className={hasDriver ? styles.driverAvatarSvgOn : styles.driverAvatarSvgOff}
          />
        </div>
        <div className={styles.driverMiniText}>
          <div className={styles.driverMiniHeader}>Votre chauffeur</div>
          {hasDriver ? (
            <>
              <p className={styles.driverMiniName}>{getClientDriverDisplayName(booking)}</p>
              {booking.company_name ? (
                <p className={styles.driverMiniMeta}>{booking.company_name}</p>
              ) : null}
              {etaLine ? <p className={styles.driverMiniEta}>{etaLine}</p> : null}
            </>
          ) : (
            <p className={styles.driverMiniPlaceholder}>
              Le nom du chauffeur et l’horaire d’arrivée seront affichés dès l’attribution.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function ReservationCard({
  booking,
  isSelected,
  showCarrier,
  formatDateTime,
  formatAmount,
  showTimeline = true,
  showDriverBlock = false,
  variant = 'default',
  primaryAction = null,
  secondaryActions = null,
  showClientBookingPolicy = false,
  inlineEditPanel = null,
}) {
  const displayStatus = resolveClientBookingDisplayStatus(booking);
  const statusUx = getClientBookingUx(displayStatus);
  const scheduled = booking.scheduled_time;
  const dateTimeAttr =
    scheduled && Number.isFinite(Date.parse(scheduled))
      ? new Date(scheduled).toISOString()
      : undefined;
  const rb = booking.return_booking;
  const requestStops = Array.isArray(booking.route_request_stops) ? booking.route_request_stops : [];
  const isGroupedRequest = requestStops.length > 1;
  const isRoundTripCard = !isGroupedRequest && isClientRoundTripOutbound(booking);
  const rbAmount = rb != null ? Number(rb.amount || 0) : 0;
  const splitRoundTripAmounts =
    isRoundTripCard && Number.isFinite(rbAmount) && rbAmount > 0;
  const amountDisplay = portalReservationAmountDisplay(booking);
  const returnScheduled = rb?.scheduled_time;
  const returnDateTimeAttr =
    returnScheduled && Number.isFinite(Date.parse(returnScheduled))
      ? new Date(returnScheduled).toISOString()
      : undefined;

  const cardClass = [
    styles.reservationCard,
    isSelected ? styles.reservationCardSelected : '',
    variant === 'spotlight' ? styles.reservationCardSpotlight : '',
  ]
    .filter(Boolean)
    .join(' ');

  const displayNorm = normalizeClientBookingStatus(displayStatus);
  const companyName = String(booking?.company_name || '').trim().toLowerCase();
  const carrierConfirmed =
    Boolean(booking?.company_id) ||
    (companyName.length > 0 &&
      companyName !== 'non assignée' &&
      companyName !== 'non assigné');
  const awaitingCarrierConfirmation = isGroupedRequest && !carrierConfirmed;
  const showPendingBroadcastHint =
    displayNorm === 'pending' && !bookingNeedsClientOnlinePayment(booking);

  const spotlightTime = formatSpotlightTime(scheduled);
  const relDay = formatRelativeDayLabel(scheduled);
  const longDay = formatLongWeekdayDateFr(scheduled);
  const spotlightDateLine =
    variant === 'spotlight' && !isRoundTripCard && relDay && longDay ? `${relDay}, ${longDay}` : '';

  return (
    <article id={`booking-${booking.id}`} className={cardClass}>
      <header className={variant === 'spotlight' ? styles.cardTopSpotlight : styles.cardTop}>
        <div className={styles.cardTopMain}>
          {variant === 'spotlight' && !isRoundTripCard && !isGroupedRequest ? (
            <>
              <div className={styles.spotlightEyebrow}>Prochaine course</div>
              <div className={styles.spotlightWhenRow}>
                <time className={styles.spotlightTime} dateTime={dateTimeAttr}>
                  {spotlightTime}
                </time>
                {spotlightDateLine ? (
                  <span className={styles.spotlightDateMuted}>— {spotlightDateLine}</span>
                ) : null}
              </div>
            </>
          ) : (
            <div className={styles.cardWhenStack}>
              {variant === 'spotlight' && (isRoundTripCard || isGroupedRequest) ? (
                <div className={styles.spotlightEyebrow}>Prochaine course</div>
              ) : null}
              <time
                className={isGroupedRequest ? styles.glanceDate : styles.cardWhen}
                dateTime={dateTimeAttr}
              >
                {isGroupedRequest
                  ? [relDay, longDay].filter(Boolean).join(' · ') || formatDateTime(scheduled)
                  : isRoundTripCard
                    ? `Aller · ${formatDateTime(scheduled)}`
                    : formatDateTime(scheduled)}
              </time>
              {isRoundTripCard && returnScheduled ? (
                <time className={styles.cardWhenSecondary} dateTime={returnDateTimeAttr}>
                  Retour · {formatDateTime(returnScheduled)}
                </time>
              ) : null}
            </div>
          )}
        </div>
        <div className={styles.cardTopRight}>
          {!isGroupedRequest && isRoundTripCard ? (
            <span className={styles.tripKindBadge} title="Demande aller et retour liées.">
              Aller-retour
            </span>
          ) : null}
          <span
            className={`${styles.statusPill} ${styles.statusPillWithDot} ${getClientBookingToneClass(statusUx.label, styles)}`}
          >
            <span className={styles.statusPillDot} aria-hidden />
            {statusUx.label}
          </span>
        </div>
      </header>

      {showPendingBroadcastHint && !isGroupedRequest ? (
        <div className={styles.pendingHint} role="status">
          <span className={styles.pendingHintIcon} aria-hidden>
            <SvgIconInfo size={14} />
          </span>
          <span className={styles.pendingHintText}>
            Votre demande est en cours de traitement. Une entreprise de transport sera sélectionnée dans
            les prochaines minutes.
          </span>
        </div>
      ) : null}

      <div className={isGroupedRequest ? styles.routeSplit : undefined}>
        <ClientRouteStops
          stops={
            isGroupedRequest
              ? requestStops
              : [
                  {
                    key: 'pickup',
                    label: 'Départ',
                    place: booking.pickup_location || 'Départ non renseigné',
                  },
                  {
                    key: 'dropoff',
                    label: 'Destination',
                    place: booking.dropoff_location || 'Destination non renseignée',
                  },
                ]
          }
        />
        {isGroupedRequest ? <ReservationRouteMap stops={requestStops} /> : null}
      </div>

      {showTimeline && !isGroupedRequest ? (
        <div className={styles.timelineSection}>
          <div className={styles.timelineSectionTitle}>Avancement de la prise en charge</div>
          <BookingStatusTimeline booking={booking} layout="horizontal" />
        </div>
      ) : null}
      {showDriverBlock && hasDriverAssigned(booking) ? (
        <DriverMiniCard booking={booking} emphasized={variant === 'spotlight'} />
      ) : null}

      <dl className={styles.metaGrid}>
        <div className={styles.metaItem}>
          <dt>
            <span className={styles.metaDtIcon} aria-hidden>
              <SvgIconCreditCard size={11} />
            </span>
            {awaitingCarrierConfirmation ? 'Prix maximum autorisé' : amountDisplay.label}
          </dt>
          <dd>
            {awaitingCarrierConfirmation ? (
              <>
                {formatAmount(booking.route_request_amount)}
                <span className={styles.metaAmountHint}>
                  {' '}
                  En attente de confirmation de l’entreprise de transport
                </span>
              </>
            ) : isGroupedRequest ? (
              formatAmount(booking.route_request_amount)
            ) : splitRoundTripAmounts ? (
              <>
                <div>
                  {formatAmount(booking.amount)}{' '}
                  <span className={styles.metaAmountHint}>(aller)</span>
                </div>
                <div>
                  {formatAmount(rbAmount)}{' '}
                  <span className={styles.metaAmountHint}>(retour)</span>
                </div>
                <div className={styles.metaAmountHint}>
                  Total dossier : {formatAmount(clientRoundTripTotalChf(booking))}
                </div>
              </>
            ) : amountDisplay.isCeiling && amountDisplay.lines.length > 0 ? (
              <>
                {amountDisplay.lines.map((line) => (
                  <div key={line.key}>
                    {line.primary ? (
                      <>
                        <strong>{formatAmount(line.amount)}</strong>
                        <span className={styles.metaAmountHint}> — {line.label}</span>
                      </>
                    ) : (
                      <>
                        {formatAmount(line.amount)}
                        <span className={styles.metaAmountHint}> — {line.label}</span>
                      </>
                    )}
                  </div>
                ))}
              </>
            ) : (
              <>
                {formatAmount(amountDisplay.amount)}
                {isRoundTripCard ? (
                  <span className={styles.metaAmountHint}> (total aller + retour)</span>
                ) : null}
              </>
            )}
          </dd>
        </div>
        {showCarrier && !awaitingCarrierConfirmation ? (
          <div className={styles.metaItem}>
            <dt>
              <span className={styles.metaDtIcon} aria-hidden>
                <SvgIconTruck size={11} />
              </span>
              Transporteur
            </dt>
            <dd>{booking.company_name || 'Non assigné'}</dd>
          </div>
        ) : null}
        {isGroupedRequest ? null : (
        <div className={styles.metaItem}>
          <dt>
            <span className={styles.metaDtIcon} aria-hidden>
              <SvgIconShield size={11} />
            </span>
            Couverture
          </dt>
          <dd>{getBillingCoverageLabel(booking)}</dd>
        </div>
        )}
      </dl>

      {showClientBookingPolicy ? (
        <ModificationAnnulationPolicy
          portalDoubleValidation={isPortalContractFlow(booking)}
        />
      ) : null}

      {inlineEditPanel}

      <div className={styles.bookingActions}>
        {primaryAction ? <div className={styles.primaryActionSlot}>{primaryAction}</div> : null}
        {secondaryActions ? (
          <div className={styles.secondaryActionsRow}>{secondaryActions}</div>
        ) : null}
      </div>
    </article>
  );
}

const ReservationsPage = () => {
  const { public_id: publicIdFromRoute } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedBookingId = Number.parseInt(
    searchParams.get('bookingId') || searchParams.get('booking') || '',
    10
  );
  const selectedBookingIdSafe = Number.isNaN(selectedBookingId) ? null : selectedBookingId;
  const effectivePublicId = publicIdFromRoute || getActivePublicId();

  const [bookings, setBookings] = useState([]);
  const [_clientData, setClientData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState('scheduled_time');
  const [filter, setFilter] = useState('all');
  const [exportPeriod, setExportPeriod] = useState('this_month');
  const [exportingPdf, setExportingPdf] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [pastExpandedId, setPastExpandedId] = useState(null);
  const [liveTrackBookingId, setLiveTrackBookingId] = useState(null);
  const [inlineEditBookingId, setInlineEditBookingId] = useState(null);
  const [contactModalBooking, setContactModalBooking] = useState(null);
  const [eventReportBooking, setEventReportBooking] = useState(null);
  const [carrierMessages, setCarrierMessages] = useState([]);
  const [readCarrierMessages, setReadCarrierMessages] = useState(() => readClientMessageIds());
  const [balanceDueChf, setBalanceDueChf] = useState(null);

  const reloadBookings = useCallback(async (quiet = false) => {
    if (!effectivePublicId) {
      setError("Impossible de charger les réservations : identifiant client introuvable.");
      if (!quiet) setLoading(false);
      return;
    }
    if (quiet) setIsRefreshing(true);
    else setLoading(true);
    if (!quiet) setError(null);
    try {
      const data = await fetchBookings(effectivePublicId);
      setBookings(data);
    } catch (err) {
      const msg = 'Impossible de charger les réservations.';
      setError(msg);
      if (quiet) toast.error(msg);
      console.error('Erreur chargement réservations :', err);
    } finally {
      if (quiet) setIsRefreshing(false);
      else setLoading(false);
    }
  }, [effectivePublicId]);

  const { pollError } = useHybridDataSync({
    fetchFn: () => reloadBookings(true),
    enabled: process.env.NODE_ENV !== 'test',
    staleThreshold: 120000,
    pollIntervalDisconnected: 45000,
    pollIntervalConnected: 180000,
    dependencies: [effectivePublicId],
  });

  useClientBookingSocketRefresh(reloadBookings, Boolean(effectivePublicId));

  const loadClientData = useCallback(async () => {
    try {
      const client = await fetchClient(effectivePublicId);
      setClientData(client);
    } catch (err) {
      console.error('Erreur lors du chargement du profil client :', err);
    }
  }, [effectivePublicId]);

  useEffect(() => {
    loadClientData();
  }, [loadClientData]);

  useEffect(() => {
    if (effectivePublicId) {
      reloadBookings(false);
    }
  }, [effectivePublicId, reloadBookings]);

  useEffect(() => {
    let cancelled = false;
    Promise.resolve(apiClient.get('/clients/me/portal-receivables'))
      .then((res) => {
        const rows = Array.isArray(res?.data?.data) ? res.data.data : [];
        const sum = rows.reduce((acc, row) => acc + Number(row.balance_due || 0), 0);
        if (!cancelled) setBalanceDueChf(Number.isFinite(sum) ? sum : 0);
      })
      .catch(() => {
        if (!cancelled) setBalanceDueChf(null);
      });
    return () => {
      cancelled = true;
    };
  }, [effectivePublicId]);

  // 🎯 Annuler une réservation
  const onlinePaymentCompleted = (booking) => {
    const st = (booking.online_payment?.status || '').toLowerCase();
    return st === 'completed';
  };

  const canPayOnline = (booking) => {
    const status = (booking.status || '').toLowerCase();
    if (status === 'canceled') return false;
    if (status !== 'pending' && status !== 'awaiting_client_payment') return false;
    if (onlinePaymentCompleted(booking)) return false;
    if (!requiresPrivateOnlinePaymentAtBooking(booking)) return false;
    return true;
  };

  const handlePayOnline = (booking) => {
    const bookingId = booking?.id;
    if (!bookingId) return;
    trackClientKpiEvent('pay_now_clicked', { bookingId, clientPublicId: effectivePublicId });
    const statusUx = getClientBookingUx(resolveClientBookingDisplayStatus(booking));
    navigate(
      `/client/payment/saferpay/start?bookingId=${encodeURIComponent(String(bookingId))}`,
      {
        state: {
          amountChf: clientRoundTripTotalChf(booking),
          payerLabel: booking.payer_label || booking.coverage_label || 'Client',
          lifecycleLabel: statusUx.label,
          fromDashboardPath: `/reservations/${encodeURIComponent(String(effectivePublicId))}`,
        },
      }
    );
  };

  const scrollToBooking = useCallback((bookingId) => {
    const el = document.getElementById(`booking-${bookingId}`);
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, []);

  const openInlineEditFor = useCallback((bookingId) => {
    setInlineEditBookingId((cur) => {
      const next = cur === bookingId ? null : bookingId;
      if (next != null) {
        requestAnimationFrame(() => {
          document.getElementById(`booking-${bookingId}`)?.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest',
          });
        });
      }
      return next;
    });
  }, []);

  const goDashboardPrefill = useCallback(
    (booking) => {
      if (!effectivePublicId || !booking) return;
      const stops = historyRouteStops(booking);
      const hasReturn = stops.some((stop) => stop.label === 'Retour');
      const routeStops = (hasReturn ? stops.slice(0, -1) : stops).filter((stop) =>
        String(stop?.place || '').trim()
      );
      const dropoffs = routeStops.slice(1);
      const roundTrip =
        hasReturn ||
        booking.is_round_trip === true ||
        booking.is_round_trip === 1 ||
        booking.is_round_trip === '1' ||
        booking.is_round_trip === 'true' ||
        booking.has_return === true ||
        booking.has_return === 1 ||
        booking.has_return === '1' ||
        booking.has_return === 'true';
      navigate(`/dashboard/client/${encodeURIComponent(effectivePublicId)}`, {
        state: {
          prefillFromBooking: {
            pickup_location: routeStops[0]?.place || booking.pickup_location || '',
            dropoff_location: dropoffs[0]?.place || booking.dropoff_location || '',
            dropoff_detail: dropoffs[0]?.detail || '',
            extra_stops: dropoffs.slice(1).map((stop) => ({
              place: stop.place,
              detail: stop.detail || '',
            })),
            round_trip: roundTrip,
          },
        },
      });
    },
    [navigate, effectivePublicId]
  );

  const handleOpenSupportFromContactModal = useCallback(() => {
    setContactModalBooking(null);
    navigate('/contact/support');
  }, [navigate]);

  const handleConfirmReturnAfterCall = useCallback(
    async (booking) => {
      if (!effectivePublicId || !booking?.id) return;
      try {
        await apiClient.post(
          `/clients/${encodeURIComponent(effectivePublicId)}/bookings/${booking.id}/confirm-return-time`
        );
        await reloadBookings(true);
        toast.success(
          "L’heure du retour est enregistrée comme confirmée pour le transporteur. Merci."
        );
      } catch (err) {
        console.error(err);
        toast.error(
          String(
            err?.response?.data?.message ||
              err?.response?.data?.error ||
              "Impossible d’enregistrer la confirmation pour le moment."
          ),
          { duration: 7000 }
        );
        return;
      }
      const raw = String(booking?.company_contact_phone || '').replace(/\s/g, '');
      if (raw.length >= 8 && /^\+?\d/.test(raw)) {
        window.location.href = `tel:${raw}`;
      } else {
        toast.info(
          'Numéro du transporteur non affiché ici : utilisez les coordonnées reçues par SMS ou courriel.',
          { duration: 7000 }
        );
      }
    },
    [effectivePublicId, reloadBookings]
  );

  const handleRequestUrgentReturn = useCallback(
    async (booking) => {
      if (!effectivePublicId || !booking?.id) return;
      try {
        const { data } = await apiClient.post(
          `/clients/${encodeURIComponent(effectivePublicId)}/bookings/${booking.id}/request-urgent-return`,
          { minutes_offset: 15 }
        );
        const body = data?.data ?? data;
        await reloadBookings(true);
        if (body?.assigned_driver_id) {
          toast.success(
            'Retour programmé : un chauffeur a été assigné. Les détails de la course se mettent à jour sur cette page.'
          );
        } else {
          toast.success(
            'Retour programmé : prise en charge visée dans environ 15 minutes. Le transporteur en est informé.'
          );
        }
      } catch (err) {
        console.error(err);
        toast.error(
          String(
            err?.response?.data?.message ||
              err?.response?.data?.error ||
              err?.response?.data?.description ||
              'Impossible de programmer le retour pour le moment.'
          ),
          { duration: 8000 }
        );
      }
    },
    [effectivePublicId, reloadBookings]
  );

  const handleTrackMap = useCallback((booking) => {
    const u = String(booking?.tracking_url || '').trim();
    if (u && /^https?:\/\//i.test(u)) {
      window.open(u, '_blank', 'noopener,noreferrer');
      return;
    }
    if (canClientLiveTrack(booking)) {
      setLiveTrackBookingId(booking.id);
      return;
    }
    const norm = normalizeClientBookingStatus(booking?.status);
    if (
      (norm === 'driver_on_the_way' || norm === 'in_progress') &&
      hasDriverAssigned(booking)
    ) {
      document.getElementById(`booking-driver-${booking.id}`)?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }
    const noLiveMapDriverEnRoute =
      'Le transporteur n’a pas encore activé le partage de position sur carte. Votre chauffeur est déjà en route : le bloc « Votre chauffeur » ci-dessous est à jour. Réessayez plus tard pour ouvrir le lien, ou contactez le secrétariat si besoin.';
    const noLiveMapInProgress =
      'Aucun suivi carte en direct n’est proposé pour cette course en cours. Les informations sur votre chauffeur restent affichées sur cette page ; en cas de question, contactez le transporteur ou le secrétariat.';
    const noLiveMapDefault =
      'Le suivi sur carte sera disponible lorsque le transporteur activera le lien de partage.';
    const msg =
      norm === 'driver_on_the_way'
        ? noLiveMapDriverEnRoute
        : norm === 'in_progress'
          ? noLiveMapInProgress
          : noLiveMapDefault;
    toast.info(msg, { duration: 7000 });
  }, []);

  const liveTrackBooking = useMemo(
    () =>
      liveTrackBookingId != null ? bookings.find((b) => b.id === liveTrackBookingId) || null : null,
    [bookings, liveTrackBookingId]
  );

  useEffect(() => {
    if (liveTrackBookingId == null) return undefined;
    const t = setInterval(() => {
      void reloadBookings(true).catch(() => {});
    }, 25000);
    return () => clearInterval(t);
  }, [liveTrackBookingId, reloadBookings]);

  useEffect(() => {
    if (liveTrackBookingId == null) return;
    if (!bookings.some((b) => b.id === liveTrackBookingId)) {
      setLiveTrackBookingId(null);
    }
  }, [bookings, liveTrackBookingId]);

  const handleCancelBooking = async (booking) => {
    const ids = (
      Array.isArray(booking?.route_request_ids) && booking.route_request_ids.length
        ? booking.route_request_ids
        : [booking?.id]
    ).filter((id) => Number(id) > 0);
    const confirmMessage =
      ids.length > 1
        ? `Confirmer l’annulation de cette demande ? Les ${ids.length} transports seront annulés ensemble.`
        : 'Confirmer l’annulation de cette réservation ? Les conditions de remboursement (délais 24 h / 4 h, aller-retour) sont rappelées dans la section « Modification et annulation » au-dessus des boutons.';
    if (!window.confirm(confirmMessage)) {
      return;
    }

    setBookings((prevBookings) =>
      prevBookings.map((b) => (ids.includes(b.id) ? { ...b, isCancelling: true } : b))
    );

    try {
      for (const id of ids) {
        const response = await apiClient.delete(`/bookings/${id}`);
        if (response.status !== 200) {
          throw new Error("L'annulation a échoué.");
        }
      }
      setInlineEditBookingId((cur) => (ids.includes(cur) ? null : cur));
      await reloadBookings(true);
      toast.success(ids.length > 1 ? 'Demande annulée.' : 'Réservation annulée.');
    } catch (error) {
      console.error("Erreur lors de l'annulation :", error);
      setBookings((prev) =>
        prev.map((b) => (ids.includes(b.id) ? { ...b, isCancelling: false } : b))
      );
      toast.error("Une erreur s'est produite lors de l'annulation.", { duration: 6000 });
    }
  };

  const clientBookingsForDisplay = useMemo(
    () => foldClientRouteRequests(bookings),
    [bookings]
  );

  const loadCarrierMessages = useCallback(async () => {
    try {
      const data = await fetchClientMessageNotifications();
      setCarrierMessages(Array.isArray(data?.notifications) ? data.notifications : []);
    } catch (loadError) {
      console.error('[ReservationsPage] messages transporteur', loadError);
    }
  }, []);

  useEffect(() => {
    void loadCarrierMessages();
  }, [loadCarrierMessages]);

  useEffect(() => {
    const syncRead = () => setReadCarrierMessages(readClientMessageIds());
    window.addEventListener(CLIENT_MESSAGE_READ_EVENT, syncRead);
    return () => window.removeEventListener(CLIENT_MESSAGE_READ_EVENT, syncRead);
  }, []);

  useEffect(() => {
    let socket;
    let cancelled = false;
    const onMessage = (payload) => {
      const sender = String(payload?.message?.sender_type || '').toUpperCase();
      if (sender && sender !== 'COMPANY') return;
      void loadCarrierMessages();
    };
    ensureClientPortalSocket().then((connected) => {
      if (cancelled || !connected) return;
      socket = connected;
      socket.on('booking_message', onMessage);
    });
    return () => {
      cancelled = true;
      socket?.off('booking_message', onMessage);
    };
  }, [loadCarrierMessages]);

  const openContact = useCallback((booking) => {
    setContactModalBooking(booking);
    const ids = new Set(
      [booking?.id, ...(booking?.route_request_ids || [])].map(Number)
    );
    const unreadIds = carrierMessages
      .filter((item) => ids.has(Number(item?.booking_id)) && !readCarrierMessages.has(Number(item?.id)))
      .map((item) => item.id);
    if (unreadIds.length) {
      setReadCarrierMessages(rememberClientMessageRead(unreadIds));
    }
  }, [carrierMessages, readCarrierMessages]);

  const contactBookingId = Number.parseInt(searchParams.get('contact') || '', 10);
  useEffect(() => {
    if (!Number.isFinite(contactBookingId) || contactBookingId <= 0) return;
    const match = clientBookingsForDisplay.find((booking) => (
      Number(booking?.id) === contactBookingId
      || (booking?.route_request_ids || []).some((id) => Number(id) === contactBookingId)
    ));
    if (!match) return;
    openContact(match);
    const next = new URLSearchParams(searchParams);
    next.delete('contact');
    setSearchParams(next, { replace: true });
  }, [contactBookingId, clientBookingsForDisplay, searchParams, setSearchParams, openContact]);

  // 📌 Tri et filtrage des réservations
  const sortedBookings = [...clientBookingsForDisplay].sort((a, b) => {
    if (!a || !b) return 0;

    if (sortBy === 'scheduled_time') {
      return new Date(a.scheduled_time) - new Date(b.scheduled_time);
    } else if (sortBy === 'amount') {
      return clientRoundTripTotalChf(b) - clientRoundTripTotalChf(a);
    } else if (sortBy === 'status') {
      return a.status.localeCompare(b.status);
    }
    return 0;
  });

  const statusForClientFilter = (raw) => {
    const t = String(raw || '').toLowerCase();
    if (t === 'awaiting_client_payment') return 'pending';
    if (t === 'round_trip_return_pending') return 'confirmed';
    return t;
  };

  const filteredBookings = sortedBookings.filter((booking) => {
    if (filter === 'all') return true;
    const disp = resolveClientBookingDisplayStatus(booking);
    return statusForClientFilter(disp) === filter;
  });

  const nowTimestamp = Date.now();
  const upcomingBookings = filteredBookings
    .filter((booking) => clientTripListSection(booking, nowTimestamp) === 'upcoming')
    .sort((a, b) => clientUpcomingSortKey(a) - clientUpcomingSortKey(b));
  const nextBooking = upcomingBookings[0] || null;
  const otherUpcoming = upcomingBookings.slice(1);
  const pastBookings = filteredBookings
    .filter((booking) => clientTripListSection(booking, nowTimestamp) === 'past')
    .sort((a, b) => Date.parse(b.scheduled_time) - Date.parse(a.scheduled_time));

  const hasUpcoming = upcomingBookings.length > 0;
  const hasHistory = pastBookings.length > 0;

  const bookingStats = useMemo(() => {
    const nowTs = Date.now();
    const list = clientBookingsForDisplay;
    const upcomingN = list.filter((b) => clientTripListSection(b, nowTs) === 'upcoming').length;
    const pastOnly = list.filter((b) => clientTripListSection(b, nowTs) === 'past');
    const completedN = pastOnly.filter(
      (b) => normalizeClientBookingStatus(resolveClientBookingDisplayStatus(b)) === 'completed'
    ).length;
    const canceledN = pastOnly.filter((b) => {
      const n = normalizeClientBookingStatus(resolveClientBookingDisplayStatus(b));
      return n === 'cancelled' || n === 'canceled';
    }).length;
    return {
      upcomingN,
      completedN,
      canceledN,
    };
  }, [clientBookingsForDisplay]);

  const filterPills = useMemo(
    () => [
      { value: 'all', label: 'Toutes' },
      { value: 'completed', label: 'Terminées' },
      { value: 'canceled', label: 'Annulées' },
      { value: 'pending', label: 'En attente' },
      { value: 'confirmed', label: 'Confirmées' },
    ],
    []
  );

  const bookingActions = useCallback((booking) => getEffectiveClientBookingActions(booking), []);

  const formatDateTime = (iso) => {
    const parsed = Date.parse(iso);
    if (!Number.isFinite(parsed)) return 'Date inconnue';
    return new Date(parsed).toLocaleString('fr-CH', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatAmount = (amount) => {
    const numeric = Number(amount);
    if (!Number.isFinite(numeric)) return '-- CHF';
    return `${numeric.toFixed(2)} CHF`;
  };

  const renderPrimaryAction = (booking) => {
    if (!booking) return null;
    if (canPayOnline(booking)) {
      return (
        <button type="button" className="primaryButton" onClick={() => handlePayOnline(booking)}>
          Payer maintenant
        </button>
      );
    }
    const norm = normalizeClientBookingStatus(resolveClientBookingDisplayStatus(booking));
    if (norm === 'cancelled') {
      return (
        <button type="button" className="primaryButton" onClick={() => goDashboardPrefill(booking)}>
          Recommander ce trajet
        </button>
      );
    }
    if (norm === 'round_trip_return_pending') {
      return (
        <button
          type="button"
          className="primaryButton"
          onClick={() => handleRequestUrgentReturn(booking)}
        >
          Programmer le retour d&apos;urgence (~15 min)
        </button>
      );
    }
    if (norm === 'completed') {
      return (
        <button type="button" className="primaryButton" onClick={() => setEventReportBooking(booking)}>
          Signaler un événement
        </button>
      );
    }
    if (norm === 'pending') {
      return (
        <button type="button" className="primaryButton" onClick={() => openInlineEditFor(booking.id)}>
          Modifier la demande
        </button>
      );
    }
    if (norm === 'awaiting_payment') {
      return (
        <button type="button" className="primaryButton" onClick={() => scrollToBooking(booking.id)}>
          Voir le détail
        </button>
      );
    }
    if (norm === 'confirmed') {
      return null;
    }
    if (norm === 'driver_on_the_way' || norm === 'in_progress') {
      const label =
        hasTrackingUrl(booking) || canClientLiveTrack(booking)
          ? 'Suivre sur la carte'
          : 'Suivre la course';
      return (
        <button type="button" className="primaryButton" onClick={() => handleTrackMap(booking)}>
          {label}
        </button>
      );
    }
    return (
      <button type="button" className="primaryButton" onClick={() => scrollToBooking(booking.id)}>
        Voir le détail
      </button>
    );
  };

  const renderSecondaryUpcoming = (booking) => {
    const acts = bookingActions(booking);
    const norm = normalizeClientBookingStatus(resolveClientBookingDisplayStatus(booking));
    const primaryIsTrack = norm === 'driver_on_the_way' || norm === 'in_progress';
    return (
      <>
        {norm === 'round_trip_return_pending' ? (
          <button
            type="button"
            className="secondaryButton"
            onClick={() => setContactModalBooking(booking)}
          >
            Appeler le transporteur
          </button>
        ) : null}
        {norm === 'round_trip_return_pending' &&
        booking?.return_booking?.scheduled_time &&
        booking?.return_booking?.time_confirmed === false ? (
          <button
            type="button"
            className="secondaryButton"
            onClick={() => handleConfirmReturnAfterCall(booking)}
          >
            Confirmer l&apos;heure du retour (après appel)
          </button>
        ) : null}
        {acts.includes('Modifier') ? (
          <button
            type="button"
            className="secondaryButton"
            onClick={() => openInlineEditFor(booking.id)}
            aria-expanded={inlineEditBookingId === booking.id}
          >
            {inlineEditBookingId === booking.id ? 'Fermer le formulaire' : 'Modifier'}
          </button>
        ) : null}
        {acts.includes('Suivre') && !primaryIsTrack ? (
          <button type="button" className="secondaryButton" onClick={() => handleTrackMap(booking)}>
            Suivre
          </button>
        ) : null}
        <button type="button" className="secondaryButton" onClick={() => openContact(booking)}>
          Contacter
          {bookingHasUnreadCarrierMessage(booking, carrierMessages, readCarrierMessages) ? (
            <span className={styles.contactNewDot} aria-label="Nouveau message" />
          ) : null}
        </button>
        {acts.includes('Annuler') ? (
          <button
            type="button"
            className={styles.cancelLinkBtn}
            onClick={() => handleCancelBooking(booking)}
            disabled={booking.isCancelling}
          >
            {booking.isCancelling ? 'Annulation…' : 'Annuler'}
          </button>
        ) : null}
      </>
    );
  };

  const renderSecondaryHistory = (booking) => {
    const norm = normalizeClientBookingStatus(resolveClientBookingDisplayStatus(booking));
    if (norm === 'round_trip_return_pending') {
      return (
        <>
          <button
            type="button"
            className="secondaryButton"
            onClick={() => handleRequestUrgentReturn(booking)}
          >
            Programmer le retour (~15 min)
          </button>
          <button
            type="button"
            className="secondaryButton"
            onClick={() => setContactModalBooking(booking)}
          >
            Appeler le transporteur
          </button>
        </>
      );
    }
    if (norm !== 'completed') return null;
    return (
      <button type="button" className="secondaryButton" onClick={() => goDashboardPrefill(booking)}>
        Recommander
      </button>
    );
  };

  const renderInlineEditForBooking = (booking) =>
    inlineEditBookingId === booking.id ? (
      <ClientBookingInlineEditForm
        key={booking.id}
        booking={booking}
        clientPublicId={effectivePublicId}
        onClose={() => setInlineEditBookingId(null)}
        onSaved={async () => {
          setInlineEditBookingId(null);
          await reloadBookings(true);
          toast.success('Modifications enregistrées.');
        }}
      />
    ) : null;

  const handleExportPdf = async () => {
    if (!effectivePublicId || exportingPdf) return;
    if (!hasHistory) {
      toast.info('Aucune donnée exportable pour la période sélectionnée.');
      return;
    }
    setExportingPdf(true);
    trackClientKpiEvent('history_export_clicked', {
      clientPublicId: effectivePublicId,
      period: exportPeriod,
    });
    try {
      const result = await exportBookingsPDF(exportPeriod, pastBookings, _clientData, null);
      if (result?.pdfUrl) {
        window.open(result.pdfUrl, '_blank', 'noopener,noreferrer');
      }
      toast.success('Export PDF prêt.');
    } catch (exportError) {
      const code = exportError?.code || exportError?.response?.status;
      if (code === 404) {
        toast.info("Le service d'export est temporairement indisponible. Réessayez plus tard.");
      } else if (code === 'empty_period') {
        toast.info('Aucune donnée exportable pour la période sélectionnée.');
      } else if (code === 'custom_period_invalid') {
        toast.error('La période personnalisée est invalide. Vérifiez les dates.');
      } else {
        toast.error("Impossible de lancer l'export PDF pour le moment.");
      }
    } finally {
      setExportingPdf(false);
    }
  };

  useEffect(() => {
    if (!selectedBookingIdSafe || bookings.length === 0) return;
    const target = bookings.find((b) => b.id === selectedBookingIdSafe);
    if (target) {
      const ts = Date.parse(target.scheduled_time);
      if (Number.isFinite(ts) && ts <= Date.now()) {
        setPastExpandedId(selectedBookingIdSafe);
      }
    }
    const el = document.getElementById(`booking-${selectedBookingIdSafe}`);
    if (!el) return;
    const t = setTimeout(() => {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 50);
    return () => clearTimeout(t);
  }, [bookings, selectedBookingIdSafe]);

  return (
    <div className="container">
      <HeaderDashboard />

      <main className={styles.reservationsMain}>
        <div className={styles.reservationsInner}>
          <div className={styles.masterCard}>
            <header className={styles.heroHeader}>
              <div className={styles.heroMastheadGrid}>
                <div className={styles.heroLead}>
                  <div className={styles.heroTitleBlock}>
                    <div className={styles.heroEyebrow}>
                      <span className={styles.heroEyebrowDot} aria-hidden />
                      Suivi en temps réel
                    </div>
                    <h1 className="title bookingHeaderTitle info-level-1-action">Mes courses</h1>
                    <p className={styles.heroSubtitle}>
                      Suivez l&apos;avancement de vos transports médicaux, gérez vos réservations et consultez
                      votre historique.
                    </p>
                    <div className={styles.heroStats} aria-label="Synthèse de vos courses">
                      <div className={styles.heroStat}>
                        <span className={styles.heroStatValue}>{bookingStats.upcomingN}</span>
                        <span className={styles.heroStatLabel}>À venir</span>
                      </div>
                      <div className={styles.heroStatDivider} aria-hidden />
                      <div className={styles.heroStat}>
                        <span className={styles.heroStatValue}>{bookingStats.completedN}</span>
                        <span className={styles.heroStatLabel}>Terminées</span>
                      </div>
                      <div className={styles.heroStatDivider} aria-hidden />
                      <div className={styles.heroStat}>
                        <span className={styles.heroStatValue}>
                          {balanceDueChf == null
                            ? '—'
                            : `${balanceDueChf.toFixed(2)} CHF`}
                        </span>
                        <span className={styles.heroStatLabel}>Reste à payer</span>
                      </div>
                      <div className={styles.heroStatDivider} aria-hidden />
                      <div className={styles.heroStat}>
                        <span className={styles.heroStatValue}>{bookingStats.canceledN}</span>
                        <span className={styles.heroStatLabel}>Annulées</span>
                      </div>
                    </div>
                  </div>
                  {isRefreshing ? (
                    <span className={styles.refreshBadge} aria-live="polite">
                      Actualisation…
                    </span>
                  ) : null}
                </div>
                <div className={styles.heroAlerts}>
                  {loading && <div className="loadingSkeleton" aria-hidden />}
                  {error && (
                    <p className="error" role="alert">
                      {error}
                    </p>
                  )}
                  {pollError && !error ? (
                    <p className={styles.pollHint} role="status">
                      Mise à jour temporairement ralentie. Vos informations reviennent automatiquement.
                    </p>
                  ) : null}
                </div>
              </div>
            </header>

            <div className={styles.listControlsBar} role="toolbar" aria-label="Filtres et tri des courses">
              <div className={styles.listControlsInner}>
                <div className={styles.filterPillRow}>
                  {filterPills.map((p) => (
                    <button
                      key={p.value}
                      type="button"
                      className={`${styles.filterPill} ${filter === p.value ? styles.filterPillActive : ''}`}
                      onClick={() => setFilter(p.value)}
                      aria-pressed={filter === p.value}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
                <div className={styles.sortControl}>
                  <span className={styles.sortIcon} aria-hidden>
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                      <line x1="4" y1="6" x2="20" y2="6" stroke="currentColor" strokeWidth="2" />
                      <line x1="4" y1="12" x2="14" y2="12" stroke="currentColor" strokeWidth="2" />
                      <line x1="4" y1="18" x2="8" y2="18" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </span>
                  <label htmlFor="reservations-sort" className={styles.visuallyHidden}>
                    Trier
                  </label>
                  <select
                    id="reservations-sort"
                    className={styles.sortSelectCompact}
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                  >
                    <option value="scheduled_time">Par date</option>
                    <option value="amount">Par montant</option>
                    <option value="status">Par statut</option>
                  </select>
                </div>
              </div>
            </div>

            <section className={styles.sectionBlock} aria-labelledby="upcoming-heading">
              <h2 id="upcoming-heading" className={styles.sectionLabel}>
                À venir
              </h2>
              {!hasUpcoming ? (
                <div className={styles.upcomingEmptyWrap}>
                  <UpcomingEmptyCard clientPublicId={effectivePublicId} />
                </div>
              ) : (
                <div className={styles.reservationList}>
                  {nextBooking ? (
                    <div className={styles.upcomingSpotlightBlock}>
                      <ReservationCard
                        booking={nextBooking}
                        isSelected={selectedBookingIdSafe === nextBooking.id}
                        showCarrier
                        showTimeline
                        showDriverBlock
                        variant="spotlight"
                        showClientBookingPolicy
                        formatDateTime={formatDateTime}
                        formatAmount={formatAmount}
                        primaryAction={renderPrimaryAction(nextBooking)}
                        secondaryActions={renderSecondaryUpcoming(nextBooking)}
                        inlineEditPanel={renderInlineEditForBooking(nextBooking)}
                      />
                    </div>
                  ) : null}
                  {otherUpcoming.map((booking) => (
                    <ReservationCard
                      key={booking.id}
                      booking={booking}
                      isSelected={selectedBookingIdSafe === booking.id}
                      showCarrier
                      showTimeline
                      showDriverBlock
                      showClientBookingPolicy
                      formatDateTime={formatDateTime}
                      formatAmount={formatAmount}
                      primaryAction={renderPrimaryAction(booking)}
                      secondaryActions={renderSecondaryUpcoming(booking)}
                      inlineEditPanel={renderInlineEditForBooking(booking)}
                    />
                  ))}
                  {nextBooking &&
                  otherUpcoming.length === 0 &&
                  !nextBooking.route_request_stops?.length ? (
                    <p className={styles.upcomingOnlyNextHint}>Aucune autre course programmée.</p>
                  ) : null}
                </div>
              )}
            </section>

            <section className={styles.sectionBlock} aria-labelledby="past-heading">
              <h2 id="past-heading" className={styles.sectionLabel}>
                Historique
              </h2>
              <div className={styles.exportBar}>
                <div className={styles.exportPeriod}>
                  <label htmlFor="export-period">Exporter :</label>
                  <select
                    id="export-period"
                    className={styles.monthSelect}
                    value={exportPeriod}
                    onChange={(event) => setExportPeriod(event.target.value)}
                    aria-label="Période d'export PDF"
                  >
                    <option value="this_month">Ce mois</option>
                    <option value="previous_month">Mois précédent</option>
                    <option value="this_year">Cette année</option>
                    <option value="custom">Période personnalisée</option>
                  </select>
                </div>
                <button
                  type="button"
                  className={`secondaryButton ${styles.exportBtn}`}
                  onClick={handleExportPdf}
                  disabled={exportingPdf}
                >
                  {exportingPdf ? 'Export en cours…' : 'Exporter en PDF'}
                </button>
              </div>
              {hasHistory ? (
                <div className={styles.pastRowsList}>
                  {pastBookings.map((booking) => {
                    const displayPast = resolveClientBookingDisplayStatus(booking);
                    const statusUx = getClientBookingUx(displayPast);
                    const normPast = normalizeClientBookingStatus(displayPast);
                    const tripCount = Number(booking.route_request_transport_count) || 0;
                    const destination =
                      String(booking.dropoff_location || '').trim() ||
                      String(booking.pickup_location || '').trim() ||
                      'Course';
                    const title =
                      tripCount > 1 ? `${tripCount} trajets · ${destination}` : destination;
                    const historyStops = historyRouteStops(booking);
                    const historyHeading =
                      historyStops.length > 2 ? 'Étapes du trajet' : 'Trajet';
                    const totalChf = clientRoundTripTotalChf(booking);
                    const parsed = Date.parse(booking.scheduled_time);
                    const timeLabel = Number.isFinite(parsed)
                      ? new Date(parsed).toLocaleTimeString('fr-CH', {
                          hour: '2-digit',
                          minute: '2-digit',
                        })
                      : '';
                    const relativeDayLabel = Number.isFinite(parsed)
                      ? formatRelativeDayLabel(booking.scheduled_time)
                      : '';
                    const amountContent =
                      normPast === 'cancelled' || !Number.isFinite(totalChf) || totalChf <= 0 ? (
                        <span className={styles.pastAmountDash}>—</span>
                      ) : (
                        formatAmount(totalChf)
                      );
                    const statusPill = (
                      <span
                        className={`${styles.statusPill} ${styles.statusPillWithDot} ${getClientBookingToneClass(statusUx.label, styles)}`}
                      >
                        <span className={styles.statusPillDot} aria-hidden />
                        {statusUx.label}
                      </span>
                    );
                    const historySecondary = renderSecondaryHistory(booking);
                    return (
                      <PastBookingRow
                        key={booking.id}
                        booking={booking}
                        isSelected={selectedBookingIdSafe === booking.id}
                        isExpanded={pastExpandedId === booking.id}
                        onToggleDetails={() =>
                          setPastExpandedId((prev) => (prev === booking.id ? null : booking.id))
                        }
                        onContact={() => openContact(booking)}
                        hasNewMessage={bookingHasUnreadCarrierMessage(
                          booking,
                          carrierMessages,
                          readCarrierMessages
                        )}
                        title={title}
                        relativeDayLabel={relativeDayLabel}
                        timeLabel={timeLabel}
                        statusPill={statusPill}
                        amountContent={amountContent}
                      >
                        {historyStops.length > 0 ? (
                          <div className={styles.pastRowPanelTimelineSection}>
                            <h4 className={styles.pastRowPanelSubheading}>{historyHeading}</h4>
                            <div className={styles.routeSplit}>
                              <ClientRouteStops stops={historyStops} label={historyHeading} />
                              <ReservationRouteMap stops={historyStops} />
                            </div>
                          </div>
                        ) : null}
                        {normPast === 'completed' ? (
                          <div className={styles.pastHistoryActions}>
                            <button
                              type="button"
                              className={styles.pastHistoryPrimary}
                              onClick={() => setEventReportBooking(booking)}
                            >
                              Signaler un événement
                            </button>
                            <button
                              type="button"
                              className={styles.pastHistorySecondary}
                              onClick={() => goDashboardPrefill(booking)}
                            >
                              Recommander
                            </button>
                          </div>
                        ) : (
                          <div className={styles.pastRowPanelActions}>
                            <div className={styles.primaryActionSlot}>
                              {renderPrimaryAction(booking)}
                            </div>
                            {historySecondary ? (
                              <div className={styles.secondaryActionsRow}>{historySecondary}</div>
                            ) : null}
                          </div>
                        )}
                      </PastBookingRow>
                    );
                  })}
                </div>
              ) : (
                <div className={styles.emptyCard}>
                  <p>Aucune course passée.</p>
                </div>
              )}
            </section>
          </div>
        </div>
      </main>

      {liveTrackBooking ? (
        <ClientBookingLiveTrackModal
          booking={liveTrackBooking}
          etaLine={formatDriverEtaLine(liveTrackBooking)}
          onClose={() => setLiveTrackBookingId(null)}
        />
      ) : null}

      <ClientTransportContactModal
        booking={contactModalBooking}
        open={Boolean(contactModalBooking)}
        onClose={() => setContactModalBooking(null)}
        onOpenSupport={handleOpenSupportFromContactModal}
      />

      <ClientTripEventReportModal
        booking={eventReportBooking}
        client={_clientData}
        open={Boolean(eventReportBooking)}
        onClose={() => setEventReportBooking(null)}
      />

      <Footer />
    </div>
  );
};

export default ReservationsPage;
