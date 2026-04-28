import { trackClientKpiEvent } from './clientKpi';
import { CLIENT_SURFACE_CONTRACTS } from './clientSurfaceContracts';

export const STATUS_DICTIONARY_VERSION = CLIENT_SURFACE_CONTRACTS.statusDictionaryVersion;

const STATUS_ALIASES = {
  pending: 'pending',
  requested: 'pending',
  awaiting: 'pending',
  waiting: 'pending',
  awaiting_client_payment: 'awaiting_payment',
  confirmed: 'confirmed',
  accepted: 'confirmed',
  validated: 'confirmed',
  assigned: 'confirmed',
  driver_on_the_way: 'driver_on_the_way',
  driver_en_route: 'driver_on_the_way',
  driver_arriving: 'driver_on_the_way',
  en_route: 'driver_on_the_way',
  in_progress: 'in_progress',
  ongoing: 'in_progress',
  started: 'in_progress',
  completed: 'completed',
  done: 'completed',
  finished: 'completed',
  terminated: 'completed',
  cancelled: 'cancelled',
  canceled: 'cancelled',
  rejected: 'cancelled',
  return_completed: 'completed',
  /** Statut de synthèse (aller terminé, retour encore actif) — défini côté UI. */
  round_trip_return_pending: 'round_trip_return_pending',
};

const STATUS_CONFIG = {
  awaiting_payment: {
    label: 'Paiement requis',
    tone: 'warning',
    actions: ['Annuler'],
  },
  pending: {
    label: 'Demande envoyée',
    tone: 'warning',
    actions: ['Voir', 'Annuler'],
  },
  confirmed: {
    label: 'Course confirmée',
    tone: 'success',
    actions: ['Voir', 'Modifier', 'Annuler'],
  },
  driver_on_the_way: {
    label: 'Chauffeur en route',
    tone: 'info',
    actions: ['Suivre'],
  },
  in_progress: {
    label: 'En cours',
    tone: 'info',
    actions: ['Suivre'],
  },
  completed: {
    label: 'Terminée',
    tone: 'success',
    actions: ['Recommander'],
  },
  round_trip_return_pending: {
    label: 'Aller terminé — retour à venir',
    tone: 'info',
    actions: ['Voir', 'Contacter'],
  },
  cancelled: {
    label: 'Annulée',
    tone: 'danger',
    actions: ['Recommander'],
  },
  unknown: {
    label: 'Statut en cours de mise à jour',
    tone: 'warning',
    actions: ['Rafraîchir'],
  },
};

export function normalizeClientBookingStatus(rawStatus) {
  const normalized = String(rawStatus || '').trim().toLowerCase();
  const resolved = STATUS_ALIASES[normalized] || 'unknown';
  if (resolved === 'unknown') {
    trackClientKpiEvent('status_dictionary_mismatch_event', {
      surface: 'web',
      status: normalized || null,
      statusDictionaryVersion: STATUS_DICTIONARY_VERSION,
    });
  }
  return resolved;
}

/**
 * Statut affiché pour une ligne « aller » d’un aller-retour : tant que le retour
 * n’est pas terminé (ou annulé), le dossier reste « en cours » côté client.
 */
export function resolveClientBookingDisplayStatus(booking) {
  if (!booking || booking.is_return) {
    return booking?.status || '';
  }
  const rb = booking.return_booking;
  const outNorm = normalizeClientBookingStatus(booking.status);
  if (rb && outNorm === 'completed') {
    const rsn = normalizeClientBookingStatus(rb.status);
    if (rsn !== 'completed' && rsn !== 'cancelled') {
      return 'round_trip_return_pending';
    }
  }
  return booking.status || '';
}

export function getClientBookingUx(rawStatus) {
  const status = normalizeClientBookingStatus(rawStatus);
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.unknown;
  return {
    status,
    label: cfg.label,
    tone: cfg.tone,
    actions: cfg.actions,
  };
}

/** Chauffeur en route vers le client ou course déjà commencée : plus d’annulation / modification libre côté client. */
export function clientBookingDriverOrTripInMotion(booking) {
  if (!booking) return false;
  const norm = normalizeClientBookingStatus(resolveClientBookingDisplayStatus(booking));
  return norm === 'driver_on_the_way' || norm === 'in_progress';
}

/**
 * Actions affichées (Mes courses, tableau de bord) : retire Modifier et Annuler
 * dès que le chauffeur est en route ou que la course est en cours.
 */
export function getEffectiveClientBookingActions(booking) {
  const ux = getClientBookingUx(resolveClientBookingDisplayStatus(booking));
  if (clientBookingDriverOrTripInMotion(booking)) {
    return ux.actions.filter((a) => a !== 'Modifier' && a !== 'Annuler');
  }
  return [...ux.actions];
}

export function getClientBookingToneClass(label, cssClasses = {}) {
  const toneMap = {
    'Paiement requis': cssClasses.statusPending || cssClasses.statusDefault || '',
    'Demande envoyée': cssClasses.statusPending || cssClasses.statusDefault || '',
    'Course confirmée': cssClasses.statusConfirmed || cssClasses.statusCompleted || cssClasses.statusDefault || '',
    'Chauffeur en route': cssClasses.statusOnRoute || cssClasses.statusInProgress || cssClasses.statusDefault || '',
    'En cours': cssClasses.statusInProgress || cssClasses.statusDefault || '',
    Terminée: cssClasses.statusCompleted || cssClasses.statusDefault || '',
    'Aller terminé — retour à venir':
      cssClasses.statusOnRoute || cssClasses.statusInProgress || cssClasses.statusDefault || '',
    Annulée: cssClasses.statusCancelled || cssClasses.statusCanceled || cssClasses.statusDefault || '',
  };
  return toneMap[label] || cssClasses.statusDefault || '';
}

