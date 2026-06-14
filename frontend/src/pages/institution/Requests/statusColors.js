/**
 * Taxonomie statut partagée — liste + panneau détail institution.
 * STOP GATE UX Couleurs : couleur = statut (ou alerte transverse retard).
 */

import {
  EXTERNAL_STATUSES,
  getRequestStatusLabel,
  isConvertedLirie,
  isExternalRequest,
} from '../../../utils/requestStatus';

export const STATUS_TONES = Object.freeze({
  neutral: 'neutral',
  info: 'info',
  success: 'success',
  warning: 'warning',
  error: 'error',
});

export const BOOKING_STATUS_LABELS = {
  PENDING: 'En attente',
  ACCEPTED: 'Accepté',
  ASSIGNED: 'Chauffeur assigné',
  EN_ROUTE: 'En route',
  IN_PROGRESS: 'En cours',
  OUTBOUND_COMPLETED: 'Retour en cours',
  COMPLETED: 'Terminé',
  RETURN_COMPLETED: 'Aller-retour OK',
  CANCELED: 'Annulé',
};

export const BOOKING_STATUS_TONES = {
  PENDING: STATUS_TONES.warning,
  ACCEPTED: STATUS_TONES.info,
  ASSIGNED: STATUS_TONES.info,
  EN_ROUTE: STATUS_TONES.info,
  IN_PROGRESS: STATUS_TONES.info,
  OUTBOUND_COMPLETED: STATUS_TONES.info,
  COMPLETED: STATUS_TONES.success,
  RETURN_COMPLETED: STATUS_TONES.success,
  CANCELED: STATUS_TONES.error,
};

export const REQUEST_STATUS_LABELS = {
  DRAFT: 'Brouillon',
  SENT: 'Envoyée',
  ACCEPTED: 'Acceptée',
  CONVERTED: 'Confirmée',
  CANCELLED: 'Annulée',
  EXPIRED: 'Expirée',
  [EXTERNAL_STATUSES.ASSIGNED]: 'Transporteur externe affecté',
  [EXTERNAL_STATUSES.COMPLETED]: 'Déclarée réalisée par l\'institution',
};

export const REQUEST_STATUS_TONES = {
  DRAFT: STATUS_TONES.neutral,
  SENT: STATUS_TONES.info,
  ACCEPTED: STATUS_TONES.info,
  CONVERTED: STATUS_TONES.success,
  CANCELLED: STATUS_TONES.error,
  EXPIRED: STATUS_TONES.neutral,
  [EXTERNAL_STATUSES.ASSIGNED]: STATUS_TONES.warning,
  [EXTERNAL_STATUSES.COMPLETED]: STATUS_TONES.success,
};

/**
 * @param {string} tone
 * @returns {string}
 */
export function statusToneToBadgeClass(tone) {
  const key = Object.values(STATUS_TONES).includes(tone) ? tone : STATUS_TONES.neutral;
  return `badgeStatus${key.charAt(0).toUpperCase()}${key.slice(1)}`;
}

/**
 * @param {string} tone
 * @returns {string}
 */
export function statusToneToIndicatorClass(tone) {
  const key = Object.values(STATUS_TONES).includes(tone) ? tone : STATUS_TONES.neutral;
  return `indicatorStatus${key.charAt(0).toUpperCase()}${key.slice(1)}`;
}

/**
 * @param {object|null|undefined} req
 * @param {(summary: object) => string} resolveBookingStatusKey
 * @returns {{ label: string, fullLabel: string, statusTone: string, badgeClass: string, indicatorClass: string }}
 */
export function resolveStatusDisplay(req, resolveBookingStatusKey) {
  if (!req) {
    return {
      label: '—',
      fullLabel: '—',
      statusTone: STATUS_TONES.neutral,
      badgeClass: statusToneToBadgeClass(STATUS_TONES.neutral),
      indicatorClass: statusToneToIndicatorClass(STATUS_TONES.neutral),
    };
  }

  const fullLabel = getRequestStatusLabel(req);

  if (isExternalRequest(req)) {
    const statusTone = REQUEST_STATUS_TONES[req.status] || STATUS_TONES.warning;
    const label = REQUEST_STATUS_LABELS[req.status] || fullLabel;
    return {
      label,
      fullLabel,
      statusTone,
      badgeClass: statusToneToBadgeClass(statusTone),
      indicatorClass: statusToneToIndicatorClass(statusTone),
    };
  }

  if (isConvertedLirie(req) && req.booking_summary?.status) {
    const bookingKey = resolveBookingStatusKey(req.booking_summary);
    const statusTone = BOOKING_STATUS_TONES[bookingKey] || REQUEST_STATUS_TONES.CONVERTED;
    const label = BOOKING_STATUS_LABELS[bookingKey] || REQUEST_STATUS_LABELS.CONVERTED;
    return {
      label,
      fullLabel,
      statusTone,
      badgeClass: statusToneToBadgeClass(statusTone),
      indicatorClass: statusToneToIndicatorClass(statusTone),
    };
  }

  const statusTone = REQUEST_STATUS_TONES[req.status] || STATUS_TONES.neutral;
  const label = REQUEST_STATUS_LABELS[req.status] || fullLabel || req.status;
  return {
    label,
    fullLabel,
    statusTone,
    badgeClass: statusToneToBadgeClass(statusTone),
    indicatorClass: statusToneToIndicatorClass(statusTone),
  };
}

/**
 * Compose les lignes meta neutres (niveaux 2 et 3) pour une carte liste.
 *
 * @param {object} params
 * @returns {{ carrierLine: string|null, detailsLine: string|null }}
 */
export function buildCardMeta({
  req,
  companyName,
  carrierModeLabel,
  isExternal,
  tripTypeLabel,
  billingLabel,
  timeTypeLabel,
}) {
  let carrierLine = null;

  if (companyName) {
    carrierLine = isExternal ? `Transporteur : ${companyName}` : companyName;
  } else if (req?.status === 'SENT' && !isExternal) {
    carrierLine = 'En attente d\'offre';
  } else if (carrierModeLabel) {
    carrierLine = carrierModeLabel;
  }

  const details = [tripTypeLabel, billingLabel, timeTypeLabel].filter(Boolean);
  const detailsLine = details.length > 0 ? details.slice(0, 2).join(' · ') : null;

  return { carrierLine, detailsLine };
}

/**
 * Libellé facturation pour metaDetails (lecture seule).
 *
 * @param {object} req
 * @returns {string|null}
 */
export function resolveBillingMetaLabel(req) {
  if (!req) return null;
  const isConverted = isConvertedLirie(req) && req.booking_summary;
  const isPatient = isConverted
    ? req.booking_summary.billed_to_type !== 'clinic'
    : (req.billing_intent || 'patient') === 'patient';
  return isPatient ? 'Facturé patient' : 'Facturé institution';
}
