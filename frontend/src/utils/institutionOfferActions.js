import { canRespondToInstitutionOffer } from './institutionOfferResponse';

const LABELS = {
  validate: 'Valider',
  plan: 'Planifier',
  acceptNow: 'Départ immédiat',
  reject: 'Refuser',
};

function parseWallClock(value) {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function hasConfirmedDeparture(req) {
  if (!req) return false;
  const stType = req.scheduled_time_type ?? 'departure';
  return Boolean(
    req.pickup_time_confirmed && req.scheduled_time && stType !== 'arrival'
  );
}

export function isDepartureStale(req, now = new Date()) {
  if (!hasConfirmedDeparture(req) || !req?.scheduled_time) return false;
  const dep = parseWallClock(req.scheduled_time);
  if (!dep) return false;
  return dep.getTime() < now.getTime();
}

export function hasConfirmedRdvOnly(req) {
  if (!req || hasConfirmedDeparture(req)) return false;

  const legs = [...(req.legs ?? [])].sort(
    (a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0)
  );
  for (const leg of legs) {
    if (leg.time_confirmed !== false && leg.scheduled_time) return true;
  }

  if (req.scheduled_time_type === 'arrival' && req.scheduled_time) {
    const apptConfirmed = req.appointment_time_confirmed ?? true;
    return Boolean(apptConfirmed);
  }
  return false;
}

/** Matrice Valider / Planifier / Départ immédiat / Refuser (offre PENDING non expirée). */
export function resolveInstitutionOfferActions(offer, now = new Date()) {
  const empty = {
    canRespond: false,
    canValidate: false,
    canPlan: false,
    canAcceptNow: false,
    canReject: false,
    ...LABELS,
    validateLabel: LABELS.validate,
    planLabel: LABELS.plan,
    acceptNowLabel: LABELS.acceptNow,
    rejectLabel: LABELS.reject,
  };

  if (!offer || !canRespondToInstitutionOffer(offer, now)) {
    return empty;
  }

  const req = offer.transport_request;
  const isUrgent = Boolean(req?.is_urgent);
  const departureConfirmed = hasConfirmedDeparture(req);
  const departureStale = isDepartureStale(req, now);
  const rdvOnly = hasConfirmedRdvOnly(req);

  let canValidate = false;
  let canPlan = true;
  let canAcceptNow = false;
  let hint;

  if (departureConfirmed && !departureStale) {
    canValidate = true;
  } else if (departureConfirmed && departureStale) {
    if (isUrgent) canAcceptNow = true;
    else {
      hint = "L'horaire de départ est dépassé — planifiez une nouvelle prise en charge.";
    }
  } else if (isUrgent) {
    canAcceptNow = true;
  } else if (rdvOnly) {
    hint = "Seul le rendez-vous est connu — planifiez l'heure de prise en charge pour garantir l'arrivée.";
  } else if (!departureConfirmed) {
    hint = "Planifiez l'heure de prise en charge pour accepter cette demande.";
  }

  return {
    canRespond: true,
    canValidate,
    canPlan,
    canAcceptNow,
    canReject: true,
    validateLabel: LABELS.validate,
    planLabel: LABELS.plan,
    acceptNowLabel: LABELS.acceptNow,
    rejectLabel: LABELS.reject,
    hint,
  };
}

export const INSTITUTION_ACCEPT_NOW_OFFSET_MINUTES = 15;

/** Départ immédiat : now + délai opérationnel. */
export function computeAcceptNowPickupIso(
  offsetMinutes = INSTITUTION_ACCEPT_NOW_OFFSET_MINUTES
) {
  const next = new Date();
  next.setMinutes(next.getMinutes() + offsetMinutes);
  const pad = (n) => String(n).padStart(2, '0');
  return `${next.getFullYear()}-${pad(next.getMonth() + 1)}-${pad(next.getDate())}T${pad(next.getHours())}:${pad(next.getMinutes())}:00`;
}
