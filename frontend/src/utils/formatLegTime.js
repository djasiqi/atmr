/**
 * Affichage horaire institution — legs et retours sans sentinelle 00:00.
 * S'appuie sur time_confirmed (API) ; heures murales Genève (Europe/Zurich).
 */

import {
  extractWallClockTime,
  formatWallClockDateShort,
} from './missionTimeDisplay';

const fmtTime = (value) => extractWallClockTime(value);

const fmtDateShort = (value) => formatWallClockDateShort(value);

/** Dernier leg = retour institution uniquement s'il y a plusieurs étapes. */
export const isReturnLegIndex = (request, legs, index) =>
  Boolean(request?.return_to_institution) && legs.length > 1 && index === legs.length - 1;

/** Préfixe systématique Départ / RDV devant l'heure ou « À définir ». */
export function prefixScheduleTimeLabel(prefix, rawTime) {
  const p = String(prefix || '').trim();
  if (!rawTime || rawTime === 'À définir') {
    return `${p} · À définir`;
  }
  if (String(rawTime).includes('(non confirmé)')) {
    const base = String(rawTime).replace(' (non confirmé)', '').trim();
    return `${p} ${base} (non confirmé)`;
  }
  return `${p} ${rawTime}`;
}

/**
 * Libellé horaire pour une étape du parcours (panneau détail institution).
 * @param {{ kind: 'start'|'departure'|'destination'|'return', request?: object, leg?: object }} params
 */
export function formatRouteStopTime({ kind, request, leg }) {
  if (kind === 'start' || kind === 'departure') {
    return prefixScheduleTimeLabel('Départ', formatDepartureTime(request));
  }

  if (kind === 'return') {
    const raw = leg?.scheduled_time != null
      ? formatLegTime(leg)
      : (formatReturnTimeHint(request) || 'À définir');
    return prefixScheduleTimeLabel('Départ', raw);
  }

  const raw = leg ? formatLegTime(leg) : 'À définir';
  return prefixScheduleTimeLabel('RDV', raw);
}

/** Leg multi-stop : heure confirmée, indicative ou à définir. */
export function formatLegTime(leg) {
  if (!leg) return 'À définir';
  if (leg.scheduled_time == null) return 'À définir';
  const timeStr = fmtTime(leg.scheduled_time);
  if (!timeStr) return 'À définir';
  if (leg.time_confirmed === true) return timeStr;
  return `${timeStr} (non confirmé)`;
}

/** Heure de départ confirmée sur le booking lié (après acceptation transporteur). */
export function getBookingDepartureTime(request) {
  const bs = request?.booking_summary;
  if (!bs?.scheduled_time) return null;
  const time = fmtTime(bs.scheduled_time);
  return time || null;
}

/** Départ mission (pickup_time_confirmed ou booking converti). */
export function formatDepartureTime(request) {
  if (!request) return 'À définir';

  if (request.pickup_time_confirmed === true && request.scheduled_time) {
    const timeStr = fmtTime(request.scheduled_time);
    if (timeStr) return timeStr;
  }

  if (request.pickup_time_confirmed === false && request.scheduled_time) {
    const timeStr = fmtTime(request.scheduled_time);
    if (timeStr) return `${timeStr} (non confirmé)`;
  }

  // Legacy : départ sans flag explicite
  if (request.scheduled_time_type === 'departure' && request.scheduled_time) {
    const timeStr = fmtTime(request.scheduled_time);
    if (timeStr) return timeStr;
  }

  // Après acceptation : heure de prise en charge sur le booking
  const bookingDep = getBookingDepartureTime(request);
  if (bookingDep) return bookingDep;

  return 'À définir';
}

/** Retour A/R classique sur une TransportRequest. */
export function formatReturnTimeHint(req) {
  if (!req) return '';
  if (req.return_time_confirmed === false) {
    if (req.return_time) {
      const t = fmtTime(req.return_time);
      return t ? `${t} (non confirmé)` : (req.return_date ? 'À définir' : '');
    }
    return req.return_date ? 'À définir' : '';
  }
  if (req.return_time) return fmtTime(req.return_time);
  if (req.return_date) return 'À définir';
  return '';
}

export function formatReturnTimeLabel(req) {
  const hint = formatReturnTimeHint(req);
  if (!hint) return '';
  if (hint === 'À définir') return 'retour · À définir';
  if (hint.includes('(non confirmé)')) return `retour ${hint}`;
  return `retour ${hint}`;
}

const isOperational = (scheduledTime, timeConfirmed) =>
  timeConfirmed === true && scheduledTime != null;

const missionDayStr = (request) => {
  if (request?.mission_date) return String(request.mission_date).slice(0, 10);
  if (request?.scheduled_time) return String(request.scheduled_time).slice(0, 10);
  return null;
};

const sameMissionDay = (iso, missionDay) => {
  if (!iso || !missionDay) return true;
  const day = String(iso).slice(0, 10);
  return day === missionDay;
};

/** Prochaine heure confirmée (aligné get_effective_dispatch_time backend). */
export function getNextConfirmedLegTime(request) {
  const info = getNextConfirmedScheduleInfo(request);
  return info?.iso || null;
}

/**
 * Prochaine heure confirmée avec nature (départ / RDV / retour).
 * @returns {{ iso: string, time: string, kind: 'departure'|'destination'|'return', label: string } | null}
 */
export function getNextConfirmedScheduleInfo(request) {
  if (!request) return null;
  const missionDay = missionDayStr(request);
  const candidates = [];

  const addCandidate = (iso, kind, label) => {
    if (!iso) return;
    if (!sameMissionDay(iso, missionDay)) return;
    const time = fmtTime(iso);
    if (!time) return;
    candidates.push({
      iso: String(iso),
      ms: new Date(iso).getTime(),
      time,
      kind,
      label,
    });
  };

  if (isOperational(request.scheduled_time, request.pickup_time_confirmed)) {
    addCandidate(request.scheduled_time, 'departure', 'Départ');
  }

  const legs = Array.isArray(request.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];

  legs.forEach((leg, index) => {
    if (!isOperational(leg.scheduled_time, leg.time_confirmed)) return;
    const isReturn = isReturnLegIndex(request, legs, index);
    const label = isReturn
      ? 'Retour'
      : (leg.dropoff_establishment || leg.dropoff_service || `RDV ${index + 1}`);
    addCandidate(
      leg.scheduled_time,
      isReturn ? 'return' : 'destination',
      isReturn ? 'Retour' : 'RDV',
    );
  });

  if (
    !legs.length
    && request.scheduled_time_type === 'arrival'
    && isOperational(request.scheduled_time, request.appointment_time_confirmed ?? true)
  ) {
    addCandidate(request.scheduled_time, 'destination', 'RDV');
  }

  if (!candidates.length) return null;
  const next = candidates.reduce((min, c) => (c.ms < min.ms ? c : min));
  return {
    iso: next.iso,
    time: next.time,
    kind: next.kind,
    label: next.label,
  };
}

/** Parties horaires confirmées pour affichage liste (départ + RDV + retour). */
export function getConfirmedScheduleParts(request) {
  if (!request) return [];
  const parts = [];

  const dep = formatDepartureTime(request);
  const hasRequestDeparture = request.pickup_time_confirmed === true && request.scheduled_time;
  const hasBookingDeparture = Boolean(getBookingDepartureTime(request));
  if (dep !== 'À définir' && (hasRequestDeparture || hasBookingDeparture)) {
    parts.push({ label: 'Départ', time: dep.replace(' (non confirmé)', '') });
  }

  const legs = Array.isArray(request.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];

  legs.forEach((leg, index) => {
    if (!isOperational(leg.scheduled_time, leg.time_confirmed)) return;
    const isReturn = isReturnLegIndex(request, legs, index);
    const time = formatLegTime(leg);
    if (time === 'À définir') return;
    parts.push({
      label: isReturn ? 'Retour' : 'RDV',
      time: time.replace(' (non confirmé)', ''),
    });
  });

  if (
    !legs.length
    && request.scheduled_time_type === 'arrival'
    && isOperational(request.scheduled_time, request.appointment_time_confirmed ?? true)
  ) {
    const t = formatLegTime({
      scheduled_time: request.scheduled_time,
      time_confirmed: true,
    });
    if (t !== 'À définir') {
      parts.push({ label: 'RDV', time: t });
    }
  }

  return parts;
}

/** Libellé court pour une ligne horaire (ex. « Départ 19:00 »). */
export const formatSchedulePartLabel = ({ label, time }) => `${label} ${time}`;
export function formatLegScheduleSummary(request) {
  if (!request) return '';
  const parts = [];

  const dep = formatDepartureTime(request);
  if (dep !== 'À définir') parts.push(`${dep} Départ`);

  const legs = Array.isArray(request.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];

  if (legs.length > 0) {
    legs.forEach((leg, index) => {
      const isReturn = isReturnLegIndex(request, legs, index);
      const label = isReturn ? 'Retour' : 'RDV';
      const time = formatLegTime(leg);
      if (isReturn && time === 'À définir') {
        parts.push('Retour à définir');
      } else {
        parts.push(`${time} ${label}`);
      }
    });
  } else if (request.scheduled_time_type === 'arrival' && request.scheduled_time) {
    const t = formatLegTime({
      scheduled_time: request.scheduled_time,
      time_confirmed: true,
    });
    parts.push(`${t} RDV`);
  }

  return parts.join(' · ');
}

/** Date mission + horaires confirmés pour listes (ex. « 15 juin · Départ 19:00 · RDV 20:00 »). */
export function formatMissionScheduleListLabel(request) {
  if (!request) return '—';
  const dateSrc = request.mission_date || request.scheduled_time;
  const datePart = dateSrc ? fmtDateShort(dateSrc) : '—';
  const timeParts = getConfirmedScheduleParts(request);
  if (timeParts.length) {
    return `${datePart} · ${timeParts.map(formatSchedulePartLabel).join(' · ')}`;
  }
  return datePart;
}

/**
 * Affichage carte liste — départ mis en avant, RDV/retour en secondaire.
 * @returns {{ dateLabel: string, primary: { label: string, time: string } | null, secondary: Array<{ label: string, time: string }> }}
 */
export function getMissionScheduleCardDisplay(request) {
  const parts = getConfirmedScheduleParts(request);
  const dateSrc = request?.mission_date || request?.scheduled_time;
  const dateLabel = dateSrc ? fmtDateShort(dateSrc) : '—';

  const departure = parts.find((p) => p.label === 'Départ') || null;
  const nonDeparture = parts.filter((p) => p.label !== 'Départ');

  if (departure) {
    return { dateLabel, primary: departure, secondary: nonDeparture };
  }

  if (nonDeparture.length) {
    return {
      dateLabel,
      primary: nonDeparture[0],
      secondary: nonDeparture.slice(1),
    };
  }

  return { dateLabel, primary: null, secondary: [] };
}

/** Structure pour panneaux détaillés transporteur / institution. */
export function formatMissionScheduleDetail(request) {
  const legs = Array.isArray(request?.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];

  const rows = [
    { label: 'Départ', time: formatDepartureTime(request), kind: 'departure' },
  ];

  if (legs.length > 0) {
    legs.forEach((leg, index) => {
      const isReturn = isReturnLegIndex(request, legs, index);
      rows.push({
        label: isReturn ? 'Retour' : 'RDV',
        time: formatLegTime(leg),
        kind: isReturn ? 'return' : 'destination',
        address: leg.dropoff_location,
        establishment: leg.dropoff_establishment || null,
      });
    });
  } else if (request?.dropoff_location) {
    const rdvTime = request.scheduled_time_type === 'arrival' && request.scheduled_time
      ? formatLegTime({ scheduled_time: request.scheduled_time, time_confirmed: true })
      : 'À définir';
    rows.push({ label: 'RDV', time: rdvTime, kind: 'destination', address: request.dropoff_location });
  }

  return {
    missionDate: request?.mission_date || (request?.scheduled_time ? String(request.scheduled_time).slice(0, 10) : null),
    nextConfirmed: getNextConfirmedLegTime(request),
    summary: formatLegScheduleSummary(request),
    rows,
  };
}
