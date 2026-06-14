/**
 * Affichage horaire institution — legs et retours sans sentinelle 00:00.
 * S'appuie sur time_confirmed (API) ; ne jamais inférer « À définir » via hour === 0.
 */

const fmtTime = (value) => {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
};

const fmtDateShort = (value) => {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 10);
  return d.toLocaleDateString('fr-CH', { day: '2-digit', month: 'short' });
};

/** Leg multi-stop : heure confirmée, indicative ou à définir. */
export function formatLegTime(leg) {
  if (!leg) return 'À définir';
  if (leg.scheduled_time == null) return 'À définir';
  const timeStr = fmtTime(leg.scheduled_time);
  if (!timeStr) return 'À définir';
  if (leg.time_confirmed === true) return timeStr;
  return `${timeStr} (non confirmé)`;
}

/** Départ mission (pickup_time_confirmed). */
export function formatDepartureTime(request) {
  if (!request) return 'À définir';
  const dep = request.scheduled_time;
  if (!dep) return 'À définir';
  const timeStr = fmtTime(dep);
  if (!timeStr) return 'À définir';
  if (request.pickup_time_confirmed === true) return timeStr;
  if (request.pickup_time_confirmed === false) return `${timeStr} (non confirmé)`;
  // Legacy : départ sans flag explicite
  if (request.scheduled_time_type === 'departure') return timeStr;
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

/** Prochaine heure confirmée (aligné get_effective_dispatch_time backend). */
export function getNextConfirmedLegTime(request) {
  if (!request) return null;
  const missionDay = request.mission_date || (request.scheduled_time
    ? String(request.scheduled_time).slice(0, 10)
    : null);
  const candidates = [];

  const addCandidate = (iso) => {
    if (!iso) return;
    const day = String(iso).slice(0, 10);
    if (missionDay && day !== missionDay) return;
    candidates.push(new Date(iso).getTime());
  };

  if (isOperational(request.scheduled_time, request.pickup_time_confirmed)) {
    addCandidate(request.scheduled_time);
  }

  const legs = Array.isArray(request.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];

  for (const leg of legs) {
    if (isOperational(leg.scheduled_time, leg.time_confirmed)) {
      addCandidate(leg.scheduled_time);
    }
  }

  if (!candidates.length) return null;
  return new Date(Math.min(...candidates)).toISOString();
}

/** Résumé compact : « 13:15 Départ · 14:00 Imagerie · Retour à définir » */
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
      const isReturn = Boolean(request.return_to_institution) && index === legs.length - 1;
      const label = isReturn
        ? 'Retour'
        : (leg.dropoff_establishment || leg.dropoff_service || `Dest. ${index + 1}`);
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

/** Date mission + prochain RDV confirmé pour listes. */
export function formatMissionScheduleListLabel(request) {
  if (!request) return '—';
  const dateSrc = request.mission_date || request.scheduled_time;
  const datePart = dateSrc ? fmtDateShort(dateSrc) : '—';
  const next = getNextConfirmedLegTime(request);
  if (next) {
    return `${datePart} · ${fmtTime(next)}`;
  }
  return datePart;
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
      const isReturn = Boolean(request?.return_to_institution) && index === legs.length - 1;
      rows.push({
        label: isReturn ? 'Retour' : (leg.dropoff_establishment || `RDV ${index + 1}`),
        time: formatLegTime(leg),
        kind: isReturn ? 'return' : 'destination',
        address: leg.dropoff_location,
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
