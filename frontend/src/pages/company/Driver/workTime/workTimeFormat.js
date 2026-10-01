/** Affichage des durées du temps de travail (jamais un forfait déguisé en heures réelles). */

export function formatClock(iso) {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return new Intl.DateTimeFormat('fr-CH', {
    timeZone: 'Europe/Zurich',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
  }).format(date);
}

/** Valeur `datetime-local` en heure de Genève. */
export function formatDateTimeLocal(iso) {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Europe/Zurich',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
  }).formatToParts(date);
  const pick = (type) => parts.find((part) => part.type === type)?.value || '';
  const day = `${pick('year')}-${pick('month')}-${pick('day')}`;
  const clock = `${pick('hour')}:${pick('minute')}`;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day) || !/^\d{2}:\d{2}$/.test(clock)) return '';
  return `${day}T${clock}`;
}

function courseDay(entry, knownLocal) {
  if (typeof entry?.date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(entry.date)) {
    return entry.date;
  }
  return knownLocal.slice(0, 10);
}

/**
 * Préremplit Arrivée et Fin. Si l'arrivée n'a pas été enregistrée,
 * la date de la course est quand même posée (l'heure reprend celle de la fin connue).
 */
export function adjustmentFormDefaults(entry) {
  const arrived = formatDateTimeLocal(entry?.effective_arrived_at);
  const completed = formatDateTimeLocal(entry?.effective_completed_at);
  const known = completed || arrived;
  const day = courseDay(entry, known);
  const time = known.length >= 16 ? known.slice(11, 16) : '00:00';
  const fallback = day ? `${day}T${time}` : '';
  return {
    arrived: arrived || fallback,
    completed: completed || fallback,
    arrivalRecorded: Boolean(arrived),
  };
}

export function formatMinutes(value) {
  if (value == null || value === '') return '—';
  const mins = Math.round(Number(value));
  if (!Number.isFinite(mins) || mins < 0) return '—';
  const hours = Math.floor(mins / 60);
  const rest = mins % 60;
  return `${hours}h${String(rest).padStart(2, '0')}`;
}

function plainMinutes(value) {
  const mins = Math.round(Number(value));
  if (!Number.isFinite(mins) || mins < 0) return null;
  return mins;
}

/** Distance routière OSRM, en kilomètres (ex. 3,3 km). */
export function formatRouteKm(meters) {
  const km = Number(meters) / 1000;
  if (!Number.isFinite(km) || km < 0) return '—';
  return `${km.toFixed(1).replace('.', ',')} km`;
}

/**
 * Temps de trajet estimé (même calcul que la réservation) et minutes offertes
 * au chauffeur. Ces minutes ne corrigent pas le trajet.
 * Retourne null si aucune route n’a été obtenue.
 */
export function routeProposalCopy(entry) {
  const route = plainMinutes(entry?.route_minutes);
  const margin = plainMinutes(entry?.margin_minutes);
  const proposed = plainMinutes(entry?.proposed_worked_minutes);
  if (entry?.routing_status !== 'ok' || route == null || margin == null || proposed == null) {
    return null;
  }
  const distance = formatRouteKm(entry.route_distance_m);
  return {
    proposedLabel: `${proposed} min proposées`,
    reference: `Temps de trajet estimé : ${route} min`,
    trace: `Voiture · ${distance}`,
    margin: `Offertes au chauffeur : +${margin} min`,
    facts: [
      ['Temps de trajet estimé', `${route} min`],
      ['Distance routière', distance],
      ['Minutes offertes par course', `+${margin} min`],
      ['Temps proposé', `${proposed} min`],
    ],
  };
}

export function qualityBadge(quality) {
  if (quality === 'estimated_historical') return '≈ Estimé';
  if (quality === 'adjusted') return 'Rectifié';
  if (quality === 'incomplete') return 'Incomplet';
  if (quality === 'manual') return 'Manuel';
  return null;
}

export function workTimeBadge(entry) {
  const status = entry?.work_time_status;
  if (status === 'pending_validation') return { label: 'À valider', warn: true };
  if (status === 'validated_estimate') return { label: 'Validé', warn: false };
  if (status === 'adjusted') return { label: 'Rectifié', warn: false };
  if (status === 'verified') return { label: 'Vérifié', warn: false };
  if (entry?.quality === 'estimated_historical') return { label: '≈ Estimé', warn: true };
  if (status === 'incomplete' || entry?.quality === 'incomplete') return { label: 'Incomplet', warn: true };
  if (entry?.is_manual || entry?.quality === 'manual') return { label: 'Manuel', warn: false };
  return null;
}

const ANOMALY_LABELS = {
  completion_time_missing: 'Heure de fin manquante',
  arrival_not_recorded: 'Arrivée non enregistrée',
  completed_before_arrival: 'Fin avant l’arrivée',
  journey_incomplete: 'Trajet incomplet',
  journey_structure_undetermined: 'Structure de trajet indéterminée',
  journey_split_across_drivers: 'Trajet réparti entre plusieurs chauffeurs',
  compensation_rule_missing: 'Règle de rémunération manquante',
  cancelled_after_arrival: 'Annulé après arrivée sur place',
  manual_compensation_rule_missing: 'Règle du temps ajouté manquante',
  transport_not_completed: 'Transport non terminé',
  schedule_adjusted: 'Horaire corrigé',
  overlap: 'Chevauchement',
  duration_exceeds_threshold: 'Durée au-dessus du seuil',
  outside_finalized_snapshot: 'Hors snapshot de paie',
};

export function anomalyLabel(code) {
  return ANOMALY_LABELS[code] || code;
}

function countLabel(count, singular, plural) {
  const n = Number(count) || 0;
  return `${n} ${n > 1 ? plural : singular}`;
}

export function activityLines(source) {
  const journeys = Number(source?.compensated_journeys_count || 0);
  const segments = Number(source?.completed_segments_count || 0);
  const incomplete = Number(source?.incomplete_segments_count || 0);
  const pending = Number(source?.pending_journeys_count || 0);
  const main = [countLabel(journeys, 'trajet', 'trajets')];
  if (segments > 0) main.push(countLabel(segments, 'segment', 'segments'));
  const extra = [];
  if (incomplete > 0) extra.push(countLabel(incomplete, 'incomplet', 'incomplets'));
  if (pending > 0) extra.push(countLabel(pending, 'trajet en attente', 'trajets en attente'));
  return { main: main.join(' · '), extra };
}

export function workedHints(source) {
  const verified = Number(source?.worked_verified_minutes || 0);
  const estimated = Number(source?.worked_estimated_minutes || 0);
  const adjusted = Number(source?.worked_adjusted_minutes || 0);
  const manual = Number(source?.manual_minutes || 0);
  const transport = verified + estimated + adjusted;
  const hints = [];
  if (transport > 0 && manual > 0) {
    hints.push(`${formatMinutes(transport)} courses + ${formatMinutes(manual)} ajouté`);
  } else if (manual > 0) {
    hints.push(`${formatMinutes(manual)} ajouté`);
  }
  if (estimated > 0) hints.push(`dont ≈ ${formatMinutes(estimated)} estimées`);
  if (adjusted > 0) hints.push(`${formatMinutes(adjusted)} corrigées`);
  return hints;
}

export function transportMinutes(source, mode) {
  if (mode === 'flat') return Number(source?.flat_transport_minutes || 0);
  return Number(source?.real_transport_minutes || 0);
}

export function addedMinutes(source, mode) {
  if (mode === 'flat') return Number(source?.flat_added_minutes || 0);
  return Number(source?.real_added_minutes || 0);
}

export function totalMinutes(source, mode) {
  return transportMinutes(source, mode) + addedMinutes(source, mode);
}

export function reviewCount(source, mode) {
  if (mode === 'flat') return Number(source?.review_count_flat || 0);
  return Number(source?.review_count_real || 0);
}

export function policyPhrase(version) {
  if (!version) return 'aucune version';
  const real = version.mode === 'validated_work_time' || version.mode === 'real_time';
  if (real) return 'Temps réel';
  const minutes = Number(version.transport_flat_minutes);
  return `Forfait par transport · ${Number.isFinite(minutes) ? minutes : 0} min`;
}

export function contractualCopy(rules) {
  const versions = rules?.versions || [];
  if (versions.length === 0) return 'Règle contractuelle : aucune version sur cette période';
  if (versions.length > 1) return 'Règle contractuelle : plusieurs versions sur cette période';
  return `Règle contractuelle : ${policyPhrase(versions[0])}`;
}

export function closurePhrase(rules) {
  const versions = rules?.versions || [];
  if (versions.length === 0) return 'Aucune version sur cette période';
  if (versions.length > 1) return 'Plusieurs versions sur cette période';
  return policyPhrase(versions[0]);
}

export function formatCivilDate(iso) {
  if (!iso || !/^\d{4}-\d{2}-\d{2}$/.test(iso)) return iso || '';
  const [year, month, day] = iso.split('-').map(Number);
  return new Intl.DateTimeFormat('fr-CH', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  }).format(new Date(year, month - 1, day));
}

export function periodHeading(preset, range) {
  if (!range?.from) return '';
  if (preset === 'month') {
    const [year, month] = range.from.split('-').map(Number);
    const label = new Intl.DateTimeFormat('fr-CH', { month: 'long', year: 'numeric' })
      .format(new Date(year, month - 1, 1));
    return label.charAt(0).toUpperCase() + label.slice(1);
  }
  if (preset === 'today') return formatCivilDate(range.from);
  if (preset === 'week') return `Semaine du ${formatCivilDate(range.from)}`;
  if (range.from === range.to) return formatCivilDate(range.from);
  return `${formatCivilDate(range.from)} — ${formatCivilDate(range.to)}`;
}

export function fleetSentence(source, mode) {
  const count = Number(source?.transport_count || 0);
  const rawTransport = transportMinutes(source, mode);
  const added = addedMinutes(source, mode);
  const transportUnknown = reviewCount(source, mode) > 0 && count > 0 && rawTransport === 0;
  const trips = transportUnknown
    ? `${count} transport${count > 1 ? 's' : ''} · temps à vérifier`
    : `${count} transport${count > 1 ? 's' : ''} · ${formatMinutes(rawTransport)} transport`;
  if (transportUnknown) {
    return added > 0 ? `${trips} · +${formatMinutes(added)} ajouté` : trips;
  }
  const total = formatMinutes(rawTransport + added);
  if (added > 0) return `${trips} · +${formatMinutes(added)} ajouté = ${total}`;
  return `${trips} = ${total}`;
}

export function formatDayLabel(iso) {
  if (!iso || !/^\d{4}-\d{2}-\d{2}$/.test(iso)) return iso || '';
  const [year, month, day] = iso.split('-').map(Number);
  const short = new Intl.DateTimeFormat('fr-CH', { month: 'short' })
    .format(new Date(year, month - 1, day))
    .replace(/\.$/, '');
  return `${day} ${short}.`;
}

export function driverNarrative(name, source, mode) {
  const who = name || 'Ce chauffeur';
  const count = Number(source?.transport_count || 0);
  const pendingTransport = source?.transport_pending === true;
  const pendingAdded = source?.added_pending === true;
  const transport = formatMinutes(transportMinutes(source, mode));
  const added = addedMinutes(source, mode);
  if (count === 0 && pendingTransport) {
    return `Durant cette période, ${who} n’a effectué aucun transport. Le temps reste à vérifier.`;
  }
  if (count === 0) {
    if (pendingAdded) {
      return `Durant cette période, ${who} n’a effectué aucun transport. Le temps supplémentaire reste à vérifier.`;
    }
    const extra = added > 0
      ? ` Le temps supplémentaire s’élève à ${formatMinutes(added)}.`
      : '';
    return `Durant cette période, ${who} n’a effectué aucun transport.${extra}`;
  }
  const noun = count > 1 ? 'transports' : 'transport';
  if (pendingTransport) {
    const time = count > 1
      ? 'Le temps de ces transports reste à vérifier.'
      : 'Le temps de ce transport reste à vérifier.';
    const extra = !pendingAdded && added > 0
      ? ` À cela s’ajoute ${formatMinutes(added)} de temps supplémentaire.`
      : pendingAdded
        ? ' Le temps supplémentaire reste à vérifier.'
        : '';
    return `Durant cette période, ${who} a effectué ${count} ${noun}. ${time}${extra}`;
  }
  const link = !pendingAdded && added > 0
    ? `, ${count > 1 ? 'auxquels s’ajoutent' : 'auquel s’ajoute'} ${formatMinutes(added)} de temps supplémentaire`
    : '';
  const pendingNote = pendingAdded ? ' Le temps supplémentaire reste à vérifier.' : '';
  return `Durant cette période, ${who} a effectué ${count} ${noun} représentant ${transport} de temps de transport${link}.${pendingNote}`;
}

export function entryNeedsReview(entry, mode) {
  if (!entry) return false;
  const wanted = mode === 'flat' ? 'flat' : 'real';
  if (Array.isArray(entry.review_modes)) return entry.review_modes.includes(wanted);
  if (entry.kind === 'open') return true;
  if (entry.kind === 'transport') {
    if (mode === 'flat') return entry.flat_status === 'requires_review';
    return !Number.isInteger(entry.real_minutes);
  }
  if (mode === 'flat') return entry.flat_status !== 'calculated';
  return false;
}

export function reviewItemsForMode(items, mode) {
  const wanted = mode === 'flat' ? 'flat' : 'real';
  return (items || []).filter((item) => (item.modes || []).includes(wanted));
}

export function reviewActivity(item) {
  if (item?.description) return item.description;
  if (item?.pickup_label && item?.dropoff_label) {
    return `${item.pickup_label} → ${item.dropoff_label}`;
  }
  return item?.pickup_label || item?.dropoff_label || 'Activité';
}

export function reviewReason(item) {
  const labels = (item?.reasons || []).map((code) => anomalyLabel(code)).filter(Boolean);
  return labels.join(' · ') || 'Temps à vérifier';
}

export function dayMetrics(day, mode, finalized) {
  const entries = day?.entries || [];
  const transports = entries.filter((entry) => entry.kind === 'transport');
  const addedEntries = entries.filter((entry) => entry.kind === 'manual' || entry.is_manual);
  const unknownTransport = transports.some((entry) => entryDuration(entry, mode, finalized) == null);
  const hasOpen = entries.some((entry) => entry.kind === 'open');
  const pendingTransport = unknownTransport || (hasOpen && transports.length === 0);
  const pendingAdded = addedEntries.some((entry) => entryDuration(entry, mode, finalized) == null);
  const transportSum = transports.reduce(
    (sum, entry) => sum + (Number(entryDuration(entry, mode, finalized)) || 0),
    0
  );
  const addedSum = addedEntries.reduce(
    (sum, entry) => sum + (Number(entryDuration(entry, mode, finalized)) || 0),
    0
  );
  const review = entries.some((entry) => entryNeedsReview(entry, mode));
  return {
    transportCount: transports.length,
    transportMinutes: transportSum,
    addedMinutes: addedSum,
    transportPending: pendingTransport,
    addedPending: pendingAdded,
    transportLabel: pendingTransport ? 'À vérifier' : (transports.length === 0 ? '—' : formatMinutes(transportSum)),
    addedLabel: pendingAdded ? 'À vérifier' : (addedSum === 0 ? '—' : formatMinutes(addedSum)),
    totalLabel: pendingTransport || pendingAdded ? '—' : formatMinutes(transportSum + addedSum),
    statusLabel: review ? 'À vérifier' : 'Validé',
    review,
  };
}

/** Totaux lus sur les journées affichées. Une cellule « À vérifier » n’est pas une durée. */
export function aggregateDisplayedDays(days, mode, finalized) {
  const rows = (days || []).map((day) => dayMetrics(day, mode, finalized));
  const transportPending = rows.some((row) => row.transportPending);
  const addedPending = rows.some((row) => row.addedPending);
  const transportMinutesSum = rows.reduce(
    (sum, row) => sum + (row.transportPending ? 0 : row.transportMinutes),
    0,
  );
  const addedMinutesSum = rows.reduce(
    (sum, row) => sum + (row.addedPending ? 0 : row.addedMinutes),
    0,
  );
  return {
    transport_count: rows.reduce((sum, row) => sum + row.transportCount, 0),
    transport_minutes: transportPending ? null : transportMinutesSum,
    added_minutes: addedPending ? null : addedMinutesSum,
    transport_pending: transportPending,
    added_pending: addedPending,
    review_count: (days || []).reduce(
      (sum, day) => sum + (day.entries || []).filter((entry) => entryNeedsReview(entry, mode)).length,
      0,
    ),
  };
}

export function narrativeSource(aggregate, mode) {
  const transportKey = mode === 'flat' ? 'flat_transport_minutes' : 'real_transport_minutes';
  const addedKey = mode === 'flat' ? 'flat_added_minutes' : 'real_added_minutes';
  return {
    transport_count: aggregate.transport_count,
    [transportKey]: aggregate.transport_minutes ?? 0,
    [addedKey]: aggregate.added_minutes ?? 0,
    transport_pending: aggregate.transport_pending,
    added_pending: aggregate.added_pending,
  };
}

export function entryDuration(entry, mode, finalized) {
  if (!entry) return null;
  if (entryNeedsReview(entry, mode)) return null;
  if (entry.kind === 'open') return null;
  if (mode === 'flat') {
    if (entry.flat_status === 'calculated' && Number.isInteger(entry.flat_minutes)) {
      return entry.flat_minutes;
    }
    if (finalized && entry.compensated_minutes != null) return entry.compensated_minutes;
    return null;
  }
  if (entry.work_time_status === 'pending_validation') return null;
  if (Number.isInteger(entry.real_minutes)) return entry.real_minutes;
  if (Number.isInteger(entry.worked_minutes)) return entry.worked_minutes;
  if (finalized && entry.compensated_minutes != null) return entry.compensated_minutes;
  return null;
}

export function rowBadge(entry, mode) {
  if ((entry?.anomalies || []).includes('cancelled_after_arrival')) {
    return { label: 'Annulé sur place', warn: false };
  }
  if (entry?.kind === 'open') return { label: 'À vérifier', warn: true };
  if (mode === 'flat' && entry?.flat_status === 'requires_review') {
    return { label: 'À vérifier', warn: true };
  }
  return workTimeBadge(entry);
}

export function segmentsLabel(driver) {
  const done = Number(driver?.completed_segments_count || 0);
  const incomplete = Number(driver?.incomplete_segments_count || 0);
  const journeys = Number(driver?.compensated_journeys_count || 0);
  const pending = Number(driver?.pending_journeys_count || 0);
  return {
    segments: `${done} segment${done > 1 ? 's' : ''} réalisé${done > 1 ? 's' : ''}`,
    incomplete: incomplete > 0 ? `${incomplete} incomplet${incomplete > 1 ? 's' : ''}` : '',
    journeys: `${journeys} trajet${journeys > 1 ? 's' : ''} forfaitaire${journeys > 1 ? 's' : ''}`,
    pending: pending > 0 ? `${pending} en attente` : '',
  };
}
