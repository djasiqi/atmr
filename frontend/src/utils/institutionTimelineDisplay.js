/**
 * Projection métier de la timeline institution — chronologie opérationnelle lisible.
 * L'audit technique reste en base ; seule la couche affichée est filtrée ici.
 */

/** Groupes de champs alignés sur transport_timeline_service._FIELD_CHANGE_GROUPS */
const FIELD_CHANGE_GROUPS = [
  ['itinéraire', new Set([
    'pickup_location', 'dropoff_location', 'dropoff_establishment', 'dropoff_service',
    'dropoff_doctor', 'intermediate_stops', 'multi_stop', 'return_to_institution',
    'is_round_trip', 'pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng',
    'leg_appointments', 'appointment_time', 'return_appointment_time',
  ])],
  ['horaires', new Set([
    'mission_date', 'scheduled_time', 'scheduled_time_type', 'pickup_time_confirmed',
    'appointment_time_confirmed', 'return_time', 'return_date', 'return_time_confirmed',
    'return_scheduled_time', 'appointment_time',
  ])],
  ['mobilité', new Set([
    'mobility', 'requires_wheelchair', 'requires_assistance', 'wheelchair_need',
    'wheelchair_client_has',
  ])],
  ['notes', new Set(['notes', 'notes_medical', 'pickup_access_notes', 'dropoff_access_notes'])],
  ['patient', new Set(['patient_id', 'customer_name', 'external_reference'])],
  ['facturation', new Set(['billing_intent', 'billing_details'])],
];

const ITINERAIRE_GROUP = 'itinéraire';
const HORAIRES_GROUP = 'horaires';
const IMPORTANT_CHANGE_GROUPS = new Set([ITINERAIRE_GROUP, HORAIRES_GROUP]);

const ALWAYS_HIDDEN_TYPES = new Set([
  'booking_created',
  'status_changed',
  'driver_reassigned',
  'driver_reassignment_attempted',
  'change_confirmation_requested',
  'change_accepted_by_company',
  'change_refused_by_company',
  'change_refused_by_driver',
  'change_expired',
  'escalation_required',
  'redispatched',
  'billing_changed',
  'offer_expired',
  'offer_rejected',
]);

const SEND_EVENT_TYPES = new Set(['request_sent', 'offer_sent']);

/**
 * Même minute calendaire (fuseau local navigateur).
 * Consolidation volontaire à la minute — pas à la seconde — pour masquer
 * création + envoi quasi simultanés sans valeur métier supplémentaire.
 */
export const isSameTimelineMinute = (a, b) => {
  if (!a || !b) return false;
  const da = new Date(a);
  const db = new Date(b);
  if (Number.isNaN(da.getTime()) || Number.isNaN(db.getTime())) return false;
  return (
    da.getFullYear() === db.getFullYear()
    && da.getMonth() === db.getMonth()
    && da.getDate() === db.getDate()
    && da.getHours() === db.getHours()
    && da.getMinutes() === db.getMinutes()
  );
};

const normalizeChangedFields = (raw) => {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw.map(String);
  if (typeof raw === 'object') return Object.keys(raw).filter((k) => raw[k]);
  return [];
};

const classifyChangedFields = (fields) => {
  const fieldSet = new Set(fields.map(String));
  const groups = [];
  let matched = new Set();
  FIELD_CHANGE_GROUPS.forEach(([label, keys]) => {
    const hit = [...fieldSet].filter((f) => keys.has(f));
    if (hit.length) {
      groups.push(label);
      matched = new Set([...matched, ...hit]);
    }
  });
  const remaining = [...fieldSet].filter((f) => !matched.has(f));
  if (remaining.length) groups.push('autres détails');
  return groups;
};

const hasItineraryFields = (fields) => classifyChangedFields(fields).includes(ITINERAIRE_GROUP);

const hasOnlyUnimportantFields = (fields) => {
  const groups = classifyChangedFields(fields);
  if (!groups.length) return true;
  return groups.every((g) => !IMPORTANT_CHANGE_GROUPS.has(g) && g !== 'autres détails');
};

const resolveCarrierName = (event, context) => (
  event?.payload?.company_name
  || context.offerAccepted?.payload?.company_name
  || context.request?.accepted_by_company?.name
  || null
);

const resolveExternalCarrierName = (event, context) => (
  event?.payload?.carrier_name
  || context.externalCarrierAssigned?.payload?.carrier_name
  || context.request?.external_carrier?.name
  || null
);

const buildTimelineContext = (apiEvents, request) => {
  const events = Array.isArray(apiEvents) ? apiEvents : [];
  return {
    request,
    apiEvents: events,
    hasRequestConverted: events.some((e) => e.event_type === 'request_converted'),
    offerAccepted: events.find((e) => e.event_type === 'offer_accepted') || null,
    externalCarrierAssigned: events.find((e) => e.event_type === 'external_carrier_assigned') || null,
    hasRouteJourney: Boolean(
      request?.booking_summary?.route_journey?.length,
    ),
  };
};

/** Agrégat unique « Demande envoyée » depuis request_sent + offer_sent. */
const buildSendAggregate = (apiEvents) => {
  const sendEvents = (apiEvents || []).filter((e) => SEND_EVENT_TYPES.has(e.event_type));
  if (!sendEvents.length) return null;
  const timestamps = sendEvents.map((e) => e.created_at).filter(Boolean);
  if (!timestamps.length) return null;
  const timestamp = timestamps.reduce((min, t) => (new Date(t) < new Date(min) ? t : min));
  const requestSent = sendEvents.find((e) => e.event_type === 'request_sent');
  const actorName = requestSent?.payload?.actor_name
    || sendEvents.map((e) => e.payload?.actor_name).find(Boolean)
    || null;
  const label = actorName ? `Demande envoyée — ${actorName}` : 'Demande envoyée';
  return { label, timestamp, source: 'api' };
};

const shouldHideRequestCreated = (event, context, sendAggregate) => {
  if (event.event_type !== 'request_created') return false;
  if (!sendAggregate) return false;
  return isSameTimelineMinute(event.created_at, sendAggregate.timestamp);
};

const shouldHideFieldUpdated = (event, context) => {
  if (event.event_type !== 'field_updated') return false;
  const fields = normalizeChangedFields(event.payload?.changed_fields);
  if (hasOnlyUnimportantFields(fields)) return true;
  if (hasItineraryFields(fields)) {
    return context.apiEvents.some(
      (other) => other.event_type === 'route_legs_reorganized'
        && isSameTimelineMinute(other.created_at, event.created_at),
    );
  }
  return false;
};

const shouldHideApiEvent = (event, context, sendAggregate) => {
  const type = event.event_type;

  if (ALWAYS_HIDDEN_TYPES.has(type)) return true;
  if (SEND_EVENT_TYPES.has(type)) return Boolean(sendAggregate);
  if (type === 'offer_accepted' && context.hasRequestConverted) return true;
  if (shouldHideRequestCreated(event, context, sendAggregate)) return true;
  if (shouldHideFieldUpdated(event, context)) return true;

  if (context.hasRouteJourney && ['patient_boarded', 'patient_completed', 'status_changed'].includes(type)) {
    return true;
  }

  return false;
};

const withActorName = (label, payload) => {
  const name = payload?.actor_name;
  if (!name) return label;
  return `${label} — ${name}`;
};

const labelForFieldUpdated = (event) => {
  const fields = normalizeChangedFields(event.payload?.changed_fields);
  if (hasItineraryFields(fields) && !classifyChangedFields(fields).includes(HORAIRES_GROUP)) {
    return 'Parcours modifié';
  }
  if (classifyChangedFields(fields).includes(HORAIRES_GROUP) && !hasItineraryFields(fields)) {
    return 'Horaire modifié';
  }
  if (hasItineraryFields(fields)) return 'Parcours modifié';
  return null;
};

/**
 * Projette un événement API en entrée d'affichage métier.
 * @returns {{ category: string, label: string, timestamp: string, importance: string, source: string, type?: string, eventId?: number } | null}
 */
export function getTimelineDisplayEvent(event, context, sendAggregate) {
  if (!event?.created_at) return null;
  if (shouldHideApiEvent(event, context, sendAggregate)) return null;

  const type = event.event_type;
  const payload = event.payload || {};
  let label = null;
  let importance = 'primary';
  let category = 'request';

  if (type === 'request_created') {
    label = 'Demande créée';
  } else if (type === 'request_converted') {
    const name = resolveCarrierName(event, context);
    label = name ? `Réservation confirmée — ${name}` : 'Réservation confirmée';
  } else if (type === 'driver_assigned') {
    const driverName = payload.driver_name;
    if (!driverName) return null;
    label = `Chauffeur assigné — ${driverName}`;
    importance = 'secondary';
    category = 'assignment';
  } else if (type === 'route_legs_reorganized') {
    label = withActorName('Parcours modifié', payload);
    category = 'change';
  } else if (type === 'field_updated') {
    label = labelForFieldUpdated(event);
    if (!label) return null;
    label = withActorName(label, payload);
    category = 'change';
  } else if (type === 'cancelled') {
    label = payload.cancellation_display_label
      ? `Transport annulé — ${payload.cancellation_display_label}`
      : 'Transport annulé';
    label = withActorName(label, payload);
    category = 'cancellation';
  } else if (type === 'external_carrier_assigned') {
    const name = resolveExternalCarrierName(event, context);
    label = name ? `Transporteur externe affecté — ${name}` : 'Transporteur externe affecté';
    category = 'assignment';
  } else if (type === 'external_mission_completed') {
    const name = resolveExternalCarrierName(event, context);
    label = name ? `Mission terminée — ${name}` : 'Mission terminée — Transporteur externe';
    category = 'execution';
  } else if (type === 'external_carrier_switched') {
    const name = resolveExternalCarrierName(event, context);
    label = name ? `Transporteur externe affecté — ${name}` : 'Mission basculée vers transporteur externe';
    category = 'assignment';
  } else {
    return null;
  }

  return {
    category,
    label,
    timestamp: event.created_at,
    importance,
    source: 'api',
    type: type === 'cancelled' ? 'cancel' : undefined,
    eventId: event.id,
  };
}

/**
 * Libellés terrain enrichis depuis route_journey (métadonnées leg).
 * @param {object} ev — événement journey { type, date, leg_index, is_return, is_final_leg, leg_count }
 * @param {object} [request]
 */
export function formatRouteJourneyEvent(ev, request = null) {
  if (!ev?.type) return null;
  const isPickup = ev.type === 'pickup';
  const isDropoff = ev.type === 'dropoff';
  const isReturn = Boolean(ev.is_return);
  const legIndex = ev.leg_index;
  const isFinal = Boolean(ev.is_final_leg);
  const multi = (ev.leg_count || 0) > 1;
  const isRoundTrip = Boolean(request?.is_round_trip ?? request?.booking_summary?.is_round_trip);

  if (multi && legIndex != null) {
    if (isPickup) return { label: `Prise en charge — Étape ${legIndex}`, type: 'pickup' };
    if (isDropoff && isFinal) return { label: 'Transport terminé — Destination finale', type: 'dropoff' };
    if (isDropoff) return { label: `Dépose — Étape ${legIndex}`, type: 'dropoff' };
  }

  if (isReturn) {
    if (isPickup) return { label: 'Patient repris en charge — Retour', type: 'pickup' };
    if (isDropoff) return { label: 'Retour terminé — Institution', type: 'dropoff' };
  }

  if (isPickup) {
    const suffix = isRoundTrip === false ? 'Départ' : 'Aller';
    return { label: `Patient pris en charge — ${suffix}`, type: 'pickup' };
  }
  if (isDropoff) {
    return { label: 'Patient déposé — Destination', type: 'dropoff' };
  }

  return { label: ev.event || '', type: ev.type };
}

const enrichCancellationFromBooking = (items, bookingSummary) => {
  if (!bookingSummary) return items;
  const reason = bookingSummary.cancellation_display_label;
  const hasCancel = items.some((it) => it.type === 'cancel');
  if (hasCancel && reason) {
    return items.map((it) => {
      if (it.type !== 'cancel') return it;
      // Ne pas écraser un libellé déjà enrichi (motif et/ou auteur depuis l'API).
      if (it.label && it.label !== 'Transport annulé') return it;
      return { ...it, label: `Transport annulé — ${reason}` };
    });
  }
  if (!bookingSummary.cancelled_at) return items;
  const reasonLabel = bookingSummary.cancellation_display_label || '';
  let label = 'Transport annulé';
  if (reasonLabel) label += ` — ${reasonLabel}`;
  return [
    ...items,
    {
      label,
      timestamp: bookingSummary.cancelled_at,
      importance: 'primary',
      source: 'fallback',
      type: 'cancel',
      category: 'cancellation',
    },
  ];
};

const buildFallbackAdminEvents = (request) => {
  if (!request) return [];
  const items = [];
  const company = request.accepted_by_company?.name;

  if (request.sent_at) {
    const sentBy = request.sent_by_name || request.created_by_name;
    items.push({
      label: sentBy ? `Demande envoyée — ${sentBy}` : 'Demande envoyée',
      timestamp: request.sent_at,
      importance: 'primary',
      source: 'fallback',
      category: 'request',
    });
    if (request.created_at && !isSameTimelineMinute(request.created_at, request.sent_at)) {
      const creator = request.created_by_name;
      items.push({
        label: creator ? `Demande créée par ${creator}` : 'Demande créée',
        timestamp: request.created_at,
        importance: 'primary',
        source: 'fallback',
        category: 'request',
      });
    }
  } else if (request.created_at) {
    const creator = request.created_by_name;
    items.push({
      label: creator ? `Demande créée par ${creator}` : 'Demande créée',
      timestamp: request.created_at,
      importance: 'primary',
      source: 'fallback',
      category: 'request',
    });
  }
  if (request.converted_at && company) {
    items.push({
      label: `Réservation confirmée — ${company}`,
      timestamp: request.converted_at,
      importance: 'primary',
      source: 'fallback',
      category: 'request',
    });
  } else if (request.accepted_at && company) {
    items.push({
      label: `Réservation confirmée — ${company}`,
      timestamp: request.accepted_at,
      importance: 'primary',
      source: 'fallback',
      category: 'request',
    });
  }

  return items;
};

const toPanelItem = (entry) => ({
  event: entry.label,
  date: entry.timestamp,
  type: entry.type,
  eventId: entry.eventId,
  importance: entry.importance,
  source: entry.source,
});

const dedupeKey = (entry) => `${entry.label}|${entry.timestamp}`;

/**
 * Construit la chronologie opérationnelle affichée dans le panneau détail.
 * @param {{ apiEvents?: Array, request?: object, bookingSummary?: object }} params
 * @returns {Array<{ event: string, date: string, type?: string, eventId?: number }>}
 */
export function buildOperationalTimeline({ apiEvents = [], request = null, bookingSummary = null } = {}) {
  const bs = bookingSummary || request?.booking_summary || null;
  const effectiveRequest = request
    ? { ...request, booking_summary: bs || request.booking_summary }
    : (bs ? { booking_summary: bs } : null);
  const context = buildTimelineContext(apiEvents, effectiveRequest);
  const sendAggregate = buildSendAggregate(apiEvents);
  const entries = [];

  if (sendAggregate) {
    entries.push({
      category: 'request',
      label: sendAggregate.label,
      timestamp: sendAggregate.timestamp,
      importance: 'primary',
      source: 'api',
    });
  }

  (apiEvents || []).forEach((ev) => {
    const projected = getTimelineDisplayEvent(ev, context, sendAggregate);
    if (projected) entries.push(projected);
  });

  if (!apiEvents?.length) {
    buildFallbackAdminEvents(request).forEach((item) => entries.push(item));
  }

  const hasJourney = Array.isArray(bs?.route_journey) && bs.route_journey.length > 0;
  if (hasJourney) {
    bs.route_journey.forEach((jev) => {
      const formatted = formatRouteJourneyEvent(jev, request);
      if (!formatted?.label) return;
      entries.push({
        category: 'execution',
        label: formatted.label,
        timestamp: jev.date,
        importance: 'primary',
        source: 'journey',
        type: formatted.type,
      });
    });
  } else if (!apiEvents?.length && bs) {
    if (bs.boarded_at) {
      entries.push({
        category: 'execution',
        label: 'Patient pris en charge',
        timestamp: bs.boarded_at,
        importance: 'primary',
        source: 'fallback',
      });
    }
    if (bs.completed_at) {
      entries.push({
        category: 'execution',
        label: 'Transport terminé',
        timestamp: bs.completed_at,
        importance: 'primary',
        source: 'fallback',
      });
    }
  }

  let merged = enrichCancellationFromBooking(entries, bs);

  // Repli annulation : si la demande/booking est annulé mais qu'aucun événement
  // d'annulation n'est présent dans la timeline, on l'ajoute pour la traçabilité.
  const cancelledAt = request?.cancelled_at || bs?.cancelled_at || null;
  const isCancelledStatus = ['cancelled', 'canceled', 'annulee'].includes(
    String(request?.status || '').toLowerCase(),
  );
  if ((cancelledAt || isCancelledStatus) && !merged.some((it) => it.type === 'cancel')) {
    const reason = bs?.cancellation_display_label
      || request?.cancellation_display_label
      || request?.cancel_reason
      || null;
    merged.push({
      label: reason ? `Transport annulé — ${reason}` : 'Transport annulé',
      timestamp: cancelledAt || request?.updated_at,
      importance: 'primary',
      source: 'fallback',
      type: 'cancel',
      category: 'cancellation',
    });
  }

  const seen = new Set();
  const visible = merged.filter((entry) => {
    if (!entry.timestamp || !entry.label) return false;
    const key = dedupeKey(entry);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return visible
    .map(toPanelItem)
    .sort((a, b) => new Date(b.date) - new Date(a.date));
}

/** @deprecated Utiliser buildOperationalTimeline */
export function consolidateTimelineApiEvents(apiEvents, request = null) {
  return buildOperationalTimeline({ apiEvents, request, bookingSummary: request?.booking_summary });
}
