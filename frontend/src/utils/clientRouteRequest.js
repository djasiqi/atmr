/** Montant technique des étapes intermédiaires : le tarif client est sur l'aller et le retour. */
const INTERMEDIATE_LEG_PLACEHOLDER_CHF = 0.5;
/** Plancher indicatif d'un transport (CHF), aligné sur la confirmation de demande. */
const MIN_TRANSPORT_FARE_CHF = 45;

function roundChfToFiveRappen(value) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return 0;
  return Math.round((amount + Number.EPSILON) * 20) / 20;
}

function blankMedical(value) {
  const text = String(value || '').trim();
  if (!text || text === 'Non spécifié' || text === 'Aucune note') return '';
  return text;
}

function formatRequestDay(iso) {
  const parsed = Date.parse(iso);
  if (!Number.isFinite(parsed)) return '';
  return new Date(parsed).toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
}

function formatRequestStamp(prefix, iso, { withTime = true } = {}) {
  const parsed = Date.parse(iso);
  if (!Number.isFinite(parsed)) return prefix;
  const date = new Date(parsed);
  const datePart = date.toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
  if (!withTime) return `${prefix} · ${datePart}`;
  const timePart = date.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
  return `${prefix} · ${datePart} · ${timePart}`;
}

function mapPoint(lat, lon) {
  const latitude = Number(lat);
  const longitude = Number(lon);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
  if (latitude === 0 && longitude === 0) return null;
  return { lat: latitude, lng: longitude };
}

function sequenceOf(booking) {
  const sequence = Number(booking?.route_sequence_number);
  return Number.isFinite(sequence) ? sequence : 0;
}

function agreedCarrierTotalChf(legs) {
  const total = legs.reduce((sum, leg) => {
    const amount = Number(leg?.amount);
    if (!Number.isFinite(amount) || amount <= 0) return sum;
    if (Math.abs(amount - INTERMEDIATE_LEG_PLACEHOLDER_CHF) < 0.001) return sum;
    return sum + amount;
  }, 0);
  return roundChfToFiveRappen(total);
}

function maximumAuthorizedChf(legs, transportCount) {
  const storedUnits = legs
    .filter((leg) => leg?.is_return || Number(leg?.route_sequence_number) === 1)
    .map((leg) => Number(leg?.amount))
    .filter(
      (amount) =>
        Number.isFinite(amount) &&
        amount > 0 &&
        Math.abs(amount - INTERMEDIATE_LEG_PLACEHOLDER_CHF) > 0.001
    );
  const storedUnit = storedUnits.length ? Math.max(...storedUnits) : 0;
  const unit = Math.max(storedUnit, MIN_TRANSPORT_FARE_CHF);
  return roundChfToFiveRappen(unit * transportCount);
}

function foldOneGroup(legs) {
  const ordered = [...legs].sort((left, right) => {
    const gap = sequenceOf(left) - sequenceOf(right);
    if (gap !== 0) return gap;
    return Number(left?.id || 0) - Number(right?.id || 0);
  });
  const anchor = ordered.find((leg) => !leg?.is_return) || ordered[0];
  if (!anchor || ordered.length < 2) return anchor || null;

  const outbound = ordered.filter((leg) => !leg?.is_return);
  const returnLeg = ordered.find((leg) => leg?.is_return) || null;
  const first = outbound[0] || anchor;
  const daySource = first?.scheduled_time || returnLeg?.scheduled_time || '';
    const stops = [
    {
      key: `pickup-${first.id}`,
      label: 'Prise en charge',
      place: first.pickup_location || '',
      detail: '',
      point: mapPoint(first.pickup_lat, first.pickup_lon),
      boarded_at: first.boarded_at || null,
      completed_at: null,
      when: first.time_confirmed
        ? formatRequestStamp('Prise en charge', first.scheduled_time)
        : 'Prise en charge à déterminer',
    },
  ];

  outbound.forEach((leg, index) => {
    const detail = [blankMedical(leg.hospital_service), blankMedical(leg.doctor_name)]
      .filter(Boolean)
      .join(' · ');
    const when = leg.scheduled_time
      ? formatRequestStamp(leg.time_confirmed ? 'Heure de départ' : 'Rendez-vous', leg.scheduled_time)
      : '';
    stops.push({
      key: `step-${leg.id}`,
      label: outbound.length > 1 ? `Étape ${index + 1}` : 'Destination',
      place: leg.dropoff_location || '',
      detail,
      point: mapPoint(leg.dropoff_lat, leg.dropoff_lon),
      boarded_at: null,
      completed_at: leg.completed_at || null,
      when,
    });
  });

  if (returnLeg) {
    const day = formatRequestDay(returnLeg.scheduled_time || daySource);
    stops.push({
      key: `return-${returnLeg.id}`,
      label: 'Retour',
      place: returnLeg.dropoff_location || first.pickup_location || '',
      detail: '',
      point: mapPoint(returnLeg.dropoff_lat, returnLeg.dropoff_lon),
      boarded_at: null,
      completed_at: returnLeg.completed_at || null,
      when:
        returnLeg.scheduled_time && returnLeg.time_confirmed
          ? formatRequestStamp('Heure de départ', returnLeg.scheduled_time)
          : day
            ? `Heure à définir · ${day}`
            : 'Heure à définir',
    });
  }

  // Avant acceptation : plafond client. Ensuite : somme du tarif entreprise par trajet.
  const transportCount = outbound.length + (returnLeg ? 1 : 0);
  const requestAmount = anchor.company_id
    ? agreedCarrierTotalChf(ordered)
    : maximumAuthorizedChf(ordered, transportCount);
  const latestMs = ordered.reduce((best, leg) => {
    const parsed = Date.parse(leg?.scheduled_time);
    return Number.isFinite(parsed) && parsed > best ? parsed : best;
  }, Number.NEGATIVE_INFINITY);

  return {
    ...anchor,
    route_request_latest_time: Number.isFinite(latestMs)
      ? new Date(latestMs).toISOString()
      : anchor.scheduled_time,
    route_request_stops: stops,
    route_request_transport_count: outbound.length + (returnLeg ? 1 : 0),
    route_request_amount: requestAmount,
    route_request_ids: ordered.map((leg) => leg.id),
  };
}

/**
 * Une demande multi-transports (route_group_id) devient une seule ligne.
 * Les retours simples déjà imbriqués dans return_booking restent masqués.
 */
export function foldClientRouteRequests(bookings) {
  const list = Array.isArray(bookings) ? bookings : [];
  const hideReturnIds = new Set();
  for (const booking of list) {
    const returnId = Number(booking?.return_booking?.id);
    if (Number.isFinite(returnId) && returnId > 0) hideReturnIds.add(returnId);
  }

  const groups = new Map();
  const order = [];
  for (const booking of list) {
    const groupId = String(booking?.route_group_id || '').trim();
    if (!groupId) {
      if (!hideReturnIds.has(Number(booking?.id))) order.push(booking);
      continue;
    }
    if (!groups.has(groupId)) {
      groups.set(groupId, []);
      order.push({ __group: groupId });
    }
    groups.get(groupId).push(booking);
  }

  return order
    .map((item) => (item?.__group ? foldOneGroup(groups.get(item.__group) || []) : item))
    .filter(Boolean);
}
