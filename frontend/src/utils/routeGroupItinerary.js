import { hasScheduledPickupTime } from './bookingScheduling';

function legSequence(leg) {
  return Number(leg?.sequence ?? leg?.route_sequence_number) || 0;
}

function isReturnLeg(leg) {
  return Boolean(leg?.is_return || leg?.trip_flags?.return_leg);
}

function appointmentClock(leg) {
  const raw = String(leg?.appointment_time || leg?.scheduling?.appointment_time || '').trim();
  const match = raw.match(/(\d{2}):(\d{2})/);
  return match ? `${match[1]}:${match[2]}` : '';
}

function clockOf(leg) {
  const time = String(leg?.display_time || leg?.scheduling?.display_time || '').trim();
  if (!time || time === 'À définir' || time === 'À confirmer') return '';
  return time.replace(/\s*\(non confirmé\)\s*$/i, '').trim();
}

function legHasTime(leg) {
  if (leg?.time_scheduled === true) return true;
  if (leg?.time_scheduled === false) return false;
  return hasScheduledPickupTime(leg);
}

function blankMedical(value) {
  const text = String(value || '').trim();
  if (!text || /^non sp[eé]cifi[eé]$/i.test(text)) return '';
  return text;
}

/** Demande en attente dont l'heure de prise en charge n'est pas encore confirmée. */
export function pickupNeedsCompanyConfirmation(booking) {
  if (!booking || booking.__institutionOffer) return false;
  if (String(booking.status || '').toLowerCase() !== 'pending') return false;
  if (isReturnLeg(booking)) return false;
  if (booking.scheduling?.time_defined === false) return true;
  return booking.time_confirmed === false;
}

/** Étapes ordonnées : résumé embarqué, sinon les lignes déjà chargées. */
export function orderedRouteLegs(booking, allReservations) {
  const embedded = Array.isArray(booking?.route_group_legs) ? booking.route_group_legs : [];
  if (embedded.length >= 2) {
    return [...embedded].sort((left, right) => legSequence(left) - legSequence(right));
  }
  const groupId = booking?.route_group_id;
  if (!groupId || !Array.isArray(allReservations)) return [];
  const members = allReservations
    .filter((row) => row && !row.__institutionOffer && row.route_group_id === groupId)
    .sort((left, right) => legSequence(left) - legSequence(right));
  return members.length >= 2 ? members : [];
}

export function journeyPlaces(booking, allReservations) {
  const fallback = [booking?.pickup_location, booking?.dropoff_location]
    .map((place) => String(place || '').trim())
    .filter(Boolean);
  const members = orderedRouteLegs(booking, allReservations);
  if (members.length < 2) return fallback;
  const places = [];
  const push = (place) => {
    const text = String(place || '').trim();
    if (!text || places[places.length - 1] === text) return;
    places.push(text);
  };
  members.forEach((leg, index) => {
    if (index === 0) push(leg.pickup_location);
    push(leg.dropoff_location);
  });
  return places.length >= 2 ? places : fallback;
}

export function otherJourneyScheduleLines(booking, allReservations) {
  const members = orderedRouteLegs(booking, allReservations);
  if (members.length < 2) return [];
  const lines = [];
  const ownAppointment = appointmentClock(booking);
  if (ownAppointment) {
    lines.push({ key: 'own-appointment', text: `Rendez-vous ${ownAppointment}` });
  }
  members.forEach((leg) => {
    if (Number(leg?.id) === Number(booking?.id)) return;
    if (!legHasTime(leg)) return;
    const time = clockOf(leg);
    if (!time) return;
    const kind = isReturnLeg(leg) ? 'Retour' : leg.time_confirmed === false ? 'Rendez-vous' : 'Départ';
    const text = `${kind} ${time}`;
    if (lines.some((line) => line.text === text)) return;
    lines.push({ key: `sched-${leg.id}`, text });
  });
  return lines;
}

/**
 * Points du panneau détail : prise en charge, chaque étape, retour.
 * L'heure d'un rendez-vous est portée par l'arrivée, pas par la prise en charge.
 */
export function routeTimelinePoints(booking, allReservations) {
  const members = orderedRouteLegs(booking, allReservations);
  if (members.length < 2) return null;
  const outbound = members.filter((leg) => !isReturnLeg(leg));
  const first = outbound[0] || members[0];
  const firstClock = legHasTime(first) ? clockOf(first) : '';
  const firstIsAppointment = first?.time_confirmed === false && Boolean(firstClock);
  const points = [
    {
      key: `pickup-${first?.id || 'start'}`,
      label: 'Prise en charge',
      timeLabel: firstIsAppointment || !firstClock ? 'À déterminer' : `Départ ${firstClock}`,
      address: first?.pickup_location || '',
      details: '',
    },
  ];
  outbound.forEach((leg, index) => {
    const clock = legHasTime(leg) ? clockOf(leg) : '';
    const keptAppointment = appointmentClock(leg);
    const timeLabel = keptAppointment && leg.time_confirmed !== false
      ? `RDV ${keptAppointment}`
      : !clock
        ? ''
        : leg.time_confirmed === false
          ? `RDV ${clock}`
          : `Départ ${clock}`;
    const details = [blankMedical(leg.hospital_service), blankMedical(leg.doctor_name)]
      .filter(Boolean)
      .join(' · ');
    points.push({
      key: `step-${leg.id || index}`,
      label: outbound.length > 1 ? `Étape ${index + 1}` : 'Destination',
      timeLabel,
      address: leg.dropoff_location || '',
      details,
    });
  });
  const returnLeg = members.find((leg) => isReturnLeg(leg));
  if (returnLeg) {
    const clock = legHasTime(returnLeg) ? clockOf(returnLeg) : '';
    points.push({
      key: `return-${returnLeg.id || 'back'}`,
      label: 'Retour',
      timeLabel: clock ? `Départ ${clock}` : 'Heure à définir',
      address: returnLeg.dropoff_location || first?.pickup_location || '',
      details: '',
    });
  }
  return points;
}
