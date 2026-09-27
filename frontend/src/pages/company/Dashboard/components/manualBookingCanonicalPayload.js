/** Construit le corps canonique d'une réservation entreprise. */

const ensureSeconds = (value) => {
  if (!value) return null;
  const text = String(value).trim();
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(text)) return `${text}:00`;
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(text)) return text;
  return null;
};

export const combineDateAndTime = (date, time) => {
  if (!date || !time) return null;
  return ensureSeconds(`${date}T${time}`);
};

const stepLocation = (location, coords) => ({
  location: String(location || '').trim(),
  latitude: coords?.lat ?? null,
  longitude: coords?.lon ?? null,
});

/**
 * @returns {{ error: string | null, payload: object | null }}
 */
export function buildCanonicalReservationPayload(input) {
  const pickupAt = ensureSeconds(input.scheduledTime);
  if (!pickupAt) {
    return { error: 'Veuillez sélectionner la date et l\'heure de départ.', payload: null };
  }
  if (!String(input.pickupLocation || '').trim() || !String(input.dropoffLocation || '').trim()) {
    return { error: 'Le départ et la destination sont obligatoires.', payload: null };
  }

  const extras = Array.isArray(input.extraStops) ? input.extraStops : [];
  const material = Boolean(input.isMaterialDelivery);
  const roundTrip = Boolean(input.isRoundTrip);
  const destinationKind = material ? null : input.destinationKind || 'other';

  const finalArrival = ensureSeconds(input.destinationArrival) || pickupAt;
  // Heure de retour vide = « À définir ». Ne pas recopier l'heure de l'aller.
  const explicitReturnDeparture = roundTrip
    ? ensureSeconds(input.destinationDeparture)
    : null;
  const finalDeparture = explicitReturnDeparture;
  const returnArrival = roundTrip
    ? ensureSeconds(input.returnArrival) || explicitReturnDeparture
    : null;

  const steps = [
    {
      position: 0,
      kind: 'pickup',
      ...stepLocation(input.pickupLocation, input.pickupCoords),
      arrival_at: null,
      departure_at: pickupAt,
      access_notes: input.pickupAccessNotes || null,
    },
  ];

  extras.forEach((stop) => {
    const departure = ensureSeconds(stop.departure);
    const arrival = ensureSeconds(stop.arrival) || departure;
    if (!String(stop.location || '').trim() || !departure) {
      steps.push(null);
      return;
    }
    steps.push({
      kind: 'destination',
      ...stepLocation(stop.location, stop.coords),
      arrival_at: arrival,
      departure_at: departure,
      destination_kind: material ? null : stop.destinationKind || 'other',
      establishment: material ? null : stop.establishment || null,
      service: material ? null : stop.service || null,
      doctor: material ? null : stop.doctor || null,
      access_notes: stop.accessNotes || null,
    });
  });
  if (steps.includes(null)) {
    return {
      error: 'Chaque étape intermédiaire a une adresse et une heure de départ.',
      payload: null,
    };
  }

  steps.push({
    kind: 'destination',
    ...stepLocation(input.dropoffLocation, input.dropoffCoords),
    arrival_at: finalArrival,
    departure_at: finalDeparture,
    destination_kind: destinationKind,
    establishment: input.establishment || null,
    service: input.service || null,
    doctor: input.doctor || null,
    access_notes: input.dropoffAccessNotes || null,
  });

  if (roundTrip) {
    steps.push({
      kind: 'return',
      location: String(input.pickupLocation).trim(),
      latitude: input.pickupCoords?.lat ?? null,
      longitude: input.pickupCoords?.lon ?? null,
      arrival_at: returnArrival,
      departure_at: null,
      access_notes: input.pickupAccessNotes || null,
    });
  }

  steps.forEach((step, index) => {
    step.position = index;
  });

  const segmentCount = steps.length - 1;
  const mode = input.pricingMode;
  if (!['automatic', 'manual', 'preferential'].includes(mode)) {
    return { error: 'Choisissez un mode de tarification.', payload: null };
  }

  const payload = {
    client_id: input.clientId,
    route_steps: steps,
    pricing_mode: mode,
    idempotency_key: input.idempotencyKey,
    ...(String(input.passengerName || '').trim()
      ? { passenger_name: String(input.passengerName).trim() }
      : {}),
    requester_name: input.requesterName || null,
    requester_phone: input.requesterPhone || null,
    is_urgent: Boolean(input.isUrgent),
    needs_assistance: Boolean(input.needsAssistance),
    mission_type: material ? 'material_delivery' : 'patient_transport',
    wheelchair_client_has: Boolean(input.wheelchairClientHas) || undefined,
    wheelchair_need: Boolean(input.wheelchairNeed) || undefined,
    notes_medical: input.notesMedical || undefined,
    ...(input.billToPatient ? { bill_to_patient: true } : {}),
  };

  if (material) {
    const description = String(input.deliveryDescription || '').trim();
    if (!description) {
      return { error: 'Veuillez saisir la description de la livraison.', payload: null };
    }
    payload.delivery_description = description;
  }

  if (mode === 'manual') {
    const amounts = input.manualAmounts || [];
    if (amounts.length !== segmentCount || amounts.some((value) => !(Number(value) > 0))) {
      return { error: 'Chaque tronçon doit avoir un montant.', payload: null };
    }
    payload.segment_amounts = amounts.map((value, index) => ({
      from_position: index,
      to_position: index + 1,
      amount: Number(value).toFixed(2),
    }));
  }

  if (mode === 'preferential') {
    const forfait = Number(input.preferentialAmount);
    if (!(forfait > 0)) {
      return { error: 'Le forfait doit être strictement positif.', payload: null };
    }
    payload.preferential_amount = forfait.toFixed(2);
  }

  if (input.isRecurring) {
    payload.is_recurring = true;
    payload.recurrence_type = input.recurrenceType || 'weekly';
    payload.occurrences = input.occurrences;
    if (input.recurrenceType === 'custom') {
      payload.recurrence_days = input.recurrenceDays || [];
    }
    if (input.recurrenceEndDate) {
      payload.recurrence_end_date = input.recurrenceEndDate;
    }
  }

  return { error: null, payload };
}
