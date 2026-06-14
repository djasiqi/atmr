/**
 * Filtre les étapes multi-stop avec une destination non vide.
 * @param {Array<{ dropoff_location?: string, scheduled_time?: string }>} intermediateStops
 */
export function filterValidMultiStopDestinations(intermediateStops) {
  return (intermediateStops || []).filter((stop) => stop?.dropoff_location?.trim());
}

/**
 * Construit la prévisualisation ordonnée des legs multi-stop.
 * Source unique pour le résumé UI, le payload et les tests.
 * Reproduit build_legs_chain backend : origine → étapes → retour institution.
 *
 * @param {{
 *   pickupLocation?: string,
 *   validStops?: Array<{ dropoff_location: string, scheduled_time?: string }>,
 *   returnToInstitution?: boolean,
 * }} params
 * @returns {Array<{ sequence: number, from: string, to: string, isReturn?: boolean }>}
 */
export function buildMultiStopLegsPreview({
  pickupLocation,
  validStops,
  returnToInstitution = true,
}) {
  const origin = (pickupLocation || '').trim();
  const stops = (validStops || []).filter((stop) => stop?.dropoff_location?.trim());
  const legs = [];
  let currentFrom = origin;

  for (let i = 0; i < stops.length; i += 1) {
    const to = stops[i].dropoff_location.trim();
    legs.push({
      sequence: i + 1,
      from: currentFrom,
      to,
      isReturn: false,
    });
    currentFrom = to;
  }

  if (returnToInstitution !== false && origin) {
    legs.push({
      sequence: stops.length + 1,
      from: currentFrom,
      to: origin,
      isReturn: true,
    });
  }

  return legs;
}

/**
 * Construit le payload intermediate_stops à partir des étapes valides.
 * @param {Array<{ dropoff_location: string, scheduled_time?: string }>} validStops
 */
export function buildMultiStopPayloadStops(validStops) {
  return validStops.map((stop, index) => {
    const entry = {
      sequence: index + 1,
      dropoff_location: stop.dropoff_location.trim(),
    };
    const raw = stop.scheduled_time?.trim();
    if (raw) {
      const parsed = new Date(raw);
      if (!Number.isNaN(parsed.getTime())) {
        entry.scheduled_time = parsed.toISOString();
      }
    }
    if (typeof stop.time_confirmed === 'boolean') {
      entry.time_confirmed = stop.time_confirmed;
    }
    const establishment = stop.dropoff_establishment?.trim();
    if (establishment) entry.dropoff_establishment = establishment;
    const service = stop.dropoff_service?.trim();
    if (service) entry.dropoff_service = service;
    const doctor = stop.dropoff_doctor?.trim();
    if (doctor) entry.dropoff_doctor = doctor;
    return entry;
  });
}
