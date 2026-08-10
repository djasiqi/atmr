/**
 * Initialisation parcours institution (demande / booking opérationnel).
 */

/** Types soins à domicile : départ au domicile patient (IMAD, curatelle). */
export const HOME_CARE_INSTITUTION_TYPES = new Set(['imad', 'curatelle']);

export const TRIP_TYPE_DOM_TO_DEST = 'dom_to_dest';

/** « Domicile → Dest. » réservé aux institutions de soins à domicile. */
export const institutionSupportsDomicilePickupTrip = (institutionType) =>
  HOME_CARE_INSTITUTION_TYPES.has(String(institutionType || '').toLowerCase());

export const filterTripTypesForInstitution = (tripTypes, institutionType) => {
  if (institutionSupportsDomicilePickupTrip(institutionType)) {
    return tripTypes;
  }
  return tripTypes.filter((tt) => tt.value !== TRIP_TYPE_DOM_TO_DEST);
};

const textOrEmpty = (value) => (value == null ? '' : String(value));

export const routingDropoffDetails = (request) => {
  const routing = request?.billing_details?.routing || {};
  return {
    establishment: routing.dropoff_establishment || '',
    service: routing.dropoff_service || '',
    doctor: routing.dropoff_doctor || '',
  };
};

/** Destinations (hors retour A/R) depuis les legs ou champs legacy. */
export const buildInitialDestinations = (request, bookingSummary = null) => {
  const bs = bookingSummary || request?.booking_summary || {};
  const legs = Array.isArray(request?.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];
  const hasReturn = Boolean(request?.return_to_institution);
  const routingDetails = routingDropoffDetails(request);

  if (legs.length > 0) {
    const destLegs = hasReturn ? legs.slice(0, -1) : legs;
    return destLegs.map((leg) => ({
      address: leg.dropoff_location || '',
      establishment: leg.dropoff_establishment || textOrEmpty(bs.medical_facility) || '',
      service: leg.dropoff_service || textOrEmpty(bs.hospital_service) || '',
      doctor: leg.dropoff_doctor || textOrEmpty(bs.doctor_name) || '',
      scheduled_time: leg.scheduled_time || '',
      time_confirmed: Boolean(leg.time_confirmed),
      booking_id: leg.booking_id || null,
      leg_index: leg.sequence_index ?? null,
      use_custom_billing: Boolean(leg.destination_billing_override),
      destination_billing_override: leg.destination_billing_override || 'patient',
    }));
  }

  const arrivalOnRequest = request?.scheduled_time_type === 'arrival';
  return [{
    address: textOrEmpty(bs.dropoff_location || request?.dropoff_location),
    establishment: textOrEmpty(bs.medical_facility) || routingDetails.establishment,
    service: textOrEmpty(bs.hospital_service) || routingDetails.service,
    doctor: textOrEmpty(bs.doctor_name) || routingDetails.doctor,
    scheduled_time: arrivalOnRequest ? (request?.scheduled_time || '') : '',
    time_confirmed: arrivalOnRequest && Boolean(request?.scheduled_time),
    booking_id: bs.id || request?.booking_id || null,
    leg_index: 0,
    use_custom_billing: false,
    destination_billing_override: 'patient',
  }];
};

/** Override facturation du leg retour (A/R) depuis les legs existants. */
export const buildInitialReturnBilling = (request) => {
  const legs = Array.isArray(request?.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : [];
  if (!request?.return_to_institution || legs.length === 0) {
    return { use_custom_billing: false, destination_billing_override: 'patient' };
  }
  const returnLeg = legs.find((leg) => leg.is_return_stop) || legs[legs.length - 1];
  return {
    use_custom_billing: Boolean(returnLeg?.destination_billing_override),
    destination_billing_override: returnLeg?.destination_billing_override || 'patient',
  };
};

/**
 * Mapping destinations UI → intermediate_stops API (préserve la facturation spécifique).
 */
export const mapDestinationsToIntermediateStops = (destinations) =>
  (destinations || []).map((d, i) => {
    const entry = {
      dropoff_location: d.address,
      sequence: i,
      time_confirmed: Boolean(d.time_confirmed),
    };
    if (d.scheduled_time) entry.scheduled_time = d.scheduled_time;
    if (d.establishment) entry.dropoff_establishment = d.establishment;
    if (d.service) entry.dropoff_service = d.service;
    if (d.doctor) entry.dropoff_doctor = d.doctor;
    if (d.use_custom_billing) {
      entry.use_custom_billing = true;
      entry.destination_billing_override = d.destination_billing_override || 'patient';
    }
    return entry;
  });

export const DOCTOR_NAME_PATTERN = /^(dr\.?|prof\.?|méd\.?|med\.?|docteur|professeur)\s/i;

export const extractAddressFromPlace = (item) =>
  item?.label || item?.address || item?.formatted_address || item?.description || '';

export const extractPlaceDetails = (item) => {
  const placeName = item?.name || '';
  const details = { establishment: '', doctor: '' };
  if (placeName && placeName !== item?.address) {
    if (DOCTOR_NAME_PATTERN.test(placeName)) {
      details.doctor = placeName;
    } else {
      details.establishment = placeName;
    }
  }
  return details;
};
