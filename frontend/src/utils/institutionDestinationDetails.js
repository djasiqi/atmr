/**
 * Destination institution : service OU médecin uniquement si type médical.
 * Ne pas poser `required` HTML sur les deux champs (validation croisée).
 */

export const MEDICAL_DESTINATION_OR_ERROR =
  'Veuillez renseigner au moins le service ou le médecin.';

export const MEDICAL_DESTINATION_OR_HINT =
  'Renseignez au moins le service ou le médecin.';

export const DESTINATION_TYPE_MEDICAL = 'medical';
export const DESTINATION_TYPE_OTHER = 'other';

const EXPLICIT_DESTINATION_TYPES = new Set([
  DESTINATION_TYPE_MEDICAL,
  DESTINATION_TYPE_OTHER,
  'domicile',
  'institution',
]);

const MEDICAL_PLACE_TYPES = new Set([
  'hospital',
  'doctor',
  'health',
  'dentist',
  'pharmacy',
  'physiotherapist',
  'medical_lab',
  'clinic',
]);

export function hasServiceOrDoctor(service, doctor) {
  return Boolean(String(service || '').trim() || String(doctor || '').trim());
}

export function isMedicalDestinationType(destinationType) {
  return destinationType === DESTINATION_TYPE_MEDICAL;
}

/**
 * Suggestion optionnelle depuis les types d'un POI.
 * Médical uniquement via types hospital/clinic/etc.
 * Un POI typé non médical (restaurant, hôtel…) bascule en Autre lieu.
 * Sans types : pas d'inférence (le défaut UI reste médical).
 */
export function suggestDestinationTypeFromPlace(item) {
  const types = item?.types;
  if (!Array.isArray(types)) return null;
  if (types.some((t) => MEDICAL_PLACE_TYPES.has(String(t)))) {
    return DESTINATION_TYPE_MEDICAL;
  }
  return DESTINATION_TYPE_OTHER;
}

/**
 * Hydratation édition : type explicite, sinon service/médecin déjà remplis.
 * Un type omis / other n'est jamais médical.
 */
export function inferDestinationType({
  destinationType,
  service,
  doctor,
} = {}) {
  const explicit = String(destinationType || '').trim();
  if (EXPLICIT_DESTINATION_TYPES.has(explicit)) {
    return explicit;
  }
  return hasServiceOrDoctor(service, doctor)
    ? DESTINATION_TYPE_MEDICAL
    : DESTINATION_TYPE_OTHER;
}

function stopAddress(stop) {
  return String(stop?.dropoff_location || stop?.address || '').trim();
}

function stopService(stop) {
  return stop?.dropoff_service || stop?.service;
}

function stopDoctor(stop) {
  return stop?.dropoff_doctor || stop?.doctor;
}

function stopDestinationType(stop) {
  return stop?.destination_type || stop?.destinationType;
}

/**
 * Destinations médicales incomplètes (service et médecin vides).
 * Les étapes sans adresse sont ignorées (non soumises).
 *
 * @returns {{ principal: boolean, extraStopIndexes: number[] }}
 */
export function findMissingMedicalDestinationDetails({
  missionType,
  destinationType,
  dropoffService,
  dropoffDoctor,
  extraStops = [],
} = {}) {
  if (missionType === 'material_delivery') {
    return { principal: false, extraStopIndexes: [] };
  }

  const extraStopIndexes = (extraStops || [])
    .map((stop, idx) => {
      if (!stopAddress(stop)) return -1;
      if (!isMedicalDestinationType(stopDestinationType(stop))) return -1;
      if (hasServiceOrDoctor(stopService(stop), stopDoctor(stop))) return -1;
      return idx;
    })
    .filter((idx) => idx >= 0);

  const principal = isMedicalDestinationType(destinationType)
    && !hasServiceOrDoctor(dropoffService, dropoffDoctor);

  return { principal, extraStopIndexes };
}

export function hasMissingMedicalDestinationDetails(params) {
  const found = findMissingMedicalDestinationDetails(params);
  return found.principal || found.extraStopIndexes.length > 0;
}
