/**
 * Validation formulaire création demande institution.
 * Collecte toutes les erreurs (ordre du formulaire) pour affichage inline.
 */

import {
  MEDICAL_DESTINATION_OR_ERROR,
  findMissingMedicalDestinationDetails,
  isMedicalDestinationType,
} from './institutionDestinationDetails';
import { filterValidMultiStopDestinations } from './buildMultiStopLegsPreview';
import {
  MIN_ARRIVAL_LEAD_MINUTES,
  combineMissionDateTime,
  derivePickupTimeConfirmed,
  extractHHMM,
  isInstantBeforeLead,
  isInstantInPast,
  normalizeMissionDate,
} from './missionScheduleForm';

export const REQUIRED_FIELDS_TOAST = 'Veuillez compléter les champs obligatoires.';
export const MISSION_DATE_REQUIRED = 'Veuillez renseigner la date de mission.';
export const MISSION_DATE_INVALID = "Date de mission invalide pour l'heure de départ.";
export const PICKUP_ADDRESS_REQUIRED = "Veuillez renseigner l'adresse de départ.";
export const DROPOFF_ADDRESS_REQUIRED = "Veuillez renseigner l'adresse d'arrivée.";
export const PICKUP_TIME_PAST = 'Le départ ne peut pas être dans le passé.';
export const CONFIRMED_TIME_REQUIRED =
  'Confirmez au moins une heure.';
export const DELIVERY_DESCRIPTION_REQUIRED = 'Veuillez décrire le matériel à livrer.';
export const ASSISTANCE_NOTES_REQUIRED =
  "Décrivez le besoin d'assistance (Pathologie / Difficultés).";
export const EXTRA_STOP_ADDRESS_REQUIRED =
  "Veuillez renseigner l'adresse de cette destination.";
export const PATIENT_REQUIRED_FOR_DOMICILE =
  'Sélectionnez un patient pour renseigner le domicile.';

const leadHours = MIN_ARRIVAL_LEAD_MINUTES / 60;
export const APPOINTMENT_LEAD_REQUIRED =
  `Le rendez-vous doit être au minimum ${leadHours}h après l'heure actuelle.`;
export const RETURN_LEAD_REQUIRED =
  `Le retour doit être au minimum ${leadHours}h après l'heure actuelle.`;

export function formErrorId(fieldId) {
  return `${fieldId}-error`;
}

function extraStopNeedsAddress(stop) {
  if (!stop) return false;
  if (String(stop.dropoff_location || '').trim()) return false;
  return Boolean(
    isMedicalDestinationType(stop.destination_type)
    || String(stop.dropoff_service || '').trim()
    || String(stop.dropoff_doctor || '').trim()
    || String(stop.dropoff_establishment || '').trim()
    || String(stop.scheduled_time || '').trim(),
  );
}

function hasConfirmedTime(data, extraStops, returnEnabled) {
  if (derivePickupTimeConfirmed(data.pickup_time)) return true;
  if (data.dropoff_time?.trim()) return true;
  if (returnEnabled && data.return_time?.trim()) return true;
  return extraStops.some((s) => s.scheduled_time?.trim());
}

/**
 * @returns {Array<{ key: string, message: string, fieldId: string }>}
 */
export function collectInstitutionRequestFormErrors({
  formData,
  flushedSchedule = {},
  institutionAddress = '',
  isLirieSendMode = false,
  isExternalMode = false,
  billingHasBlockingError = false,
  externalCarrierError = null,
} = {}) {
  const errors = [];
  const push = (key, message, fieldId) => {
    errors.push({ key, message, fieldId });
  };

  const missionDate = normalizeMissionDate(
    flushedSchedule.mission_date
      || formData.mission_date
      || (formData.scheduled_time ? formData.scheduled_time.split('T')[0] : ''),
  );
  const pickupTime = flushedSchedule.pickup_time ?? formData.pickup_time;
  const dropoffTime = flushedSchedule.dropoff_time ?? formData.dropoff_time;
  const returnTime = flushedSchedule.return_time ?? formData.return_time;
  const merged = { ...formData, ...flushedSchedule, pickup_time: pickupTime, dropoff_time: dropoffTime, return_time: returnTime };

  const needsPatientForDomicile = formData.mission_type !== 'material_delivery'
    && (formData.pickup_type === 'domicile' || formData.dropoff_type === 'domicile')
    && !formData.patient_id;
  if (needsPatientForDomicile) {
    push('patient_id', PATIENT_REQUIRED_FOR_DOMICILE, 'patient-select');
  }

  if (!missionDate) {
    push('mission_date', MISSION_DATE_REQUIRED, 'mission_date');
  } else if (
    derivePickupTimeConfirmed(pickupTime)
    && !combineMissionDateTime(missionDate, pickupTime)
  ) {
    push('mission_date', MISSION_DATE_INVALID, 'mission_date');
  }

  const pickupIso = combineMissionDateTime(missionDate, pickupTime);
  if (pickupIso && isInstantInPast(pickupIso)) {
    push('pickup_time', PICKUP_TIME_PAST, 'pickup_time');
  }

  const dropoffIso = combineMissionDateTime(missionDate, dropoffTime);
  if (dropoffIso && isInstantBeforeLead(dropoffIso)) {
    push('dropoff_time', APPOINTMENT_LEAD_REQUIRED, 'dropoff_time');
  }

  const extraStops = formData.intermediate_stops || [];
  extraStops.forEach((stop, idx) => {
    if (extraStopNeedsAddress(stop)) {
      push(`extra_stop_location_${idx}`, EXTRA_STOP_ADDRESS_REQUIRED, `intermediate_stop_${idx}`);
    }
    const hhmm = extractHHMM(stop.scheduled_time)
      || stop.scheduled_time?.split('T')[1]?.slice(0, 5);
    const stopIso = combineMissionDateTime(missionDate, hhmm);
    if (stopIso && isInstantBeforeLead(stopIso)) {
      push(`extra_stop_time_${idx}`, APPOINTMENT_LEAD_REQUIRED, `intermediate_stop_time_${idx}`);
    }
  });

  const returnEnabled = formData.return_to_institution === true;
  if (returnEnabled) {
    const returnIso = combineMissionDateTime(missionDate, returnTime);
    if (returnIso && isInstantBeforeLead(returnIso)) {
      push('return_time', RETURN_LEAD_REQUIRED, 'return_time');
    }
  }

  const extraValid = filterValidMultiStopDestinations(extraStops);
  if (isLirieSendMode && !hasConfirmedTime(merged, extraValid, returnEnabled)) {
    push('confirmed_time', CONFIRMED_TIME_REQUIRED, 'confirmed_time');
  }

  const effectivePickup = formData.pickup_location
    || (formData.pickup_type === 'institution' ? institutionAddress : '');
  if (!String(effectivePickup || '').trim()) {
    push('pickup_location', PICKUP_ADDRESS_REQUIRED, 'pickup_location');
  }

  const effectiveDropoff = formData.dropoff_location
    || (formData.dropoff_type === 'institution' ? institutionAddress : '');
  if (!String(effectiveDropoff || '').trim()) {
    push('dropoff_location', DROPOFF_ADDRESS_REQUIRED, 'dropoff_location');
  }

  const medicalGaps = findMissingMedicalDestinationDetails({
    missionType: formData.mission_type,
    destinationType: formData.destination_type,
    dropoffService: formData.dropoff_service,
    dropoffDoctor: formData.dropoff_doctor,
    extraStops,
  });
  if (medicalGaps.principal) {
    push('medical_principal', MEDICAL_DESTINATION_OR_ERROR, 'dropoff_service');
  }
  medicalGaps.extraStopIndexes.forEach((idx) => {
    push(`medical_extra_${idx}`, MEDICAL_DESTINATION_OR_ERROR, `stop_service_${idx}`);
  });

  if (formData.mission_type === 'material_delivery' && !String(formData.delivery_description || '').trim()) {
    push('delivery_description', DELIVERY_DESCRIPTION_REQUIRED, 'delivery_description');
  }
  if (formData.requires_assistance && !String(formData.notes || '').trim()) {
    push('notes', ASSISTANCE_NOTES_REQUIRED, 'patient_notes');
  }
  if (isLirieSendMode && billingHasBlockingError) {
    push('billing', 'Corrigez les problèmes de facturation avant d\'envoyer la demande.', 'billing_intent');
  }
  if (isExternalMode && externalCarrierError) {
    push('external_carrier', externalCarrierError, 'external-carrier');
  }

  return errors;
}

export function fieldErrorsMap(errors) {
  return (errors || []).reduce((acc, err) => {
    acc[err.key] = err.message;
    return acc;
  }, {});
}

export function scrollToFirstFormError(errors, focusByFieldId = {}) {
  const first = errors?.[0];
  if (!first) return;
  requestAnimationFrame(() => {
    const el = typeof document !== 'undefined' ? document.getElementById(first.fieldId) : null;
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const customFocus = focusByFieldId[first.fieldId];
    if (typeof customFocus === 'function') {
      customFocus();
      return;
    }
    el?.focus?.({ preventScroll: true });
  });
}
